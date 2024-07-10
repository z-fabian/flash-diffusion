import torch
from ldm.ldm.modules.diffusionmodules.model import Encoder
from scripts.utils import load_config_from_yaml
from data_utils.operators import create_operator
from abc import ABC, abstractmethod


class SevEncoder(ABC):
# Subclass SevEncoder to add a custom severity encoder.
# See below the functions that have to be implemented.

    @abstractmethod
    def __call__(self, y):
        # Takes degraded image y and returns predicted latent (mean, \hat{z}) and severity (var, \hat{\sigma}^2)
        pass
    # return mean, var 

    @abstractmethod
    def get_fwd_operator(self):
        # Returns the forward operator used to train the model
        pass

    @staticmethod
    @abstractmethod
    def load_pretrained(self, ckpt_path):
        # Loads pretrained model from checkpoint.
        pass
    # return model


class LDMSevEncoder(ABC, torch.nn.Module):
    def __init__(
            self,
            config,
        ):
        torch.nn.Module.__init__(self)
        ddconfig = config['params']['ddconfig']
        self.embed_dim = config['params']['embed_dim']
        self.encoder = Encoder(**ddconfig)
        ch_mult = 2 if ddconfig["double_z"] else 1
        self.quant_conv = torch.nn.Conv2d(ch_mult * ddconfig["z_channels"], ch_mult * self.embed_dim, 1)
        self.encoder_type = config['target']
        self.sigma_max = 1.0
        if self.encoder_type in ['ldm.models.autoencoder.VQModel', 'ldm.models.autoencoder.VQModelInterface']:
            # need to map latent to variances
            self.std_conv = torch.nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.num_resolutions = self.encoder.num_resolutions
        self.fwd_operator = None
        
    def cov_to_var(self, cov):
        b = cov.shape[0]
        return cov.view(b, -1).mean(dim=1)

    def get_embedding(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        if self.encoder_type == 'ldm.models.autoencoder.AutoencoderKL':
            mean, std = torch.chunk(z, 2, dim=1)            
        elif self.encoder_type in ['ldm.models.autoencoder.VQModel', 'ldm.models.autoencoder.VQModelInterface']:
            mean, std = z, self.std_conv(z)
        var = std.pow(2)    
        var_single = self.cov_to_var(var)
        return mean, var_single
    
    def __call__(self, x):
        mean, var = self.get_embedding(x) 
        return mean, var    
    
    def get_fwd_operator(self):
        return self.fwd_operator

    @staticmethod
    def load_pretrained(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # Check checkpoint type
        if 'sev_encoder_config_path' in checkpoint['hyper_parameters']:
            loaded_config = load_config_from_yaml(checkpoint['hyper_parameters']['ldm_model_config_path'])
            model = LDMSevEncoder(loaded_config['model']['params']['first_stage_config'])
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('encoder.', '', 1): v for k,v in state_dict.items() if ('encoder.' in k and 'pretrained' not in k)}
            model.load_state_dict(state_dict)
        else:
            assert 'model_config' in checkpoint['hyper_parameters']
            model = LDMSevEncoder(checkpoint['hyper_parameters']['model_config'])
            model.load_state_dict(checkpoint['state_dict'])
        model.fwd_operator = create_operator(checkpoint['hyper_parameters']['operator_config'])
        return model