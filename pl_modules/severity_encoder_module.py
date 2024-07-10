from argparse import ArgumentParser
import torch
import lightning.pytorch as pl
import wandb
from ldm.ldm.util import instantiate_from_config
from data_utils.operators import create_operator, create_noise_schedule
from data_utils.metrics import ssim
from flash_diffusion import LDMSevEncoder
from scripts.utils import load_config_from_yaml, rescale_to_minusone_one, rescale_to_zero_one



class SeverityEncoderModule(pl.LightningModule):

    def __init__(
        self,
        operator_config,
        noise_config,
        ldm_model_ckpt_path,
        sev_encoder_ckpt_path,
        sev_encoder_config_path,
        pretrained_encoder_ckpt_path,
        pretrained_encoder_config_path,
        lr,
        lr_step_size,
        lr_gamma,
        ldm_model_config_path=None,
        sigma_reg=0.0,
        img_space_reg=0.0,
        lr_milestones=None,
        weight_decay=0.0,
        logger_type='wandb',
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.logger_type = logger_type
        self.fwd_operator = create_operator(operator_config)
        if noise_config is None:
            self.noise_schedule = None
            self.fwd_sigma_max = 0.0
        else:
            self.noise_schedule = create_noise_schedule(noise_config)    
            self.fwd_sigma_max = noise_config['sigma_max']
            
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        
        # Check if given checkpoint-config combination is valid
        if ldm_model_ckpt_path is not None:
            if ldm_model_config_path is None:
                ldm_model_config_path = '/'.join(ldm_model_ckpt_path.split('/')[:-1]) + '/config.yaml'
            print("Setting up encoders/decoders from pretrained LDM.")
        else:
            assert self.pretrained_autoencoder_config is not None and self.pretrained_encoder_ckpt_path is not None
            print('Loading ground truth encoder from pretrained autoencoder.')
                
        # Set up pretrained encoder
        self.ldm_model_ckpt_path = ldm_model_ckpt_path
        self.ldm_model_config_path = ldm_model_config_path
        if ldm_model_ckpt_path is None:
            print('Initializing pretrained autoencoder from config {} and checkpoint {}.'.format(self.pretrained_autoencoder_config, self.pretrained_encoder_ckpt_path))
            self.pretrained_autoencoder_config = load_config_from_yaml(pretrained_encoder_config_path)['model']
            self.pretrained_encoder_ckpt_path = pretrained_encoder_ckpt_path
            self.pretrained_autoencoder = instantiate_from_config(self.pretrained_autoencoder_config)
            self.pretrained_autoencoder = self.pretrained_autoencoder.load_from_checkpoint(checkpoint_path=self.pretrained_encoder_ckpt_path, 
                                **self.pretrained_autoencoder_config['params'])
            self.pretrained_encoder_type = self.pretrained_autoencoder_config['target']
        else:
            print('Initializing autoencoder from pretrained LDM with config {} and checkpoint {}.'.format(self.ldm_model_config_path, self.ldm_model_ckpt_path))
            ldm_config = load_config_from_yaml(self.ldm_model_config_path)
            ldm_checkpoint = torch.load(self.ldm_model_ckpt_path, map_location='cpu')['state_dict']
            autoencoder_checkpoint = {k.replace('first_stage_model.', ''): v for k,v in ldm_checkpoint.items() if 'first_stage_model.' in k}
            self.pretrained_autoencoder = instantiate_from_config(ldm_config['model']['params']['first_stage_config'])
            self.pretrained_autoencoder.load_state_dict(autoencoder_checkpoint)
            self.pretrained_encoder_type = ldm_config['model']['params']['first_stage_config']['target']
        self.pretrained_autoencoder.eval()
        for param in self.pretrained_autoencoder.parameters(): # Freeze ae
            param.requires_grad = False
        
        
        # Set up encoder
        if ldm_model_ckpt_path is None:
            self.sev_encoder_config_path = sev_encoder_config_path
            self.sev_encoder_config = load_config_from_yaml(sev_encoder_config_path)['model']
            self.sev_encoder_ckpt_path = sev_encoder_ckpt_path
            self.encoder = LDMSevEncoder(self.sev_encoder_config)
            if self.sev_encoder_ckpt_path is not None:      
                print('Initializing Severity Encoder with pretrained model from ', self.ldm_model_ckpt_path)
                checkpoint = torch.load(self.sev_encoder_ckpt_path, map_location='cpu')['state_dict']
                checkpoint = {k.replace('encoder.', ''): v for k,v in checkpoint.items() if 'encoder.' in k}
                self.encoder.encoder.load_state_dict(checkpoint)
        else:
            print('Initializing Severity Encoder with pretrained LDM encoder from ', self.ldm_model_config_path)
            self.sev_encoder_config = load_config_from_yaml(self.ldm_model_config_path)['model']['params']['first_stage_config']
            self.encoder = LDMSevEncoder(self.sev_encoder_config)
            encoder_checkpoint = {k.replace('first_stage_model.encoder.', ''): v for k,v in ldm_checkpoint.items() if 'first_stage_model.encoder.' in k}
            quant_conv_checkpoint = {k.replace('first_stage_model.quant_conv.', ''): v for k,v in ldm_checkpoint.items() if 'first_stage_model.quant_conv.' in k}
            self.encoder.encoder.load_state_dict(encoder_checkpoint)
            self.encoder.quant_conv.load_state_dict(quant_conv_checkpoint)
            del ldm_checkpoint, autoencoder_checkpoint, encoder_checkpoint, quant_conv_checkpoint
            
        # Regularization
        self.sigma_reg = sigma_reg
        self.img_space_reg = img_space_reg

        self.num_eval_levels = 10 # for calculating ordering accuracy. 
    
    def encode(self, x, get_var=True):
        z_mean, z_var  = self.encoder(x)
        if get_var:
            return z_mean, z_var
        else:
            return z_mean
        
    def decode(self, x, force_not_quantize=False):
        if self.pretrained_encoder_type in ['ldm.models.autoencoder.VQModel', 'ldm.models.autoencoder.VQModelInterface']:
            # also go through quantization layer
            if not force_not_quantize:
                quant, _, _ = self.pretrained_autoencoder.quantize(x)
            else:
                quant = x
            quant2 = self.pretrained_autoencoder.post_quant_conv(quant)
            dec = self.pretrained_autoencoder.decoder(quant2)
            return dec
        elif self.pretrained_encoder_type == 'ldm.models.autoencoder.AutoencoderKL':
            return self.pretrained_autoencoder.decode(x)
        else:
            raise ValueError('Unknown model type.')
        
    def get_z0(self, x):
        if self.pretrained_encoder_type == 'ldm.models.autoencoder.AutoencoderKL':
            return self.pretrained_autoencoder.encode(x).mode().detach()
        elif self.pretrained_encoder_type in ['ldm.models.autoencoder.VQModel', 'ldm.models.autoencoder.VQModelInterface']:
            h = self.pretrained_autoencoder.encoder(x)
            h = self.pretrained_autoencoder.quant_conv(h)
            return h.detach()
        else:
            raise ValueError('Unknown encoder type')
        
    def get_loss(self, batch):
        b = batch['clean'].shape[0]
        d_img = batch['clean'].view(b, -1).shape[1]
        z0 = self.get_z0(batch['clean'])
        z_mean, z_var = self.encoder(batch['degraded_noisy'])
        mean_term = (z_mean - z0).pow(2).view(b, -1).sum(1)
        var_term = z_var
        
        # Image space loss
        if self.img_space_reg > 0.0:
            recon_ssim = 0
            x_pred = self.decode(z_mean)
            img_space_loss = (x_pred - batch['clean']).view(b, -1).pow(2).sum()
            for i in range(b):
                recon_ssim += ssim(rescale_to_zero_one(x_pred[i]),
                                   rescale_to_zero_one(batch['clean'][i]),
                                  )
        else:
            img_space_loss = 0
            recon_ssim = 0
            
        recon_ssim /= b
        
        # Discrepancy metrics
        e_i = (z_mean - z0).view(b, -1)
        d = e_i.shape[1] # dimension of latent vec
        mu_i = e_i.sum(1) / d
        var_i = (e_i - mu_i.unsqueeze(1)).pow(2).sum(1) / (d - 1)
        var_discrep_sq = (var_i - z_var).pow(2)

        loss = 1 / d * mean_term + self.sigma_reg * var_discrep_sq + self.img_space_reg * 1 / d_img * img_space_loss
        return {
                'loss': loss.mean(),
                'mean_term': mean_term.mean(), 
                'var_term': var_term.mean(),
                'img_space_loss': img_space_loss.mean(),
                'recon_ssim': recon_ssim
               }
                                                     
    def training_step(self, batch, batch_idx):
        losses = self.get_loss(batch)
        for k, v in losses.items():
            self.log('train/{}'.format(k), losses[k])
        return losses['loss']
        
    def validation_step(self, batch, batch_idx):
        val_losses = self.get_loss(batch)
        for k in val_losses:
            self.log('val/{}'.format(k), val_losses[k], sync_dist=True)
        _, ord_acc = self.eval_ordering(batch)
        self.log('val/ord_acc', ord_acc)
        if batch_idx == 0:
            b = batch['clean'].shape[0]
            _, z_var = self.encoder(batch['degraded_noisy'])
            for i in range(b):
                self.log_image('val/images/img_{}'.format(i), batch["degraded_noisy"][i].unsqueeze(0), 'var_pred: {}, t_gt: {}'.format(str(z_var[i].detach().cpu().numpy()),str(batch["t"][i].detach().cpu().numpy())))
    
    def eval_ordering(self, batch):
        # Evaluates whether the ordering of predicted sigmas corresponds to the ordering of true degradation severity
        # This metric only makes sense if degradation severity is defined for the operator
        x0 = rescale_to_zero_one(batch['clean'])
        b = x0.shape[0]
        var_preds = torch.zeros(b, self.num_eval_levels)
        ts = torch.linspace(0.0, 1.0, self.num_eval_levels)
        for i in range(self.num_eval_levels):
            y = self.fwd_operator(x0, torch.tensor(ts[i]).to(x0.device)) 
            if self.noise_schedule is not None:
                z, _ = self.noise_schedule(ts[i], y.shape)
                y += z.to(y.device)
            y = rescale_to_minusone_one(y)
            var_preds[:, i] = self.encoder(y)[1]
            
        ord_true = self.ordering_mx(torch.linspace(0.0, 1.0, self.num_eval_levels).repeat(b, 1))
        ord_pred = self.ordering_mx(var_preds)
        ord_loss = 0.5 * (ord_true-ord_pred).pow(2).mean()
        ord_acc = self.ordering_acc(ord_pred, ord_true).mean()
        return ord_loss, ord_acc
            
    def ordering_acc(self, ordering_mx, gt_mx):
        num_vals = (self.num_eval_levels - 1) * self.num_eval_levels / 2 
        return torch.triu(torch.where(ordering_mx == gt_mx, 1.0, 0.0), diagonal=1).sum(dim=[1, 2]) / num_vals   
        
    def ordering_mx(self, vals):
        # Assume vals shape (b, num_levels)
        # Compare each pair of output entries, assign 0/1 to each pair if less/greater than.
        assert vals.shape[1] == self.num_eval_levels
        x = vals.unsqueeze(2).expand(vals.size(0), self.num_eval_levels, self.num_eval_levels)
        xT = vals.unsqueeze(1).expand(vals.size(0), self.num_eval_levels, self.num_eval_levels)
        ord_mx = torch.where(x - xT > 0, 0.0, 1.0)
        mask = 1.0 - torch.eye(self.num_eval_levels, self.num_eval_levels, dtype=vals.dtype, device=vals.device)
        ord_mx = ord_mx * mask  # Zero out diagonals to avoid numerical errors when comparing the same float to itself
        return ord_mx
    
    def configure_optimizers(self):

        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.lr_step_size is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim, self.lr_step_size, self.lr_gamma
            )
            return [optim], [scheduler]
        elif self.lr_milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim, self.lr_milestones, self.lr_gamma
            )
            return [optim], [scheduler]
        return optim
        
    
    def log_image(self, name, image, caption=None):
        if self.logger_type == 'wandb':
            # wandb logging
            self.logger.experiment.log({name:  wandb.Image(image, caption=caption)})
        else:
            # tensorboard logging
            self.logger.experiment.add_image(name, image, global_step=self.global_step)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # configs/model setup
        parser.add_argument(
            "--ldm_model_config_path", 
            type=str, 
            default=None,
            help="Config file for LDM."
        )
        parser.add_argument(
            "--ldm_model_ckpt_path", 
            type=str, 
            default=None,
            help="Path to pretrained LDM to initialize sev encoder."
        )
        parser.add_argument(
            "--sev_encoder_config_path", 
            type=str, 
            default=None,
            help="Config file for encoder arch."
        )
        parser.add_argument(
            "--sev_encoder_ckpt_path", 
            type=str, 
            default=None,
            help="Path to pretrained encoder to initialize sev encoder with."
        )
        parser.add_argument(
            "--pretrained_encoder_ckpt_path", 
            type=str, 
            default=None,
            help="Path to pretrained AE model checkpoint."
        )
        parser.add_argument(
            "--pretrained_encoder_config_path", 
            type=str, 
            default=None,
            help="Config file for pretrained autoencoder"
        )

        # training params (opt)       
        parser.add_argument(
            "--lr", 
            default=0.0001, 
            type=float, 
            help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=None,
            type=int,
            help="Number of epochs to decrease step size",
        )
        parser.add_argument(
            "--lr_milestones",
            default=None,
            type=int,
            nargs='+',
            help="List of epochs to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", 
            default=0.1, 
            type=float, 
            help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--sigma_reg", 
            default=0.0,
            type=float, 
            help="Weight of sigma regularization."
        )
        parser.add_argument(
            "--img_space_reg", 
            default=0.0,
            type=float, 
            help="Weight of image domain regularization."
        )

        return parser