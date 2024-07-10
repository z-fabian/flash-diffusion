import torch 
from ldm.ldm.util import instantiate_from_config
import pathlib
import yaml
import torch
import numpy as np
import os

def rescale_to_minusone_one(x):
    return x * 2. - 1.

def rescale_to_zero_one(x):
    return (x + 1.) / 2.

def load_config_from_yaml(path):
    config_file = pathlib.Path(path)
    if config_file.exists():
        with config_file.open('r') as f:
            d = yaml.safe_load(f)
        return d
    else:
        raise ValueError(f"Config file {path} does not exist.")
    
def load_np_to_tensor(file, device=None):
    if device is None:
        device = 'cpu'
    return torch.from_numpy(np.load(file)).to(device)

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
    else:
        pl_sd = {"state_dict": None}
    global_step = None
    model = load_model_from_config(config['model'],
                                   pl_sd["state_dict"])

    return model, global_step

def str2int(s):
    return int.from_bytes(s.encode(), 'little') % (2 ** 32 - 1)

def extract_sev_encoder_ckpt(ckpt_path, ouput_dir=None):
    ckpt = torch.load(ckpt_path)
    sev_encoder_ckpt = {}
    sev_encoder_ckpt['state_dict'] = {'.'.join(k.split('.')[1:]):v for k, v in ckpt['state_dict'].items() if k.startswith('encoder')}
    model_config = load_config_from_yaml(ckpt['hyper_parameters']['ldm_model_config_path'])['model']['params']['first_stage_config']
    sev_encoder_ckpt['hyper_parameters'] = {"operator_config": ckpt['hyper_parameters']['operator_config'], "noise_config": ckpt['hyper_parameters']['noise_config'], "model_config": model_config}
    if ouput_dir is not None:
        torch.save(sev_encoder_ckpt, os.path.join(ouput_dir, ckpt_path.split('/')[-1].replace('.ckpt', '_light.ckpt')))
    else:
        torch.save(sev_encoder_ckpt, ckpt_path.replace('.ckpt', '_light.ckpt'))