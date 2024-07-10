import argparse, os, sys, yaml, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, os.path.join(os.path.dirname(pathlib.Path(__file__).parent.absolute()), 'ldm'))
import random 
import torch
import numpy as np
from samplers.latent_recon import get_baseline_sampler
from flash_diffusion.adaptive_sampler import AdaptiveSampler
from scripts.utils import load_config_from_yaml
from data_utils.image_data import get_dataloader 
        
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Logging directory.",
    )
    parser.add_argument(
        "--recon_config_path",
        type=str,
        help="Path to reconstruction config file.",
    )
    return parser

if __name__ == "__main__":

    # Load configs
    parser = get_parser()
    opt = parser.parse_args()
    config = load_config_from_yaml(opt.recon_config_path)
    data_config = config['data']
    flash_config = config['adaptation']
    baseline_config = config['baseline']

    # Fix random seeds
    g = torch.Generator()
    g.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)

    # Set up logging
    logdir = opt.output_dir
    print("logging to:", logdir)
    os.makedirs(logdir, exist_ok=True)

    # Set up model
    baseline_sampler = get_baseline_sampler(
                 degradation_config_path=data_config['degradation_config'],
                 **baseline_config
    )
    flash_sampler = AdaptiveSampler(**flash_config)
    flash_sampler.attach(baseline_sampler)
        
    # Set up dataset
    dataloader = get_dataloader(data_config, g)

    # Save reconstruction config
    sampling_file = os.path.join(logdir, "recon_config.yaml")
    with open(sampling_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run reconstruction
    result = flash_sampler.run(data=dataloader, 
                    output_dir=logdir,
                    )

    # Print and save results
    for k, v in result.items():
        print('{}: {}'.format(k, v))
    
    results_file = os.path.join(logdir, "results_summary.yaml")
    with open(results_file, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)
    print("Done.")