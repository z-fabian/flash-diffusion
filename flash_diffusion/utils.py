# Utils for logging and evaluation
from pathlib import Path
import numpy as np
import yaml
import torch_fidelity
import matplotlib.pyplot as plt
from data_utils.metrics import psnr, LPIPS, ssim, nmse
from scripts.utils import load_config_from_yaml, load_np_to_tensor, rescale_to_zero_one 

lpips = LPIPS('vgg')
    
def save_logs(log, output_dir, image_id, expname='recon'):
    tensor_formatter = lambda im: rescale_to_zero_one(im.cpu().numpy())
    image_formatter = lambda im: rescale_to_zero_one(im[0].permute(1,2,0).cpu().numpy()).clip(0.0, 1.0)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'target').mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'noisy').mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / expname).mkdir(parents=True, exist_ok=True)
    image_id = str(image_id)
    
    np.save(str((Path(output_dir) / 'target' / (image_id+'.npy'))), tensor_formatter(log["clean_img"]))
    plt.imsave(str((Path(output_dir) / 'target' / (image_id+'.png'))), image_formatter(log["clean_img"]))

    np.save(str((Path(output_dir) / 'noisy' / (image_id+'.npy'))), tensor_formatter(log["degraded_img"]))
    plt.imsave(str((Path(output_dir) / 'noisy' / (image_id+'.png'))), image_formatter(log["degraded_img"]))

    np.save(str(Path(output_dir) / expname / (image_id+'.npy')), tensor_formatter(log["recon"]))    
    plt.imsave(str(Path(output_dir)/ expname / (image_id+'.png')), image_formatter(log["recon"]))
    
    recon_info = {"severity": float(log["severity"][0].cpu().numpy()), "start_T": log["start_T"], "wall_time": log["time"]}
    with open(Path(output_dir)/ expname / (image_id +'_recon_info.yml'), 'w') as outfile:
        yaml.dump(recon_info, outfile)

def evaluate_results(output_dir, device='cpu'):
    (Path(output_dir) / 'recon' / 'eval').mkdir(parents=True, exist_ok=True)
    
    target_files = sorted(list((Path(output_dir) / 'target').glob('*.npy')))
    degraded_files = sorted(list((Path(output_dir) / 'noisy').glob('*.npy')))
    recon_files = sorted(list((Path(output_dir) / 'recon').glob('*.npy')))
    recon_info_files = sorted(list((Path(output_dir) / 'recon').glob('*_recon_info.yml')))
    assert len(target_files) == len(recon_files) == len(degraded_files)
    
    ssim_arr = []
    psnr_arr = []
    nmse_arr = []
    lpips_arr = []
    start_T_arr = []
    
    for target, recon, recon_info, noisy in zip(target_files, recon_files, recon_info_files, degraded_files):
        assert str(target.stem) == str(recon.stem) == str(noisy.stem)
        target_arr = load_np_to_tensor(target, device)
        recon_arr = load_np_to_tensor(recon, device).clip(0.0, 1.0)
        start_T = load_config_from_yaml(recon_info)['start_T']
        
        ssim_arr.append(ssim(target_arr, recon_arr).cpu().numpy())
        psnr_arr.append(psnr(target_arr, recon_arr).cpu().numpy())
        nmse_arr.append(nmse(target_arr, recon_arr).cpu().numpy())
        lpips_arr.append(lpips(target_arr, recon_arr).cpu().numpy())
        start_T_arr.append(float(start_T))
        
    # Compute generative/distributional metrics
    recon_path = str(Path(output_dir) / 'recon')
    target_path = str(Path(output_dir) / 'target')
    gen_res = torch_fidelity.calculate_metrics(input1=recon_path,
                                        input2=target_path,
                                        cuda=True,
                                        isc=False,
                                        fid=True,
                                        kid=False,
                                        verbose=False,
                                        )
        
    # Aggregate results
    ssim_final = np.array(ssim_arr).mean()
    psnr_final = np.array(psnr_arr).mean()
    nmse_final = np.array(nmse_arr).mean()
    lpips_final = np.array(lpips_arr).mean()
    start_T_mean_final = np.array(start_T_arr).mean()
    results = {'ssim': float(ssim_final), 
            'psnr': float(psnr_final), 
            'nmse': float(nmse_final), 
            'lpips': float(lpips_final), 
            'start_T_mean': float(start_T_mean_final)
            }
    
    for k,v in gen_res.items():
        results[k] = v
        
    with open(Path(output_dir) / 'recon' / 'eval' / 'final_metrics.yml', 'w') as outfile:
        yaml.dump(results, outfile)
        
    return results