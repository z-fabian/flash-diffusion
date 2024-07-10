import time
from pathlib import Path
import yaml
import torch
from ldm.ldm.util import get_obj_from_str
from .utils import save_logs, evaluate_results

class AdaptiveSampler:

    def __init__(self, 
                 sev_enc_model_class, 
                 sev_enc_ckpt_path, 
                 corr_mult=None, 
                 var_mult=None, 
                 init_mode='sev_enc_with_corr'
                 ):
        self.baseline_sampler = None
        self.snrs = None
        self.device = None
        self.initialized = False

        self.init_mode = init_mode
        self.var_mult = var_mult
        self.corr_mult = corr_mult

        self.severity_encoder = get_obj_from_str(sev_enc_model_class).load_pretrained(sev_enc_ckpt_path)
        self.fwd_operator = self.severity_encoder.get_fwd_operator()

    def attach(self, baseline_sampler):
        self.baseline_sampler = baseline_sampler
        self.snrs = self.baseline_sampler.get_snrs()
        self.device = self.baseline_sampler.get_device()
        self.severity_encoder.to(self.device)
        self.initialized = True

    def var_to_t(self, var, legacy=False):
        if legacy:
            t_pred = [1 / var >= snr for snr in self.snrs].index(True)
        else:
            diffs = [abs(snr - 1 / var) for snr in self.snrs]
            t_pred = diffs.index(min(diffs))
        return t_pred

    def find_ldm_start(self, y):
        z_mean, var = self.severity_encoder(y)
        
        if self.init_mode == 'sev_enc_with_corr':
            if self.var_mult is not None:
                var = var * self.var_mult
                
            if self.corr_mult is not None:
                var_corr = var * self.corr_mult
                z_start = torch.sqrt(1 - var_corr) * z_mean + torch.randn_like(z_mean) * torch.sqrt(var_corr)
            else:
                z_start = z_mean
            
            t = self.var_to_t(var)
        elif self.init_mode == 'sev_enc_ddim_inv':
            ddim_sampler = self.baseline_sampler.get_ddim_sampler()
            assert ddim_sampler is not None
            assert ddim_sampler.model.parameterization == "eps"
            t = self.var_to_t(var)
            timesteps = ddim_sampler.ddim_timesteps[:t]
            zt = z_mean.clone()
            for index, t_curr in enumerate(timesteps):
                ts = torch.full((1,), t_curr, device=z_mean.device, dtype=torch.long)
                a_cumprod_t = ddim_sampler.ddim_alphas[index]
                a_cumprod_t_prev = torch.tensor(ddim_sampler.ddim_alphas_prev[index]).to(a_cumprod_t.device)
                e_t = ddim_sampler.model.apply_model(zt, ts, cond=None)
                zt = torch.sqrt(a_cumprod_t) * (zt - torch.sqrt(1 - a_cumprod_t_prev) * e_t) / torch.sqrt(a_cumprod_t_prev) + torch.sqrt(1 - a_cumprod_t) * e_t # DDIM inversion formula from https://openreview.net/pdf?id=6ALuy19mPa
            z_start = zt.clone()

        return z_start, t, z_mean, var
    
    @torch.no_grad()
    def run(self, data, output_dir, verbose=False):
        if not self.initialized:
            raise ValueError("AdaptiveSampler not initialized. Please attach a baseline sampler first.")
        
        tstart_run = time.time()
        fnames = {}

        if verbose:
            print(f"Baseline sampler: {self.baseline_sampler}")
        
        # Reconstruct
        for i, item in enumerate(data):
            print("{}/{}".format(i+1, len(data)))
            degraded_img = item["degraded_noisy"].cuda()
            tstart_run = time.time()
            recon_dict = self.run_single(degraded_img, item["t"], item["fname"][0])
            outs = {"clean_img": item["clean"], "degraded_img": degraded_img, "recon": recon_dict['recon'], "severity": item['t'], "start_T": recon_dict['t_start'], "time": time.time() - tstart_run}
            fnames[i] = item['fname'][0]
            save_logs(outs, output_dir, i, "recon")
        
        if verbose:
            print(f"reconstruction of {len(data)} images finished in {(time.time() - tstart_run) / 60.:.2f} minutes.")
        
        # Save list of images in root dir
        with open(Path(output_dir) / 'images.yml', 'w') as outfile:
            yaml.dump(fnames, outfile)    

        if verbose:    
            print('Evaluating results.')
        results = evaluate_results(output_dir)
        return results
    
    @torch.no_grad()
    def run_single(self, degraded_img, t=None, fname=None):
        y = degraded_img.cuda()
        z_start, t_start, _, _ = self.find_ldm_start(y)
        self.baseline_sampler.update_fwd_operator(y, t, fname)
        x_recon = self.baseline_sampler.reconstruct_sample(z_start, t_start) 
        return {'recon': x_recon, 't_start': t_start, 'z_start': z_start}
