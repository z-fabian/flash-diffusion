from tqdm import tqdm
import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from ldm.ldm.models.diffusion.ddim import DDIMSampler
from ldm.ldm.modules.diffusionmodules.util import noise_like
from ldm.ldm.util import get_obj_from_str
from data_utils.operators import create_noise_schedule, create_operator
from samplers.data_consistency import LatentDataConsistency
from scripts.utils import load_config_from_yaml, str2int, load_model 



class LatentSampler(ABC):
    # Subclass to add a custom latent reconstruction method.
    # See below the functions that have to be implemented.

    @abstractmethod
    def reconstruct_sample(self, z_start, t_start):
        # Takes starting latent z_start and starting timestep t_start and returns the reconstructed image.
        pass

    @abstractmethod
    def get_snrs(self):
        # Returns the SNR for each timestep.
        # For a forward process with N(a_t * z_0, b_t^2 * I), the SNRs are is a_t^2 / b_t^2.
        pass

    @abstractmethod
    def get_device(self):
        # Return the device where the model is placed.
        pass

    def get_ddim_sampler(self):
        # Optional
        # Returns the DDIM sampler if it is used.
        return None
    
    def update_fwd_operator(self, y, t, fname):
        # Optional
        # Make any necessary internal updates based on the sample to be reconstructed.
        # This can be used to update the forward model for data consistency steps.
        pass

class LatentReconAlgo(LatentSampler):

    def __init__(self, 
                 degradation_config_path,
                 ldm_ckpt_path,
                 data_consistency,
                 ddim_steps=None,
                 ddim_eta=None,
                 **kwargs,
                 ):

        # Set up diffusion model
        self.configure_model(ldm_ckpt_path)

        # Add DDIM wrapper if needed
        self.configure_ddim_sampler(ddim_steps, ddim_eta)

        # Set up degradation operator
        self.configure_operator(degradation_config_path)

        # Set up data consistency correction
        self.configure_dc_corrector(data_consistency)

    def __str__(self):
        return "Custom sampler"

    def get_snrs(self):
        if self.use_ddim_sampling:
            alphas = self.ddim.ddim_alphas
        else:
            alphas = self.model.alphas_cumprod
        return [alpha / (1 - alpha) for alpha in alphas]
    
    def get_device(self):
        return self.model.device
    
    def get_ddim_sampler(self):
        return self.ddim
    
    def update_fwd_operator(self, y, t, fname=None):
        if self.dc_corrector is not None:
            seed = str2int(fname) if fname is not None else None
            self.dc_corrector.update_fwd_seed(seed)
            self.dc_corrector.update_y(y.cuda(), t.cuda())

    def configure_model(self, model_ckpt):
        model_config_path = os.path.join('/'.join(model_ckpt.split('/')[:-1]), "config.yaml")
        model_config = load_config_from_yaml(model_config_path)
        self.model, _ = load_model(model_config, model_ckpt) 
        self.model.eval()

        # Switch to EMA weights for inference
        if self.model.use_ema:
            self.model.model_ema.store(self.model.model.parameters())
            self.model.model_ema.copy_to(self.model.model)

    def configure_ddim_sampler(self, ddim_steps, ddim_eta):
        self.custom_steps = ddim_steps
        self.eta = ddim_eta
        self.use_ddim_sampling = self.custom_steps is not None and self.eta is not None   
        if self.use_ddim_sampling:
            self.ddim = DDIMSampler(self.model)
            self.ddim.make_schedule(ddim_num_steps=self.custom_steps, ddim_eta=self.eta, verbose=False)
        else:
            self.ddim = None

    def configure_operator(self, config_path):
        degradation_config = load_config_from_yaml(config_path)
        self.fwd_operator = create_operator(degradation_config['operator'])
        self.noise_schedule = create_noise_schedule(degradation_config['noise'])

    def configure_dc_corrector(self, config):
        if 'dc_correct_stage' not in config:
            config['dc_correct_stage'] = ['chaotic', 'semantic', 'refinement']
        use_dc = (config['dc_step'] > 0.0 and config['dc_correct_freq'] > 0.0 and len(config['dc_correct_stage']) > 0) or ('z0_pred_corr_freq' in config and config['z0_pred_corr_freq'] > 0.0) or ('z0_correct_last_n' in config and config['z0_correct_last_n'] > 0) or ('z0_pred_corr_every_n' in config and config['z0_pred_corr_every_n'] > 0) 
        if use_dc:
            self.dc_corrector = LatentDataConsistency(
                fwd_operator=self.fwd_operator,
                noise_schedule=self.noise_schedule,
                encode_fn=lambda x: self.differentiable_encode_first_stage(x), 
                decode_fn=lambda x: self.model.differentiable_decode_first_stage(x), 
                dc_step=config['dc_step'],
                ae_weight=config['ae_weight'] if 'ae_weight' in config else 0.0,
                use_psld=config['use_psld'] if 'use_psld' in config else False,
                z0_pred_corr_steps=config['z0_pred_corr_steps'] if 'z0_pred_corr_steps' in config else 0,
                z0_pred_corr_lam=config['z0_pred_corr_lam'] if 'z0_pred_corr_lam' in config else 0.0,
                z0_pred_corr_error_weighting=config['z0_pred_corr_error_weighting'] if 'z0_pred_corr_error_weighting' in config else False,
                z0_pred_corr_domain=config['z0_pred_corr_domain'] if 'z0_pred_corr_domain' in config else 'image',
                z0_pred_corr_init_mode=config['z0_pred_corr_init_mode'] if 'z0_pred_corr_init_mode' in config else 'randn',
                z0_pred_corr_stop_eps=config['z0_pred_corr_stop_eps'] if 'z0_pred_corr_stop_eps' in config else None,
                stochastic_resample=config['stochastic_resample'] if 'stochastic_resample' in config else False,
                resample_step=config['resample_step'] if 'resample_step' in config else 0,
                scaling_method=config['scaling_method'],
                scale_with_alphas=config['scale_with_alphas'],
            )        
            self.dc_correct_freq = config['dc_correct_freq'] if 'dc_correct_freq' in config else 0.0
            config['dc_correct_freq'] = self.dc_correct_freq
            self.dc_correct_stage = config['dc_correct_stage'] if 'dc_correct_stage' in config else ['chaotic', 'semantic', 'refinement']
            self.z0_correct_freq = config['z0_pred_corr_freq'] if 'z0_pred_corr_freq' in config else 0.0
            self.z0_correct_last_n = config['z0_correct_last_n'] if 'z0_correct_last_n' in config else 0
            self.z0_correct_every_n = config['z0_pred_corr_every_n'] if 'z0_pred_corr_every_n' in config else 0
        else:
            self.dc_corrector = None
            self.dc_correct_freq = 0.0
            self.dc_correct_stage = []
            self.z0_correct_freq = 0.0
            self.z0_correct_last_n = 0
            self.z0_correct_every_n = 0
                    

    @torch.no_grad()
    def reconstruct_sample(self, z_start, t_start):        
        shape = [1,
                self.model.model.diffusion_model.in_channels,
                self.model.model.diffusion_model.image_size,
                self.model.model.diffusion_model.image_size]
        
        self.model.zero_grad()

        if self.ddim is None:
            if self.z0_correct_every_n > 0:
                z0_corr_iterations = list(range(0, t_start, self.z0_correct_every_n))
                z0_corr_iterations.append(t_start - 1)
                z0_corr_iterations = set(z0_corr_iterations)
            elif self.z0_correct_freq > 0.0:
                z0_corr_iterations = list(range(0, t_start, int(1/self.z0_correct_freq)))
                z0_corr_iterations.append(t_start - 1)
                z0_corr_iterations = set(z0_corr_iterations)
            else:
                z0_corr_iterations = []

            if self.z0_correct_last_n > 0:
                z0_corr_iterations = list(z0_corr_iterations)
                z0_corr_iterations.extend(list(range(0, self.z0_correct_last_n)))
                z0_corr_iterations = set(z0_corr_iterations)

        if not self.use_ddim_sampling:
            sample, _ = self.latent_reconstruction(x_T=z_start, start_T=t_start, shape=shape)
        else:
            bs = shape[0]
            shape_ddim = shape[1:]
            sample, _ = self.latent_reconstruction_ddim(x_T=z_start, 
                                                        start_T=t_start, 
                                                        batch_size=bs, 
                                                        shape=shape_ddim, 
                                                        )
        x_recon = self.model.decode_first_stage(sample)  
        
        return x_recon
    
    @torch.no_grad()
    def latent_reconstruction(self, 
                              x_T, 
                              start_T, 
                              shape, 
                              cond=None,
                              quantize_denoised=False, 
                              temperature=1., 
                              noise_dropout=0.,
                              score_corrector=None,
                              corrector_kwargs=None, 
                              batch_size=None,
                              log_every_t=None,
                             ):
        if not log_every_t:
            log_every_t = self.model.log_every_t
            
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
            
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
            
        img = x_T
        
        intermediates = []
        timesteps = min(self.model.num_timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Reconstruction',
                        total=timesteps)
        

        # iters with L-DPS correction
        if self.dc_correct_freq > 0.0 and len(self.dc_correct_stage) > 0 and self.dc_corrector is not None:
            if self.dc_corrector.dc_step == 0.0:
                dc_corr_iterations = []
            else:
                dc_corr_iterations = list(range(0, timesteps, int(1/self.dc_correct_freq)))
                dc_corr_iterations.append(timesteps - 1)
                dc_corr_iterations = list(set(dc_corr_iterations))
                dc_corr_iterations = [iter for iter in dc_corr_iterations if self.dc_corrector.get_stage_from_step(iter) in self.dc_correct_stage]
        else:
            dc_corr_iterations = []

        # iters with correction on posterior mean
        if self.z0_correct_every_n > 0:
            z0_corr_iterations = list(range(0, start_T, self.z0_correct_every_n))
            z0_corr_iterations.append(start_T - 1)
            z0_corr_iterations = set(z0_corr_iterations)
        elif self.z0_correct_freq > 0.0:
            z0_corr_iterations = list(range(0, start_T, int(1/self.z0_correct_freq)))
            z0_corr_iterations.append(start_T - 1)
            z0_corr_iterations = set(z0_corr_iterations)
        else:
            z0_corr_iterations = []

        if self.z0_correct_last_n > 0:
            z0_corr_iterations = list(z0_corr_iterations)
            z0_corr_iterations.extend(list(range(0, self.z0_correct_last_n)))
            z0_corr_iterations = set(z0_corr_iterations)

        
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for iternum, i in enumerate(iterator):
            ts = torch.full((b,), i, device=self.model.device, dtype=torch.long)
            use_dc = (i in dc_corr_iterations) or (i in z0_corr_iterations)
            # Apply data consistency step
            if use_dc:
                with torch.enable_grad():
                    self.model.zero_grad()
                    img.requires_grad = True
                    img_next, x0_partial = self.p_sample_with_dc(img, cond, ts,
                                            clip_denoised=self.model.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,
                                            )
                    if i in z0_corr_iterations:
                        img_next = self.dc_corrector.resample(img_next, x0_partial, self.model.alphas_cumprod[i], self.model.alphas_cumprod_prev[i], i)
                    if i in dc_corr_iterations:
                        img_next = self.dc_corrector.modify_score(z_past=img, z_next=img_next, z0_pred=x0_partial)
                img = img_next.clone()
            else:
                img, x0_partial = self.model.p_sample(img, cond, ts,
                                                clip_denoised=self.model.clip_denoised,
                                                quantize_denoised=quantize_denoised, return_x0=True,
                                                temperature=temperature[i], noise_dropout=noise_dropout,
                                                score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
        return img, intermediates
    
    def p_sample_with_dc(self, x, c, t, clip_denoised=False, repeat_noise=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device


        outputs = self.model.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs
                    
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        x_next = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        if return_x0:
            return x_next, x0
        else:
            return x_next
    
    def differentiable_encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.model.split_input_params["patch_distributed_vq"]:
                ks = self.model.split_input_params["ks"]  # eg. (128, 128)
                stride = self.model.split_input_params["stride"]  # eg. (64, 64)
                df = self.model.split_input_params["vqf"]
                self.model.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.model.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.model.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.model.first_stage_model.encode(x)
        else:
            return self.model.first_stage_model.encode(x)
        

    @torch.no_grad()
    def latent_reconstruction_ddim(self,
                batch_size,
                shape,
                x_T,
                start_T,
                conditioning=None,
                unconditional_guidance_scale=None,
                unconditional_conditioning=None,
                temperature=1.,
                noise_dropout=0.,
                score_corrector=None,
                corrector_kwargs=None,
                log_every_t=100,
                ):
        C, H, W = shape
        device = self.ddim.model.betas.device

        img = x_T
        timesteps = start_T
        ddim_use_original_steps = False

        if timesteps is None:
            timesteps = self.ddim.ddpm_num_timesteps if ddim_use_original_steps else self.ddim.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim.ddim_timesteps.shape[0], 1) * self.ddim.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim.ddim_timesteps[:subset_end]

            # Make sure we are doing at least one step
            if len(timesteps) == 0:
                timesteps = np.array([self.ddim.ddim_timesteps[0]])
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        iterator = tqdm(time_range, desc='DDIM Reconstruction.', total=total_steps)
        
        # Set up dc correction iterations
        if self.dc_correct_freq > 0.0 and len(self.dc_correct_stage) > 0 and self.dc_corrector is not None:
            if self.dc_corrector.dc_step == 0.0:
                dc_corr_iterations = []
            else:
                dc_corr_iterations =  list(range(0, timesteps, int(1/self.dc_correct_freq))) if isinstance(timesteps, int) else timesteps.tolist()[:: int(1/self.dc_correct_freq)]
                if isinstance(timesteps, int):
                    dc_corr_iterations.append(timesteps - 1)
                else:
                    dc_corr_iterations.append(timesteps[0])
                dc_corr_iterations = list(set(dc_corr_iterations))
                dc_corr_iterations = [iter for iter in dc_corr_iterations if self.dc_corrector.get_stage_from_step(iter) in self.dc_correct_stage]
        else:
            dc_corr_iterations = []

        # Set up z0 correction iterations
        if self.z0_correct_every_n > 0:
            z0_correct_iterations =  list(range(0, timesteps, self.z0_correct_every_n)) if isinstance(timesteps, int) else timesteps.tolist()[::self.z0_correct_every_n]
        elif self.z0_correct_freq > 0.0:
            z0_correct_iterations =  list(range(0, timesteps, int(1/self.z0_correct_freq))) if isinstance(timesteps, int) else timesteps.tolist()[:: int(1/self.z0_correct_freq)]
        else:
            z0_correct_iterations = []

        if self.z0_correct_last_n > 0:
            z0_correct_iterations = list(z0_correct_iterations)
            z0_correct_iterations.extend(timesteps[:self.z0_correct_last_n])
            z0_correct_iterations = set(z0_correct_iterations)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            use_dc = (step in dc_corr_iterations) or (step in z0_correct_iterations)

            # Apply data consistency step
            if use_dc:
                with torch.enable_grad():
                    self.ddim.model.zero_grad()
                    img.requires_grad = True
                    img_next, pred_x0 = self.p_sample_ddim_with_dc(img, 
                                                                    c=conditioning, 
                                                                    t=ts, 
                                                                    index=index,         
                                                                    use_original_steps=ddim_use_original_steps,
                                                                    temperature=temperature,
                                                                    noise_dropout=noise_dropout,
                                                                    score_corrector=score_corrector,
                                                                    corrector_kwargs=corrector_kwargs,
                                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                                    unconditional_conditioning=unconditional_conditioning,
                                                                    )
                    if step in z0_correct_iterations:
                        img_next = self.dc_corrector.resample(img_next, pred_x0, self.ddim.alphas_cumprod[step], self.ddim.alphas_cumprod_prev[step], step)
                    if step in dc_corr_iterations:
                        img_next = self.dc_corrector.modify_score(z_past=img, z_next=img_next, z0_pred=pred_x0)
                img = img_next.clone()
            else:
                img, pred_x0 = self.ddim.p_sample_ddim(img, c=conditioning, t=ts, index=index, use_original_steps=ddim_use_original_steps, temperature=temperature,
                                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning)
            
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
    
    def p_sample_ddim_with_dc(self, x, c, t, index, repeat_noise=False, use_original_steps=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None, x0_corrector=None):
        b, *_, device = *x.shape, x.device
        if x0_corrector is None:
            x0_corrector = lambda x: x

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.ddim.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.ddim.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.ddim.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.ddim.model.alphas_cumprod if use_original_steps else self.ddim.ddim_alphas
        alphas_prev = self.ddim.model.alphas_cumprod_prev if use_original_steps else self.ddim.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        pred_x0_corr = x0_corrector(pred_x0)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0_corr + dir_xt + noise
        return x_prev, pred_x0
    
class ReSample(LatentReconAlgo):

    def __str__(self):
        return "ReSample"

    def configure_dc_corrector(self, config):
        # Checking validity of config file
        assert ('z0_pred_corr_every_n' in config and config['z0_pred_corr_every_n'] > 0) or ('z0_pred_corr_freq' in config and config['z0_pred_corr_freq'] > 0.0), "Either z0_pred_corr_every_n or z0_pred_corr_freq has to be set for ReSample."
        assert 'resample_step' in config, "resample_step has to be set for ReSample."

        if 'dc_correct_stage' not in config:
            config['dc_correct_stage'] = ['chaotic', 'semantic', 'refinement']
        
        self.dc_corrector = LatentDataConsistency(
            fwd_operator=self.fwd_operator,
            noise_schedule=self.noise_schedule,
            encode_fn=lambda x: self.differentiable_encode_first_stage(x), 
            decode_fn=lambda x: self.model.differentiable_decode_first_stage(x), 
            dc_step=0.0,
            ae_weight=0.0,
            use_psld=False,
            z0_pred_corr_steps=config['z0_pred_corr_steps'] if 'z0_pred_corr_steps' in config else 100,
            z0_pred_corr_lam=config['z0_pred_corr_lam'] if 'z0_pred_corr_lam' in config else 0.0,
            z0_pred_corr_error_weighting=False,
            z0_pred_corr_domain='staged',
            z0_pred_corr_init_mode=config['z0_pred_corr_init_mode'] if 'z0_pred_corr_init_mode' in config else 'x_init',
            z0_pred_corr_stop_eps=config['z0_pred_corr_stop_eps'] if 'z0_pred_corr_stop_eps' in config else None,
            stochastic_resample=True,
            resample_step=config['resample_step'],
        )        
        self.dc_correct_freq =  0.0
        config['dc_correct_freq'] = self.dc_correct_freq
        self.dc_correct_stage = config['dc_correct_stage']
        self.z0_correct_freq = config['z0_pred_corr_freq'] if 'z0_pred_corr_freq' in config else 0.0
        self.z0_correct_last_n = config['z0_correct_last_n'] if 'z0_correct_last_n' in config else 0
        self.z0_correct_every_n = config['z0_pred_corr_every_n'] if 'z0_pred_corr_every_n' in config else 0

class LDPS(LatentReconAlgo):

    def __str__(self):
        return "LDPS"

    def configure_dc_corrector(self, config):
        # Checking validity of config file
        if 'dc_correct_stage' not in config:
            config['dc_correct_stage'] = ['chaotic', 'semantic', 'refinement']
        if 'dc_correct_freq' not in config:
            config['dc_correct_freq'] = 1.0
        assert (config['dc_step'] > 0.0 and config['dc_correct_freq'] > 0.0 and len(config['dc_correct_stage']) > 0)

        self.dc_corrector = LatentDataConsistency(
            fwd_operator=self.fwd_operator,
            noise_schedule=self.noise_schedule,
            encode_fn=lambda x: self.differentiable_encode_first_stage(x), 
            decode_fn=lambda x: self.model.differentiable_decode_first_stage(x), 
            dc_step=config['dc_step'],
            ae_weight=0.0,
            use_psld=False,
            z0_pred_corr_steps=0,
            z0_pred_corr_lam=0.0,
            z0_pred_corr_error_weighting=False,
            scaling_method='error',
            scale_with_alphas=False,
        )        
        self.dc_correct_freq = config['dc_correct_freq']
        self.dc_correct_stage = config['dc_correct_stage']
        self.z0_correct_freq = 0.0
        self.z0_correct_last_n = 0
        self.z0_correct_every_n = 0

class GML_DPS(LatentReconAlgo):

    def __str__(self):
        return "GML_DPS"

    def configure_dc_corrector(self, config):
        # Checking validity of config file
        if 'dc_correct_stage' not in config:
            config['dc_correct_stage'] = ['chaotic', 'semantic', 'refinement']
        if 'dc_correct_freq' not in config:
            config['dc_correct_freq'] = 1.0
        assert (config['dc_step'] > 0.0 and config['dc_correct_freq'] > 0.0 and len(config['dc_correct_stage']) > 0)
        assert 'ae_weight' in config

        self.dc_corrector = LatentDataConsistency(
            fwd_operator=self.fwd_operator,
            noise_schedule=self.noise_schedule,
            encode_fn=lambda x: self.differentiable_encode_first_stage(x), 
            decode_fn=lambda x: self.model.differentiable_decode_first_stage(x), 
            dc_step=config['dc_step'],
            ae_weight=config['ae_weight'],
            use_psld=False,
            z0_pred_corr_steps=0,
            z0_pred_corr_lam=0.0,
            z0_pred_corr_error_weighting=False,
            scaling_method='error',
            scale_with_alphas=False,
        )        
        self.dc_correct_freq = config['dc_correct_freq']
        self.dc_correct_stage = config['dc_correct_stage']
        self.z0_correct_freq = 0.0
        self.z0_correct_last_n = 0
        self.z0_correct_every_n = 0

class PSLD(LatentReconAlgo):

    def __str__(self):
        return "PSLD"

    def configure_dc_corrector(self, config):
        # Checking validity of config file
        if 'dc_correct_stage' not in config:
            config['dc_correct_stage'] = ['chaotic', 'semantic', 'refinement']
        if 'dc_correct_freq' not in config:
            config['dc_correct_freq'] = 1.0
        assert (config['dc_step'] > 0.0 and config['dc_correct_freq'] > 0.0 and len(config['dc_correct_stage']) > 0)
        assert 'ae_weight' in config

        self.dc_corrector = LatentDataConsistency(
            fwd_operator=self.fwd_operator,
            noise_schedule=self.noise_schedule,
            encode_fn=lambda x: self.differentiable_encode_first_stage(x), 
            decode_fn=lambda x: self.model.differentiable_decode_first_stage(x), 
            dc_step=config['dc_step'],
            ae_weight=config['ae_weight'],
            use_psld=True,
            z0_pred_corr_steps=0,
            z0_pred_corr_lam=0.0,
            z0_pred_corr_error_weighting=False,
            scaling_method='error',
            scale_with_alphas=False,
        )        
        self.dc_correct_freq = config['dc_correct_freq']
        self.dc_correct_stage = config['dc_correct_stage']
        self.z0_correct_freq = 0.0
        self.z0_correct_last_n = 0
        self.z0_correct_every_n = 0


def get_baseline_sampler(degradation_config_path, **kwargs):
    if 'class' not in kwargs:
        # General sampler for latent reconstruction that can combine all the different methods
        return LatentReconAlgo(degradation_config_path=degradation_config_path, **kwargs)
    else:
        return get_obj_from_str(kwargs['class'])(degradation_config_path=degradation_config_path, **kwargs)