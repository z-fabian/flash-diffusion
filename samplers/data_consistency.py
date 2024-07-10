import torch
from scripts.utils import rescale_to_minusone_one, rescale_to_zero_one

class InverseSolver:

    def __init__(self, num_steps, lam, init_mode='randn'):
        self.num_steps = num_steps
        self.lam = lam
        self.init_mode = init_mode

    def solve(self, x_init, y, fwd_fn, decode_fn=None, stop_eps=None):
        if self.init_mode == 'x_init':
            x = torch.nn.Parameter(x_init)
        elif self.init_mode == 'zeros':
            x = torch.nn.Parameter(torch.zeros_like(x_init))
        elif self.init_mode == 'randn':
            x = torch.nn.Parameter(torch.randn_like(x_init))
        else:
            raise ValueError("Unknown initialization in InverseSolver.")

        if decode_fn is None:
            decode_fn = lambda x: x

        optimizer = torch.optim.Adam([x], lr=0.1)
        for t in range(self.num_steps):
            error_sq = (fwd_fn(decode_fn(x)) - y.detach()).pow(2)
            if stop_eps is not None and error_sq.mean() <= stop_eps:
                # Stopping condition reached
                break
            loss = error_sq.sum() + self.lam * (x_init.detach() - x).pow(2).sum()
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return x.data

class LatentDataConsistency:
    def __init__(self, 
                fwd_operator, 
                noise_schedule, 
                encode_fn, 
                decode_fn, 
                dc_step, 
                ae_weight=0.0, 
                use_psld=False,
                z0_pred_corr_steps=0, 
                z0_pred_corr_lam=1.0, 
                z0_pred_corr_error_weighting=True, 
                z0_pred_corr_domain='image',
                z0_pred_corr_init_mode='randn',
                z0_pred_corr_stop_eps=None,
                stochastic_resample=False,
                resample_step=0.0,
                scaling_method='std', 
                scale_with_alphas=False, 
                fwd_seed=None
                ):
        self.dc_step = dc_step
        self.ae_weight = ae_weight
        self.use_psld = use_psld
        self.fwd_operator = fwd_operator
        self.fwd_seed = fwd_seed
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.scaling_method = scaling_method
        self.sigma = None
        self.noise_schedule = noise_schedule
        self.scale_with_alphas = scale_with_alphas
        self.z0_pred_corr_steps = z0_pred_corr_steps
        self.z0_pred_corr_lam = z0_pred_corr_lam
        self.z0_pred_corr_error_weighting = z0_pred_corr_error_weighting
        self.set_up_z0_pred_corr_domain(z0_pred_corr_domain)
        self.z0_pred_corr_init_mode = z0_pred_corr_init_mode
        self.z0_pred_corr_stop_eps = z0_pred_corr_stop_eps
        self.solver = InverseSolver(self.z0_pred_corr_steps, self.z0_pred_corr_lam, self.z0_pred_corr_init_mode)
        self.stochastic_resample = stochastic_resample
        self.resample_step = resample_step

    def set_up_z0_pred_corr_domain(self, z0_pred_corr_domain):
        if isinstance(z0_pred_corr_domain, str):
            if z0_pred_corr_domain == 'staged':
                self.z0_pred_corr_domain = {'chaotic': 'no_prox', 'semantic': 'image', 'refinement': 'latent'}
            elif z0_pred_corr_domain == 'image':
                self.z0_pred_corr_domain = {'chaotic': 'image', 'semantic': 'image', 'refinement': 'image'}
            elif z0_pred_corr_domain == 'latent':
                self.z0_pred_corr_domain = {'chaotic': 'latent', 'semantic': 'latent', 'refinement': 'latent'}
            else:
                raise ValueError('Unknown z0_pred_corr_domain value.')
        elif isinstance(z0_pred_corr_domain, list):
            assert len(z0_pred_corr_domain) == 3
            self.z0_pred_corr_domain = {'chaotic': z0_pred_corr_domain[0], 'semantic': z0_pred_corr_domain[1], 'refinement': z0_pred_corr_domain[2]}
        
    def update_y(self, y, t):
        self.y = y
        self.t = t
        if self.noise_schedule: # If noise schedule is None, don't need to rescale
            self.sigma = self.noise_schedule.get_std(self.t) * 2 # Scale noise from [0, 1] to [-1, 1]
        
    def update_z(self, z_mean, z_var_pred):
        self.z_mean = z_mean
        self.z_var_pred = z_var_pred
        
    def update_fwd_seed(self, seed):
        self.fwd_seed = seed
        
    def get_noised_z(self, noise_fn):
        return noise_fn(self.z_mean, self.z_var_pred)
    
    def apply_fwd(self, x):
        y_pred = self.fwd_operator(rescale_to_zero_one(x), self.t * torch.ones(1).to(x.device), seed=self.fwd_seed)
        y_pred = rescale_to_minusone_one(y_pred)
        return y_pred
    
    def apply_fwd_transpose(self, x):
        y_pred = self.fwd_operator.forward_transpose(rescale_to_zero_one(x), self.t * torch.ones(1).to(x.device), seed=self.fwd_seed)
        y_pred = rescale_to_minusone_one(y_pred)
        return y_pred
    
    def apply_z0_corr(self, z0_pred, domain):
        if self.z0_pred_corr_steps == 0:
            return z0_pred

        if domain == 'image' or self.z0_pred_corr_error_weighting:
            x0_pred = self.decode_fn(z0_pred)

        stop_eps = self.z0_pred_corr_stop_eps if self.z0_pred_corr_stop_eps is not None else self.sigma**2
        if domain == 'image':
            x0_corr = self.solver.solve(x0_pred, self.y, lambda x: self.apply_fwd(x), stop_eps=stop_eps)
            z0_corr = self.encode_fn(x0_corr)
        elif domain == 'latent':
            z0_corr = self.solver.solve(z0_pred, self.y, lambda x: self.apply_fwd(x), lambda x: self.decode_fn(x), stop_eps=stop_eps)
        else:
            raise ValueError('Unknown z0_pred_corr_domain value.')

        if self.z0_pred_corr_error_weighting:
            error = (self.y - self.apply_fwd(x0_pred)).pow(2).mean()
            beta = 1 - self.sigma**2 / torch.maximum(error, self.sigma**2)
            z0_corr = (1 - beta) * z0_pred + beta * z0_corr
        return z0_corr

    def get_dc_error(self, z0_pred):
        x0_pred = self.decode_fn(z0_pred)
        error = (self.y - self.apply_fwd(x0_pred)).pow(2).mean()
        return error
            
    def modify_score(self, z_past, z_next, z0_pred, scale=None):
        if self.scale_with_alphas:
            assert scale is not None
        else:
            scale = 1.0     

        x0_pred = self.decode_fn(z0_pred)
        y_pred = self.apply_fwd(x0_pred)
        error = (self.y - y_pred).pow(2).sum()
        if self.scaling_method == 'error':
            error = error / torch.sqrt(error)
        elif self.scaling_method == 'std':
            error = error / (self.sigma ** 2)
        elif self.scaling_method == 'none':
            error = error

        error = self.dc_step * error

        if self.ae_weight > 0.0:
            if self.use_psld:
                x0_corr = self.apply_fwd_transpose(self.y) + x0_pred - self.apply_fwd_transpose(self.apply_fwd(x0_pred))
            else:
                x0_corr = x0_pred
            ae_error = (z0_pred - self.encode_fn(x0_corr)).pow(2).sum()
            error += self.ae_weight * ae_error / torch.sqrt(ae_error)

        grad = torch.autograd.grad(error, z_past)[0]

        z_out = z_next - scale * grad
        return z_out
    
    def resample(self, z_next, z0_pred, alphas_cumprod, alphas_cumprod_prev, current_step):
        assert current_step is not None
        domain = self.get_pred_corr_domain_from_step(current_step)

        if domain == 'no_prox':
            return z_next

        resample_var = self.resample_step * (1.0 - alphas_cumprod_prev) / alphas_cumprod * (1.0 - alphas_cumprod / alphas_cumprod_prev)
        z0_corr = self.apply_z0_corr(z0_pred, domain)
        mean = (resample_var * alphas_cumprod ** 0.5 * z0_corr + (1.0 - alphas_cumprod) * z_next) / (resample_var + 1 - alphas_cumprod)
        z_next = mean + torch.randn_like(mean) * torch.sqrt(resample_var * (1 - alphas_cumprod) / (resample_var + 1 - alphas_cumprod))
        return z_next
    
    def get_stage_from_step(self, current_step):
        if current_step > 750:
            return 'chaotic'
        elif current_step > 300:
            return 'semantic'
        else:
            return 'refinement'
    
    def get_pred_corr_domain_from_step(self, current_step):
        current_stage = self.get_stage_from_step(current_step)
        return self.z0_pred_corr_domain[current_stage]
