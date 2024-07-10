import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import resize, center_crop
from scripts.utils import str2int, rescale_to_minusone_one
from data_utils.operators import create_operator, create_noise_schedule


class SevEncInputTransform:

    def __init__(self):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
        
    def __call__(self, image):
        image = self.transform(image)
        return image.unsqueeze(0)

class ImageDataTransform:
    
    def __init__(self, 
                 is_train, 
                 operator_schedule,
                 noise_schedule=None,
                 fixed_t=None,
                 t_range=None,
                 range_zero_one=False,
                ):
        self.is_train = is_train   
        self.range_zero_one = range_zero_one
        if isinstance(operator_schedule, dict):
            self.fwd_operator = create_operator(operator_schedule)
        else:
            self.fwd_operator = operator_schedule
        
        if noise_schedule is None:
            self.noise_scheduler = None
        elif isinstance(noise_schedule, dict):
            self.noise_scheduler = create_noise_schedule(noise_schedule)
        else:
            self.noise_scheduler = noise_schedule
        self.fixed_t = fixed_t
        self.t_range = t_range

    @torch.no_grad()
    def __call__(self, 
                 image, 
                 fname=None
                ):

        # Crop image to square 
        shorter = min(image.size)
        image = center_crop(image, shorter)
        
        # Resize images to uniform size
        image = resize(image, (256, 256))
        
        # Convert to ndarray and permute dimensions to C, H, W
        image = np.array(image)
        image = image.transpose(2, 0, 1)
        
        # Normalize image to range [0, 1]
        image = image / 255.
    
        # Convert to tensor
        image = torch.from_numpy(image.astype(np.float32))
        image = image.unsqueeze(0)
        
        if not self.is_train:  # deterministic forward model for validation
            assert fname is not None
            seed = str2int(fname)
        else:
            seed = None
            
        # Generate degraded noisy images
        if self.fixed_t:
            t = torch.tensor(self.fixed_t)
        elif self.t_range is not None:
            if not self.is_train:
                    g = torch.Generator()
                    g.manual_seed(seed)
                    t = torch.rand(1, generator=g) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
            else:
                t = torch.rand(1) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        else:
            if not self.is_train:
                g = torch.Generator()
                g.manual_seed(seed)
                t = torch.rand(1, generator=g)
            else:
                t = torch.rand(1)
            
        degraded = self.fwd_operator(image, t, seed=seed).squeeze(0) 
        
        if self.noise_scheduler:
            z, noise_std = self.noise_scheduler(t, image.shape, seed=seed)
            degraded_noisy = degraded + z.to(image.device)
            noisy = image + z.to(image.device)
        else: 
            degraded_noisy = degraded
            noisy = image
            noise_std = 0.0
        
        image = image.squeeze(0)
        degraded_noisy = degraded_noisy.squeeze(0)
        noisy = noisy.squeeze(0)
        return {
                'clean': image if self.range_zero_one else rescale_to_minusone_one(image), 
                'degraded': degraded if self.range_zero_one else rescale_to_minusone_one(degraded), 
                'degraded_noisy': degraded_noisy if self.range_zero_one else rescale_to_minusone_one(degraded_noisy), 
                'noise_std': noise_std,
                'noisy': noisy if self.range_zero_one else rescale_to_minusone_one(noisy),
                't': t,
                'fname': fname,
               }