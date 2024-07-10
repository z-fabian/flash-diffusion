from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Subset
from ldm.ldm.util import get_obj_from_str
from scripts.utils import load_config_from_yaml
from data_utils.data_transforms import ImageDataTransform

DATA_CONFIG_PATH = 'flash_configs/data_configs/dataset_config.yaml'

def load_ids_from_txt(path):
    with open(path) as f:
        ids = list(f)
    ids = [l.rstrip('\n') for l in ids]
    return ids

def get_fname(full_path):
    full_path = Path(full_path)
    return str(full_path).replace(str(full_path.parents[1]), '').replace(str(full_path.suffix), '')

class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        split,
        sample_rate=None,
        num_images_per_class=None,
        transform=None,
        shuffle_seed=0
    ):

        self.transform = transform
        self.examples = []
        self.idx_to_synset = load_config_from_yaml('ldm/data/index_synset.yaml')

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        else:
            assert num_images_per_class is None # either use subsampling, or give number of images per class
        
        root_folder = (Path(root) / 'train') if split == 'train' else (Path(root) / 'val') # create test data out of validation dataset
                
        for subfolder in sorted(list(root_folder.iterdir())):
            if subfolder.is_dir():
                files_in_class = [file for file in sorted(list(Path(subfolder).iterdir())) if file.suffix in ['.JPG', '.JPEG', '.jpg', '.jpeg']]
                if num_images_per_class is None:
                    self.examples.extend(files_in_class)
                else:
                    if split in ['train', 'val']: # use start of the dataset
                        self.examples.extend(files_in_class[:num_images_per_class])
                    elif split == 'test': # use end of dataset
                        self.examples.extend(files_in_class[-num_images_per_class:])
                    else:
                        raise ValueError("Invalid split.")

        # only keep synsets defined in the config
        self.examples = [file for file in self.examples if str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '').split('/')[-2] in list(self.idx_to_synset.values())]
        
        # shuffle 
        state = random.getstate()
        random.seed(shuffle_seed)
        random.shuffle(self.examples)
        
        # subsample if desired
        if sample_rate < 1.0: 
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        random.setstate(state)
            
        print('{} images loaded from {} for {} split.'.format(len(self.examples), str(root), split))
                  
    def class_id_from_synset(self, s):
        for cid, syn in self.idx_to_synset.items():
            if s == syn:
                return cid
        raise ValueError('Synset {} not in config file.'.format(s))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        file = self.examples[i]
        fname = str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '')
        im = Image.open(file).convert("RGB") # Will load grayscale images as RGB!

        if self.transform is None:
            raise ValueError('Must define forward model and pass in DataTransform.')
        else:
            sample = self.transform(im, fname)
        synset = fname.split('/')[-2]
        cid = self.class_id_from_synset(synset)
        sample['cid'] = cid
        return sample
    
    def get_filenames(self):
        filenames = [str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '') for file in self.examples]
        return filenames
    
class CelebaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        split,
        transform,
        sample_rate=None,
    ):

        self.transform = transform
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0

        for file in list(Path(root).iterdir()):
            if file.suffix in ['.JPG', '.JPEG', '.jpg', '.jpeg']:
                suffix = file.suffix
                break
        
        data_config = load_config_from_yaml(DATA_CONFIG_PATH)
        img_ids = load_ids_from_txt(data_config['celeba256'][split+'_split'])
        for i in img_ids:
            file = Path(root)/(i + suffix)
            if file.is_file():
                self.examples.append(file)
            else:
                raise ValueError("CelebA image {} is missing.".format(i))

        # subsample if desired
        if sample_rate < 1.0: 
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        print('{} images loaded from {} as {} split.'.format(len(self.examples), str(root), split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        file = self.examples[i]
        fname = get_fname(file)
        im = Image.open(file).convert("RGB")

        if self.transform is None:
            raise ValueError('Must define forward model and pass in DataTransform.')
        else:
            sample = self.transform(im, fname)
            
        return sample
    
class FFHQDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        split,
        transform,
        sample_rate=None,
    ):

        self.transform = transform
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
            
        suffix = '.png'
            
        data_config = load_config_from_yaml(DATA_CONFIG_PATH)
        img_ids = load_ids_from_txt(data_config['ffhq'][split+'_split'])
        
        for i in img_ids:
            folder = str(int(i) // 1000).rjust(5, '0')
            file = Path(root)/folder/('img' + i.rjust(8, '0') + suffix)
            if file.is_file():
                self.examples.append(file)
            else:
                raise ValueError("FFHQ image {} is missing.".format(i))
        
        # subsample if desired
        if sample_rate < 1.0: 
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        print('{} images loaded from {} as {} split.'.format(len(self.examples), str(root), split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        file = self.examples[i]
        fname = str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '')
        im = Image.open(file).convert("RGB")

        if self.transform is None:
            raise ValueError('Must define forward model and pass in DataTransform.')
        else:
            sample = self.transform(im, fname)
            
        return sample
    
    def get_filenames(self):
        filenames = [str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '') for file in self.examples]
        return filenames
    
class LSUNBedroomDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        split,
        transform,
        sample_rate=None,
    ):

        self.transform = transform
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
            
        data_config = load_config_from_yaml(DATA_CONFIG_PATH)
        if split in ['val', 'test']:
            img_ids = load_ids_from_txt(data_config['lsun_bedroom'][split+'_split'])
        else:
            assert split == 'train' # Load complement of val/test splits
            val_ids = load_ids_from_txt(data_config['lsun_bedroom']['val_split'])
            test_ids = load_ids_from_txt(data_config['lsun_bedroom']['test_split'])
            all_ids = [str(p.name) for p in Path(root).iterdir() if p.is_file() and p.suffix == '.webp']
            img_ids = [id for id in all_ids if (id not in val_ids and id not in test_ids)]
        
        for i in img_ids:
            file = Path(root)/i
            if file.is_file():
                self.examples.append(file)
            else:
                raise ValueError("LSUN Bedroom image {} is missing.".format(i))
        
        # subsample if desired
        if sample_rate < 1.0: 
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        print('{} images loaded from {} as {} split.'.format(len(self.examples), str(root), split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        file = self.examples[i]
        fname = str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '')
        im = Image.open(file).convert("RGB")

        if self.transform is None:
            raise ValueError('Must define forward model and pass in DataTransform.')
        else:
            sample = self.transform(im, fname)
            
        return sample
    
    def get_filenames(self):
        filenames = [str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '') for file in self.examples]
        return filenames

def get_dataloader(config, rnd_generator=None):
    dataset_config = load_config_from_yaml('flash_configs/data_configs/dataset_config.yaml')
    dataset_key = config['dataset']
    dataset_path = dataset_config[dataset_key]['path']
    dataset_class = dataset_config[dataset_key]['dataset_class']

    degradation_config = load_config_from_yaml(config['degradation_config'])

    if 0.0 <= config['fixed_severity'] <= 1.0:
        fixed_t = config['fixed_severity']
    else:
        fixed_t = None
    data_transform = ImageDataTransform(
        is_train=False, 
        operator_schedule=degradation_config['operator'],
        noise_schedule=degradation_config['noise'],
        fixed_t=fixed_t,
        t_range=config['t_range'] if 't_range' in config else None,
    )
        
    dataset = get_obj_from_str(dataset_class)(
        root=dataset_path,
        split=config['split'],
        transform=data_transform,
        )
    
    dataset = Subset(dataset, range(config['num_images']))
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=4,
            generator=rnd_generator,
            worker_init_fn=seed_worker,
        ) 
    return dataloader