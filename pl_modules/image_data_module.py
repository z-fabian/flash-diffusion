from argparse import ArgumentParser
import lightning.pytorch as pl
import torch
from data_utils.image_data import CelebaDataset, FFHQDataset, LSUNBedroomDataset

class CelebaDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        data_path, 
        train_transform, 
        val_transform, 
        test_transform, 
        batch_size, 
        sample_rate_dict, 
        distributed_sampler,
        num_workers=4,
    ):

        super().__init__()

        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.sample_rate_dict = sample_rate_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_split,
        data_transform,
        batch_size=None
    ) :
        sample_rate = self.sample_rate_dict[data_split]
        dataset = CelebaDataset(
            root=self.data_path,
            split=data_split,
            transform=data_transform,
            sample_rate=sample_rate,
        )

        batch_size = self.batch_size if batch_size is None else batch_size
        sampler = torch.utils.data.DistributedSampler(dataset) if self.distributed_sampler else None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=False
        )
        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(data_split='train', data_transform=self.train_transform)

    def val_dataloader(self):
        return self._create_data_loader(data_split='val', data_transform=self.val_transform)

    def test_dataloader(self):
        return self._create_data_loader(data_split='test', data_transform=self.test_transform)

    
class FFHQDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        data_path, 
        train_transform, 
        val_transform, 
        test_transform, 
        batch_size, 
        sample_rate_dict, 
        distributed_sampler,
        num_workers=4,
    ):

        super().__init__()

        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.sample_rate_dict = sample_rate_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_split,
        data_transform,
        batch_size=None
    ) :
        sample_rate = self.sample_rate_dict[data_split]
        dataset = FFHQDataset(
            root=self.data_path,
            split=data_split,
            transform=data_transform,
            sample_rate=sample_rate,
        )

        batch_size = self.batch_size if batch_size is None else batch_size
        sampler = torch.utils.data.DistributedSampler(dataset) if self.distributed_sampler else None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=False
        )
        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(data_split='train', data_transform=self.train_transform)

    def val_dataloader(self):
        return self._create_data_loader(data_split='val', data_transform=self.val_transform)

    def test_dataloader(self):
        return self._create_data_loader(data_split='test', data_transform=self.test_transform)

    
class LSUNBedroomDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        data_path, 
        train_transform, 
        val_transform, 
        test_transform, 
        batch_size, 
        sample_rate_dict, 
        distributed_sampler,
        num_workers=4,
    ):

        super().__init__()

        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.sample_rate_dict = sample_rate_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_split,
        data_transform,
        batch_size=None
    ) :
        sample_rate = self.sample_rate_dict[data_split]
        dataset = LSUNBedroomDataset(
            root=self.data_path,
            split=data_split,
            transform=data_transform,
            sample_rate=sample_rate,
        )

        is_train = (data_split == 'train')
        batch_size = self.batch_size if batch_size is None else batch_size
        sampler = torch.utils.data.DistributedSampler(dataset) if self.distributed_sampler else None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=False
        )
        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(data_split='train', data_transform=self.train_transform)

    def val_dataloader(self):
        return self._create_data_loader(data_split='val', data_transform=self.val_transform)

    def test_dataloader(self):
        return self._create_data_loader(data_split='test', data_transform=self.test_transform)


def add_data_specific_args(parent_parser):
    """
    Define parameters that only apply to this model
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # dataset arguments
    parser.add_argument(
        "--sample_rates",
        type=float,
        nargs='+', 
        default=[1.0, 1.0, 1.0],
        help="Fraction of images in the dataset to use in the following order: train, val, test. If not given all will be used.",
    )
        
    # data loader arguments
    parser.add_argument(
        "--batch_size", 
        default=32, 
        type=int, 
        help="Data loader batch size"
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=float,
        help="Number of workers to use in data loader",
    )

    return parser