import os, sys
import pathlib
from argparse import ArgumentParser
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )
import lightning.pytorch as pl
import yaml
import torch.distributed
from pl_modules.image_data_module import add_data_specific_args
from  pl_modules.severity_encoder_module import SeverityEncoderModule
from data_utils.data_transforms import ImageDataTransform
from data_utils.image_data import DATA_CONFIG_PATH
from scripts.utils import load_config_from_yaml
from ldm.ldm.util import get_obj_from_str

def cli_main(args):
    
    if args.verbose:
        print(args.__dict__)
        print('pytorch-lightning version: {}'.format(pl.__version__))
                
    pl.seed_everything(args.seed)
    
    # Set up schedules
    exp_config = load_config_from_yaml(args.experiment_config_file)
    operator_config = exp_config['operator']
    noise_config = exp_config['noise']
        
    # ------------
    # model
    # ------------
    model = SeverityEncoderModule(
        operator_config=operator_config,
        noise_config=noise_config,
        ldm_model_ckpt_path=args.ldm_model_ckpt_path,
        ldm_model_config_path=args.ldm_model_config_path,
        sev_encoder_config_path=args.sev_encoder_config_path,
        sev_encoder_ckpt_path=args.sev_encoder_ckpt_path,
        pretrained_encoder_ckpt_path=args.pretrained_encoder_ckpt_path,
        pretrained_encoder_config_path=args.pretrained_encoder_config_path,
        sigma_reg=args.sigma_reg,
        img_space_reg=args.img_space_reg,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_milestones=args.lr_milestones,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        logger_type=args.logger_type,
    )
    
    # ------------
    # data
    # ------------
    data_config = load_config_from_yaml(DATA_CONFIG_PATH) 
    assert args.dataset in data_config
    data_config = data_config[args.dataset]
    dataset_class = data_config['pl_module_class']
    data_path = data_config['path']
    
    train_transform = ImageDataTransform(is_train=True, operator_schedule=operator_config, noise_schedule=noise_config)
    val_transform = ImageDataTransform(is_train=False, operator_schedule=operator_config, noise_schedule=noise_config)
    test_transform = ImageDataTransform(is_train=False, operator_schedule=operator_config, noise_schedule=noise_config)
    
    # ptl data module - this handles data loaders
    data_module = get_obj_from_str(dataset_class)(
        data_path=data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        sample_rate_dict={'train': args.sample_rates[0], 'val': args.sample_rates[1], 'test': args.sample_rates[2]},
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=True,
    )

    # ------------
    # trainer
    # ------------
    # set up logger
    if args.output_dir is not None:
        default_root_dir = args.output_dir
    else:
        default_root_dir = os.path.join('outputs', args.experiment_name)
    
    if args.logger_type == 'tb':
        logger = True
    elif args.logger_type == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.experiment_name, save_dir=default_root_dir)
    else:
        raise ValueError('Unknown logger type.')
    
    if args.output_dir is not None:
        default_root_dir = args.output_dir
    else:
        default_root_dir = os.path.join('outputs', args.experiment_name)
        
    callbacks=[]
    callbacks.append(args.checkpoint_callback)
    trainer = pl.Trainer(default_root_dir=default_root_dir,
                        max_epochs=args.max_epochs,
                        devices=args.devices,
                        accelerator=args.accelerator, 
                        enable_checkpointing=True,
                        callbacks=callbacks,
                        logger=logger,
                        strategy='ddp_find_unused_parameters_false', 
)
    
    # Save all hyperparameters to .yaml file in the current log dir
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                save_all_hparams(trainer, args)
    else: 
         save_all_hparams(trainer, args)     
            
    # ------------
    # run
    # ------------
    trainer.fit(model, datamodule=data_module)

def save_all_hparams(trainer, args):
    if not os.path.exists(trainer.logger.log_dir):
        os.makedirs(trainer.logger.log_dir)
    save_dict = args.__dict__
    save_dict.pop('checkpoint_callback')
    with open(trainer.logger.log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)
    
def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        '--dataset', 
        type=str,          
        help='Dataset to train on. Loads data path from config file. Options: celeba256, ffhq, lsun_bedroom',
    )
    parser.add_argument(
        '--experiment_config_file', 
        type=pathlib.Path,          
        help='Experiment configuration will be loaded from this file.',
    )
    parser.add_argument(
        '--verbose', 
        default=False,   
        action='store_true',          
        help='If set, print all command line arguments at startup.',
    )
    parser.add_argument(
        '--logger_type', 
        default='wandb',   
        type=str,          
        help='Set Pytorch Lightning training logger. Options "tb" - Tensorboard, "wandb" - Weights and Biases (default)',
    )
    parser.add_argument(
        '--experiment_name', 
        default='sev-encoder-train',   
        type=str,          
        help='Used with wandb logger to define the project name.',
    )
    parser.add_argument(
        '--output_dir', 
        default=None,   
        type=str, 
        help='Output directory for logs and checkpoints. Defaults to the outputs folder.',         
    )
    parser.add_argument(
        '--accelerator', 
        default='gpu',   
        type=str,          
    )
    parser.add_argument(
        '--devices', 
        default=8,   
        type=int,
        help='Number of GPUs to train on.',          
    )
    parser.add_argument(
        '--seed', 
        default=42,   
        type=int,
        help='Global seed used to seed everything on start.',          
    )
    parser.add_argument(
        '--max_epochs', 
        default=100,   
        type=int,
        help='Number of training epochs.',          
    )

    # data config
    parser = add_data_specific_args(parser)
    parser.set_defaults(
        test_path=None, 
    )

    # module config
    parser = SeverityEncoderModule.add_model_specific_args(parser)
    
    args = parser.parse_args()

    args.checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/loss",
        mode="min",
        filename='epoch{epoch}-val-loss{val/loss:.4f}',
        auto_insert_metric_name=False,
        save_last=True
    )

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()