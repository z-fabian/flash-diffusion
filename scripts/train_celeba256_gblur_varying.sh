#!/bin/bash
python scripts/train_severity_encoder.py --dataset celeba256 \
                                    --devices 8 \
                                    --batch_size 4 \
                                    --lr_gamma 0.1 \
                                    --max_epochs 100 \
                                    --lr 0.0001 \
                                    --weight_decay 0.0 \
                                    --logger_type wandb \
                                    --experiment_name sev-enc-celeba256 \
                                    --experiment_config_file flash_configs/inverse_configs/gaussian_blur_fixed_noise.yaml \
                                    --ldm_model_ckpt_path ldm/models/ldm/celeba256/model.ckpt \
                                    --ldm_model_config_path ldm/models/ldm/celeba256/config.yaml \
                                    --sigma_reg 1.0 \
                                    --img_space_reg 1.0