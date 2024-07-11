#!/bin/bash
mkdir -p checkpoints
gdown --id 1dHEPyKhZOUxwoRK_R6DuBWhi-5cBTVV9 -O checkpoints/sevenc_gblur_lsun_bedroom.ckpt
gdown --id 1W0e5talN1UpOPNHYoFuJEK6dX8rimF4X -O checkpoints/sevenc_nl_blur_lsun_bedroom.ckpt
gdown --id 1KDY9NuhXW3UVRzz5toek3z3nYNTsJ4Kn -O checkpoints/sevenc_gblur_celeba256.ckpt
gdown --id 17d1VsNQS95-noq_uZO1uzqQOw_DsyWy4 -O checkpoints/sevenc_nl_blur_celeba256.ckpt
gdown --id 1TsvxKIT6Y4LsReWouAPDbxp90cn0ZGS5 -O checkpoints/sevenc_random_inpaint_celeba256.ckpt
gdown --id 1wtZjTZcUTUrHIagLaSWjNPJ_v1dahQay -O checkpoints/sevenc_random_inpaint_lsun_bedroom.ckpt