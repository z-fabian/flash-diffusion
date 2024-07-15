# FlashDiffusion
## Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models  (ICML 2024 Spotlight)
This is the official repository for the paper [Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models](https://openreview.net/pdf?id=V3OpGwo68Z).

> [**Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models**](https://openreview.net/pdf?id=V3OpGwo68Z),  
> Zalan Fabian<sup>\*</sup>, Berk Tınaz<sup>\*</sup>, Mahdi Soltanolkotabi  
> *ICML 2024*  
> <sup>\*</sup> equal contribution

We introduce *FlashDiffusion*, a reconstruction framework that automatically adapts inference time to the corruption level of the input. We leverage a *severity encoder* that estimates the corruption level in the latent space of autoencoders. Based on the estimated severity, we adjust the sampling trajectory of a latent diffusion process. FlashDiffusion acts as a wrapper that can be added to any latent diffusion-based inverse problem solver. FlashDiffusion enhances the baseline solver with sample-adaptivity and accelerated inference (up to a factor of 10x).

![](assets/flash_diffusion.gif)

The above example depicts the reconstruction of an easy, lightly corrupted sample (*top row*) and a difficult, heavily corrupted sample (*bottom row*). Existing solvers expend the same amount of compute to reconstruct both of these samples. FlashDiffusion dynamically adapts the number of reverse diffusion steps to the degradation severity of the sample, expending half as much compute for the easier reconstruction task than for the more difficult one.

## Installation
Follow these steps to install dependencies and download pretrained models for FlashDiffusion.

### 1. Clone repo and submodules
```bash
git clone --recurse-submodules https://github.com/z-fabian/flash-diffusion
cd flash-diffusion
```

### 2. Install packages
```bash
conda create -n flash_diffusion python=3.10 -y
conda activate flash_diffusion
pip install --upgrade pip
pip install -r requirements.txt
```
To download pretrained model checkpoints, you will also need `gdown`:
```bash
pip install gdown
```
To run the demo Notebook, also install `ipykernel`:
```bash
conda install ipykernel
```

### 3. Download pretrained models
We leverage pretrained models from the official [latent-diffusion](https://github.com/CompVis/latent-diffusion) repo for baseline solvers and as initialization for severity encoder training. The following script downloads LDMs and autoencoders for `celeba256`, `ffhq` and `lsun-bedroom` datasets into `ldm/models`:
```bash
bash ./scripts/download_first_stages.sh 
bash ./scripts/download_ldms.sh 
```
To download pretrained severity encoder checkpoints into the `checkpoints` folder (approx. 500MB) run the following script:
```bash
bash ./scripts/download_sev_encoders.sh
```
You can also download them one by one from the [links below](#pretrained-severity-encoders).

(Optional) To run nonlinear blur experiments, you will need to download the model (from [blur-kernel-space-exploring](https://github.com/VinAIResearch/blur-kernel-space-exploring)) that simulates realistic motion blur:
```bash
./scripts/download_nlblur_model.sh 
```
Now you are ready to [run the demo Notebook](demo.ipynb)! The demo will give you a high-level idea of severity encoding and walks you through the steps to deploying adaptive FlashDiffusion reconstruction on a sample image. 

### 4. (Optional) Download datasets
If you are planning on running experiments from the paper or training your own severity encoder, please follow the instructions in the next section to set up the datasets. You will have to modify [the dataset config file](flash_configs/data_configs/dataset_config.yaml) with the corresponding containing directories on your machine.



## Datasets
In order to avoid training data leakage from pretrained LDMs we match our train/val/test splits with the [official LDM paper repo](https://github.com/CompVis/latent-diffusion) as closely as possible. Here, we provide instructions how to download and set up each of the datasets. Once you downloaded the datasets, update the [dataset config file](flash_configs/data_configs/dataset_config.yaml) `path` field with your containing directory for each dataset.

### CelebA-HQ 256x256
There are two ways to obtain the dataset. You can directly download the dataset resized to 256x256 resolution in `.jpg` format from [Kaggle](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256/data). We created train/val/test splits based on the file naming used in this source. You can also follow the instructions in [progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans) to download and resize the images. Eventually, the library structure should look like this:
```
.../celeba_hq_256
├── 00000.jpg
├── 00001.jpg
├── ...
├── 29998.jpg
├── 29999.jpg
```
### FFHQ 256x256
To download the data follow the intructions in [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) and resize to 256x256 resolution. We create custom train/val/test splits following [latent-diffusion](https://github.com/CompVis/latent-diffusion). The dataset should follow this folder structure:
```
.../ffhq
├── 00000
│   ├── img00000000.png
│   ├── img00000001.png
│   ├── ...
│   ├── img00000999.png
├── 00001
│   ├── img00001000.png
│   ├── img00001001.png
│   ├── ...
│   ├── img00001999.png
├── ...
├── 00069
│   ├── img00069000.png
│   ├── img00069001.png
│   ├── ...
│   ├── img00069999.png
```

### LSUN Bedrooms
Follow instructions [here](https://github.com/fyu/lsun) to download the dataset and extract images from the downloaded `.mdb` files. Following [latent-diffusion](https://github.com/CompVis/latent-diffusion), we split the training folder of LSUN bedrooms into custom train/val/test splits.
The library structure should look like this:
```
.../bedroom_train
├── 000038527b455eaccd15e623f2e229ecdbceba2b.webp
├── 0000779b2a12face117e71cea6e0a60ef1a7faee.webp
├── ...
├── fffffa900959150cb53ac851b355ec4adbc22e4e.webp
├── fffffbb9225d069b7f47e464bdd75e6eff82b61c.webp
```

## Reconstruction
We have implemented some latent diffusion solvers, such as L-DPS, [GML-DPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/9c70cfa2e7d9328c649c94d50cbf8faf-Paper-Conference.pdf), [PSLD](https://proceedings.neurips.cc/paper_files/paper/2023/file/9c70cfa2e7d9328c649c94d50cbf8faf-Paper-Conference.pdf) and [ReSample](https://openreview.net/pdf?id=j8hdRqOUhN). These can be used as baseline solvers and enhanced with adaptivity through FlashDiffusion. Setting up baseline solver and Flash hyperparameters is done through config files. You can find configs for each main experiment in the paper [here](flash_configs/reconstruction_configs). To set up a custom config, take a look at the [annotated config file](flash_configs/reconstruction_configs/annotated_config.yaml).

As an example, we provide a script to reconstruct FFHQ samples under varying amounts of Gaussian blur using Flash(LDPS):
```bash
bash ./scripts/recon_ffhq_gblur_varying_ldps.sh
```

### Custom latent diffusion solvers
Beyond the provided solvers, one can add their own baseline solver by subclassing `samplers.laten_recon.LatentSampler`. The key is to implement `reconstruct_sample(z_start, t_start)` of the baseline solver that runs reconstruction starting at reverse diffusion time `t_start` from starting latent `z_start`. More details on the interface to be implemented can be found in [samplers/latent_recon.py](samplers/latent_recon.py). Once the new latent solver is implemented, one can simply update the `class` key in the experiment config file.

## Severity encoder training
We provide an example script to train a severity encoder from LDM autoencoder initialization:
```bash
bash ./scripts/train_celeba256_gblur_varying.sh
```
We tested the training code on 8x RTX A6000 (48GB) and 8x Titan RTX (24GB) GPUs.

## Pretrained severity encoders
Each checkpoint is approximately 85MB.
| Train dataset      | Operator  | Link |
| ------------ | ---------- | ---- |
| CelebA-HQ | Gaussian blur + noise | [Download](https://drive.google.com/file/d/1KDY9NuhXW3UVRzz5toek3z3nYNTsJ4Kn/view?usp=sharing) |
| CelebA-HQ |Nonlinear blur + noise | [Download](https://drive.google.com/file/d/17d1VsNQS95-noq_uZO1uzqQOw_DsyWy4/view?usp=sharing) |
| CelebA-HQ | Random inpainting + noise | [Download](https://drive.google.com/file/d/1TsvxKIT6Y4LsReWouAPDbxp90cn0ZGS5/view?usp=sharing) |
| LSUN Bedrooms | Gaussian blur + noise | [Download](https://drive.google.com/file/d/1dHEPyKhZOUxwoRK_R6DuBWhi-5cBTVV9/view?usp=sharing) |
| LSUN Bedrooms |Nonlinear blur + noise | [Download](https://drive.google.com/file/d/1W0e5talN1UpOPNHYoFuJEK6dX8rimF4X/view?usp=sharing) |
| LSUN Bedrooms | Random inpainting + noise | [Download](https://drive.google.com/file/d/1wtZjTZcUTUrHIagLaSWjNPJ_v1dahQay/view?usp=sharing) |

## Citation

If you find our paper useful, please cite

```bibtex
@inproceedings{fabianadapt,
  title={Adapt and Diffuse: Sample-adaptive Reconstruction via Latent Diffusion Models},
  author={Fabian, Zalan and Tinaz, Berk and Soltanolkotabi, Mahdi},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

## Acknowledgments
This repository builds upon code from
- [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion) for diffusion models and autoencoders.
- [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://github.com/DPS2022/diffusion-posterior-sampling) for some operators.
- [Exploring Image Deblurring via Encoded Blur Kernel Space](https://github.com/VinAIResearch/blur-kernel-space-exploring) for nonlinear blur model.
