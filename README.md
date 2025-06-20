# HCMNet: A Hybrid CNN-Mamba Architecture for Medical Ultrasound Image  Segmentation

Official repository for: *[HCMNet: A Hybrid CNN-Mamba Architecture for Medical Ultrasound Image  Segmentation](https://arxiv.org/abs/2402.03302)*

[network](https://github.com/ScholarChen20/HCMNet/tree/main/assets/HCMNet.png)

The source code will be released to the public in the near future.

If you have any questions, please contact: chen_wenqin2002@163.com

# Main Results

- BUSI
- STU
- DDTI
- SZU-BCH-TUS983

# Installation

**Step-1:** Create a new conda environment & install requirements

```shell
conda create -n swin_umamba python=3.10
conda activate swin_umamba

pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm
pip install torchinfo timm numba

**ImageNet pretrained model:** 

We use the ImageNet pretrained VMamba-Tiny model from [VMamba](https://github.com/MzeroMiko/VMamba). You need to download the model checkpoint and put it into `data/pretrained/vmamba/vmamba_tiny_e292.pth`

```
wget https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth
mv vssmtiny_dp01_ckpt_epoch_292.pth data/pretrained/vmamba/vmamba_tiny_e292.pth
```


## Acknowledgements

We thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba), [UMamba](https://github.com/bowang-lab/U-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba),[Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba) and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) for making their valuable code & data publicly available.

