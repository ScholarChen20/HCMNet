# HCMNet: A Hybrid CNN-Mamba Architecture for Medical Ultrasound Image  Segmentation

Official repository for: *[HCMNet: A Hybrid CNN-Mamba Architecture for Medical Ultrasound Image  Segmentation](https://arxiv.org/abs/2402.03302)*

![network]()

The source code will be released to the public in the near future.

If you have any questions, please contact: chen_wenqin2002@163.com

## Installation

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


## Citation

```
@article{HCMNet,
    title={HCMNet: A Hybrid CNN-Mamba Architecture for Medical Ultrasound Image  Segmentation},
    author={Wenqin Chen},
    journal={arXiv preprint arXiv:2402.03302},
    year={2024}
}
```
