"""
Author: Omid Nejati Manzari
Date: Jun  2023
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .Encoder import All2Cross
from .Decoder import ConvUpsample, SegmentationHead


class BEFUnet(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 32]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)

        self.ConvUp_s = ConvUpsample(in_chans=768, out_chans=[128, 128, 128], upsample=True)  # 1
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)  # 0

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):
            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size // self.patch_size[i]),
                              w=(self.img_size // self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)

            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        out = self.segmentation_head(C)

        return out

import ml_collections
import os
import wget

os.makedirs('./pretrained_ckpt', exist_ok=True)

# BEFUnet Configs
def get_BEFUnet_configs():
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9
    if not os.path.isfile('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth'):
        print('Downloading Swin-transformer model ...')
        wget.download(
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
            "./pretrained_ckpt/swin_tiny_patch4_window7_224.pth")
    cfg.swin_pretrained_path = './pretrained_ckpt/swin_tiny_patch4_window7_224.pth'

    # CNN Configs
    cfg.cnn_backbone = "pidinet_small_converted"
    cfg.pdcs = 'carv4'
    cfg.cnn_pyramid_fm = [30, 60, 120, 120]
    cfg.pidinet_pretrained = False
    cfg.PDC_pretrained_path = './pretrained_ckpt/table5_pidinet-small.pth'

    # DLF Configs
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio = (2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg

def get_BEFUNet():
    cfg = get_BEFUnet_configs()
    model = BEFUnet(config=cfg, img_size=224, in_chans=3, n_classes=1).cuda()
    return model

if __name__ == '__main__':
    cfg = get_BEFUnet_configs()
    model = BEFUnet(config=cfg,img_size=224, in_chans=3, n_classes=1).cuda()
    x = torch.randn(1,3,224,224).cuda()
    out = model(x)
    print(out.shape)
