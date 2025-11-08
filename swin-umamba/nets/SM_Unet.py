import re
from einops import rearrange
from mamba_ssm import Mamba
from sympy import false
from timm.models.layers import trunc_normal_, DropPath,to_2tuple
from torch.nn import Upsample
from torchvision import models
from torchvision.models import resnet18
from torchvision.ops import DeformConv2d
import math
from .SwinUMambaD import VSSMEncoder,UNetResDecoder,TIF
from .MedMamba import MedMamba
from ptflops import get_model_complexity_info
# from torch import autocast

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        skip = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            skip.append(x)
        return self.norm(x.mean([-2, -1])),skip  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x,skip = self.forward_features(x)
        x = self.head(x)
        return x,skip


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=True, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):#dilation是
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x


class EncoderBlock(nn.Module):
    """Encoding then downsampling"""

    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.GELU()
        self.block = nn.Sequential(
            AxialMambaBlock(in_c),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        skip = self.bn(self.dw(x))
        # x = self.act(self.down(self.pw(skip)))
        # print(x.shape)   B C H W
        x = self.block(skip)
        assert not torch.isnan(x).any(),"EncoderBlock出现Nan"
        return x, skip

class DecoderBlock(nn.Module):
    """Upsampling then decoding"""

    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.pw = nn.Conv2d(in_c+out_c, out_c, kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Upsample(scale_factor=2)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw1 = AxialDW(out_c, mixer_kernel=(7, 7),dilation=1)
        self.dw2 = AxialDW(out_c, mixer_kernel=(7, 7),dilation=3)
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
    def forward(self, x, skip):

        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        # x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))

        x = self.bn(self.pw(x))
        x = self.act(self.pw2(self.dw1(x)+self.dw2(x)))
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    # @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)   # B C L
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        # print(out.shape)
        return out


class MambaMLPLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=4):
        super(MambaMLPLayer, self).__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

        self.mlp_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        y = self.mamba_norm(x)
        y = self.mamba(y)

        x = x + y
        y = self.mlp_norm(x)
        y = self.mlp(y)
        x = x + y

        return x

### 用于多尺度结合
class TriMamba(nn.Module):
    """Axial dilated DW convolution"""

    def __init__(self, norm_layer = nn.LayerNorm,dim=512, width=8):
        super().__init__()
        self.norm = norm_layer(normalized_shape=dim)
        self.mamba1 = MambaLayer(dim=dim)
        self.mamba2 = MambaLayer(dim=width)
        self.act = nn.Sigmoid()


    def forward(self, x):
        B, C, H, W = x.shape    #  1 512 8  8

        x1 = x.flatten(2)  # (B,C,L)  1,512,64
        x_per1 = x.permute(0, 2, 3, 1).contiguous().flatten(2)  # B H W C -> B H L
        x_per2 = x.permute(0, 3, 1, 2).contiguous().flatten(2)  # B W C H -> B W L
        # print(x1.shape,x_per1.shape,x_per2.shape)# torch.Size([1, 512, 64]) torch.Size([1, 8, 4096]) torch.Size([1, 8, 4096])

        out1 = self.mamba1(x1)
        out2 = self.mamba2(x_per1)
        out3 = self.mamba2(x_per2)


        out1 = self.norm(torch.transpose(out1, dim0=1, dim1=2).contiguous().view(B, H, W, -1))
        out2 = self.norm(torch.transpose(out2, dim0=1, dim1=2).contiguous().view(B, H, W, -1))
        out3 = self.norm(torch.transpose(out3, dim0=1, dim1=2).contiguous().view(B, H, W, -1))

        out1 = self.act(out1)
        out2 = self.act(out2)
        out3 = self.act(out3)

        out = (out1+out2+out3)/3
        out = out.view(B, C, H, W)

        return out

class AxialMambaBlock(nn.Module):
    def __init__(self, dim, mixer_kernel=(3, 3), expansion=2):
        super().__init__()
        # 轴向卷积分支
        self.axial_conv = AxialDW(dim,(7,7))
        # Mamba分支
        self.mamba = MambaLayer(dim)

        # 动态门控融合
        self.gate = nn.Conv2d(dim * 2, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        conv_path = self.axial_conv(x)
        # conv_path = self.axial_conv1(x)+self.axial_conv2(x)+self.axial_conv3(x)
        x_seq = rearrange(x, 'b c h w -> b (h w) c') # B L C
        mamba_path = self.mamba(x_seq.permute(0,2,1))
        mamba_path = rearrange(mamba_path, 'b c (h w) -> b c h w',h=H,w=W)

        # 自适应融合
        fused = torch.cat([conv_path, mamba_path], dim=1)
        gates = torch.softmax(self.gate(fused), dim=1)
        out = gates[:, 0:1] * conv_path + gates[:, 1:2] * mamba_path
        return self.bn(out) + x

class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""

    def __init__(self, dim):
        super().__init__()

        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)

        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.act = nn.GELU()

        # self.trimamba  = TriMamba(dim=dim,width=8)
    def forward(self, x):
        # x1= self.trimamba(x)
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        # return x+x1
        return x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SM_Unet(nn.Module):
    def __init__(self,in_channels=3,n_classes=1):
        super().__init__()
        # 初始化所有卷积层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Mamba层特殊初始化
        for m in self.modules():
            if isinstance(m, Mamba):
                nn.init.normal_(m.A_log, mean=0, std=0.01)  # 状态矩阵初始化
                nn.init.normal_(m.D, mean=0.5, std=0.02)  # 跳跃连接初始化
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    if not getattr(m.bias, "_no_reinit", False):
                        nn.init.zeros_(m.bias)

        self.apply(initialize_weights)

        """Encoder"""
        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')# 16 256 256
        self.e1 = EncoderBlock(16, 32,)  # 32 128 128
        self.e2 = EncoderBlock(32, 64,)   # 64 64 64
        self.e3 = EncoderBlock(64, 128,)  # 128 32 32
        self.e4 = EncoderBlock(128, 256,)  # 256 64 64
        self.e5 = EncoderBlock(256, 512,)  # 512 8 8

        """Bottle Neck"""
        self.b5 = BottleNeckBlock(dim=512)

        """Decoder"""
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)  # (512, 8, 8)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x
        # triMamba 3.5M 2.17/3.14M 2.17   #SMFormer 34.61M 6.36    34.51M 6.39


##########SMFormer新架构%%%%%%%%%%%%%%%
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)          # 通道注意力
        x = x * self.spatial_attention(x)             # 空间注意力
        return x

class DBFM(nn.Module):
    def __init__(self, in_channels):
        super(DBFM, self).__init__()
        self.cbr=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.CBAM = AttentionModule(in_channels)
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, y):
        xy  = self.cbr(self.cbr(x) * self.cbr(y)) # 残差连接
        x = self.CBAM(x)#  通道注意力
        y = self.sigmoid(y)*self.cbr(self.channel_pool(x))  #cnn分支通道特征池化，与sigmoid激活函数结合使用
        fuse = torch.concat([x, xy, y], dim=1)
        out = self.residual(fuse)+self.cbr(self.residual(fuse))
        return out


class PyramidPoolUnit(nn.Module):
    """ Pyramid Pooling Unit (PPU) """
    def __init__(self, in_channels, pool_sizes=[1, 3, 6]):
        super(PyramidPoolUnit, self).__init__()
        self.pooling = nn.ModuleList([nn.AdaptiveAvgPool2d(size) for size in pool_sizes])  #  [1,3,6] 池化后维度是多少[B,C,H,W]
        self.conv = nn.Conv2d(in_channels * (len(pool_sizes) + 1), in_channels, kernel_size=1)

    def forward(self, x):
        pooled_features = [F.interpolate(pool(x), size=x.shape[2:], mode='bilinear', align_corners=False) for pool in
                           self.pooling]  #  [B,C,H,W]
        return self.conv(torch.cat([x] + pooled_features, dim=1))   # 768 8 8


class MCFFFD(nn.Module):
    def __init__(self, in_channels, in_channel_list=None,deep_supervised=True):
        super(MCFFFD, self).__init__()
        if in_channel_list is None:
            in_channel_list = [96, 192, 384]
        self.up = Upsample(scale_factor=4)
        # self.SDU = nn.ModuleList()
        # for ch in in_channel_list:
        #     self.SDU.append(BottleNeckBlock(ch))

        self.PPU = PyramidPoolUnit(in_channels)
        self.decoder3= DecoderBlock(768,384)
        self.decoder2 = DecoderBlock(384,192)
        self.decoder1 = DecoderBlock(192,96)
        self.final_conv = nn.Conv2d(96, 1, kernel_size=1)
        self.deep_supervised = deep_supervised
        self.seg_outs = nn.ModuleList([
            nn.Conv2d(ch, 1, 1, 1) for ch in in_channel_list])

    def forward(self, features):
        # sdu_out = []
        # for i,s in enumerate(self.SDU):
        #     sdu_out.append(features[i])
            # f = self.SDU[i](features[i])
            # sdu_out.append(f)  #[1, 96, 64, 64]), torch.Size([1, 192, 32, 32]), torch.Size([1, 384, 16, 16])]

        ppu_out = self.PPU(features[-1])  #768 8 8  self.PPU(features[-1])

        d3 = self.decoder3(ppu_out, features[-2]) #384 16 16
        d2 = self.decoder2(d3, features[-3]) #192 32 32
        d1 = self.decoder1(d2, features[-4]) #96 64 64
        d0 = self.final_conv(self.up(d1))
        seg_output = [d0, d1, d2, d3]
        if self.deep_supervised:
            for i in range(len(self.seg_outs)):
                seg_output[i+1] = self.seg_outs[i](seg_output[i+1])
            return seg_output
        else:
            return d0

#####  混合架构
class SMFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth_dim=None,deep_supervised=True):
        super().__init__()
        if depth_dim is None:
            depth_dim = [96, 192, 384, 768]
        self.cnnnet = convnext_tiny()
        """Encoder"""
        self.vssm_encoder  = VSSMEncoder(drop_path_rate=0.2)
        self.Fuse = nn.ModuleList()
        for i in range(len(depth_dim)):
            self.Fuse.append(TIF(depth_dim[i],depth_dim[i]))
        self.decoder = MCFFFD(768,[96,192,384],deep_supervised=deep_supervised)
        self.deep_supervised = deep_supervised
        """Decoder"""
        # self.decoder1 = UNetResDecoder(out_channels,deep_supervised,[96,192,384,768])
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv1d):
    #         n = m.kernel_size[0] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        """Encoder"""
        cnn_x,cnn_out = self.cnnnet(x)  #([1, 96, 64, 64]),([1, 192, 32, 32]),([1, 384, 16, 16]),([1, 768, 8, 8])]
        vss_out = self.vssm_encoder(x) #96 64 64 ,192 32 32 , 384 16 16, 768 8 8
        res_out = []
        for i in range(len(self.Fuse)):
            # if i==0: res_out.append(vss_out[i])  # UNetResDecoder解码器时需要该特征
            fuse = self.Fuse[i](vss_out[i+1], cnn_out[i]) #融合模块  self.Fuse[i](vss_out[i+1],cnn_out[i])
            res_out.append(fuse)
        """Decoder"""
        seg_out =  self.decoder(res_out)
        if self.deep_supervised:
            for i,o in enumerate(seg_out):
                seg_out[i] = F.interpolate(o,(256,256),mode='bilinear',align_corners=True)
            # print([o.shape for o in seg_out])
            return seg_out
        else:
            return seg_out

    # @torch.no_grad()
    # def freeze_encoder(self):
    #     for name, param in self.vssm_encoder.named_parameters():
    #         if "patch_embed" not in name:
    #             param.requires_grad = False
    #
    # @torch.no_grad()
    # def unfreeze_encoder(self):
    #     for param in self.vssm_encoder.parameters():
    #         param.requires_grad = True


class MedFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth_dim=None,deep_supervised=True):
        super().__init__()
        if depth_dim is None:
            depth_dim = [96, 192, 384, 768]
        self.cnnnet = convnext_tiny()
        """Encoder"""
        self.vmunet  = MedMamba()
        self.Fuse = nn.ModuleList()
        """Encoder"""
        self.vmunet  = MedMamba()
        self.Fuse = nn.ModuleList()
        for i in range(len(depth_dim)):
            self.Fuse.append(TIF(depth_dim[i],depth_dim[i]))
        self.decoder = MCFFFD(768,[96,192,384],deep_supervised=deep_supervised)
        self.deep_supervised = deep_supervised
        """Decoder"""
    def forward(self, x):
        """Encoder"""
        cnn_x,cnn_out = self.cnnnet(x)  #([1, 96, 64, 64]),([1, 192, 32, 32]),([1, 384, 16, 16]),([1, 768, 8, 8])]
        vss_out = self.vmunet(x) #96 64 64 ,192 32 32 , 384 16 16, 768 8 8
        res_out = []
        for i in range(len(self.Fuse)):
            # if i==0: res_out.append(vss_out[i])
            fuse = self.Fuse[i](vss_out[i+1],cnn_out[i])
            res_out.append(fuse)

        """Decoder"""
        seg_out =  self.decoder(res_out)
        if self.deep_supervised:
            for i,o in enumerate(seg_out):
                seg_out[i] = F.interpolate(o,(256,256),mode='bilinear',align_corners=True)
            # print([o.shape for o in seg_out])
            return seg_out
        else:
            return seg_out

class res(nn.Module):
    def __init__(self,in_channel=3,out_channels=1):
        super(res, self).__init__()
        resnet = models.resnet34(pretrained=True)   #resnet18 11.18M 2.38G    resnet34 21.28M 4.81G
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        x = self.inc(x)
        x = self.pool(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        return x

def load_pretrained_ckpt(
        model,
        num_input_channels=1,
        ckpt_path="./pretrained_ckpt/vmamba_tiny_e292.pth"
):
    print(f"Loading weights from: {ckpt_path}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias"]
    ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)
    model_dict = model.state_dict()
    for k, v in ckpt['model'].items():
        if k in skip_params:
            print(f"Skipping weights: {k}")
            continue
        kr = f"vssm_encoder.{k}"
        if "patch_embed" in k and ckpt['model']["patch_embed.proj.weight"].shape[1] != num_input_channels:
            print(f"Passing weights: {k}")
            continue
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape},{kr}"
            model_dict[kr] = v
        else:
            print(f"Passing weights: {k}")

    model.load_state_dict(model_dict)
    return model

def load_from(model,ckpt_path="./pretrained_ckpt/Breast_MedMamba.pth"):
    print(f"Loading weights from: {ckpt_path}")
    model_dict = model.state_dict()
    # print(model_dict.keys())   #vmunet.vmunet.layers.0.blocks.0.attn.relative_position_bias_table
    modelCheckpoint = torch.load(ckpt_path, weights_only=False)
    # print(modelCheckpoint.keys())            print(model_dict.keys())
    pretrained_dict = modelCheckpoint
    new_dict = {}
    for k, v in pretrained_dict.items():  # fc.weight,fc.bias
        kr = f"vmunet.vmunet.{k}"
        # kr = kr.replace(f"vmunet", f"")#
        if kr in model_dict.keys():
            if "head" in kr:
                continue
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            new_dict[kr] = v
        else:
            print(f"Passing weights: {k}")
    model_dict.update(new_dict) # 打印出来，更新了多少的参数
    print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                               len(pretrained_dict),
                                                                               len(new_dict)))
    load_partial_state_dict(model, model_dict)  # total 117 pretrained 276 update 4
    print("encoder loaded finished!")

def load_partial_state_dict(model, state_dict):  #
    own_state = model.state_dict()

    for name, param in state_dict.items():  #
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                print(f'Skipping {name} due to size mismatch.')
                print(own_state[name].shape, param.shape)
        else:
            print(f'Skipping {name} as it is not in the model.')



def get_sm_model():
    model = SMFormer().cuda()
    model = load_pretrained_ckpt(model)   #57.88M 11.26G
    return model
def get_med_model():
    model = MedFormer().cuda()
    load_from(model)
    return model

if __name__ == '__main__':
    model = res(3,1).cuda()
    # load_from(model)
    # x = torch.randn(2,3,256,256).cuda()
    # out = model(x)
    # print([o.shape for o in out])
    macs, params = get_model_complexity_info(model, (3,256,256), as_strings=True, print_per_layer_stat=True)
    print(f"模型 Params and FLOPs:{params}, {macs}")

    # encoder = EncoderBlock(16, 32).cuda()
    # test_input = torch.randn(2, 16, 256, 256).cuda()
    # test_output, _ = encoder(test_input)
    # print(f"编码器输出范围: {test_output.min():.3f} ~ {test_output.max():.3f}")  #  -2.050 ~ 5.666

    # mamba_layer = MambaLayer(dim=16).cuda()
    # test_seq = torch.randn(1, 16, 65536).cuda()  # (batch, seq_len, dim)
    # output = mamba_layer(test_seq)
    # assert not torch.isnan(output).any()
    # print(output.shape)

    #test channel attention
    # model  = CrossAttentionFusion(96).cuda()
    # x = torch.randn(2,96,64,64).cuda()
    # y = torch.randn(2,96,64,64).cuda()
    # out = model(x,y)
    # print(out.shape)

    # test PPU model
    # model = PyramidPoolUnit(768).cuda()
    # x = torch.randn(2, 768, 8, 8).cuda()
    # out  = model(x)
    # print(out.shape)

    # test MCFFFD
    # model = MCFFFD(768,[96,192,384]).cuda()
    # x= torch.rand(1,96,64,64).cuda()
    # x1 = torch.rand(1,192,32,32).cuda()
    # x2 = torch.rand(1,384,16,16).cuda()
    # x3 = torch.rand(1,768,8,8).cuda()
    # input = [x,x1,x2,x3]
    # #
    # out  = model(input)
    # print([o.shape for o in out ])
    #
    # model = PVMLayer(192, 384, d_state=16, d_conv=4, expand=2).cuda()
    # x = torch.randn(2, 192, 32, 32).cuda()
    # out = model(x)
    # print(out.shape)
