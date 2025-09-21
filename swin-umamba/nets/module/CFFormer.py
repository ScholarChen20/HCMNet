# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from typing import Union, Type, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from thop import profile
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from mamba_ssm import Mamba
from torchvision import models
import timm


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedResidualBlocks(
                n_blocks=n_conv_per_stage[s - 1],
                conv_op=encoder.conv_op,
                input_channels=2 * input_features_skip,
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=1,
                conv_bias=encoder.conv_bias,
                norm_op=encoder.norm_op,
                norm_op_kwargs=encoder.norm_op_kwargs,
                dropout_op=encoder.dropout_op,
                dropout_op_kwargs=encoder.dropout_op_kwargs,
                nonlin=encoder.nonlin,
                nonlin_kwargs=encoder.nonlin_kwargs,
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output

  #  CFFormer core module
class CFCA(nn.Module):
    def __init__(self,t_channels,c_channels,out_channels):
        super().__init__()
        self.AAP=nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.excitation = nn.Linear(c_channels,t_channels)
        self.compression = nn.Linear(t_channels,c_channels)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x,y):
        b,c,_,_ = x.size()
        b1,c1,_,_ = y.size()
        u1 = self.AAP(x).view(b,c)    #b,64
        v1 = self.AAP(y).view(b1,c1)   #b,96
        u1 = self.relu(self.excitation(u1))
        u_attention = self.sigmoid(self.compression(self.relu(self.excitation(u1))))    #b,64
        v_attention = self.sigmoid(self.excitation(self.relu(self.compression(v1))))    #b,96
        q = torch.einsum('bc,ob->co',u_attention,v_attention.T)    #64 * 96
        u_v = torch.einsum('bchw,oc->bohw',x,self.softmax(q.T))     #b,96,h,w
        v_u = torch.einsum('bchw,oc->bohw',y,self.softmax(q))   #b,64,h,w
        u_fuse = v_u + x
        v_fuse = u_v + y
        return u_fuse,v_fuse

# 优化后的，重构通道注意生成逻辑，添加top-k稀疏化。优化特征融合方式
class DualBranchCFCA(nn.Module):
    def __init__(self, t_channels, c_channels, reduction=16, k_ratio=0.3):
        super().__init__()
        self.k = int(t_channels * k_ratio)

        # 双分支通道注意力
        self.aap_t = nn.AdaptiveAvgPool2d(1)  # Transformer分支
        self.aap_c = nn.AdaptiveAvgPool2d(1)  # CNN分支

        # Transformer分支注意力
        self.fc_t = nn.Sequential(
            nn.Linear(t_channels, t_channels // reduction),
            nn.ReLU(),
            nn.Linear(t_channels // reduction, t_channels),
            nn.Sigmoid()
        )

        # CNN分支注意力
        self.fc_c = nn.Sequential(
            nn.Linear(c_channels, c_channels // reduction),
            nn.ReLU(),
            nn.Linear(c_channels // reduction, c_channels),
            nn.Sigmoid()
        )

        # 跨分支交互权重
        self.cross_att = nn.Parameter(torch.randn(t_channels, c_channels))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_t, x_c):
        # 分支内通道注意力
        b, ct, ht, wt = x_t.shape
        b, cc, hc, wc = x_c.shape

        # Transformer分支注意力
        att_t = self.aap_t(x_t).view(b, ct)
        att_t = self.fc_t(att_t)  # [B, T]

        # CNN分支注意力
        att_c = self.aap_c(x_c).view(b, cc)
        att_c = self.fc_c(att_c)  # [B, C]

        # Top-k稀疏化（Transformer侧）
        topk_val, topk_idx = torch.topk(att_t, self.k, dim=1)
        sparse_att_t = torch.zeros_like(att_t).scatter(1, topk_idx, 1.0)

        # 跨分支交互
        cross_weight = self.softmax(self.cross_att)  # [T, C]
        cross_t = torch.einsum('bt,tc,bchw->bthw', sparse_att_t, cross_weight, x_c)
        cross_c = torch.einsum('bc,tc,bthw->bchw', att_c, cross_weight.T, x_t)

        # 双分支输出（保留独立特征流）
        out_t = x_t * att_t.unsqueeze(-1).unsqueeze(-1) + cross_t
        out_c = x_c * att_c.unsqueeze(-1).unsqueeze(-1) + cross_c

        return out_t, out_c

class XFF(nn.Module):
    def __init__(self,t_channels,c_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(t_channels,c_channels,3,padding=1)
        self.conv2 = nn.Conv2d(c_channels,t_channels,5,padding=2)
        self.conv3 = nn.Conv2d(c_channels+t_channels,out_channels,3,padding=1)
    def forward(self,x,y):
        u_skip = self.conv1(y)+x
        v_skip = self.conv2(x)+y
        skip  = self.conv3(torch.concat([u_skip,v_skip],dim=1))
        return skip

#修改后的
class SkipConnectionXFF(nn.Module):
    def __init__(self, t_channels, c_channels, out_channels):
        super().__init__()
        # 双分支多尺度提取
        self.t_branch = nn.Sequential(
            nn.Conv2d(t_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 5, padding=2),
            nn.BatchNorm2d(out_channels)
        )
        self.out_channels = out_channels
        self.c_branch = nn.Sequential(
            nn.Conv2d(c_channels, out_channels, 5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # 全局上下文融合
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )

        # 特征压缩
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU()
        )

    def forward(self, x_t, x_c):
        # 处理双分支输入
        feat_t = self.t_branch(x_t)  # [B, C, H, W]
        feat_c = self.c_branch(x_c)  # [B, C, H, W]

        # 全局上下文引导
        global_feat = torch.cat([
            self.aap(feat_t).squeeze(-1).squeeze(-1),
            self.aap(feat_c).squeeze(-1).squeeze(-1)
        ], dim=1)  # [B, 2C]
        gate = self.global_fc(global_feat).view(-1, self.out_channels, 1, 1)

        # 门控融合
        fused = feat_t * gate + feat_c * (1 - gate)
        return self.fusion(fused)

class Encoder(nn.Module):
    def __init__(self,t_channels=64,c_channels=64,skip_channels=64):
        super().__init__()
        self.CFCA = CFCA(t_channels,c_channels,skip_channels)
        self.XFF = XFF(t_channels,c_channels,skip_channels)
    def forward(self,x,y):
        x1,y1 = self.CFCA(x,y)
        skip = self.XFF(x1,y1)
        return x1,y1,skip


class CNNDecoder(nn.Module):
    def __init__(self,in_channels,out_channels,end=False):
        super().__init__()
        self.end = end
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(in_channels,out_channels,2,stride=2)
    def forward(self,x):
        x1 = self.cbr(x)
        x2 = self.cbr(x1)
        if not self.end:
            out = self.up(x2)
        else:
            out = x2
        return out

class OutDecoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.decoder1 = CNNDecoder(in_channels,in_channels//2,False)
        self.decoder2 = CNNDecoder(in_channels//2,in_channels//2,True)
        self.conv1 = nn.Conv2d(in_channels//2,out_channels,1,padding=0)
    def forward(self,x):
        x1 = self.decoder1(x)
        x2 = self.decoder2(x1)
        out = self.conv1(x2)
        return out



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        # if downsample is not None:
        #     self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        # else:
        #     self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # if self.downsample is not None:
        #     x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [64, 64]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],downsample=True, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,          #modify
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        self.downsample1 = PatchMerging(input_resolution=(64,64),dim=96)
        self.downsample2 = PatchMerging(input_resolution=(32,32),dim=192)
        self.downsample3 = PatchMerging(input_resolution=(16,16),dim=384)
        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        resnet = models.resnet34(pretrained=True)
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # CFCA+XFF
        self.cross1 = Encoder(96, 64, 64)
        self.cross2 = Encoder(192, 128, 128)
        self.cross3 = Encoder(384, 256, 256)
        self.cross4 = Encoder(768, 512, 512)
        # decoder
        self.decoder4 = CNNDecoder(512, 256, False)
        self.decoder3 = CNNDecoder(512, 128, False)
        self.decoder2 = CNNDecoder(256, 64, False)
        self.decoder1 = CNNDecoder(128, 64, False)
        self.out = OutDecoder(128, 1)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, x):
        x1 = self.inc(x)
        xx = self.pool(x1)   # 64 64 64
        y = self.patch_embed(x)    # 4096 96  ->  64 64 96
        if self.ape:
            y = y + self.absolute_pos_embed
        y = self.pos_drop(y)    # 4096 96
        skip = []
        for inx,layer in enumerate(self.layers):
            y = layer(y)
            B,L,C = y.shape
            H = W = L // (2 ** (6-inx))
            y = y.view(B,H,W,C)       # b 64 64 96   b h w c
            if inx==0:
                xx = self.encoder1(xx)
                xx,y,out = self.cross1(xx,y.permute(0,3,1,2))    # b c h w
                y = y.permute(0, 2, 3, 1)  # b h w c
                y = y.view(B, H * W, C)  # # B H*W C
                y = self.downsample1(y)   # B H/2*W/2 2C
            elif inx==1:
                xx = self.encoder2(xx)
                xx,y,out = self.cross2(xx,y.permute(0,3,1,2))
                y = y.permute(0, 2, 3, 1)  # b h w c
                y = y.view(B, H * W, C)  # # B H*W C
                y = self.downsample2(y)   # B H/2*W/2 2C
            elif inx==2:
                xx = self.encoder3(xx)
                xx,y,out = self.cross3(xx,y.permute(0,3,1,2))
                y = y.permute(0, 2, 3, 1)  # b h w c
                y = y.view(B, H * W, C)  # B H*W C
                y = self.downsample3(y)  # B H/2*W/2 2C
            else:
                xx = self.encoder4(xx)
                xx,y,out = self.cross4(xx,y.permute(0,3,1,2))

            skip.append(out)   #1 64 64 64   1 128 32 32  1 256 16 16  1 512 8 8
        seg4 = self.decoder4(skip[3])
        seg3 = self.decoder3(torch.concat([seg4,skip[2]],1))
        seg2 = self.decoder2(torch.concat([seg3,skip[1]],1))
        seg1 = self.decoder1(torch.concat([seg2,skip[0]],1))
        seg = self.out(torch.concat([seg1,x1],1))
        return seg

        # x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
    #
    # def forward(self, x):
    #     x = self.forward_features(x)
    #     # x = self.head(x)
    #     return x   #1 1 256 256


class MambaMLPLayer(nn.Module):
    def __init__(self, hidden_size, d_state=64, d_conv=4, expand=4, mha_head=8):
        super(MambaMLPLayer, self).__init__()
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # self.mha = MHA(hidden_size, mha_head, )
        # self.dropout = nn.Dropout(dropout_rate)
        self.mamba_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        # self.mha_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                                 nn.Linear(hidden_size * 4, hidden_size))

        self.mlp_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        # y = self.mha_norm(x)
        # y, _ = self.mha(y, y, y)
        # x = x + y

        y = self.mamba_norm(x)
        y = self.mamba(y)
        x = x + y

        y = self.mlp_norm(x)
        y = self.mlp(y)
        x = x + y

        return x


class CFFormer(nn.Module):
    def __init__(self,embed_dim=64,t_channel=3,c_channels=3,patch_norm=True,norm_layer=nn.LayerNorm,drop_rate=0.1,depth=4,ape=False):
        super(CFFormer, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = depth
        self.patch_norm = patch_norm
        self.ape = ape
        #resnet34
        resnet = models.resnet34(pretrained=True)
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Mamba
        self.patch_embed = PatchEmbed(
            img_size=256, patch_size=4, in_chans=c_channels, embed_dim=64,
            norm_layer=norm_layer if self.patch_norm else None)
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 4096, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.downsample1 = PatchMerging(input_resolution=(64, 64), dim=64)
        self.downsample2 = PatchMerging(input_resolution=(32, 32), dim=128)
        self.downsample3 = PatchMerging(input_resolution=(16, 16), dim=256)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MambaMLPLayer(hidden_size = embed_dim * 2**i_layer)
            self.layers.append(layer)
        #CFCA+XFF
        self.cross1 = Encoder(64,64,64)
        self.cross2 = Encoder(128,128,128)
        self.cross3 = Encoder(256,256,256)
        self.cross4 = Encoder(512,512,512)
        #decoder
        self.decoder4 = CNNDecoder(512,256,False)
        self.decoder3 = CNNDecoder(512,128,False)
        self.decoder2 = CNNDecoder(256,64,False)
        self.decoder1 = CNNDecoder(128,64,False)
        self.out = OutDecoder(128,1)

    def forward_features(self,x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x      # b l c

    def forward(self,x):
        x1 = self.inc(x)  # 64 128 128
        xx = self.pool(x1)  # 64 56 56
        y = self.forward_features(x)
        skip = []

        # xx1 = self.encoder1(xx)
        # xx2 = self.encoder2(xx1)
        # xx3 = self.encoder3(xx2)
        # xx4 = self.encoder4(xx3)

        for i,layer in enumerate(self.layers):
            y = layer(y)
            B, L, C = y.shape
            H = W = L // (2 ** (6 - i))
            y = y.view(B, H, W, C)  # b 64 64 96   b h w c
            if i==0:
                xx = self.encoder1(xx)
                # out = xx + y.permute(0,3,1,2)
                xx,y,out = self.cross1(xx, y.permute(0, 3, 1, 2))  # b c h w
                y = y.permute(0, 2, 3, 1)  # b h w c
                y = y.reshape(B, H * W, C)  # # B H*W C
                y = self.downsample1(y)  # B H/2*W/2 2C
            elif i==1:
                xx = self.encoder2(xx)
                # out = xx + y.permute(0,3,1,2)
                xx,y,out = self.cross2(xx,y.permute(0, 3, 1, 2))
                y = y.permute(0, 2, 3, 1)
                y = y.reshape(B, H * W, C)
                y = self.downsample2(y)
            elif i==2:
                xx = self.encoder3(xx)
                # out = xx + y.permute(0,3,1,2)
                xx,y,out = self.cross3(xx,y.permute(0, 3, 1, 2))
                y = y.permute(0, 2, 3, 1)
                y = y.reshape(B, H * W, C)
                y = self.downsample3(y)
            else:
                xx = self.encoder4(xx)
                out = xx + y.permute(0,3,1,2)
                assert not torch.isnan(x).any()
                # xx,y,out = self.cross4(xx,y.permute(0, 3, 1, 2))
            skip.append(out)  #1 64 64 64   1 128 32 32  1 256 16 16  1 512 8 8

        # y2 = self.Mambalayer1(y)           #56 56 96
        # x22,y22,skip1 = self.cross1(x2,y2.permute(0,3,1,2))   #64 56 56
        # x3 = self.encoder2(x22)
        # y3 = self.Mambalayer2(y22.permute(0,2,3,1))
        # x33,y33,skip2 = self.cross2(x3,y3.permute(0,3,1,2))   #128 28 28
        # x4 = self.encoder3(x33)
        # y4 = self.transformer3(y33.permute(0,2,3,1))
        # x44,y44,skip3 = self.cross3(x4,y4.permute(0,3,1,2))   #256 14 14
        # x5 = self.encoder4(x44)
        # y5 = self.transformer4(y44.permute(0,2,3,1))
        # x55,y55,skip4 = self.cross4(x5,y5.permute(0,3,1,2))   #512 7  7
        # print(skip[3].shape)
        seg4 = self.decoder4(skip[3])          #256 14 14
        seg3 = self.decoder3(torch.concat([seg4,skip[2]],1))  #128 28 28
        seg2 = self.decoder2(torch.concat([seg3,skip[1]],1))  #64 56 56
        seg1 = self.decoder1(torch.concat([seg2,skip[0]],1))  #64 128 128
        out = self.out(torch.concat([seg1,x1],1))           #1  224 224
        return out


class res(nn.Module):
    def __init__(self,in_channel=3,out_channels=1):
        super(res, self).__init__()
        resnet = models.resnet18(pretrained=True)
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


class net(nn.Module):
    def __init__(self,backbone_name = 'swsl_resnet18'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.act1 = self.backbone.act1
        self.maxpool = self.backbone.maxpool
        self.layers = nn.ModuleList()
        self.layers.append(self.backbone.layer1)
        self.layers.append(self.backbone.layer2)
        self.layers.append(self.backbone.layer3)
        self.layers.append(self.backbone.layer4)
    def forward(self,x):
        x  = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x


def get_former_model():
    model = CFFormer().cuda()
    return model

from ptflops import get_model_complexity_info

if __name__ == '__main__':
    model =res().cuda()
    # model = Encoder(64,64,64)
    # x1 = torch.randn(1,64,64,64)
    # x2 = torch.randn(1,64,64,64)
    input = torch.randn(1,3,256,256).cuda()
    x = model(input)
    # print(cal_params_flops(model,256,logger))
    print(x.shape)
    macs, params = get_model_complexity_info(model, (3,256,256), as_strings=True, print_per_layer_stat=True)

    print(f"模型 FLOPs and Params: {macs},{params}")