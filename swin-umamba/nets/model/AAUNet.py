# # from AAU-net: An Adaptive Attention U-net for Breast Lesions Segmentation in Ultrasound Images
# # coding=utf-8
# import torch
# # from tensorflow.keras.layers import *
# # import cv2
# # import tensorflow.keras.backend as K
# # from tensorflow.keras.models import *
# from torch import nn
# import torch.nn.functional as F
from torch.cpu.amp import autocast
#
#
# class ChannelBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ChannelBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         # Global Average Pooling + Fully Connected layers
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(out_channels*2, out_channels)
#         self.fc2 = nn.Linear(out_channels, out_channels)
#         self.sigmoid = nn.Sigmoid()
#         self.conv_out = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)
#     def forward(self, x):
#         # Two convolutional branches
#         conv1 = self.relu(self.bn1(self.conv1(x)))
#         conv2 = self.relu(self.bn2(self.conv2(x)))
#         # Concatenate along the channel dimension
#         sum = torch.cat([conv1,conv2],1) # [B, 2*out_channels, H, W]
#         # Channel attention using global average pooling
#         pooled = self.global_avg_pool(sum).view(sum.size(0),-1)  # [B, 2*out_channels]
#         fc_out = self.relu(self.fc1(pooled))
#         attention = self.sigmoid(self.fc2(fc_out))     #(B ， out_channel)
#         # print(attention.shape)
#
#         # Reshape attention weights and apply to feature maps
#         a = attention.unsqueeze(2).unsqueeze(3)# [B, 2*out_channels, 1, 1]
#         a1 = 1 - a  # Inverse attention
#         # Split into two branches and apply attention
#         conv1_att = a * conv1
#         conv2_att = a1 * conv2
#
#         # Concatenate attention-weighted feature maps
#         sum2 =  torch.cat([conv1_att, conv2_att], dim=1)
#         out = self.conv_out(sum2)
#         return  out
#
# # Spatial Attention Block
# class SpatialBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SpatialBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         # Final convolution for spatial attention
#         self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#         # Final output processing
#         self.conv4 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x, channel_data):
#         # Spatial feature extraction
#         spatial = self.relu(self.bn1(self.conv1(x)))
#         spatial = self.relu(self.bn2(self.conv2(spatial)))
#
#         # Element-wise addition with channel attention
#         data = self.relu(channel_data + spatial)
#         attention = self.sigmoid(self.conv3(data))  # Spatial attention map   （B, out_channel)
#         # print(attention.shape)
#         a = attention
#         # Apply spatial attention
#         spatial_att = a * channel_data
#         a1 = 1-a
#         spatial_inv_att = a1 * spatial
#
#         # Concatenate and process final output
#         concat = torch.cat([spatial_att, spatial_inv_att], dim=1)
#         out = self.bn3(self.conv4(concat))
#         return out
#
#
# # HAAM Module
# class HAAM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(HAAM, self).__init__()
#         self.channel_block = ChannelBlock(in_channels, out_channels)
#         self.spatial_block = SpatialBlock(in_channels, out_channels)
#
#     def forward(self, x):
#         channel_data = self.channel_block(x)  # Channel attention
#         haam_out = self.spatial_block(x, channel_data)  # Spatial attention
#         return haam_out

# from pytorch_wavelets import DWTForward
# class HWD(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(HWD,self).__init__()
#         self.dwt = DWTForward(J=1,mode='zero',wave = 'haar')
#         self.conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_channels*4, out_channels, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         yL,yH = self.dwt(x)
#         y_HL = yH[0][:,:,0,::]
#         y_LH = yH[0][:,:,1,::]
#         y_HH = yH[0][:,:,2,::]
#         x = torch.concat([yL,y_HL,y_LH,y_HH],1)
#         out = self.conv_bn_relu(x)
#         return out


import torch
import torch.nn as nn
#论文：AAU-net: An Adaptive Attention U-net for Breast Lesions Segmentation in Ultrasound Images
#论文地址：https://arxiv.org/pdf/2204.12077

def expend_as(tensor, rep):
    return tensor.repeat(1, rep, 1, 1)


class Channelblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Channelblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)

        combined = torch.cat([conv1, conv2], dim=1)
        pooled = self.global_avg_pool(combined)
        pooled = torch.flatten(pooled, 1)
        sigm = self.fc(pooled)

        a = sigm.view(-1, sigm.size(1), 1, 1)
        a1 = 1 - sigm
        a1 = a1.view(-1, a1.size(1), 1, 1)

        y = conv1 * a
        y1 = conv2 * a1

        combined = torch.cat([y, y1], dim=1)
        out = self.conv3(combined)
        # print(out.shape)

        return out


class Spatialblock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Spatialblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=size, padding=(size // 2)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, channel_data):
        conv1 = self.conv1(x)
        spatil_data = self.conv2(conv1)

        data3 = torch.add(channel_data, spatil_data)
        data3 = torch.relu(data3)
        data3 = nn.Conv2d(data3.size(1), 1, kernel_size=1, padding=0).cuda()(data3)
        data3 = torch.sigmoid(data3)

        a = expend_as(data3, channel_data.size(1))
        y = a * channel_data

        a1 = 1 - data3
        a1 = expend_as(a1, spatil_data.size(1))
        y1 = a1 * spatil_data

        combined = torch.cat([y, y1], dim=1)
        out = self.final_conv(combined)

        return out


class HAAM(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super(HAAM, self).__init__()
        self.channel_block = Channelblock(in_channels, out_channels)
        self.spatial_block = Spatialblock(in_channels, out_channels, size)

    def forward(self, x):
        channel_data = self.channel_block(x)
        haam_data = self.spatial_block(x, channel_data)
        return haam_data

class AAUnet(nn.Module):
    def __init__(self,in_channel=3,num_classes=1):
        super(AAUnet,self).__init__()
        self.enc1 = HAAM(3, 32)
        self.enc2 = HAAM(32, 64)
        self.enc3 = HAAM(64, 128)
        self.enc4 = HAAM(128, 256)
        self.enc5 = HAAM(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.down1 = HWD(32,32)
        # self.down2 = HWD(64,64)
        # self.down3 = HWD(128,128)
        # self.down4 = HWD(256,256)

        # 解码器（上采样部分）
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = HAAM(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = HAAM(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = HAAM(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = HAAM(64, 32)

        # 最终输出层
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    @autocast(True)
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        e5 = self.enc5(p4)

        d4 = self.up4(e5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out_conv(d1)
        return out

from ptflops import get_model_complexity_info
if __name__ == '__main__':
    model = AAUnet(in_channel=3, num_classes=1).cuda()
    # model  =  HAAM(3,32).cuda()
    # x  =  torch.randn(2,3,256,256).cuda()
    # output = model(x)
    # print(output.shape)

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
