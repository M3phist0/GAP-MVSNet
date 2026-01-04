from .module import *
from .gca_module import *
from .dcn import DCN

'''
利用深度连续性设计loss，在法线连续的地方深度也应该连续

深度法线一致性为什么不加到loss里呢
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)


class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost

# Source from https://github.com/haoshao-nku/medical_seg
# 提出的注意力的实现模块
class MSCAAttention(BaseModule):
    def __init__(self,
                 channels,
                 kernel_sizes=[3, [1, 7], [1, 11], [1, 21]],
                 paddings=[1, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)

        # 多尺度特征提取
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        # 通道融合（也是通过1x1卷积）
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


# 原论文模型中带有封装MSCAAttention，可用于参考作者怎么使用这个注意力模块
class MSCASpatialAttention(BaseModule):
    """
    Spatial Attention Module in Multi-Scale Convolutional Attention Module，多尺度卷积注意力模块中的空间注意模块
    先过1x1卷积，gelu激活后过注意力，再过一次1x1卷积，最后和跳跃连接
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[1, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[0, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU')):
        """

        :param in_channels: 通道数.
        :param attention_kernel_sizes (list): 注意力核大小. 默认: [5, [1, 7], [1, 11], [1, 21]].
        :param attention_kernel_paddings (list): 注意力模块中相应填充值的个数.
        :param act_cfg (list): 注意力模块中相应填充值的个数.
        """
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 跳跃连接
        shorcut = x.clone()
        # 先过1x1卷积
        x = self.proj_1(x)
        # 激活
        x = self.activation(x)
        # 过MSCAAttention
        x = self.spatial_gating_unit(x)
        # 1x1卷积
        x = self.proj_2(x)
        # 残差融合
        x = x + shorcut
        return x

class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward function."""

        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class MSCABlock(BaseModule):
    """Basic Multi-Scale Convolutional Attention Block. It leverage the large-
    kernel attention (LKA) mechanism to build both channel and spatial
    attention. In each branch, it uses two depth-wise strip convolutions to
    approximate standard depth-wise convolutions with large kernels. The kernel
    size for each branch is set to 7, 11, and 21, respectively.

    Args:
        channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        mlp_ratio (float): The ratio of multiple input dimension to
            calculate hidden feature in MLP layer. Defaults: 4.0.
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
        drop_path (float): The ratio of drop paths.
            Defaults: 0.0.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 channels,
                 attention_kernel_sizes=[1, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[0, [0, 3], [0, 5], [0, 10]],
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = MSCASpatialAttention(channels, attention_kernel_sizes,
                                         attention_kernel_paddings, act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class LocalDeformNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dcn = nn.Sequential(
            DCN(in_channels=channels, out_channels=channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.propa = DCN(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, ret_off=False):
        x = self.dcn(x)
        
        if ret_off:
            x, offset, _ = self.propa(x, ret_off=True)
            N, C, H_out, W_out = offset.shape  # C = 2 * k * k
            k = int(C // 2)  # 卷积核的采样点数，例如 3x3 = 9
            offset_x = offset[:, :k, :, :]  # 提取前 k 个通道为 x 偏移
            offset_y = offset[:, k:, :, :]  # 提取后 k 个通道为 y 偏移
            offset = torch.stack((offset_x, offset_y), dim=2).view(N, -1, H_out, W_out)
            return x, offset
        
        x = self.propa(x)
        return x


class FastGuide(nn.Module):
    def __init__(self, input_planes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.expansion_ratio = 3
        self.conv1 = Basic2d(input_planes, input_planes, None)
        self.weight_expansion = Basic2d(input_planes, input_planes * self.expansion_ratio, norm_layer, kernel_size=1, padding=0)

        self.conv2 = Basic2d(input_planes, input_planes, norm_layer, kernel_size=1, padding=0)
        self.conv3 = Basic2d(input_planes, input_planes)

    def forward(self, input, weight):
        weight = self.conv1(weight)
        weight = self.weight_expansion(weight)

        kernels = torch.chunk(weight, self.expansion_ratio, 1)
        splits = []

        for i in range(self.expansion_ratio):
            splits.append(input*kernels[i])
        out = sum(splits)
        out = self.conv2(out)

        avg_out = torch.mean(weight, dim=1, keepdim=True)
        out = self.conv3(out * avg_out)

        return out

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(SeparableConv3d, self).__init__()
        
        # 深度卷积：3x1x1
        self.depth_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,  # 深度卷积不改变通道数
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            groups=in_channels,  # 分组卷积，通道完全分离
            bias=False
        )
        
        # 横向卷积：1x3x3
        self.point_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=False
        )
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Basic3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out

class FastGuide3d(nn.Module):
    def __init__(self, input_planes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.expansion_ratio = 3
        self.conv1 = Basic3d(input_planes, input_planes, kernel_size=(1,5,5), padding=(0,2,2))
        self.weight_expansion = Basic3d(input_planes, input_planes * self.expansion_ratio, norm_layer, kernel_size=1, padding=0)

        self.conv2 = Basic3d(input_planes, input_planes, norm_layer, kernel_size=1, padding=0)
        self.conv3 = Basic3d(input_planes, input_planes)

    def forward(self, input, weight):
        weight = self.conv1(weight)
        weight = self.weight_expansion(weight)

        kernels = torch.chunk(weight, self.expansion_ratio, 1)
        splits = []

        for i in range(self.expansion_ratio):
            splits.append(input * kernels[i])
        out = sum(splits)
        out = self.conv2(out)

        avg_out = torch.mean(weight, dim=1, keepdim=True)
        out = self.conv3(out * avg_out)

        return out
    
class GeoCostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(GeoCostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv1 = Conv3d(base_channels, base_channels * 2, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, kernel_size=(5,3,3), padding=(2,1,1))
        
        self.geo_conv0 = Conv3d(4, base_channels, kernel_size=(5,3,3), padding=(2,1,1))

        self.geo_conv1 = Conv3d(base_channels, base_channels * 2, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.geo_conv2 = Conv3d(base_channels * 2, base_channels * 2, kernel_size=(3,3,3), padding=(1,1,1))
        self.guide2= FastGuide3d(base_channels * 2)

        self.geo_conv3 = Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.geo_conv4 = Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3,3,3), padding=(1,1,1))
        self.guide4= FastGuide3d(base_channels * 4)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.prob = nn.Conv3d(base_channels, 1, kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1), bias=False)

    def forward(self, x, d, normal, *kwargs):
        min_d = d.view(d.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        max_d = d.view(d.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        d = (d - min_d) / (max_d - min_d + 1e-8)
        ndepth = d.size(1)
        geo_prior = torch.cat([d.unsqueeze(1), normal.unsqueeze(2).expand(-1, -1, ndepth, -1, -1)], dim=1)
        
        geo0 = self.geo_conv0(geo_prior)
        geo2 = self.geo_conv2(self.geo_conv1(geo0))
        geo4 = self.geo_conv4(self.geo_conv3(geo2))
        
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        
        x = conv4 + self.conv7(x)
        x = self.guide4(x, geo4) + x
        
        x = conv2 + self.conv9(x)
        x = self.guide2(x, geo2) + x
        
        x = conv0 + self.conv11(x)

        x = self.prob(x)
        
        return x
    
class SoomthGuide(nn.Module):
    def __init__(self, input_planes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.expansion_ratio = 3
        # self.conv1 = Basic2d(input_planes, input_planes, None)
        self.conv1 = MSCAAttention(input_planes)
        self.weight_expansion = Basic2d(input_planes, input_planes * self.expansion_ratio, norm_layer, kernel_size=1, padding=0)

        self.conv2 = Basic2d(input_planes, input_planes, norm_layer, kernel_size=1, padding=0)
        self.conv3 = Basic2d(input_planes, input_planes)

    def forward(self, input, weight):
        weight = self.conv1(weight)
        weight = self.weight_expansion(weight)

        kernels = torch.chunk(weight, self.expansion_ratio, 1)
        splits = []

        for i in range(self.expansion_ratio):
            splits.append(input*kernels[i])
        out = sum(splits)
        out = self.conv2(out)

        avg_out = torch.mean(weight, dim=1, keepdim=True)
        out = self.conv3(out * avg_out)

        return out

class NormalEnhancedFPN(nn.Module):
    def __init__(self, base_channels, num_stage=3):
        super().__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        
        #----
        self.normal_conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )
        self.guide0 = FastGuide(base_channels)

        self.normal_conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        self.guide1 = FastGuide(base_channels * 2)

        self.normal_conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.guide2 = FastGuide(base_channels * 4)
        #----

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]
        
        final_chs = base_channels * 4
        if num_stage == 3:
            self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
            self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

            self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
            self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
            self.out_channels.append(base_channels * 2)
            self.out_channels.append(base_channels)

        elif num_stage == 2:
            self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

            self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
            self.out_channels.append(base_channels)

    def forward(self, x, normal):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        
        normal_conv0 = self.normal_conv0(normal)
        normal_conv1 = self.normal_conv1(normal_conv0)
        normal_conv2 = self.normal_conv2(normal_conv1)

        intra_feat = self.guide2(conv2, normal_conv2)
        # intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
       
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(self.guide1(conv1, normal_conv1))
        # intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(self.guide0(conv0, normal_conv0))
        # intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs
    
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output
    
class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # DW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

        self.bn = torch.nn.BatchNorm2d(dim*2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x_1, x_2):
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))
        x = self.conv_mid(x)
        print(x.shape)

        return x

class NormalEnhancedFPN_global(nn.Module):
    def __init__(self, base_channels, num_stage=3):
        super().__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        
        #----
        self.normal_conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )
        self.guide0 = FastGuide(base_channels)

        self.normal_conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        self.guide1 = FastGuide(base_channels * 2)

        self.normal_conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.guide2 = FastGuide(base_channels * 4)
        #----
        self.DSM = nn.ModuleList([MSCABlock(32), MSCABlock(32), MSCABlock(32), nn.BatchNorm2d(32)])
        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)
        self.smooth_1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 1, base_channels * 1, 3, padding=1, bias=False)

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]
        
        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)
    
    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, x, normal):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        
        normal_conv0 = self.normal_conv0(normal)
        normal_conv1 = self.normal_conv1(normal_conv0)
        normal_conv2 = self.normal_conv2(normal_conv1)

        intra_feat = self.guide2(conv2, normal_conv2)
        outputs = {}
        out1 = self.out1(intra_feat)
        for blk in self.DSM:
            out1 = blk(out1)
        outputs["stage1"] = out1
       
        intra_feat = self._upsample_add(intra_feat, self.inner1(self.guide1(conv1, normal_conv1)))
        out2 = self.out2(intra_feat)
        out2 = self.smooth_1(self._upsample_add(self.dim_reduction_1(out1), out2))
        outputs["stage2"] = out2

        intra_feat = self._upsample_add(intra_feat, self.inner2(self.guide0(conv0, normal_conv0)))
        out3 = self.out3(intra_feat)
        out3 = self.smooth_2(self._upsample_add(self.dim_reduction_2(out2), out3))
        outputs["stage3"] = out3

        return outputs
    
class NormalEnhancedFPN_4L(nn.Module):
    def __init__(self, base_channels, num_stage=3):
        super().__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        
        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
        )
        
        #----
        self.normal_conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )
        self.guide0 = FastGuide(base_channels)

        self.normal_conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        self.guide1 = FastGuide(base_channels * 2)

        self.normal_conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.guide2 = FastGuide(base_channels * 4)
        
        self.normal_conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
        )
        self.guide3 = FastGuide(base_channels * 8)
        #----

        final_chs = base_channels * 8
        
        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels = [8 * base_channels, 4 * base_channels, 2 * base_channels, base_channels]

    def forward(self, x, normal):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        normal_conv0 = self.normal_conv0(normal)
        normal_conv1 = self.normal_conv1(normal_conv0)
        normal_conv2 = self.normal_conv2(normal_conv1)
        normal_conv3 = self.normal_conv3(normal_conv2)

        intra_feat = self.guide3(conv3, normal_conv3)
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
       
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(self.guide2(conv2, normal_conv2))
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(self.guide1(conv1, normal_conv1))
        out = self.out3(intra_feat)
        outputs["stage3"] = out
        
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(self.guide0(conv0, normal_conv0))
        out = self.out4(intra_feat)
        outputs["stage4"] = out

        return outputs

class NormalEnhancedFPN_smooth(nn.Module):
    def __init__(self, base_channels, num_stage=3):
        super().__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        
        #----
        self.normal_conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )
        self.guide0 = FastGuide(base_channels)

        self.normal_conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )
        self.guide1 = FastGuide(base_channels * 2)

        self.normal_conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.guide2 = FastGuide(base_channels * 4)
        #----

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]
        
        
        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)


    def forward(self, x, normal):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        
        normal_conv0 = self.normal_conv0(normal)
        normal_conv1 = self.normal_conv1(normal_conv0)
        normal_conv2 = self.normal_conv2(normal_conv1)

        intra_feat = self.guide2(conv2, normal_conv2)
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
       
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) \
            + self.inner1(self.guide1(conv1, normal_conv1))
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) \
            + self.inner2(self.guide0(conv0, normal_conv0))
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs

class DetailAttention(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 3)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        
        return attn

class DetailAttention(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 3)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)
        
        self.conv4 = Conv2d(dim, dim, 1)

    def forward(self, x, weight):
        attn = self.conv0(weight)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        
        x = self.conv4(attn * x)
        
        return x


class NormalEnhancedFPNv2(nn.Module):
    def __init__(self, base_channels, num_stage=3):
        super().__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        
        #----
        self.normal_conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.normal_conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.normal_conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        #----

        final_chs = base_channels * 4
        
        self.detail0 = DetailAttention(final_chs, base_channels * 8)
        self.guide0 = FastGuide(final_chs)
        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.detail1 = DetailAttention(final_chs, base_channels * 4)
        self.guide1 = FastGuide(final_chs)
        self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)
        self.detail2 = DetailAttention(final_chs, base_channels * 2)
        self.guide2 = FastGuide(final_chs)
        self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels = [4 * base_channels, 2 * base_channels, base_channels]

    def forward(self, x, normal):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        
        normal_conv0 = self.normal_conv0(normal)
        normal_conv1 = self.normal_conv1(normal_conv0)
        normal_conv2 = self.normal_conv2(normal_conv1)

        intra_feat = conv2
        intra_feat = self.guide0(self.detail0(torch.cat([conv2, normal_conv2], dim=1)), intra_feat)
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
       
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        intra_feat = self.guide1(self.detail1(torch.cat([conv1, normal_conv1], dim=1)), intra_feat)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        intra_feat = self.guide2(self.detail2(torch.cat([conv0, normal_conv0], dim=1)), intra_feat)
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs


class NormalEnhancedFPNv3(nn.Module):
    def __init__(self, base_channels, num_stage=3):
        super().__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        
        #----
        self.normal_conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.normal_conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.normal_conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        #----

        final_chs = base_channels * 4
        
        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)
        self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
        
        self.detail_attn = DetailAttention(base_channels, base_channels)
        self.detail_down1 = nn.Conv2d(base_channels, base_channels * 2, 3, 2, padding=1, bias=False)
        self.detail_down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, padding=1, bias=False)

        self.out_channels = [4 * base_channels, 2 * base_channels, base_channels]

    def forward(self, x, normal):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        
        normal_conv0 = self.normal_conv0(normal)
        normal_conv1 = self.normal_conv1(normal_conv0)
        normal_conv2 = self.normal_conv2(normal_conv1)

        intra_feat = conv2 + normal_conv2
        outputs = {}
        out1 = self.out1(intra_feat)
       
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1 + normal_conv1)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0 + normal_conv0)
        out3 = self.out3(intra_feat)

        detail_attn3 = self.detail_attn(conv0 + normal_conv0)
        detail_attn2 = self.detail_down1(detail_attn3)
        detail_attn1 = self.detail_down2(detail_attn2)
        
        out1 = out1 * detail_attn1
        out2 = out2 * detail_attn2
        out3 = out3 * detail_attn3
        
        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3

        return outputs

class NormalEnhancedFPNv4(nn.Module):
    def __init__(self, base_channels, num_stage=3):
        super().__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        
        #----
        self.normal_conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.normal_conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.normal_conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        #----

        final_chs = base_channels * 4
        
        self.detail0 = DetailAttention(base_channels, base_channels)
        self.detail1 = DetailAttention(base_channels * 2, base_channels * 2)
        self.detail2 = DetailAttention(base_channels * 4, base_channels * 4)
        
        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)
        self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels = [4 * base_channels, 2 * base_channels, base_channels]

    def forward(self, x, normal):
        normal_conv0 = self.normal_conv0(normal)
        normal_conv1 = self.normal_conv1(normal_conv0)
        normal_conv2 = self.normal_conv2(normal_conv1)
        
        conv0 = self.detail0(self.conv0(x), normal_conv0)
        conv1 = self.detail1(self.conv1(conv0), normal_conv1)
        conv2 = self.detail2(self.conv2(conv1), normal_conv2)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
       
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs

from .dcn import PlaneDCN
class NormalEnhancedFPNv5(nn.Module):
    def __init__(self, base_channels, num_stage=3):
        super().__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        
        #----
        self.normal_conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.normal_conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.normal_conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        #----

        final_chs = base_channels * 4
        
        self.detail0 = DetailAttention(base_channels, base_channels)
        self.detail1 = DetailAttention(base_channels * 2, base_channels * 2)
        self.detail2 = DetailAttention(base_channels * 4, base_channels * 4)
        self.smooth = PlaneDCN(base_channels * 4, base_channels * 4)
        
        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)
        self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels = [4 * base_channels, 2 * base_channels, base_channels]

    def forward(self, x, normal):
        normal_conv0 = self.normal_conv0(normal)
        normal_conv1 = self.normal_conv1(normal_conv0)
        normal_conv2 = self.normal_conv2(normal_conv1)
        
        conv0 = self.detail0(self.conv0(x), normal_conv0)
        conv1 = self.detail1(self.conv1(conv0), normal_conv1)
        conv2 = self.detail2(self.conv2(conv1), normal_conv2)

        outputs = {}
        
        intra_feat = self.smooth(conv2, normal)
        
        out = self.out1(intra_feat)
        outputs["stage1"] = out
       
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        
        nn.init.constant_(self.conv.weight, 0.0)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class ResampleModule(nn.Module):
    def __init__(self, base_channels, neighbor_num=8, kernel_size=3, stride=1, padding=1, use_conf=False):
        super().__init__()
        if use_conf:
            self.offset = CoordConv(base_channels + neighbor_num + 1, 2, kernel_size, stride, padding)
        else:
            self.offset = CoordConv(base_channels + neighbor_num, 2, kernel_size, stride, padding)
    
    def forward(self, x, confidence=None):
        local_sim = self.compute_similarity(x)
        _x = torch.cat([x, local_sim], 1)
        if confidence is not None:
            _x = torch.cat([_x, confidence], 1)
        offset = self.offset(_x)
        
        # off_conf = self.resample(confidence, offset)
        # diff = off_conf - confidence
        # H, W = confidence.shape[2:]
        # center_H = int(0.5 * H)  # 高度的 90%
        # center_W = int(0.5 * W)  # 宽度的 90%

        # # 计算中心区域的起始和结束索引
        # start_h = (H - center_H) // 2  # 中心区域的起始高度
        # start_w = (W - center_W) // 2  # 中心区域的起始宽度

        # # 获取中心区域的差值
        # diff = diff[:, :, start_h:start_h + center_H, start_w:start_w + center_W]
        # print((diff >= 0).float().mean())

        x = self.resample(x, offset)
        return x
        
    def compute_similarity(self, x, k=3, dilation=1, sim='cos'):
        """
        计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

        参数：
        - x: 输入张量，形状为[B, C, H, W]
        - k: 范围大小，表示周围KxK范围内的点

        返回：
        - 输出张量，形状为[B, KxK-1, H, W]
        """
        B, C, H, W = x.shape
        # 使用零填充来处理边界情况
        # padded_input = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

        # 展平输入张量中每个点及其周围KxK范围内的点
        unfold_x = F.unfold(x, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
        # print(unfold_x.shape)
        unfold_x = unfold_x.reshape(B, C, k**2, H, W)

        # 计算余弦相似度
        if sim == 'cos':
            similarity = F.cosine_similarity(unfold_x[:, :, k * k // 2:k * k // 2 + 1], unfold_x[:, :, :], dim=1)
        elif sim == 'dot':
            similarity = unfold_x[:, :, k * k // 2:k * k // 2 + 1] * unfold_x[:, :, :]
            similarity = similarity.sum(dim=1)
        else:
            raise NotImplementedError

        # 移除中心点的余弦相似度，得到[KxK-1]的结果
        similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

        # 将结果重塑回[B, KxK-1, H, W]的形状
        similarity = similarity.view(B, k * k - 1, H, W)
        return similarity
    
    def resample(self, x, offset):
        # print(offset.shape)
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h]) # (1, 2, 1, H, W)
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1,  H, W)

class ResampleModulev2(nn.Module):
    def __init__(self, base_channels, hidden_channels=32, radius=3, kernel_size=3, stride=1, padding=1, use_normal=False, use_conf=False):
        super().__init__()
        if use_normal:
            self.normal_fusion = CoordConv(base_channels + 3, base_channels, kernel_size, stride, padding)
        if use_conf:
            self.conf_fusion = CoordConv(base_channels + 1, base_channels, kernel_size, stride, padding)
            
        self.radius = radius
        self.neighbor_num = radius * radius - 1
        self.out_channels = base_channels
        self.use_normal = use_normal
        self.use_conf = use_conf
        
        self.offset_num = 4
        
        self.offset = nn.Sequential(
            Conv2d(base_channels + self.neighbor_num, hidden_channels, kernel_size, stride, padding),
            Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding),
            Conv2d(hidden_channels, 2 * self.offset_num, kernel_size, stride, padding),
        )
    
    def forward(self, x, normal=None, confidence=None):
        if self.use_normal:
            assert normal is not None
            x = self.normal_fusion(torch.cat([x, normal], 1))
        if self.use_conf is not None:
            assert confidence is not None    
            x = self.conf_fusion(torch.cat([x, confidence], 1))
            
        local_sim = self.compute_similarity(x)
        x = torch.cat([x, local_sim], 1)
        offset = self.offset(x)
        
        B, _, H, W = offset.shape
        offset = offset.view(B, -1, H, W)
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h]) # (1, 2, H, W)
                             ).transpose(1, 2).unsqueeze(1).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1)
        coords_grid = 2 * (coords + offset) / normalizer - 1
        
        x = F.grid_sample(x.reshape(B * self.offset_num, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1,  H, W)
        return x
        
    def compute_similarity(self, x, dilation=1, sim='cos'):
        """
        计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

        参数：
        - x: 输入张量，形状为[B, C, H, W]
        - k: 范围大小，表示周围KxK范围内的点

        返回：
        - 输出张量，形状为[B, KxK-1, H, W]
        """
        B, C, H, W = x.shape
        k = self.radius
        # 使用零填充来处理边界情况
        # padded_input = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

        # 展平输入张量中每个点及其周围KxK范围内的点
        unfold_x = F.unfold(x, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
        # print(unfold_x.shape)
        unfold_x = unfold_x.reshape(B, C, k**2, H, W)

        # 计算余弦相似度
        if sim == 'cos':
            similarity = F.cosine_similarity(unfold_x[:, :, k * k // 2:k * k // 2 + 1], unfold_x[:, :, :], dim=1)
        elif sim == 'dot':
            similarity = unfold_x[:, :, k * k // 2:k * k // 2 + 1] * unfold_x[:, :, :]
            similarity = similarity.sum(dim=1)
        else:
            raise NotImplementedError

        # 移除中心点的余弦相似度，得到[KxK-1]的结果
        similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

        # 将结果重塑回[B, KxK-1, H, W]的形状
        similarity = similarity.view(B, k * k - 1, H, W)
        return similarity

class ResampleModulev3(nn.Module):
    def __init__(self, base_channels, neighbor_num=8, kernel_size=3, stride=1, padding=1, use_conf=False):
        super().__init__()
        if use_conf:
            self.offset = CoordConv(base_channels + neighbor_num + 1, 2, kernel_size, stride, padding)
        else:
            self.offset = CoordConv(base_channels + neighbor_num, 2, kernel_size, stride, padding)
        self.guide = FastGuide(base_channels)
    
    def forward(self, x, confidence=None):
        local_sim = self.compute_similarity(x)
        _x = torch.cat([x, local_sim], 1)
        if confidence is not None:
            _x = torch.cat([_x, confidence], 1)
        offset = self.offset(_x)
        _x = self.resample(x, offset)
        x = self.guide(x, _x)
        return x
        
    def compute_similarity(self, x, k=3, dilation=1, sim='cos'):
        """
        计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

        参数：
        - x: 输入张量，形状为[B, C, H, W]
        - k: 范围大小，表示周围KxK范围内的点

        返回：
        - 输出张量，形状为[B, KxK-1, H, W]
        """
        B, C, H, W = x.shape
        # 使用零填充来处理边界情况
        # padded_input = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

        # 展平输入张量中每个点及其周围KxK范围内的点
        unfold_x = F.unfold(x, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
        # print(unfold_x.shape)
        unfold_x = unfold_x.reshape(B, C, k**2, H, W)

        # 计算余弦相似度
        if sim == 'cos':
            similarity = F.cosine_similarity(unfold_x[:, :, k * k // 2:k * k // 2 + 1], unfold_x[:, :, :], dim=1)
        elif sim == 'dot':
            similarity = unfold_x[:, :, k * k // 2:k * k // 2 + 1] * unfold_x[:, :, :]
            similarity = similarity.sum(dim=1)
        else:
            raise NotImplementedError

        # 移除中心点的余弦相似度，得到[KxK-1]的结果
        similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

        # 将结果重塑回[B, KxK-1, H, W]的形状
        similarity = similarity.view(B, k * k - 1, H, W)
        return similarity
        
    
    def resample(self, x, offset):
        # print(offset.shape)
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h]) # (1, 2, 1, H, W)
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1,  H, W)

# 在获取偏移量之前要保持特征的局部性，获取偏移量之后才可以做特征融合
# 讲故事方面可以再考虑一下细节恢复
class ResampleModuleN(nn.Module):
    def __init__(self, base_channels, neighbor_num=8, offset_num=4, kernel_size=3, stride=1, padding=1, use_conf=False):
        super().__init__()
        self.offset_num = offset_num
        if use_conf:
            self.offset = CoordConv(base_channels + neighbor_num + 1, 2 * offset_num, kernel_size, stride, padding)
        else:
            self.offset = CoordConv(base_channels + neighbor_num, 2 * offset_num, kernel_size, stride, padding)
        
        self.register_buffer('init_pos', self._init_pos())
        
    def _init_pos(self):
        h = torch.arange(-0.5, 0.5)
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.offset_num, 1).reshape(1, -1, 1, 1)
    
    def forward(self, x, confidence=None, ret_off=False):
        local_sim = self.compute_similarity(x)
        _x = torch.cat([x, local_sim], 1)
        if confidence is not None:
            _x = torch.cat([_x, confidence], 1)
        offset = self.offset(_x) + self.init_pos
        x = self.resample(x, offset)
        if ret_off:
            return x, offset
        return x
        
    def compute_similarity(self, x, k=3, dilation=1, sim='cos'):
        B, C, H, W = x.shape
        # 使用零填充来处理边界情况
        # padded_input = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

        # 展平输入张量中每个点及其周围KxK范围内的点
        unfold_x = F.unfold(x, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
        # print(unfold_x.shape)
        unfold_x = unfold_x.reshape(B, C, k**2, H, W)

        # 计算余弦相似度
        if sim == 'cos':
            similarity = F.cosine_similarity(unfold_x[:, :, k * k // 2:k * k // 2 + 1], unfold_x[:, :, :], dim=1)
        elif sim == 'dot':
            similarity = unfold_x[:, :, k * k // 2:k * k // 2 + 1] * unfold_x[:, :, :]
            similarity = similarity.sum(dim=1)
        else:
            raise NotImplementedError

        # 移除中心点的余弦相似度，得到[KxK-1]的结果
        similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

        # 将结果重塑回[B, KxK-1, H, W]的形状
        similarity = similarity.view(B, k * k - 1, H, W)
        return similarity
        
    
    def resample(self, x, offset):
        # print(offset.shape, x.shape)
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h]) # (1, 2, 1, H, W)
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.offset_num, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1,  H, W)


def depth_resample(depth, normal, confidence, offset, intri):
    # 置信度不够高，且邻居的置信度更高，则进行深度传播覆盖
    # print(offset.shape)
    B, _, H, W = offset.shape
    depth = depth.unsqueeze(1)
    depth = F.interpolate(depth, size=None, scale_factor=2, mode="bilinear", align_corners=False)
    
    # offset = offset.view(B, 2, H, W)
    
    offset = offset.view(B, 2, -1, H, W)
    
    with torch.no_grad():
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                            ).transpose(1, 2).unsqueeze(0).unsqueeze(2).type(offset.dtype).to(offset.device).repeat(B, 1, 1, 1, 1)
        normalizer = torch.tensor([W, H], dtype=offset.dtype, device=offset.device).view(1, 2, 1, 1, 1)

        coords_shift = 2 * (coords + offset) / normalizer - 1
        coords_shift = coords_shift.permute(0, 2, 3, 4, 1).contiguous().view(B, offset.size(2) * H, W, 2)
        
        confidence_shift = F.grid_sample(confidence, coords_shift, mode='bilinear', align_corners=False, padding_mode="border").view(B, -1, H, W)
        max_index = torch.argmax(confidence_shift, dim=1).unsqueeze(1).unsqueeze(-1).expand(B, -1, H, W, 2)

        coords_shift = torch.gather(coords_shift.view(B, -1, H, W, 2), dim=1, index=max_index).squeeze(1)
        coords = coords.squeeze(2)

        del coords_h, coords_w, normalizer, max_index
    
    normal_shift = F.grid_sample(normal, coords_shift, mode='bilinear', align_corners=False, padding_mode="border")
    
    # propagation
    fx, fy = intri[:, 0, 0].view(B, 1, 1, 1), intri[:, 1, 1].view(B, 1, 1, 1)
    cx, cy = intri[:, 0, 2].view(B, 1, 1, 1), intri[:, 1, 2].view(B, 1, 1, 1)  # B,
    nx, ny, nz = normal_shift[:, 0].unsqueeze(1), normal_shift[:, 0].unsqueeze(1), normal_shift[:, 0].unsqueeze(1)
    del normal_shift
    
    u = ((coords[:, 0].unsqueeze(1) - cx) / fx)
    v = ((coords[:, 1].unsqueeze(1) - cy) / fy)
    
    u_shift = F.grid_sample(u, coords_shift, mode='bilinear', align_corners=False, padding_mode="border")
    v_shift = F.grid_sample(v, coords_shift, mode='bilinear', align_corners=False, padding_mode="border")
    
    ddw_num = nx * u_shift + ny * v_shift + nz
    ddw_denom = nx * u + ny * v + nz  # B 1 H W
    del u, v, u_shift, v_shift
    del nx, ny, nz
    # print(ddw_num.shape, ddw_denom.shape)
    
    ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8
    ddw_denom[torch.abs(ddw_denom) < 1e-8] = 1e-8

    ddw_weights = ddw_num / ddw_denom  # (B, 1, H, W)
    del ddw_num, ddw_denom
    
    ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
    ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k2 h w

    depth_shift = F.grid_sample(depth, coords_shift, mode='bilinear', align_corners=False, padding_mode="border")
    depth_shift = ddw_weights * depth_shift  
    del ddw_weights
    #============
    
    confidence_shift = F.grid_sample(confidence, coords_shift, mode='bilinear', align_corners=False, padding_mode="border")
    # print(confidence.shape, confidence_shift.shape, coords_shift.shape)
    mask = torch.where(torch.logical_and(confidence < confidence.mean(), confidence < confidence_shift), 
                    torch.full_like(confidence, True, dtype=torch.bool), torch.full_like(confidence, False, dtype=torch.bool))
    del confidence, confidence_shift
    depth[mask] = depth_shift[mask]
    del depth_shift
    del mask
    
    return depth.squeeze(1)


class RefineNet(nn.Module):
    def __init__(self, K=3, tau=0.996):
        super().__init__()
        self.K = K
        self.tau = tau
    
    def compute_kernel(self, normal_center, normal_neighbours):
        kernel_values = normal_center.unsqueeze(2) * normal_neighbours
        kernel_values = kernel_values.sum(1)
        return kernel_values
    
    def forward(self, depth, normal, intri):
        B, H, W = depth.shape
        
        pad = self.K // 2
        kernel_size = self.K

        normal_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_unfold = F.unfold(normal_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        normal_unfold = normal_unfold.view(B, 3, kernel_size ** 2, H, W)
        
        nx, ny, nz = normal_unfold[:, 0, ...], normal_unfold[:, 1, ...], normal_unfold[:, 2, ...]

        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=normal.device),
                            torch.arange(0, W, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(B, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx.unsqueeze(1)) / fx.unsqueeze(1)  # b h*w
        posy = (xy[:, 1, :] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        pos = torch.stack([posx, posy], dim=1).reshape(B, 2, H, W)
        pos_unfold = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        pos_unfold = pos_unfold.view(B, 2, kernel_size * kernel_size, H, W)

        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        # ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        # ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k2 h w
        
        # conf_mask = confidence < confidence.mean(dim=(1, 2), keepdim=True)
        # conf_mask = conf_mask.unsqueeze(1)
        
        depth_unfold = F.pad(depth, pad=[pad, pad, pad, pad], mode='replicate')
        depth_unfold = F.unfold(depth_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        depth_unfold = depth_unfold.view(B, kernel_size ** 2, H, W)  # b k2 h w
        
        depth_propa = ddw_weights * depth_unfold  # b k2 h w
        
        kernel_values = self.compute_kernel(normal, normal_unfold)
        kernel_values[ddw_weights != ddw_weights] = 0.0  # nan
        kernel_values[torch.abs(ddw_weights) == float("Inf")] = 0.0  # inf  #b k2 h w
        
        valid_mask = kernel_values > self.tau
        
        kernel_values = kernel_values * valid_mask
        depth_propa = depth_propa * valid_mask
        
        refined_depth = (depth_propa * kernel_values / kernel_values.sum(dim=1, keepdim=True)).sum(dim=1)
        invalid_mask = (refined_depth != refined_depth) | (torch.abs(refined_depth) == float("Inf"))
        refined_depth[invalid_mask] = depth[invalid_mask]
        
        return refined_depth    

class AdaptiveWeightingNet(nn.Module):
    def __init__(self, num_candidates, in_channels=3):
        super(AdaptiveWeightingNet, self).__init__()
        self.conv1 = Conv2d(in_channels + 3, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(num_candidates + 32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, num_candidates, kernel_size=3, padding=1)
    
    def forward(self, candidates, image_features, normal):
        x = torch.cat([image_features, normal], dim=1)
        x = self.conv1(x)
        
        x = torch.cat([candidates, x], dim=1)
        x = self.conv2(x)
        
        weights = F.softmax(self.conv3(x), dim=1)
        return weights


class ResidualNet(nn.Module):
    def __init__(self, feat_channels):
        super(ResidualNet, self).__init__()
        base_channels = 8

        self.conv1_0 = nn.Sequential(
            Conv2d(1, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv1_2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 3, 2, 1, 1, bias=False)

        self.conv2_0 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(feat_channels + base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)
        
    def forward(self, depth, img_feat):
        depth_mean = torch.mean(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        depth_min, _ = torch.min(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_max,_ = torch.max(depth.reshape(depth.shape[0],-1), -1, keepdim=True)

        depth_feat = self.conv1_2(self.conv1_0(depth))
        cat = torch.cat((img_feat, depth_feat), dim=1)

        residual = cat
        x = self.conv2_1(self.conv2_0(cat))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv4_0(out1)
        x = torch.cat([x, cat], 1)
        x = self.conv4_1(x)
        residual = x
        x = self.conv4_3(self.conv4_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        res = self.final_conv(out2)

        res_ = torch.zeros_like(res)
        for i in range(res.shape[0]):
            res_[i] = torch.clamp(res[i], min=depth_min[i].cpu().item(), max=depth_max[i].cpu().item())
        depth = (res_ + F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=False)) * depth_std.unsqueeze(-1).unsqueeze(-1) + depth_mean.unsqueeze(-1).unsqueeze(-1)
        
        return res_.squeeze(1), depth.squeeze(1)
    
    
class ResidualNetv2(nn.Module):
    def __init__(self, feat_channels):
        super(ResidualNetv2, self).__init__()
        self.patch_size = 3
        base_channels = 8
            
        self.conv1_0 = nn.Sequential(
            Conv2d(9, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv1_2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1, bias=False)

        self.conv2_0 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(feat_channels + base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)
        self.act = nn.Tanh()
        
    def forward(self, depth, img_feat, normal, intri, interval, inverse_d=False):
        depth_propa = self.propagate(depth, normal, intri)
        
        if inverse_d:
            depth_propa = 1./ depth_propa
            depth = 1./ depth
        
        depth_mean = torch.mean(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        
        depth_propa = (depth_propa - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        depth_norm = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)

        depth_propa = depth_propa - depth_norm
        
        depth_feat = self.conv1_2(self.conv1_0(torch.cat((depth_norm, depth_propa), dim=1)))
        
        cat = torch.cat((img_feat, depth_feat), dim=1)

        residual = cat
        x = self.conv2_1(self.conv2_0(cat))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv4_0(out1)
        x = torch.cat((x, cat), 1)
        x = self.conv4_1(x)
        residual = x
        x = self.conv4_3(self.conv4_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        off = self.final_conv(out2).squeeze(1)
        off = self.act(off) * interval
        depth = depth + off
        
        if inverse_d:
            depth = 1./ depth
        
        return depth, off
    
    def propagate(self, depth, normal, intri):
        B, H, W = depth.shape
        
        pad = self.patch_size // 2
        kernel_size = self.patch_size

        normal_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_unfold = F.unfold(normal_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        normal_unfold = normal_unfold.view(B, 3, kernel_size ** 2, H, W)
        
        nx, ny, nz = normal_unfold[:, 0, ...], normal_unfold[:, 1, ...], normal_unfold[:, 2, ...]

        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=normal.device),
                            torch.arange(0, W, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(B, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx.unsqueeze(1)) / fx.unsqueeze(1)  # b h*w
        posy = (xy[:, 1, :] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        pos = torch.stack([posx, posy], dim=1).reshape(B, 2, H, W)
        pos_unfold = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        pos_unfold = pos_unfold.view(B, 2, kernel_size * kernel_size, H, W)

        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        
        depth_unfold = F.pad(depth, pad=[pad, pad, pad, pad], mode='replicate')
        depth_unfold = F.unfold(depth_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        depth_unfold = depth_unfold.view(B, kernel_size ** 2, H, W)  # b k2 h w
        
        depth_propa = ddw_weights * depth_unfold  # b k2 h w
        
        depth_propa = torch.cat([depth_propa[:, :kernel_size * kernel_size // 2],
                                 depth_propa[:, kernel_size * kernel_size // 2 + 1:]], dim=1)
        invalid_mask = (depth_propa != depth_propa) | (torch.abs(depth_propa) == float("Inf"))
        
        N = kernel_size * kernel_size - 1
        depth_propa[invalid_mask] = depth.unsqueeze(1).repeat(1, N, 1, 1)[invalid_mask]
        
        depth_min, _ = torch.min(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_max, _ = torch.max(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_min = depth_min.view(B, 1, 1, 1)
        depth_max = depth_max.view(B, 1, 1, 1)
        depth_propa = torch.clamp(depth_propa, min=depth_min, max=depth_max)
        
        return depth_propa

class ResidualNetv2_won(nn.Module):
    def __init__(self, feat_channels):
        super(ResidualNetv2_won, self).__init__()
        self.patch_size = 3
        base_channels = 8
            
        self.conv1_0 = nn.Sequential(
            Conv2d(1, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv1_2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1, bias=False)

        self.conv2_0 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(feat_channels + base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)
        self.act = nn.Tanh()
        
    def forward(self, depth, img_feat, normal, intri, interval, inverse_d=False):
        depth_mean = torch.mean(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        
        depth_norm = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        
        depth_feat = self.conv1_2(self.conv1_0(depth_norm))
        
        cat = torch.cat((img_feat, depth_feat), dim=1)

        residual = cat
        x = self.conv2_1(self.conv2_0(cat))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv4_0(out1)
        x = torch.cat((x, cat), 1)
        x = self.conv4_1(x)
        residual = x
        x = self.conv4_3(self.conv4_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        off = self.final_conv(out2).squeeze(1)
        off = self.act(off) * interval
        depth = depth + off
        
        if inverse_d:
            depth = 1./ depth
        
        return depth, off

    
class ResidualNetv3(nn.Module):
    def __init__(self, feat_channels):
        super(ResidualNetv3, self).__init__()
        self.patch_size = 3
        base_channels = 8
            
        self.conv1_0 = nn.Sequential(
            Conv2d(9, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1))
        self.conv1_2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1, bias=False)

        self.conv2_0 = Conv2d(feat_channels+base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels+base_channels * 4, base_channels * 8, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 3, 2, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(feat_channels + base_channels * 8, base_channels * 8, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 8, base_channels * 4, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 4, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)
        self.act = nn.Tanh()
        
    def forward(self, depth, img_feat, normal, intri, interval):
        depth_propa = self.propagate(depth, normal, intri)
        
        depth_mean = torch.mean(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        
        depth_propa = (depth_propa - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        depth_norm = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)

        depth_propa = depth_propa - depth_norm
        
        depth_feat = self.conv1_2(self.conv1_0(torch.cat((depth_norm, depth_propa), dim=1)))
        
        print(img_feat.shape, depth_feat.shape)
        cat = torch.cat((img_feat, depth_feat), dim=1)

        residual = cat
        x = self.conv2_1(self.conv2_0(cat))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv4_0(out1)
        x = torch.cat((x, cat), 1)
        x = self.conv4_1(x)
        residual = x
        x = self.conv4_3(self.conv4_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        off = self.final_conv(out2).squeeze(1)
        off = self.act(off) * interval
        depth = depth + off
        
        return depth, off
    
    def propagate(self, depth, normal, intri, dilation=3):
        B, H, W = depth.shape
        
        pad = self.patch_size // 2 * dilation
        kernel_size = self.patch_size

        # normal_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_unfold = F.unfold(normal, [kernel_size, kernel_size], padding=pad, dilation=dilation, stride=1)
        normal_unfold = normal_unfold.view(B, 3, kernel_size ** 2, H, W)
        
        nx, ny, nz = normal_unfold[:, 0, ...], normal_unfold[:, 1, ...], normal_unfold[:, 2, ...]

        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=normal.device),
                            torch.arange(0, W, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(B, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx.unsqueeze(1)) / fx.unsqueeze(1)  # b h*w
        posy = (xy[:, 1, :] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        pos = torch.stack([posx, posy], dim=1).reshape(B, 2, H, W)
        # pos_unfold = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos, [kernel_size, kernel_size], padding=pad, dilation=dilation, stride=1)
        pos_unfold = pos_unfold.view(B, 2, kernel_size * kernel_size, H, W)

        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        
        # depth_unfold = F.pad(depth, pad=[pad, pad, pad, pad], mode='replicate')
        depth_unfold = F.unfold(depth, [kernel_size, kernel_size], padding=pad, dilation=dilation, stride=1)
        depth_unfold = depth_unfold.view(B, kernel_size ** 2, H, W)  # b k2 h w
        
        depth_propa = ddw_weights * depth_unfold  # b k2 h w
        
        depth_propa = torch.cat([depth_propa[:, :kernel_size * kernel_size // 2],
                                 depth_propa[:, kernel_size * kernel_size // 2 + 1:]], dim=1)
        invalid_mask = (depth_propa != depth_propa) | (torch.abs(depth_propa) == float("Inf"))
        
        N = kernel_size * kernel_size - 1
        depth_propa[invalid_mask] = depth.unsqueeze(1).repeat(1, N, 1, 1)[invalid_mask]
        
        depth_min, _ = torch.min(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_max, _ = torch.max(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_min = depth_min.view(B, 1, 1, 1)
        depth_max = depth_max.view(B, 1, 1, 1)
        depth_propa = torch.clamp(depth_propa, min=depth_min, max=depth_max)
        
        depth_propa = torch.sort(depth_propa, 1)[0]
        
        return depth_propa


class ResidualNetv4(nn.Module):
    def __init__(self, feat_channels, patch_size, num_ch_dec = [64, 128, 256], dilation=1):
        super(ResidualNetv4, self).__init__()
        self.feat_channels = feat_channels
        self.num_ch_dec = num_ch_dec
        self.patch_size = patch_size
        self.dilation = dilation
            
        depth_channels = patch_size ** 2
        
        self.conv_fusion_1 = nn.Sequential(Conv2d(feat_channels + depth_channels, num_ch_dec[0], kernel_size=3, stride=1, padding=1),
                                           Conv2d(num_ch_dec[0], num_ch_dec[0], kernel_size=3, stride=1, padding=1))
        self.down_sample_2 = Conv2d(num_ch_dec[0], num_ch_dec[1], kernel_size=3, stride=2, padding=1, groups=depth_channels)
        self.down_sample_3 = Conv2d(num_ch_dec[1], num_ch_dec[2], kernel_size=3, stride=2, padding=1, groups=depth_channels)
        
        self.up_sample_1 = Deconv2d(num_ch_dec[2], num_ch_dec[1], kernel_size=3, stride=2, groups=depth_channels)
        self.skip_1 = Conv2d(num_ch_dec[1] * 2, num_ch_dec[1], kernel_size=3, stride=1, padding=1, groups=depth_channels)
        self.up_sample_2 = Deconv2d(num_ch_dec[1], num_ch_dec[0], kernel_size=3, stride=2, groups=depth_channels)
        self.skip_2 = Conv2d(num_ch_dec[0] * 2, num_ch_dec[0], kernel_size=3, stride=1, padding=1, groups=depth_channels)
        
        self.prob = Conv2d(num_ch_dec[0], depth_channels, kernel_size=3, stride=1, padding=1, groups=depth_channels)
        
    def forward(self, depth, img_feat, prob_volume, depth_values, normal, intri):
        if prob_volume.dim() == 4:
            prob_volume = prob_volume.unsqueeze(1)
        # B C D H W
        depth_propa = self.propagate(depth, normal, intri)
        B, D, H, W = depth_values.shape
        D_new = self.patch_size * self.patch_size
        depth_interval = depth_values[:, 1] - depth_values[:, 0]  # B H W
        depth_indices = (depth_propa - depth_values[:, 0].unsqueeze(1)) / depth_interval.unsqueeze(1)  # B D' H W

        h_coords, w_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=normal.device),
            torch.arange(W, dtype=torch.float32, device=normal.device),
            indexing='ij'
        )
        h_coords = h_coords.unsqueeze(0).unsqueeze(0).repeat(B, D_new, 1, 1)  # (B, D', H, W)
        w_coords = w_coords.unsqueeze(0).unsqueeze(0).repeat(B, D_new, 1, 1)

        grid = torch.stack((w_coords / (W - 1) * 2 - 1,  # x 坐标归一化到 [-1, 1]
                            h_coords / (H - 1) * 2 - 1,  # y 坐标归一化到 [-1, 1]
                            depth_indices / (D - 1) * 2 - 1),  # 深度索引归一化到 [-1, 1]
                        dim=-1)

        new_prob_volume = prob_volume.permute(0, 1, 3, 4, 2)  # 转换为 (B, C, H, W, D)
        new_prob_volume = F.grid_sample(
            new_prob_volume, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        depth_feat = new_prob_volume.squeeze(1)
        
        fusion_feat0 = torch.cat([img_feat, depth_feat], 1)
        fusion_feat1 = self.conv_fusion_1(fusion_feat0)
        fusion_feat2 = self.down_sample_2(fusion_feat1)
        fusion_feat3 = self.down_sample_3(fusion_feat2)
        
        x = self.skip_1(torch.cat((self.up_sample_1(fusion_feat3), fusion_feat2), 1))
        x = self.skip_2(torch.cat((self.up_sample_2(x), fusion_feat1), 1))
        
        prob_volume_rf = self.prob(x) # B, 1, D, H, W
        prob_volume_rf = torch.exp(F.log_softmax(prob_volume_rf, dim=1))
        depth_rf = depth_regression(prob_volume_rf, depth_values=depth_propa)
        
        return depth_rf, prob_volume_rf, depth_propa
    
    def interpolate_channels(self, depth_feat, depth_channels, mode='bilinear'):
        B, D, H, W = depth_feat.shape

        # 将通道维度作为插值目标
        depth_feat = depth_feat.permute(0, 2, 3, 1).unsqueeze(-1)  # (B, H, W, C, 1)
        out = F.interpolate(depth_feat.view(B, H*W, D, 1), size=(depth_channels, 1), mode=mode, align_corners=True)  # 插值通道
        out = out.view(B, H, W, depth_channels, 1).squeeze(-1).permute(0, 3, 1, 2)  # 恢复形状 (B, C, H, W)

        return out
    
    def propagate(self, depth, normal, intri):
        dilation = self.dilation
        B, H, W = depth.shape
        
        pad = self.patch_size // 2 * dilation
        kernel_size = self.patch_size

        # normal_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_unfold = F.unfold(normal, [kernel_size, kernel_size], padding=pad, dilation=dilation, stride=1)
        normal_unfold = normal_unfold.view(B, 3, kernel_size ** 2, H, W)
        
        nx, ny, nz = normal_unfold[:, 0, ...], normal_unfold[:, 1, ...], normal_unfold[:, 2, ...]

        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=normal.device),
                            torch.arange(0, W, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(B, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx.unsqueeze(1)) / fx.unsqueeze(1)  # b h*w
        posy = (xy[:, 1, :] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        pos = torch.stack([posx, posy], dim=1).reshape(B, 2, H, W)
        # pos_unfold = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos, [kernel_size, kernel_size], padding=pad, dilation=dilation, stride=1)
        pos_unfold = pos_unfold.view(B, 2, kernel_size * kernel_size, H, W)

        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        
        # depth_unfold = F.pad(depth, pad=[pad, pad, pad, pad], mode='replicate')
        depth_unfold = F.unfold(depth, [kernel_size, kernel_size], padding=pad, dilation=dilation, stride=1)
        depth_unfold = depth_unfold.view(B, kernel_size ** 2, H, W)  # b k2 h w
        
        depth_propa = ddw_weights * depth_unfold  # b k2 h w
        
        # depth_propa = torch.cat([depth_propa[:, :kernel_size * kernel_size // 2],
        #                          depth_propa[:, kernel_size * kernel_size // 2 + 1:]], dim=1)
        # N = kernel_size * kernel_size - 1
        
        N = kernel_size * kernel_size
        
        invalid_mask = (depth_propa != depth_propa) | (torch.abs(depth_propa) == float("Inf"))
        
        depth_propa[invalid_mask] = depth.unsqueeze(1).repeat(1, N, 1, 1)[invalid_mask]
        
        depth_min, _ = torch.min(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_max, _ = torch.max(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_min = depth_min.view(B, 1, 1, 1)
        depth_max = depth_max.view(B, 1, 1, 1)
        depth_propa = torch.clamp(depth_propa, min=depth_min, max=depth_max)
        
        depth_propa = torch.sort(depth_propa, 1)[0]
        
        return depth_propa

class RefineNetv2(nn.Module):
    def __init__(self, in_channels, patch_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.weight_net = AdaptiveWeightingNet(patch_size * patch_size, in_channels=in_channels)
        
    def forward(self, depth, img_feat, normal, intri):
        normalized_depth, depth_mean, depth_std = self.normalize_depth(depth)
        depth_min, _ = torch.min(normalized_depth.reshape(normalized_depth.shape[0],-1), -1, keepdim=True)
        depth_max, _ = torch.max(normalized_depth.reshape(normalized_depth.shape[0],-1), -1, keepdim=True)

        propagated_depth = self.propagate(normalized_depth, normal, intri)
        weight = self.weight_net(propagated_depth, img_feat, normal)
        refined_depth = (weight * propagated_depth).sum(dim=1, keepdim=True)
        
        depth = self.restore_depth(refined_depth, depth_mean, depth_std)
        
        return depth
        
    def propagate(self, depth, normal, intri):
        B, H, W = depth.shape
        
        pad = self.patch_size // 2
        kernel_size = self.patch_size

        normal_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_unfold = F.unfold(normal_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        normal_unfold = normal_unfold.view(B, 3, kernel_size ** 2, H, W)
        
        nx, ny, nz = normal_unfold[:, 0, ...], normal_unfold[:, 1, ...], normal_unfold[:, 2, ...]

        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=normal.device),
                            torch.arange(0, W, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(B, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx.unsqueeze(1)) / fx.unsqueeze(1)  # b h*w
        posy = (xy[:, 1, :] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        pos = torch.stack([posx, posy], dim=1).reshape(B, 2, H, W)
        pos_unfold = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        pos_unfold = pos_unfold.view(B, 2, kernel_size * kernel_size, H, W)

        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        
        depth_unfold = F.pad(depth, pad=[pad, pad, pad, pad], mode='replicate')
        depth_unfold = F.unfold(depth_unfold, [kernel_size, kernel_size], padding=0, stride=1)
        depth_unfold = depth_unfold.view(B, kernel_size ** 2, H, W)  # b k2 h w
        
        depth_propa = ddw_weights * depth_unfold  # b k2 h w
        
        return depth_propa
    
    def normalize_depth(self, depth):
        # 归一化深度
        depth_mean = torch.mean(depth.reshape(depth.shape[0], -1), dim=-1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0], -1), dim=-1, keepdim=True)
        normalized_depth = (depth - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        return normalized_depth, depth_mean, depth_std

    def restore_depth(self, normalized_depth, depth_mean, depth_std):
        # 将归一化深度还原
        return normalized_depth * depth_std.unsqueeze(-1).unsqueeze(-1) + depth_mean.unsqueeze(-1).unsqueeze(-1)

class LightGoConv3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 bn=True,
                 vertical=0):
        super(LightGoConv3D, self).__init__()

        assert kernel_size[1] == kernel_size[2]
        assert stride[1] == stride[2]
        assert padding[1] == padding[2]

        self.vertical = vertical
        self.kernel_size = kernel_size[2]
        self.stride = stride[2]
        self.conv3d = nn.Conv3d(in_channels * self.kernel_size,
                                out_channels,
                                kernel_size=(kernel_size[0], 1, 1),
                                padding=(padding[0], 0, 0),
                                stride=(stride[0], 1, 1),
                                bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
        else:
            self.bn = None

    def get_gcacost(self, x_ori, dp, normal, intri, vertical=0):
        random_value = self.vertical
        
        kernel_size = self.kernel_size
        stride = self.stride
        b, c, d, h, w = x_ori.shape
       
        h = h // stride
        w = w // stride

        pad = (kernel_size - 1) // 2

        if random_value == 0:
            dp_unfold = F.pad(dp, pad=[pad, pad, 0, 0], mode='replicate')
            dp_unfold = F.unfold(dp_unfold, [1, kernel_size], padding=0, stride=stride)
            normal_p_unfold = F.pad(normal, pad=[pad, pad, 0, 0], mode='replicate')
            normal_p_unfold = F.unfold(normal_p_unfold, [1, kernel_size], padding=0, stride=stride)
        else:
            dp_unfold = F.pad(dp, pad=[0, 0, pad, pad], mode='replicate')
            dp_unfold = F.unfold(dp_unfold, [kernel_size, 1], padding=0, stride=stride)
            normal_p_unfold = F.pad(normal, pad=[0, 0, pad, pad], mode='replicate')
            normal_p_unfold = F.unfold(normal_p_unfold, [kernel_size, 1], padding=0, stride=stride)
        
        dp_unfold = dp_unfold.view(b, d, kernel_size, h, w).squeeze(1)  # b d k h w
        normal_p_unfold = normal_p_unfold.view(b, 3, kernel_size, h, w).squeeze(1)

        nx, ny, nz = normal_p_unfold[:, 0, ...], normal_p_unfold[:, 1, ...], normal_p_unfold[:, 2, ...]
        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        h_ori = h * stride
        w_ori = w * stride
        y, x = torch.meshgrid([torch.arange(0, h_ori, dtype=torch.float32, device=normal.device),
                               torch.arange(0, w_ori, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h_ori * w_ori), x.view(h_ori * w_ori)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(b, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx.unsqueeze(1)) / fx.unsqueeze(1)  # b h*w
        posy = (xy[:, 1, :] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        # print(posx.shape, cx.shape, xy[:, 0, :].shape, h_ori, w_ori, h_ori * w_ori)
        pos = torch.stack([posx, posy], dim=1).reshape(b, 2, h_ori, w_ori)
        if random_value == 0:
            pos_p = F.pad(pos, pad=[pad, pad, 0, 0], mode='replicate')
            pos_unfold = F.unfold(pos_p, [1, kernel_size], padding=0, stride=stride)
        else:
            pos_p = F.pad(pos, pad=[0, 0, pad, pad], mode='replicate')
            pos_unfold = F.unfold(pos_p, [kernel_size, 1], padding=0, stride=stride)
        pos_unfold = pos_unfold.view(b, 2, kernel_size, h, w)
        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]

        # pos - center
        pos_u_center = pos_unfold[:, 0, kernel_size // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, kernel_size // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k h w

        dp_prog = ddw_weights.unsqueeze(1) * dp_unfold[:, :, kernel_size // 2, :, :].unsqueeze(2)  # b d k h w
        interval_unfold = dp_unfold[:, 1] - dp_unfold[:, 0] # b k h w

        # b d k2 h w  - b 1 k2 h w  / B  1 K*K H W = b d k h w
        indices = (dp_prog - dp_unfold[:, 0].unsqueeze(1)) / interval_unfold.unsqueeze(1)  # b d k h w
        #indices = indices / ((d - 1) / 2) - 1 # b d k h w
        indices = indices.reshape(b, d * kernel_size, h, w).unsqueeze(-1)

        xy_n = xy.reshape(b, 2, h_ori, w_ori)# b 2 h w .permute(0, 2, 1).reshape(0, h_ori, w_ori, 2).unsqueeze(1)  # b 1 h w 2

        if random_value == 0:
            xy_p = F.pad(xy_n, pad=[pad, pad, 0, 0], mode='replicate')
            xy_p = F.unfold(xy_p, [1, kernel_size], padding=0, stride=stride)  # (B, ps, H*W)
        else:
            xy_p = F.pad(xy_n, pad=[0, 0, pad, pad], mode='replicate')
            xy_p = F.unfold(xy_p, [kernel_size, 1], padding=0, stride=stride)  # (B, ps, H*W)
        xy_p = xy_p.reshape(b, 2, kernel_size, h, w).unsqueeze(1).repeat(1, d, 1, 1, 1, 1)  # b d 2 k h w
        xy_p = xy_p.permute(0, 1, 3, 4, 5, 2).reshape(b, d * kernel_size, h, w, 2)

        indices = (torch.cat([xy_p, indices], dim=-1))
        indices[:, :, :, :, 2] = indices[:, :, :, :, 2] / ((d - 1) / 2) - 1
        indices[:, :, :, :, 0] = indices[:, :, :, :, 0] / ((w_ori - 1) / 2) - 1
        indices[:, :, :, :, 1] = indices[:, :, :, :, 1] / ((h_ori - 1) / 2) - 1
        ans = F.grid_sample(x_ori, grid=indices, padding_mode='zeros', mode='bilinear', align_corners=True)
        ans = ans.reshape(b, c, d, kernel_size, h, w).permute(0,1,3,2,4,5)
        return ans.reshape(b, c * kernel_size, d, h, w)

    def forward(self, x, dp, normal, intri):
        # print('phase1:', x.shape)
        x = self.get_gcacost(x, dp, normal, intri)  # b c*k*k, d h w
        # print('phase2:', x.shape)
        x = self.conv3d(x)  # b c,d h w
        # print('phase3:', x.shape)
        if self.bn is not None:
            x = F.relu(self.bn(x), inplace=True)
            # print('phase4:', x.shape)
        return x


class LightGoUpConv3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 bn=True,
                 vertical=0):
        super(LightGoUpConv3D, self).__init__()

        assert kernel_size[1] == kernel_size[2]
        assert stride[1] == stride[2]
        assert padding[1] == padding[2]

        self.vertical = vertical
        self.kernel_size = kernel_size[2]
        self.stride = stride[2]
        self.out_channels = out_channels
        self.conv3d = nn.Conv3d(in_channels * self.kernel_size,
                                out_channels * 2 * 2,
                                kernel_size=(kernel_size[0], 1, 1),
                                padding=(padding[0], 0, 0),
                                stride=(stride[0], 1, 1),
                                bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
        else:
            self.bn = None

    def get_pecost(self, x_ori, dp, normal, intri):
        # intric b 3 3
        # dp: depth B D H W
        random_value = self.vertical
        
        kernel_size = self.kernel_size
        stride = self.stride
        b, c, d, h, w = x_ori.shape
        
        h = h // stride
        w = w // stride

        pad = (kernel_size - 1) // 2

        if random_value == 0:
            dp_unfold = F.pad(dp, pad=[pad, pad, 0, 0], mode='replicate')
            dp_unfold = F.unfold(dp_unfold, [1, kernel_size], padding=0, stride=stride)
            normal_p_unfold = F.pad(normal, pad=[pad, pad, 0, 0], mode='replicate')
            normal_p_unfold = F.unfold(normal_p_unfold, [1, kernel_size], padding=0, stride=stride)
        else:
            dp_unfold = F.pad(dp, pad=[0, 0, pad, pad], mode='replicate')
            dp_unfold = F.unfold(dp_unfold, [kernel_size, 1], padding=0, stride=stride)
            normal_p_unfold = F.pad(normal, pad=[0, 0, pad, pad], mode='replicate')
            normal_p_unfold = F.unfold(normal_p_unfold, [kernel_size, 1], padding=0, stride=stride)
        
        dp_unfold = dp_unfold.view(b, d, kernel_size, h, w).squeeze(1)  # b d k h w
        normal_p_unfold = normal_p_unfold.view(b, 3, kernel_size, h, w).squeeze(1)

        nx, ny, nz = normal_p_unfold[:, 0, ...], normal_p_unfold[:, 1, ...], normal_p_unfold[:, 2, ...]
        # (B, 3, ps*ps, H, W)
        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        h_ori = h * stride
        w_ori = w * stride
        y, x = torch.meshgrid([torch.arange(0, h_ori, dtype=torch.float32, device=normal.device),
                               torch.arange(0, w_ori, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h_ori * w_ori), x.view(h_ori * w_ori)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(b, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx.unsqueeze(1)) / fx.unsqueeze(1)  # b h*w
        posy = (xy[:, 1, :] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        pos = torch.stack([posx, posy], dim=1).reshape(b, 2, h_ori, w_ori)
        # print('pos,',pos)
        if random_value == 0:
            pos_p = F.pad(pos, pad=[pad, pad, 0, 0], mode='replicate')
            pos_unfold = F.unfold(pos_p, [1, kernel_size], padding=0, stride=stride)
        else:
            pos_p = F.pad(pos, pad=[0, 0, pad, pad], mode='replicate')
            pos_unfold = F.unfold(pos_p, [kernel_size, 1], padding=0, stride=stride)
        # print('pos_unfold,',pos_unfold)
        pos_unfold = pos_unfold.view(b, 2, kernel_size, h, w)

        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]

        # pos - center
        pos_u_center = pos_unfold[:, 0, kernel_size // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, kernel_size // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k h w

        dp_prog = ddw_weights.unsqueeze(1) * dp_unfold[:, :, kernel_size // 2, :, :].unsqueeze(
            2)  # b d k2 h w
        interval_unfold = dp_unfold[:, 1] - dp_unfold[:, 0]  # b k h w


        indices = (dp_prog - dp_unfold[:, 0].unsqueeze(1)) / interval_unfold.unsqueeze(1)  # b d k h w
        indices = indices.reshape(b, d * kernel_size, h, w).unsqueeze(-1)

        xy_n = xy.reshape(b, 2, h_ori,
                          w_ori) 

        if random_value == 0:
            xy_p = F.pad(xy_n, pad=[pad, pad, 0, 0], mode='replicate')
            xy_p = F.unfold(xy_p, [1, kernel_size], padding=0, stride=stride)  # (B, ps, H*W)
        else:
            xy_p = F.pad(xy_n, pad=[0, 0, pad, pad], mode='replicate')
            xy_p = F.unfold(xy_p, [kernel_size, 1], padding=0, stride=stride)  # (B, ps, H*W)
        xy_p = xy_p.reshape(b, 2, kernel_size, h, w).unsqueeze(1).repeat(1, d, 1, 1, 1, 1)  # b d 2 k h w
        xy_p = xy_p.permute(0, 1, 3, 4, 5, 2).reshape(b, d * kernel_size, h, w, 2)

        indices = (torch.cat([xy_p, indices], dim=-1))
        indices[:, :, :, :, 2] = indices[:, :, :, :, 2] / ((d - 1) / 2) - 1
        indices[:, :, :, :, 0] = indices[:, :, :, :, 0] / ((w_ori - 1) / 2) - 1
        indices[:, :, :, :, 1] = indices[:, :, :, :, 1] / ((h_ori - 1) / 2) - 1  # b d*k2 h w 3
        ans = F.grid_sample(x_ori, grid=indices, padding_mode='zeros', mode='bilinear', align_corners=True)
        ans = ans.reshape(b, c, d, kernel_size, h, w).permute(0, 1, 3, 2, 4, 5)
        return ans.reshape(b, c * kernel_size, d, h, w)

    def forward(self, x, pd, normal, intri):
        b, c, d, h, w = x.shape
        x = self.get_pecost(x, pd, normal, intri)  # b c*k*k d h w
        x = self.conv3d(x)  # b c,d h w
        x = x.reshape(b, self.out_channels, 2, 2, d, h, w)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, self.out_channels, d, h * 2,w * 2)
        x = F.relu(self.bn(x), inplace=True)
        return x


class LightGCACostRegNet(nn.Module):
    '''
    input b d h w
    output b d h w
    '''

    def __init__(self, in_channels, base_channels):
        super(LightGCACostRegNet, self).__init__()

        self.conv0 = Conv3d(in_channels, base_channels, kernel_size=(7,1,1), padding=(3,0,0))

        self.conv1 = LightGoConv3D(base_channels, base_channels * 2, stride=(1, 2, 2), vertical=0)
        self.conv2 = LightGoConv3D(base_channels * 2, base_channels * 2, vertical=1)

        self.conv3 = LightGoConv3D(base_channels * 2, base_channels * 4, stride=(1, 2, 2), vertical=0)
        self.conv4 = LightGoConv3D(base_channels * 4, base_channels * 4, vertical=1)

        self.conv5 = LightGoConv3D(base_channels * 4, base_channels * 8, stride=(1, 2, 2), vertical=0)
        self.conv6 = LightGoConv3D(base_channels * 8, base_channels * 8, vertical=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.prob = nn.Conv3d(base_channels, 1, kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1), bias=False)

    def forward(self, x, d, normal, intri):
        conv0 = self.conv0(x)
        
        d1 = F.interpolate(d, scale_factor=0.5)
        normal1 = F.interpolate(normal, scale_factor=0.5)
        intri1 = intri[:, 0:2, :] * 0.5

        conv1 = self.conv1(conv0, d, normal, intri)
        conv2 = self.conv2(conv1, d1, normal1, intri1)

        d2 = F.interpolate(d1, scale_factor=0.5)
        normal2 = F.interpolate(normal1, scale_factor=0.5)
        intri2 = intri1[:, 0:2, :] * 0.5

        conv3 = self.conv3(conv2, d1, normal1, intri1)
        conv4 = self.conv4(conv3, d2, normal2, intri2)

        d3 = F.interpolate(d2, scale_factor=0.5)
        normal3 = F.interpolate(normal2, scale_factor=0.5)
        intri3 = intri2[:, 0:2, :] * 0.5

        conv5 = self.conv5(conv4, d2, normal2, intri2)
        x = self.conv6(conv5, d3, normal3, intri3)  # b 32 d h/8 w/8
        x = conv4 + self.conv7(x)  # b 16 d h/4 w/4
        x = conv2 + self.conv9(x)  # b 8 d h/2 w/2
        x = conv0 + self.conv11(x)  # b 4 d h w
        x = self.prob(x)
        return x

class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, feat_channels, kernel_size=3, stride=1, group_num=2,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(DeformConv3d, self).__init__()
        
        # 卷积核大小
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_num = group_num
        self.stride = stride
        
        # 偏移量生成模块
        self.offset_conv = nn.Conv2d(
            feat_channels,
            2 * group_num,  # 只对 H, W 生成偏移量
            kernel_size=(kernel_size[1], kernel_size[2]), 
            stride=1,
            padding=(kernel_size[1] // 2, kernel_size[2] // 2),
            bias=True,
        )
        
        if group_num == 2:
            self.original_offset = [[0, -dilation], [0, dilation]]
        elif group_num == 4:
            self.original_offset = [[dilation, 0], [0, -dilation], [0, dilation], [dilation, 0]]

        # 卷积核
        self.conv = nn.Conv3d(in_channels * (1 + group_num), out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x, feat):
        B, C, D, H, W = x.shape

        # 1. 生成偏移量 [B, 2 * K, D, H, W]
        offsets = self.offset_conv(feat)
        
        # 2. 使用双线性插值从输入中采样
        x_offset = self.sample(x, offsets)
        S = C * self.group_num // 2
        x = torch.cat([x_offset[:, :S, ...], x, x_offset[:, S:, ...]], dim=1)
        del x_offset

        # 3. 按深度维度执行固定卷积
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def generate_grid(self, D, H, W, device):
        """
        生成 3D 网格 (x, y, z)，其中 z 按顺序生成，x 和 y 可以偏移。
        Args:
            D: 深度维度大小
            H: 高度维度大小
            W: 宽度维度大小
            device: 设备
        Returns:
            grid: 基础网格，形状为 [1, D, H, W, 3]
        """
        # 生成深度维度的顺序索引
        z = torch.linspace(0, D - 1, D, device=device)  # 深度索引 [0, 1, ..., D-1]
        z = z.view(D, 1, 1).expand(D, H, W)  # 扩展到 [D, H, W]

        # 生成高度和宽度的标准网格
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        x = x.float().unsqueeze(0).expand(D, H, W)  # 扩展到 [D, H, W]
        y = y.float().unsqueeze(0).expand(D, H, W)  # 扩展到 [D, H, W]

        # 堆叠成 [D, H, W, 3] 格式
        grid = torch.stack((x, y, z), dim=-1)  # [D, H, W, 3]
        grid = grid.unsqueeze(0)  # 添加批次维度，变为 [1, D, H, W, 3]
        return grid

    def sample(self, x, sampling_offsets):
        """
        仅对 H, W 维度加入偏移的采样方法。
        Args:
            x: 输入特征，形状为 [B, C, D, H, W]
            sampling_offsets: 偏移量，形状为 [B, 2*k, H, W]（只针对 H, W）
        Returns:
            sampled_features: 采样后的特征，形状为 [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        K = sampling_offsets.size(1) // 2
        sampling_offsets = sampling_offsets.view(B, 2, K, H, W).permute(0, 2, 3, 4, 1).unsqueeze(2).repeat(1, 1, D, 1, 1, 1)
        # [B, K, D, H, W, 2]

        # 基础网格
        base_grid = self.generate_grid(D, H, W, x.device)  # [1, D, H, W, 3]
        base_grid = base_grid.unsqueeze(1).repeat(B, K, 1, 1, 1, 1) # [B, D, K, H, W, 3]

        # 添加偏移到 H, W 维度
        sampling_grid = base_grid + torch.cat((sampling_offsets, torch.zeros_like(sampling_offsets[..., :1])), dim=-1)
        sampling_grid = sampling_grid.view(B, -1, H, W, 3).contiguous()

        # 使用 F.grid_sample 采样
        sampled_features = F.grid_sample(
            x, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        
        return sampled_features.view(B, -1, D, H, W)

class DeformCostRegNetv2(nn.Module):
    def __init__(self, in_channels, base_channels, feat_channels):
        super(DeformCostRegNetv2, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv1 = DeformConv3d(base_channels, base_channels * 2, feat_channels, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv3 = DeformConv3d(base_channels * 2, base_channels * 4, feat_channels*2, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv5 = DeformConv3d(base_channels * 4, base_channels * 8, feat_channels*4, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.prob = nn.Conv3d(base_channels, 1, kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1), bias=False)

        self.downsample1 = Conv2d(feat_channels, feat_channels*2, kernel_size=3, stride=2, padding=1)
        self.downsample2 = Conv2d(feat_channels*2, feat_channels*4, kernel_size=3, stride=2, padding=1)

    def forward(self, x, feat):
        feat1 = self.downsample1(feat)
        feat2 = self.downsample2(feat1)
        
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0, feat))
        conv4 = self.conv4(self.conv3(conv2, feat1))
        x = self.conv6(self.conv5(conv4, feat2))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

# class LightGoConv3D(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=(3, 3, 3),
#                  stride=(1, 1, 1),
#                  padding=(1, 1, 1),
#                  bn=True):
#         super(LightGoConv3D, self).__init__()

#         assert kernel_size[1] == kernel_size[2]
#         assert stride[1] == stride[2]
#         assert padding[1] == padding[2]

#         self.kernel_size = kernel_size[2]
#         self.stride = stride[2]
#         self.conv3d = nn.Conv3d(in_channels * self.kernel_size * self.kernel_size,
#                                 out_channels,
#                                 kernel_size=(kernel_size[0], 1, 1),
#                                 padding=(padding[0], 0, 0),
#                                 stride=(stride[0], 1, 1),
#                                 bias=False)
#         if bn:
#             self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
#         else:
#             self.bn = None

#     def get_gcacost(self, x_ori, dp):
#         # intric b 3 3
#         # dp: depth B D H W
#         kernel_size = self.kernel_size
#         stride = self.stride
#         b, c, d, h, w = x_ori.shape
        
#         h = h // stride
#         w = w // stride

#         pad = (kernel_size - 1) // 2

#         dp_unfold = F.pad(dp, pad=[pad, pad, pad, pad], mode='replicate')
#         dp_unfold = F.unfold(dp_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
#         dp_unfold = dp_unfold.view(b, d, kernel_size ** 2, h, w).squeeze(1)  # b d k2 h w

#         h_ori = h * stride
#         w_ori = w * stride
#         y, x = torch.meshgrid([torch.arange(0, h_ori, dtype=torch.float32, device=dp.device),
#                                torch.arange(0, w_ori, dtype=torch.float32, device=dp.device)])
#         y, x = y.contiguous(), x.contiguous()
#         y, x = y.view(h_ori * w_ori), x.view(h_ori * w_ori)
#         xy = torch.stack((x, y))  # [2, H*W]
#         xy = xy.unsqueeze(0).repeat(b, 1, 1)  # b 2 h*w

#         dp_prog = dp_unfold[:, :, (kernel_size * kernel_size) // 2, :, :].unsqueeze(2).repeat(1, 1, kernel_size * kernel_size, 1, 1)  # b d k2 h w
#         interval_unfold = dp_unfold[:, 1] - dp_unfold[:, 0]  # b k2 h w

#         indices = (dp_prog - dp_unfold[:, 0].unsqueeze(1)) / interval_unfold.unsqueeze(1)  # b d k2 h w
#         indices = indices.reshape(b, d * kernel_size * kernel_size, h, w).unsqueeze(-1)

#         xy_n = xy.reshape(b, 2, h_ori, w_ori) 

#         xy_p = F.pad(xy_n, pad=[pad, pad, pad, pad], mode='replicate')
#         xy_p = F.unfold(xy_p, [kernel_size, kernel_size], padding=0, stride=stride)  # (B, ps*ps, H*W)
#         xy_p = xy_p.reshape(b, 2, kernel_size ** 2, h, w).unsqueeze(1).repeat(1, d, 1, 1, 1, 1)  # b d 2 k2 h w
#         xy_p = xy_p.permute(0, 1, 3, 4, 5, 2).reshape(b, d * kernel_size * kernel_size, h, w, 2)

#         indices = (torch.cat([xy_p, indices], dim=-1))
#         indices[:, :, :, :, 2] = indices[:, :, :, :, 2] / ((d - 1) / 2) - 1
#         indices[:, :, :, :, 0] = indices[:, :, :, :, 0] / ((w_ori - 1) / 2) - 1
#         indices[:, :, :, :, 1] = indices[:, :, :, :, 1] / ((h_ori - 1) / 2) - 1  # b d*k2 h w 3
#         ans = F.grid_sample(x_ori, grid=indices, padding_mode='zeros', mode='bilinear', align_corners=True)
#         ans = ans.reshape(b, c, d, kernel_size * kernel_size, h, w).permute(0, 1, 3, 2, 4, 5)
#         return ans.reshape(b, c * kernel_size * kernel_size, d, h, w)

#     def forward(self, x, dp):
#         x = self.get_gcacost(x, dp)  # b c*k*k, d h w
#         x = self.conv3d(x)  # b c,d h w
#         if self.bn is not None:
#             x = F.relu(self.bn(x), inplace=True)
#         return x

# class LightGoUpConv3D(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=(3, 3, 3),
#                  stride=(1, 1, 1),
#                  padding=(1, 1, 1),
#                  bn=True):
#         super(LightGoUpConv3D, self).__init__()

#         assert kernel_size[1] == kernel_size[2]
#         assert stride[1] == stride[2]
#         assert padding[1] == padding[2]

#         self.kernel_size = kernel_size[2]
#         self.stride = stride[2]
#         self.out_channels = out_channels
#         self.conv3d = nn.Conv3d(in_channels * self.kernel_size * self.kernel_size,
#                                 out_channels * 2 * 2,
#                                 kernel_size=(kernel_size[0], 1, 1),
#                                 padding=(padding[0], 0, 0),
#                                 stride=(stride[0], 1, 1),
#                                 bias=False)
#         if bn:
#             self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
#         else:
#             self.bn = None

#     def get_pecost(self, x_ori, dp):
#         # intric b 3 3
#         # dp: depth B D H W
#         kernel_size = self.kernel_size
#         stride = self.stride
#         b, c, d, h, w = x_ori.shape
        
#         h = h // stride
#         w = w // stride

#         pad = (kernel_size - 1) // 2

#         dp_unfold = F.pad(dp, pad=[pad, pad, pad, pad], mode='replicate')
#         dp_unfold = F.unfold(dp_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
#         dp_unfold = dp_unfold.view(b, d, kernel_size ** 2, h, w).squeeze(1)  # b d k2 h w

#         h_ori = h * stride
#         w_ori = w * stride
#         y, x = torch.meshgrid([torch.arange(0, h_ori, dtype=torch.float32, device=dp.device),
#                                torch.arange(0, w_ori, dtype=torch.float32, device=dp.device)])
#         y, x = y.contiguous(), x.contiguous()
#         y, x = y.view(h_ori * w_ori), x.view(h_ori * w_ori)
#         xy = torch.stack((x, y))  # [2, H*W]
#         xy = xy.unsqueeze(0).repeat(b, 1, 1)  # b 2 h*w

#         dp_prog = dp_unfold[:, :, (kernel_size * kernel_size) // 2, :, :].unsqueeze(2).repeat(1, 1, kernel_size * kernel_size, 1, 1)  # b d k2 h w
#         interval_unfold = dp_unfold[:, 1] - dp_unfold[:, 0]  # b k2 h w

#         indices = (dp_prog - dp_unfold[:, 0].unsqueeze(1)) / interval_unfold.unsqueeze(1)  # b d k2 h w
#         indices = indices.reshape(b, d * kernel_size * kernel_size, h, w).unsqueeze(-1)

#         xy_n = xy.reshape(b, 2, h_ori, w_ori) 

#         xy_p = F.pad(xy_n, pad=[pad, pad, pad, pad], mode='replicate')
#         xy_p = F.unfold(xy_p, [kernel_size, kernel_size], padding=0, stride=stride)  # (B, ps*ps, H*W)
#         xy_p = xy_p.reshape(b, 2, kernel_size ** 2, h, w).unsqueeze(1).repeat(1, d, 1, 1, 1, 1)  # b d 2 k2 h w
#         xy_p = xy_p.permute(0, 1, 3, 4, 5, 2).reshape(b, d * kernel_size * kernel_size, h, w, 2)

#         indices = (torch.cat([xy_p, indices], dim=-1))
#         indices[:, :, :, :, 2] = indices[:, :, :, :, 2] / ((d - 1) / 2) - 1
#         indices[:, :, :, :, 0] = indices[:, :, :, :, 0] / ((w_ori - 1) / 2) - 1
#         indices[:, :, :, :, 1] = indices[:, :, :, :, 1] / ((h_ori - 1) / 2) - 1  # b d*k2 h w 3
#         ans = F.grid_sample(x_ori, grid=indices, padding_mode='zeros', mode='bilinear', align_corners=True)
#         ans = ans.reshape(b, c, d, kernel_size * kernel_size, h, w).permute(0, 1, 3, 2, 4, 5)
#         return ans.reshape(b, c * kernel_size * kernel_size, d, h, w)

#     def forward(self, x, pd):
#         b, c, d, h, w = x.shape
#         x = self.get_pecost(x, pd)  # b c*k*k d h w
#         x = self.conv3d(x)  # b c,d h w
#         x = x.reshape(b, self.out_channels, 2, 2, d, h, w)
#         x = x.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, self.out_channels, d, h * 2,w * 2)
#         x = F.relu(self.bn(x), inplace=True)
#         return x

# class LightGCACostRegNet(nn.Module):
#     '''
#     input b d h w
#     output b d h w
#     '''

#     def __init__(self, in_channels, base_channels):
#         super(LightGCACostRegNet, self).__init__()

#         self.conv0 = LightGoConv3D(in_channels, base_channels)

#         self.conv1 = LightGoConv3D(base_channels, base_channels * 2, stride=(1, 2, 2))
#         self.conv2 = LightGoConv3D(base_channels * 2, base_channels * 2)

#         self.conv3 = LightGoConv3D(base_channels * 2, base_channels * 4, stride=(1, 2, 2))
#         self.conv4 = LightGoConv3D(base_channels * 4, base_channels * 4)

#         self.conv5 = LightGoConv3D(base_channels * 4, base_channels * 8, stride=(1, 2, 2))
#         self.conv6 = LightGoConv3D(base_channels * 8, base_channels * 8)

#         self.conv7 = LightGoUpConv3D(base_channels * 8, base_channels * 4)

#         self.conv9 = LightGoUpConv3D(base_channels * 4, base_channels * 2)

#         self.conv11 = LightGoUpConv3D(base_channels * 2, base_channels * 1)

#         self.prob = LightGoConv3D(base_channels, 1, bn=False)

#     def forward(self, x, d):
#         conv0 = self.conv0(x, d)
#         d1 = F.interpolate(d, scale_factor=0.5)

#         conv1 = self.conv1(conv0, d)
#         conv2 = self.conv2(conv1, d1)
#         d2 = F.interpolate(d1, scale_factor=0.5)

#         conv3 = self.conv3(conv2, d1)
#         conv4 = self.conv4(conv3, d2)
#         d3 = F.interpolate(d2, scale_factor=0.5)

#         conv5 = self.conv5(conv4, d2)
#         x = self.conv6(conv5, d3)  # b 32 d h/8 w/8
#         x = conv4 + self.conv7(x, d3)  # b 16 d h/4 w/4
#         x = conv2 + self.conv9(x, d2)  # b 8 d h/2 w/2
#         x = conv0 + self.conv11(x, d1)  # b 4 d h w
#         x = self.prob(x, d)
#         return x