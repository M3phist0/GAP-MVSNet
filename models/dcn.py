#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable
from torchvision.ops import DeformConv2d, deform_conv2d
import torch.nn.functional as F


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()



class DCN(DCNv2):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1, bias=True):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        if bias==False:
            self.bias = None
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, ret_off=False):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        res = deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask=mask
        )
        if ret_off:
            return res, offset, mask
        return res

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

class PlaneDCN(DCNv2):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1, bias=True):
        super(PlaneDCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels + kernel_size * kernel_size - 1,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=dilation,
                                          bias=True)
        self.dilation = dilation
        
        if bias==False:
            self.bias = None
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
        
    def compute_similarity(self, x, k=3, sim='cos'):
        dilation = self.dilation
        
        B, C, H, W = x.shape

        # 展平输入张量中每个点及其周围KxK范围内的点
        unfold_x = F.unfold(x, k, padding=(k // 2) * dilation, dilation=dilation) # B, CxKxK, HW
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

    def forward(self, x):
        
        H, W = x.shape[2], x.shape[3]
        sim = self.compute_similarity(x)
        out = self.conv_offset_mask(torch.cat([x, sim], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask=mask
        )
        return x