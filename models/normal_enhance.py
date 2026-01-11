from .module import *

'''
利用深度连续性设计loss，在法线连续的地方深度也应该连续

深度法线一致性为什么不加到loss里呢
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

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

    
class ResidualNet(nn.Module):
    def __init__(self, feat_channels):
        super(ResidualNet, self).__init__()
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