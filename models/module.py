import torch
import torch.nn as nn
import torch.nn.functional as F
from .compute_normal import depth2normal

def refine_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_refined =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    Using_inverse_d = False

    _lambda, _gamma = 1.0, 2.0
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0

        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth[mask], depth_gt[mask], reduction='mean')
        
        # offset = stage_inputs["offset"]
        # interval = stage_inputs["interval"]
        # if Using_inverse_d:
        #     ratio = stage_inputs["ratio"]
        #     offset_gt = torch.clamp(1./ depth_gt - 1./ depth_entropy, min=-interval, max=interval)
        #     offset = offset / interval * ratio
        #     offset_gt = offset_gt / interval * ratio
        #     _lambda *= 2.0
        # else:
        #     offset_gt = torch.clamp(depth_gt - depth_entropy, min=-interval, max=interval)

        # refined_loss = F.smooth_l1_loss(offset[mask], offset_gt[mask], reduction='mean')
        # refined_loss *= _lambda
        refined_loss = 0.0
        
        total_entropy += entro_loss
        total_refined += refined_loss
        
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * (entro_loss + refined_loss)
        else:
            total_loss += entro_loss + refined_loss
        
    return total_loss, depth_loss, total_entropy, depth_entropy, total_refined


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)
            
            
class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        #assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] or [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                        -1)  # [B, 3, Ndepth, H*W]
        
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        invalid = (proj_xyz[:, 2:3, :, :]<1e-6).squeeze(1) # [B, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :])  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_x_normalized[invalid] = -99.
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_y_normalized[invalid] = -99.
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def homo_warping_nr(src_fea, src_proj, ref_proj, depth_values, view_dir, normal):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] or [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                        -1)  # [B, 3, Ndepth, H*W]
        
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        invalid = (proj_xyz[:, 2:3, :, :]<1e-6).squeeze(1) # [B, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :])  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_x_normalized[invalid] = -99.
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_y_normalized[invalid] = -99.
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
        
        warped_normal = F.grid_sample(normal, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
        norm = torch.norm(warped_normal, dim=1, keepdim=True)  # 范数，形状为 (B, 1, H, W)
        norm = torch.clamp(norm, min=1e-6)
        warped_normal /= norm
        dot_product = torch.sum(warped_normal * view_dir.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)
        invisible_mask = (dot_product <= -0.9).view(batch, num_depth, height*width)
        grid[invisible_mask] = -99.
        
        true_ratios_per_batch = invisible_mask.view(batch, -1).sum(dim=1) / invisible_mask.view(batch, -1).size(1)

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def homo_warping_patch(src_fea, src_proj, ref_proj, depth_values, patch_grid):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] or [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        # y, x = y.contiguous(), x.contiguous()
        # y, x = y.view(height * width), x.view(height * width)
        # xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        # xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xy = torch.stack((x, y)).unsqueeze(0).repeat(batch, 1, 1, 1)
        xy = F.grid_sample(
            xy, patch_grid, mode="bilinear", padding_mode="border", align_corners=False
        ).view(batch, 2, -1)
        xyz = torch.cat((xy, torch.ones_like(xy[:, :1, :])), dim=1)  # [3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                        -1)  # [B, 3, Ndepth, H*W]
        
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        invalid = (proj_xyz[:, 2:3, :, :]<1e-6).squeeze(1) # [B, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :])  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_x_normalized[invalid] = -99.
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_y_normalized[invalid] = -99.
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    # warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
    #                                padding_mode='zeros', align_corners=True)
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, -1, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, -1, width)

    return warped_src_fea


def homo_warping_shift(src_fea, src_proj, ref_proj, depth_values, patch_grid):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Neighbors, Ndepth] o [B, Neighbors, Ndepth, H, W]
    # out: [B, C, Neighbors, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    patch_neighbors, num_depth = depth_values.shape[1], depth_values.shape[2]
    height, width = src_fea.shape[2], src_fea.shape[3]

    patch_grid = patch_grid.view(batch, patch_neighbors, height, width, 2)
    
    warped_src_fea_total = []
    for i in range(patch_neighbors):
        with torch.no_grad():
            _grid = patch_grid[:, i]
            _depth_values = depth_values[:, i]
            # print(i, _depth_values[:, 0, 0, 0], _depth_values.shape)
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]  # [B,3,3]
            trans = proj[:, :3, 3:4]  # [B,3,1]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                                torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
            xyz = F.grid_sample(
                xyz.view(batch, 3, height, width), _grid, mode="bilinear", padding_mode="border", align_corners=False
            ).view(batch, 3, -1)
            rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * _depth_values.view(batch, 1, num_depth,
                                                                                                -1)  # [B, 3, Ndepth, H*W]
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
            invalid = (proj_xyz[:, 2:3, :, :]<1e-6).squeeze(1) # [B, Ndepth, H*W]
            proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :])  # [B, 2, Ndepth, H*W]
            # print(proj_xy[:, :, 0, width])
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_x_normalized[invalid] = -99.
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_y_normalized[invalid] = -99.
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
            grid = proj_xy

        warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                    padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.view(batch, channels, -1, num_depth, height, width)
        
        warped_src_fea_total.append(warped_src_fea)
    
    warped_src_fea_total = torch.cat(warped_src_fea_total, dim=2)

    return warped_src_fea_total


class FeatureNet(nn.Module):
    def __init__(self, base_channels):
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            # 输入通道数，输出通道数，卷积核大小，步幅，填充大小
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1))

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1))

        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 4, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
        self.out_channels = [4 * base_channels, base_channels * 2, base_channels]

    def forward(self, x):
        """forward.

        :param x: [B, C, H, W]
        :return outputs: stage1 [B, 32, 128, 160], stage2 [B, 16, 256, 320], stage3 [B, 8, 512, 640]
        """
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

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


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth

def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss

def depth_wta(p, depth_values):
    '''Winner take all.'''
    wta_index_map = torch.argmax(p, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_values, 1, wta_index_map).squeeze(1)
    return wta_depth_map

''' loss function same as TransMVSNet'''

def info_entropy_loss(prob_volume, prob_volume_pre, mask):
    # prob volume should be processed after SoftMax
    B,D,H,W = prob_volume.shape
    LSM = nn.LogSoftmax(dim=1)
    valid_points = torch.sum(mask, dim=[1,2])+1e-6
    entropy = -1*(torch.sum(torch.mul(prob_volume, LSM(prob_volume_pre)), dim=1)).squeeze(1)
    entropy_masked = torch.sum(torch.mul(mask, entropy), dim=[1,2])
    return torch.mean(entropy_masked / valid_points)


def entropy_loss(prob_volume, depth_gt, mask, depth_value, weights=None, return_prob_map=False):
    # from AA
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-12), dim=1).squeeze(1) # B, 1, H, W

    # Apply weights (1 + weights) to the cross entropy image
    if weights is not None:
        cross_entropy_image = cross_entropy_image * (1 + weights)

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map


def smooth_loss(depth, normal, mask=None, lambda_wt=1, **kwargs):
    """Computes image-aware depth smoothness loss."""
    # print('depth: {} img: {}'.format(depth.shape, img.shape))
    normal_dx = gradient_x(normal)
    normal_dy = gradient_y(normal)
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(normal_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(normal_dy), 3, keepdim=True)))
    # print('depth_dx: {} weights_x: {}'.format(depth_dx.shape, weights_x.shape))
    # print('depth_dy: {} weights_y: {}'.format(depth_dy.shape, weights_y.shape))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    
    if mask is not None:
        # 确保 mask 形状与 depth 对齐
        mask = mask.unsqueeze(1).float()  # 从 [B, H, W] -> [B, 1, H, W]
        smoothness_x = smoothness_x * mask[:, :, :, :-1]
        smoothness_y = smoothness_y * mask[:, :, :-1, :]

        # 平均损失时只考虑有效的 mask 区域
        return (torch.sum(torch.abs(smoothness_x)) + torch.sum(torch.abs(smoothness_y))) / (torch.sum(mask) + 1e-8)
    else:
        # 无 mask 时，计算全图的损失
        return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

def smooth_loss_2nd_order(depth, img, mask=None, lambda_wt=1.0, clip=2.0):
    """Computes image-aware depth smoothness loss using relaxed 2nd-order formulation."""
    depth_dx, depth_dy = gradient(depth, append_zeros=True)
    depth_dxdx, depth_dxdy = gradient(depth_dx, append_zeros=True)
    depth_dydx, depth_dydy = gradient(depth_dy, append_zeros=True)

    # enormous 2nd order gradients correspond to depth discontinuities
    depth_dxdx = torch.clip(depth_dxdx, -clip, clip)
    depth_dxdy = torch.clip(depth_dxdy, -clip, clip)
    depth_dydx = torch.clip(depth_dydx, -clip, clip)
    depth_dydy = torch.clip(depth_dydy, -clip, clip)

    image_dx, image_dy = gradient(img, append_zeros=True)

    # get weight for gradient penalty
    weights_x = torch.exp(-(lambda_wt * torch.sum(torch.abs(image_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.sum(torch.abs(image_dy), 3, keepdim=True)))

    smoothness_x = weights_x * (torch.abs(depth_dxdx) + torch.abs(depth_dydx)) / 2.
    smoothness_y = weights_y * (torch.abs(depth_dydy) + torch.abs(depth_dxdy)) / 2.
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)

def gradient_x(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(1)
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(1)
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient(pred, append_zeros=False):
    if len(pred.shape) == 3:
       pred = pred.unsqueeze(1) 
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    if append_zeros:
        zeros_row = torch.zeros_like(D_dy[:, :, :1])
        D_dy = torch.cat((D_dy, zeros_row), dim=2)
        zeros_col = torch.zeros_like(D_dx[:, :, :, :1])
        D_dx = torch.cat((D_dx, zeros_col), dim=3)
    return D_dx, D_dy

def mvsnet_loss(inputs, depth_gt_ms, mask_ms, refine_only, main_only, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_refined =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    
    _lambda = [-1, 0.1, 0.1]
    _lambda2 = 1

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0

        stage_idx = int(stage_key.replace("stage", "")) - 1
        
        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss
        if main_only:
            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * entro_loss
            else:
                total_loss += entro_loss
        elif refine_only:
            if stage_idx > 0:
                refined_depth = stage_inputs["refined_depth"]
                refined_loss = F.smooth_l1_loss(refined_depth[mask], depth_gt[mask], reduction='mean')
                total_refined += _lambda[stage_idx] * refined_loss
                if depth_loss_weights is not None:
                    total_loss += depth_loss_weights[stage_idx] * _lambda[stage_idx] * refined_loss
                else:
                    total_loss += refined_loss
        else:
            if stage_idx > 0:
                refined_depth = stage_inputs["refined_depth"]
                refined_loss = F.smooth_l1_loss(refined_depth[mask], depth_gt[mask], reduction='mean')
                total_refined += _lambda[stage_idx] * refined_loss * _lambda2
            else:
                refined_loss = None
            
            if depth_loss_weights is not None:
                if refined_loss is None:
                    total_loss += depth_loss_weights[stage_idx] * entro_loss
                else:
                    total_loss += depth_loss_weights[stage_idx] * (entro_loss + _lambda[stage_idx] * refined_loss * _lambda2)
            else:
                if refined_loss is None:
                    total_loss += entro_loss
                else:
                    total_loss += entro_loss + _lambda[stage_idx] * refined_loss * _lambda2
        
    return total_loss, depth_loss, total_entropy, depth_entropy, total_refined

def kl_smoothness_loss_3x3(prob_volume, mask=None):
    """
    Computes KL divergence smoothness loss within a 3x3 neighborhood for depth probability volume.
    
    Args:
        prob_volume (torch.Tensor): Depth probability volume of shape (B, D, H, W).
        mask (torch.Tensor): Binary mask of shape (B, H, W), indicating valid regions.
    
    Returns:
        torch.Tensor: KL smoothness loss (scalar).
    """
    # Ensure mask is broadcastable to prob_volume
    if mask is None:
        valid_pixel_count = prob_volume.shape[2] * prob_volume.shape[3]
    else:
        valid_pixel_count = torch.sum(mask, dim=[1,2]) + 1e-6
        mask = mask.unsqueeze(1).float()  # Shape: (B, 1, H, W)
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)  # (B, 1, H+2, W+2)

    # Create a padded volume for 3x3 neighborhood access
    padded_prob_volume = F.pad(prob_volume, (1, 1, 1, 1), mode="replicate")  # (B, D, H+2, W+2)

    # Define relative shifts for a 3x3 neighborhood
    shifts = [
        (0, 0),   # Center
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1),   # Right
        (-1, -1), # Up-Left
        (-1, 1),  # Up-Right
        (1, -1),  # Down-Left
        (1, 1)    # Down-Right
    ]

    # Initialize total KL loss and valid pixel count
    total_kl_loss = 0
    
    _, _, H, W = prob_volume.shape

    # Iterate over all shifts
    for dy, dx in shifts:
        # Shift probability volume and mask
        h_start, h_end = 1 + dy, H + 1 + dy
        w_start, w_end = 1 + dx, W + 1 + dx

        shifted_prob_volume = padded_prob_volume[:, :, h_start:h_end, w_start:w_end]  # (B, D, H, W)

        # Compute KL divergence for this shift
        kl_div = F.kl_div(prob_volume.log(), shifted_prob_volume, reduction="none")  # (B, D, H, W)
        kl_div = kl_div.sum(dim=1)  # Sum over depth dimension: (B, H, W)

        if mask is None:
            total_kl_loss += torch.sum(kl_div, dim=[1,2])
        else:
            # Apply mask to KL divergence
            shifted_mask = padded_mask[:, :, h_start:h_end, w_start:w_end]  # (B, 1, H, W)
            masked_kl_div = kl_div * mask.squeeze(1) * shifted_mask.squeeze(1)

            # Accumulate KL loss and valid pixel count
            total_kl_loss += torch.sum(masked_kl_div, dim=[1,2])

    # Normalize by valid pixel count
    kl_loss = torch.mean(total_kl_loss / valid_pixel_count)

    return kl_loss

def smooth_refine_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_refined =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_smooth =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    Using_inverse_d = False
    
    Using_normal = False

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth = stage_inputs["depth"]
        normal = stage_inputs["normal"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0
        
        B, H, W = mask.shape
        # Create a central region mask
        center_mask = torch.zeros_like(mask)  # Shape: (B, H, W)
        h_start, h_end = H // 10, 9 * H // 10
        w_start, w_end = W // 10, 9 * W // 10
        center_mask[:, h_start:h_end, w_start:w_end] = True
        
        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth[mask], depth_gt[mask], reduction='mean')
        
        offset = stage_inputs["offset"]
        interval = stage_inputs["interval"]
        if Using_inverse_d:
            offset_gt = torch.clamp(1./ depth_gt - 1./ depth_entropy, min=-interval, max=interval)
            offset = offset / interval
            offset_gt = offset_gt / interval
            _lambda1, _lambda2 = 1.0, 1.0
        else:
            offset_gt = torch.clamp(depth_gt - depth_entropy, min=-interval, max=interval)
            _lambda1, _lambda2 = 1.0, 1.0
        
        if Using_normal:
            refined_loss = F.smooth_l1_loss(offset[mask], offset_gt[mask], reduction='mean')
            smth_loss = kl_smoothness_loss_3x3(prob_volume, mask | center_mask)
        else:
            refined_loss = F.smooth_l1_loss(offset[mask], offset_gt[mask], reduction='mean')
            smth_loss = kl_smoothness_loss_3x3(prob_volume, mask | center_mask)
        
        refined_loss *= _lambda1
        smth_loss *= _lambda2

        total_entropy += entro_loss
        total_refined += refined_loss
        total_smooth += smth_loss
        
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * (entro_loss + refined_loss + smth_loss)
        else:
            total_loss += entro_loss + refined_loss + smth_loss
        
    return total_loss, depth_loss, total_entropy, depth_entropy, total_refined, total_smooth

# def refine_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
#     depth_loss_weights = kwargs.get("dlossw", None)
#     total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     total_refined =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

#     for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
#         stage_idx = int(stage_key.replace("stage", "")) - 1
        
#         prob_volume = stage_inputs["prob_volume"]
#         depth_values = stage_inputs["depth_values"]
#         depth = stage_inputs["depth"]
#         # offset = stage_inputs["offset"]
#         # interval = stage_inputs["interval"]
        
#         depth_gt = depth_gt_ms[stage_key]
#         mask = mask_ms[stage_key]
        
#         mask = mask > 0.5
#         entropy_weight = 2.0
        
        
#         entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
#         entro_loss = entro_loss * entropy_weight
#         depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
#         # offset_gt = torch.clamp(depth_gt - depth_entory, min=-interval, max=interval)
#         # refined_loss = F.smooth_l1_loss(offset[mask], offset_gt[mask], reduction='mean') / interval
#         refined_loss = 0
        
#         total_entropy += entro_loss
#         total_refined += refined_loss
        
#         if depth_loss_weights is not None:
#             total_loss += depth_loss_weights[stage_idx] * entro_loss
#         else:
#             total_loss += entro_loss
        
#     return total_loss, depth_loss, total_entropy, depth_entropy, total_refined

def refine_mvsnet_loss_v2(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_refined =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_rf = stage_inputs["depth_rf"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0
        
        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_rf[mask], depth_gt[mask], reduction='mean')
        
        total_entropy += entro_loss
        total_refined += depth_loss
                
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * (entro_loss + depth_loss)
        else:
            total_loss += entro_loss + depth_loss
        
    return total_loss, depth_loss, total_entropy, depth_entropy, total_refined

def trans_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0

        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss
        
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * entro_loss
        else:
            total_loss += entro_loss
        
    return total_loss, depth_loss, total_entropy, depth_entropy


def focal_loss_bld(inputs, depth_gt_ms, mask_ms, depth_interval, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0
        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * entro_loss
        else:
            total_loss += entro_loss

    abs_err = (depth_gt_ms['stage3'] - inputs["stage3"]["depth"]).abs()
    abs_err_scaled = abs_err /(depth_interval *192./128.)
    mask = mask_ms["stage3"]
    mask = mask > 0.5
    epe = abs_err_scaled[mask].mean()
    less1 = (abs_err_scaled[mask] < 1.).to(depth_gt_ms['stage3'].dtype).mean()
    less3 = (abs_err_scaled[mask] < 3.).to(depth_gt_ms['stage3'].dtype).mean()

    return total_loss, depth_loss, epe, less1, less3


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0, use_inverse_depth=False):
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        if use_inverse_depth is False:
            new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )  Shouldn't cal this if we use inverse depth
            depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                        requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)
            depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)
        else:
            # When use inverse_depth for T&T
            depth_range_samples = cur_depth.view(shape[0], -1, 1, 1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)
            
            # inverse_depth_min = 1. / cur_depth_min  # (B,)
            # inverse_depth_max = 1. / cur_depth_max
            # new_interval = (inverse_depth_min - inverse_depth_max)  / (ndepth - 1)  # 1 D H W
            # inverse_depth_range_samples = inverse_depth_max.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
            #                                                             requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)
            # depth_range_samples = 1. / inverse_depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)
    else:
        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)
        
    return depth_range_samples

def init_range(cur_depth, ndepths, device, dtype, H, W):
    if len(cur_depth.shape) == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
        new_interval = new_interval[:, None, None]  # B H W
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                         requires_grad=False).reshape(1, -1) * new_interval.squeeze(1))  # (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # (B, D, H, W)
    else:
        cur_depth_min = cur_depth[..., 0]  # (B,H,W)
        cur_depth_max = cur_depth[..., -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B,H,W)
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                         requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))  # (B, D, H, W)
    return depth_range_samples, depth_range_samples[:, 2, :, :] - depth_range_samples[:, 1, :, :]


def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    if len(cur_depth.shape) == 2:
        inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
        inverse_depth_max = 1. / cur_depth[:, -1]
        itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
        inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (inverse_depth_min - inverse_depth_max)[:, None, None, None] * itv
    else:
        inverse_depth_min = 1. / cur_depth[..., 0]  # (B,H,W)
        inverse_depth_max = 1. / cur_depth[..., -1]
        itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
        inverse_depth_hypo = inverse_depth_max[:, None, :, :] + (inverse_depth_min - inverse_depth_max)[:, None, :, :] * itv

    return 1. / inverse_depth_hypo, inverse_depth_hypo[:, 2, :, :] - inverse_depth_hypo[:, 1, :, :]


def schedule_inverse_range(depth, depth_hypo, ndepths, split_itv, H, W, shift=False):
    last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
    inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
    inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W

    if shift:  # shift is used to prevent negative depth prediction. 0.002 is set when the max depth range is 500
        is_neg = (inverse_max_depth < 0.002).float()
        inverse_max_depth = inverse_max_depth - (inverse_max_depth - 0.002) * is_neg
        inverse_min_depth = inverse_min_depth - (inverse_max_depth - 0.002) * is_neg

    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype,
                       requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (inverse_min_depth - inverse_max_depth)[:, None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1. / inverse_depth_hypo, inverse_depth_hypo[:, 2, :, :] - inverse_depth_hypo[:, 1, :, :]


def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    # shape, (B, H, W)
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    # if len(depth_inteval_pixel.shape) != 3:
    #     depth_inteval_pixel = depth_inteval_pixel[:, None, None]
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_min = torch.clamp_min(cur_depth_min, 0.001)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                                                                     requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return depth_range_samples, depth_range_samples[:, 2, :, :] - depth_range_samples[:, 1, :, :]

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x, *kwargs):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class CostRegNet_RCNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet_RCNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Rconv_3D(base_channels * 2)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Rconv_3D(base_channels * 4)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Rconv_3D(base_channels * 8)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x, *kwargs):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x
    
class CostRegNetv2(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNetv2, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv1 = Conv3d(base_channels, base_channels * 2, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.prob = nn.Conv3d(base_channels, 1, kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1), bias=False)

    def forward(self, x, *kwargs):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class GeoCostRegNetv2(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(GeoCostRegNetv2, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv1 = Conv3d(base_channels + 4, base_channels * 2, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv2 = Conv3d(base_channels * 2 + 4, base_channels * 2, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv3 = Conv3d(base_channels * 2 + 4, base_channels * 4, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv4 = Conv3d(base_channels * 4 + 4, base_channels * 4, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv5 = Conv3d(base_channels * 4 + 4, base_channels * 8, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1))
        self.conv6 = Conv3d(base_channels * 8 + 4, base_channels * 8, kernel_size=(5,3,3), padding=(2,1,1))

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2, 1, 1), output_padding=(0,1,1))

        self.prob = nn.Conv3d(base_channels, 1, kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1), bias=False)

    def forward(self, x, d, normal, *kwargs):
        min_d = d.view(d.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        max_d = d.view(d.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        d = (d - min_d) / (max_d - min_d + 1e-8)
        ndepth = d.size(1)
        
        conv0 = self.conv0(x)
        
        d1 = F.interpolate(d, scale_factor=0.5)
        normal1 = F.interpolate(normal, scale_factor=0.5)
        
        conv1 = self.conv1(torch.cat([conv0, d.unsqueeze(1), normal.unsqueeze(2).expand(-1, -1, ndepth, -1, -1)], dim=1))
        conv2 = self.conv2(torch.cat([conv1, d1.unsqueeze(1), normal1.unsqueeze(2).expand(-1, -1, ndepth, -1, -1)], dim=1))
        
        d2 = F.interpolate(d1, scale_factor=0.5)
        normal2 = F.interpolate(normal1, scale_factor=0.5)
        conv3 = self.conv3(torch.cat([conv2, d1.unsqueeze(1), normal1.unsqueeze(2).expand(-1, -1, ndepth, -1, -1)], dim=1))
        conv4 = self.conv4(torch.cat([conv3, d2.unsqueeze(1), normal2.unsqueeze(2).expand(-1, -1, ndepth, -1, -1)], dim=1))
        
        d3 = F.interpolate(d2, scale_factor=0.5)
        normal3 = F.interpolate(normal2, scale_factor=0.5)
        conv5 = self.conv5(torch.cat([conv4, d2.unsqueeze(1), normal2.unsqueeze(2).expand(-1, -1, ndepth, -1, -1)], dim=1))
        x = self.conv6(torch.cat([conv5, d3.unsqueeze(1), normal3.unsqueeze(2).expand(-1, -1, ndepth, -1, -1)], dim=1))
        
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x
    
class CostRegNetv3(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNetv3, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, kernel_size=(5,1,1), padding=(2,0,0))

        self.conv1 = Conv3d(base_channels, base_channels * 2, kernel_size=(5,1,1), stride=(1,2,2), padding=(2,0,0))
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, kernel_size=(5,1,1), padding=(2,0,0))

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, kernel_size=(5,1,1), stride=(1,2,2), padding=(2,0,0))
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, kernel_size=(5,1,1), padding=(2,0,0))

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, kernel_size=(5,1,1), stride=(1,2,2), padding=(2,0,0))
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, kernel_size=(5,1,1), padding=(2,0,0))

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, kernel_size=(5, 1, 1), stride=(1,2,2), padding=(2, 0, 0), output_padding=(0,1,1))

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, kernel_size=(5, 1, 1), stride=(1,2,2), padding=(2, 0, 0), output_padding=(0,1,1))

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, kernel_size=(5, 1, 1), stride=(1,2,2), padding=(2, 0, 0), output_padding=(0,1,1))

        self.prob = nn.Conv3d(base_channels, 1, kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0), bias=False)

    def forward(self, x, *kwargs):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x
    
class RFCostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, feat_channels):
        super(RFCostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=(1, 2, 2), padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=(1, 2, 2), padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=(1, 2, 2), padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=(1, 2, 2), padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=(1, 2, 2), padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=(1, 2, 2), padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)
        
        self.attn = AttentionModule(dim=base_channels, img_feat_dim=feat_channels)

    def forward(self, x, img_feat, *kwargs):
        conv0 = self.conv0(x)
        x = self.attn(x, img_feat)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.attn(x, img_feat)
        x = self.prob(x)
        return x

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
        return attn.unsqueeze(2) * cost
    
# class CostRegNet(nn.Module):
#     def __init__(self, in_channels, base_channels, last_layer=True):
#         super(CostRegNet, self).__init__()
#         self.last_layer = last_layer

#         self.conv1 = Conv3d(in_channels, base_channels * 2, stride=2, padding=1)
#         self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

#         self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
#         self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

#         self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
#         self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

#         self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)
#         self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)
#         self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

#         if in_channels != base_channels:
#             self.inner = nn.Conv3d(in_channels, base_channels, 1, 1)
#         else:
#             self.inner = nn.Identity()

#         if self.last_layer:
#             self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

#     def forward(self, x, *kwargs):
#         import torch.utils.checkpoint as cp
#         x = cp.checkpoint(self.forward_once, x)
#         return x

#     def forward_once(self, x, *kwargs):
#         conv0 = x
#         conv2 = self.conv2(self.conv1(conv0))
#         conv4 = self.conv4(self.conv3(conv2))
#         x = self.conv6(self.conv5(conv4))
#         x = conv4 + self.conv7(x)
#         x = conv2 + self.conv9(x)
#         x = self.inner(conv0) + self.conv11(x)
#         if self.last_layer:
#             x = self.prob(x)
#         return x


class CostRegNet2D(nn.Module):
    def __init__(self, in_channels, base_channels=8):
        super(CostRegNet2D, self).__init__()
        self.conv1 = Conv3d(in_channels, base_channels * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(base_channels, 1, 1, stride=1, padding=0)

    def forward(self, x, *kwargs):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x


class CostRegNet3D(nn.Module):
    def __init__(self, in_channels, base_channels=8, log_var=False):
        super(CostRegNet3D, self).__init__()
        self.log_var = log_var
        self.conv1 = Conv3d(in_channels, base_channels * 2, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))

        if in_channels != base_channels:
            self.inner = nn.Conv3d(in_channels, base_channels, 1, 1)
        else:
            self.inner = nn.Identity()

        self.prob = nn.Conv3d(base_channels, 2 if self.log_var else 1, 1, stride=1, padding=0)

    def forward(self, x, *kwargs):
        import torch.utils.checkpoint as cp

        x = cp.checkpoint(self.forward_once, x)
        return x

    def forward_once(self, x, *kwargs):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = self.inner(conv0) + self.conv11(x)
        x = self.prob(x)

        return x
    
class AdaGoConv3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 bn=True):
        super(AdaGoConv3D, self).__init__()

        assert kernel_size[1] == kernel_size[2]
        assert stride[1] == stride[2]
        assert padding[1] == padding[2]

        self.kernel_size = kernel_size[2]
        self.stride = stride[2]
        self.conv3d = nn.Conv3d(in_channels,
                                out_channels,
                                kernel_size=(kernel_size[0], 1, 1),
                                padding=(padding[0], 0, 0),
                                stride=(stride[0], 1, 1),
                                bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
        else:
            self.bn = None

    def get_gcacost(self, x_ori, dp, normal, intri):

        kernel_size = self.kernel_size
        stride = self.stride
        b, c, d, h, w = x_ori.shape
       
        h = h // stride
        w = w // stride

        pad = (kernel_size - 1) // 2

        dp_unfold = F.pad(dp, pad=[pad, pad, pad, pad], mode='replicate')
        dp_unfold = F.unfold(dp_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
        dp_unfold = dp_unfold.view(b, d, kernel_size ** 2, h, w).squeeze(1)  # b d k2 h w

        normal_p_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_p_unfold = F.unfold(normal_p_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
        normal_p_unfold = normal_p_unfold.view(b, 3, kernel_size ** 2, h, w).squeeze(1)

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
        pos = torch.stack([posx, posy], dim=1).reshape(b, 2, h_ori, w_ori)
        pos_p = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos_p, [kernel_size, kernel_size], padding=0, stride=stride)
        pos_unfold = pos_unfold.view(b, 2, kernel_size * kernel_size, h, w)
        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]

        # pos - center
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k2 h w

        dp_prog = ddw_weights.unsqueeze(1) * dp_unfold[:, :, (kernel_size * kernel_size) // 2, :, :].unsqueeze(2)  # b d k2 h w
        interval_unfold = dp_unfold[:, 1] - dp_unfold[:, 0] # b k2 h w

        # b d k2 h w  - b 1 k2 h w  / B  1 K*K H W = b d k*k h w
        indices = (dp_prog - dp_unfold[:, 0].unsqueeze(1)) / interval_unfold.unsqueeze(1)  # b d k2 h w
        indices = indices.reshape(b, d * kernel_size * kernel_size, h, w).unsqueeze(-1)

        xy_n = xy.reshape(b, 2, h_ori, w_ori)# b 2 h w .permute(0, 2, 1).reshape(0, h_ori, w_ori, 2).unsqueeze(1)  # b 1 h w 2

        xy_p = F.pad(xy_n, pad=[pad, pad, pad, pad], mode='replicate')
        xy_p = F.unfold(xy_p, [kernel_size, kernel_size], padding=0, stride=stride)  # (B, ps*ps, H*W)
        xy_p = xy_p.reshape(b, 2, kernel_size**2, h, w).unsqueeze(1).repeat(1, d, 1, 1, 1, 1)  # b d 2 k2 h w
        xy_p = xy_p.permute(0, 1, 3, 4, 5, 2).reshape(b, d * kernel_size * kernel_size, h, w, 2)

        indices = (torch.cat([xy_p, indices], dim=-1))
        indices[:, :, :, :, 2] = indices[:, :, :, :, 2] / ((d - 1) / 2) - 1
        indices[:, :, :, :, 0] = indices[:, :, :, :, 0] / ((w_ori - 1) / 2) - 1
        indices[:, :, :, :, 1] = indices[:, :, :, :, 1] / ((h_ori - 1) / 2) - 1
        ans = F.grid_sample(x_ori, grid=indices, padding_mode='zeros', mode='bilinear', align_corners=True)
        ans = ans.reshape(b, c, d, kernel_size*kernel_size, h, w).permute(0,1,3,2,4,5)
        
        center_normal = normal_p_unfold[:, :, kernel_size ** 2 // 2, :, :].unsqueeze(2).expand(-1, -1, kernel_size ** 2, -1, -1)
        similarity = F.cosine_similarity(normal_p_unfold, center_normal, dim=1)  # 计算余弦相似度
        weights = F.softmax(similarity, dim=1)  # 对 kernel_size**2 维度归一化，生成权重
        weights = weights.unsqueeze(1).unsqueeze(3)
        ans = torch.sum(ans * weights, dim=2)
        
        return ans

    def forward(self, x, dp, normal, intri):
        x = self.get_gcacost(x, dp, normal, intri)  # b c*k*k, d h w
        x = self.conv3d(x)  # b c,d h w
        if self.bn is not None:
            x = F.relu(self.bn(x), inplace=True)
        return x


class GoConv3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 bn=True):
        super(GoConv3D, self).__init__()

        assert kernel_size[1] == kernel_size[2]
        assert stride[1] == stride[2]
        assert padding[1] == padding[2]

        self.kernel_size = kernel_size[2]
        self.stride = stride[2]
        self.conv3d = nn.Conv3d(in_channels * self.kernel_size * self.kernel_size,
                                out_channels,
                                kernel_size=(kernel_size[0], 1, 1),
                                padding=(padding[0], 0, 0),
                                stride=(stride[0], 1, 1),
                                bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
        else:
            self.bn = None

    def get_gcacost(self, x_ori, dp, normal, intri):

        kernel_size = self.kernel_size
        stride = self.stride
        b, c, d, h, w = x_ori.shape
       
        h = h // stride
        w = w // stride

        pad = (kernel_size - 1) // 2

        dp_unfold = F.pad(dp, pad=[pad, pad, pad, pad], mode='replicate')
        dp_unfold = F.unfold(dp_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
        dp_unfold = dp_unfold.view(b, d, kernel_size ** 2, h, w).squeeze(1)  # b d k2 h w

        normal_p_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_p_unfold = F.unfold(normal_p_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
        normal_p_unfold = normal_p_unfold.view(b, 3, kernel_size ** 2, h, w).squeeze(1)

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
        pos = torch.stack([posx, posy], dim=1).reshape(b, 2, h_ori, w_ori)
        pos_p = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos_p, [kernel_size, kernel_size], padding=0, stride=stride)
        pos_unfold = pos_unfold.view(b, 2, kernel_size * kernel_size, h, w)
        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]

        # pos - center
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k2 h w

        dp_prog = ddw_weights.unsqueeze(1) * dp_unfold[:, :, (kernel_size * kernel_size) // 2, :, :].unsqueeze(2)  # b d k2 h w
        interval_unfold = dp_unfold[:, 1] - dp_unfold[:, 0] # b k2 h w

        # b d k2 h w  - b 1 k2 h w  / B  1 K*K H W = b d k*k h w
        indices = (dp_prog - dp_unfold[:, 0].unsqueeze(1)) / interval_unfold.unsqueeze(1)  # b d k2 h w
        #indices = indices / ((d - 1) / 2) - 1 # b d k2 h w
        indices = indices.reshape(b, d * kernel_size * kernel_size, h, w).unsqueeze(-1)

        xy_n = xy.reshape(b, 2, h_ori, w_ori)# b 2 h w .permute(0, 2, 1).reshape(0, h_ori, w_ori, 2).unsqueeze(1)  # b 1 h w 2

        xy_p = F.pad(xy_n, pad=[pad, pad, pad, pad], mode='replicate')
        xy_p = F.unfold(xy_p, [kernel_size, kernel_size], padding=0, stride=stride)  # (B, ps*ps, H*W)
        xy_p = xy_p.reshape(b, 2, kernel_size**2, h, w).unsqueeze(1).repeat(1, d, 1, 1, 1, 1)  # b d 2 k2 h w
        xy_p = xy_p.permute(0, 1, 3, 4, 5, 2).reshape(b, d * kernel_size * kernel_size, h, w, 2)

        indices = (torch.cat([xy_p, indices], dim=-1))
        indices[:, :, :, :, 2] = indices[:, :, :, :, 2] / ((d - 1) / 2) - 1
        indices[:, :, :, :, 0] = indices[:, :, :, :, 0] / ((w_ori - 1) / 2) - 1
        indices[:, :, :, :, 1] = indices[:, :, :, :, 1] / ((h_ori - 1) / 2) - 1
        ans = F.grid_sample(x_ori, grid=indices, padding_mode='zeros', mode='bilinear', align_corners=True)
        ans = ans.reshape(b, c, d, kernel_size*kernel_size, h, w).permute(0,1,3,2,4,5)
        return ans.reshape(b, c * kernel_size * kernel_size, d, h, w)

    def forward(self, x, dp, normal, intri):
        x = self.get_gcacost(x, dp, normal, intri)  # b c*k*k, d h w
        x = self.conv3d(x)  # b c,d h w
        if self.bn is not None:
            x = F.relu(self.bn(x), inplace=True)
        return x

class AdaGCACostRegNet(nn.Module):
    '''
    input b d h w
    output b d h w
    '''

    def __init__(self, in_channels, base_channels):
        super(AdaGCACostRegNet, self).__init__()
        
        self.conv0 = Conv3d(in_channels, base_channels, kernel_size=(3,3,3), padding=(1,1,1))

        self.conv1 = Conv3d(base_channels, base_channels * 2, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv2 = AdaGoConv3D(base_channels * 2, base_channels * 2)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv4 = AdaGoConv3D(base_channels * 4, base_channels * 4)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv6 = AdaGoConv3D(base_channels * 8, base_channels * 8)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, kernel_size=(3, 3, 3), stride=(1,2,2), padding=(1, 1, 1), output_padding=(0,1,1))

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, kernel_size=(3, 3, 3), stride=(1,2,2), padding=(1, 1, 1), output_padding=(0,1,1))

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, kernel_size=(3, 3, 3), stride=(1,2,2), padding=(1, 1, 1), output_padding=(0,1,1))

        self.prob = AdaGoConv3D(base_channels, 1, bn=False)

    def forward(self, x, d, normal, intri):
        conv0 = self.conv0(x)

        d1 = F.interpolate(d, scale_factor=0.5)
        normal1 = F.interpolate(normal, scale_factor=0.5)
        intri1 = intri[:, 0:2, :] * 0.5

        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1, d1, normal1, intri1)

        d2 = F.interpolate(d1, scale_factor=0.5)
        normal2 = F.interpolate(normal1, scale_factor=0.5)
        intri2 = intri1[:, 0:2, :] * 0.5

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3, d2, normal2, intri2)

        d3 = F.interpolate(d2, scale_factor=0.5)
        normal3 = F.interpolate(normal2, scale_factor=0.5)
        intri3 = intri2[:, 0:2, :] * 0.5

        conv5 = self.conv5(conv4)
        x = self.conv6(conv5, d3, normal3, intri3)  # b 32 d h/8 w/8
        x = conv4 + self.conv7(x)  # b 16 d h/4 w/4
        x = conv2 + self.conv9(x)  # b 8 d h/2 w/2
        x = conv0 + self.conv11(x)  # b 4 d h w
        x = self.prob(x, d, normal, intri)
        return x