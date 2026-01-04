import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .compute_normal import depth2normal
from .gca_module import GCACostRegNet
from .normal_enhance import *
Align_Corners_Range = True

class PatchDepthNetv2(nn.Module):
    def __init__(self, patch_neighbors=5):
        super(PatchDepthNetv2, self).__init__()
        self.pixel_wise_net = PixelwiseNet(patch_neighbors)
        self.patch_neighbors = patch_neighbors
    
    def get_grid(self, batch, height, width, dilation, device):
        if self.patch_neighbors == 1:
            original_offset = [[0, 0]]
        elif self.patch_neighbors == 3:
            original_offset = [[0, -dilation], [0, 0], [0, dilation]]
        elif self.patch_neighbors == 5:
            # original_offset = [[dilation, 0], [0, -dilation], [0, 0], [0, dilation], [dilation, 0]]
            original_offset = [[-dilation, -dilation], [-dilation, dilation], [0, 0], [dilation, -dilation], [dilation, dilation]]
        elif self.patch_neighbors == 9:  # if 9 neighbors to be sampled in evaluation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
        elif self.patch_neighbors == 17:  # if 17 neighbors to be sampled in evaluation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                if offset_x != 0 or offset_y != 0:
                    original_offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError

        with torch.no_grad():
            y_grid, x_grid = torch.meshgrid(
                [
                    torch.arange(0, height, dtype=torch.float32, device=device),
                    torch.arange(0, width, dtype=torch.float32, device=device),
                ]
            )
            y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
            xy = torch.stack((x_grid, y_grid))  # [2, H*W]
            xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]

        xy_list = []
        for i in range(len(original_offset)):
            original_offset_y, original_offset_x = original_offset[i]
            # 将 int 类型转换为 tensor
            original_offset_x = torch.tensor([original_offset_x], dtype=torch.float32, device=xy.device).repeat(batch, 1)
            original_offset_y = torch.tensor([original_offset_y], dtype=torch.float32, device=xy.device).repeat(batch, 1)
            xy_list.append((xy + torch.cat([original_offset_x, original_offset_y], dim=1).unsqueeze(2)).unsqueeze(2))

        xy = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]

        del xy_list
        del x_grid
        del y_grid
        
        x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
        y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
        del xy
        grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
        del x_normalized
        del y_normalized
        return grid.view(batch, len(original_offset) * height, width, 2)
    
    def get_multi_detph(self, depth_values, normal, intri, grid):
        device = depth_values.device
        B, D, H, W = depth_values.shape
        
        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        nx, ny, nz = normal[:, 0].unsqueeze(1), normal[:, 1].unsqueeze(1), normal[:, 2].unsqueeze(1)
        
        with torch.no_grad():
            y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=device),
                                torch.arange(0, W, dtype=torch.float32, device=device)])
            xy = torch.stack((x, y))  # [2, H*W]
            xy = xy.unsqueeze(0).repeat(B, 1, 1, 1)  # B 2 H, W
        
        xy_sample = F.grid_sample(
            xy, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).view(B, 2, -1)
        del xy
        
        u_sample = (xy_sample[:, 0] - cx.unsqueeze(1)) / fx.unsqueeze(1)
        v_sample = (xy_sample[:, 1] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        del xy_sample
        
        u_sample = u_sample.reshape(B, -1, H, W)
        v_sample = v_sample.reshape(B, -1, H, W)
        
        # pos - center
        u_center = u_sample[:, self.patch_neighbors // 2, :, :].unsqueeze(1)
        v_center = v_sample[:, self.patch_neighbors // 2, :, :].unsqueeze(1)

        ddw_num = nx * u_center + ny * v_center + nz  # B 1 H W
        ddw_denom = nx * u_sample + ny * v_sample + nz  # B k*k H W
        del u_sample, u_center
        del v_sample, v_center
        
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, k*k, H, W)
        del ddw_num
        del ddw_denom
        
        ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k2 h w
        
        depth_sample = ddw_weights.unsqueeze(1) * depth_values.unsqueeze(2)
        
        return depth_sample.view(B, D, -1, W)

    def forward(self,
                features,
                proj_matrices,
                depth_values,
                num_depth,
                cost_regularization,
                prob_volume_init=None,
                normal=None,
                stage_intric=None,
                view_weights=None,
                stage_idx=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shape[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]
        
        B, D, H, W = depth_values.shape
        
        if self.patch_neighbors > 1:
            C = ref_feature.size(1)
            device = ref_feature.device
            ref_intri = ref_proj[:, 1]
            patch_grid = self.get_grid(B, H, W, 1, device)
            ref_feature_patch = F.grid_sample(ref_feature, patch_grid, mode="bilinear", padding_mode="border", align_corners=False)
            multi_depth_values = self.get_multi_detph(depth_values, normal, ref_intri, patch_grid)

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            
            if self.patch_neighbors > 1:
                warped_volume = homo_warping_patch(src_fea, src_proj_new, ref_proj_new, multi_depth_values, patch_grid)
                similarity = (warped_volume * ref_feature_patch.unsqueeze(2)).mean(1, keepdim=True) \
                    .view(B, 1, D, -1, H, W).squeeze(1).permute(0, 2, 1, 3, 4)
            else:
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
                similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
        
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]
        
        if self.patch_neighbors > 1:
            del multi_depth_values, patch_grid, ref_feature_patch
            similarity = torch.cat([similarity.mean(1, keepdim=True), similarity[:, self.patch_neighbors // 2, ...].unsqueeze(1)], 1)

        cost_reg = cost_regularization(similarity)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
        if view_weights == None:
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}, view_weights.detach()
        else:
            return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}

class GAPMVSNet(nn.Module):
    def __init__(self, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], grad_method="detach", cr_base_chs=[8, 8, 8], mode="train"):
        super(GAPMVSNet, self).__init__()
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.mode = mode

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
                "stage1":{
                    "scale": 4.0,
                    },
                "stage2": {
                    "scale": 2.0,
                    },
                "stage3": {
                    "scale": 1.0,
                    }
                }

        self.feature = NormalEnhancedFPN(base_channels=8, num_stage=self.num_stage)
        
        self.resample = nn.ModuleList([None,
                ResampleModule(base_channels=16, use_conf=True),
                ResampleModule(base_channels=8, use_conf=True)])
        
        self.refine_net = nn.ModuleList([ResidualNetv2(feat_channels=32), ResidualNetv2(feat_channels=16), ResidualNetv2(feat_channels=8)])

        self.cost_regularization = nn.ModuleList([GeoCostRegNet(in_channels=1, base_channels=8),
                GeoCostRegNet(in_channels=1, base_channels=8),
                GeoCostRegNet(in_channels=1, base_channels=8)])

        self.DepthNet = VisDepthNet()

    def forward(self, imgs, proj_matrices, depth_values, normal_mono=None):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)
        
        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            normal = normal_mono[:, nview_idx]
            if normal.shape[1] != img.shape[1] or normal.shape[2] != img.shape[2]:
                normal = F.interpolate(normal.float(), [img.shape[2], img.shape[3]], mode='bilinear',
                                       align_corners=Align_Corners_Range)
            features.append(self.feature(img, normal))
            
        normal_mono = normal_mono[:, 0]

        if self.mode == "train":
            normal_mono = F.interpolate(normal_mono.float(),
                        [img.shape[2]//2**2, img.shape[3]//2**2], mode='bilinear',
                         align_corners=Align_Corners_Range)
        
        outputs = {}
        depth, cur_depth = None, None
        view_weights = None
        normal = None
        confidence = None
        for stage_idx in range(self.num_stage):
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            Using_inverse_d = False
            
            normal_stage = F.interpolate(normal_mono.float(),
                        [img.shape[2]//2**(2-stage_idx), img.shape[3]//2**(2-stage_idx)], mode='bilinear',
                         align_corners=Align_Corners_Range)

            stage_ref_proj = torch.unbind(proj_matrices_stage, 1)[0]  # to list#b n 2 4 4
            stage_ref_int = stage_ref_proj[:, 1, :3, :3]  # b 3 3
            
            if stage_idx + 1 > 1: # for stage 2 and 3
                view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")
                
                confidence = F.interpolate(outputs_stage['photometric_confidence'].unsqueeze(1),
                                           size=None, scale_factor=2, mode="bilinear", align_corners=False)
                features_stage[0] = self.resample[stage_idx](features_stage[0], confidence)

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                        [img.shape[2], img.shape[3]], mode='bilinear',
                        align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            # [B, D, H, W]
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                    ndepth=self.ndepths[stage_idx],
                    depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                    dtype=img[0].dtype,
                    device=img[0].device,
                    shape=[img.shape[0], img.shape[2], img.shape[3]],
                    max_depth=depth_max,
                    min_depth=depth_min,
                    use_inverse_depth=Using_inverse_d)

            if view_weights == None: # stage 1
                outputs_stage, view_weights = self.DepthNet(
                        features_stage,
                        proj_matrices_stage,
                        depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_idx],
                        normal=normal_stage,
                        stage_intric=stage_ref_int,
                        cost_regularization=self.cost_regularization[stage_idx], 
                        view_weights=view_weights,
                        )
            else:
                outputs_stage = self.DepthNet(
                        features_stage,
                        proj_matrices_stage,
                        depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_idx],
                        normal=normal_stage,
                        stage_intric=stage_ref_int,
                        cost_regularization=self.cost_regularization[stage_idx], 
                        view_weights=view_weights,
                        )

            wta_index_map = torch.argmax(outputs_stage['prob_volume'], dim=1, keepdim=True).type(torch.long)
            depth = torch.gather(outputs_stage['depth_values'], 1, wta_index_map).squeeze(1)
            outputs_stage['depth'] = depth

            if normal is not None:
                outputs_stage['normal'] = normal_stage #b 3 h w
            
            depth, off = self.refine_net[stage_idx](depth, features_stage[0].detach(), normal_stage, stage_ref_int,
                                                self.depth_interals_ratio[stage_idx] * depth_interval)
            
            outputs_stage['depth'] = depth
            if self.mode == 'train':
                outputs_stage['offset'] = off
                outputs_stage['interval'] = self.depth_interals_ratio[stage_idx] * depth_interval
        
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs
    
# class CasMVS(nn.Module):
#     def __init__(self, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], grad_method="detach", cr_base_chs=[8, 8, 8], mode="train"):
#         super(CasMVS, self).__init__()
#         self.ndepths = ndepths
#         self.depth_interals_ratio = depth_interals_ratio
#         self.grad_method = grad_method
#         self.cr_base_chs = cr_base_chs
#         self.num_stage = len(ndepths)
#         self.mode = mode

#         assert len(ndepths) == len(depth_interals_ratio)

#         self.stage_infos = {
#                 "stage1":{
#                     "scale": 4.0,
#                     },
#                 "stage2": {
#                     "scale": 2.0,
#                     },
#                 "stage3": {
#                     "scale": 1.0,
#                     }
#                 }

#         # self.feature = NormalEnhancedFPN(base_channels=8, num_stage=self.num_stage)
#         self.feature = FeatureNet(base_channels=8)
        
#         # self.resample = nn.ModuleList([None,
#         #         ResampleModule(base_channels=16, use_conf=True),
#         #         ResampleModule(base_channels=8, use_conf=True)])
        
#         # self.refine_net = nn.ModuleList([ResidualNetv2(feat_channels=32), ResidualNetv2(feat_channels=16), ResidualNetv2(feat_channels=8)])

#         self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=1, base_channels=8),
#                 CostRegNet(in_channels=1, base_channels=8),
#                 CostRegNet(in_channels=1, base_channels=8)])

#         self.DepthNet = VisDepthNet()

#     def forward(self, imgs, proj_matrices, depth_values, normal_mono=None):
#         depth_min = float(depth_values[0, 0].cpu().numpy())
#         depth_max = float(depth_values[0, -1].cpu().numpy())
#         depth_interval = (depth_max - depth_min) / depth_values.size(1)

#         # step 1. feature extraction
#         features = []
#         for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
#             img = imgs[:, nview_idx]
#             normal = normal_mono[:, nview_idx]
#             if normal.shape[1] != img.shape[1] or normal.shape[2] != img.shape[2]:
#                 normal = F.interpolate(normal.float(), [img.shape[2], img.shape[3]], mode='bilinear',
#                                        align_corners=Align_Corners_Range)
#             # features.append(self.feature(img, normal))
#             features.append(self.feature(img))
            
#         normal_mono = normal_mono[:, 0]
        
#         # features = self.FMT_with_pathway(features, proj_matrices)

#         if self.mode == "train":
#             normal_mono = F.interpolate(normal_mono.float(),
#                         [img.shape[2]//2**2, img.shape[3]//2**2], mode='bilinear',
#                          align_corners=Align_Corners_Range)
        
#         outputs = {}
#         depth, cur_depth = None, None
#         view_weights = None
#         normal = None
#         confidence = None
#         for stage_idx in range(self.num_stage):
#             features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            
#             proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
#             stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

#             Using_inverse_d = False
            
#             normal_stage = F.interpolate(normal_mono.float(),
#                         [img.shape[2]//2**(2-stage_idx), img.shape[3]//2**(2-stage_idx)], mode='bilinear',
#                          align_corners=Align_Corners_Range)

#             stage_ref_proj = torch.unbind(proj_matrices_stage, 1)[0]  # to list#b n 2 4 4
#             stage_ref_int = stage_ref_proj[:, 1, :3, :3]  # b 3 3
            
#             if stage_idx + 1 > 1: # for stage 2 and 3
#                 view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")
                
#                 # confidence = F.interpolate(outputs_stage['photometric_confidence'].unsqueeze(1),
#                 #                            size=None, scale_factor=2, mode="bilinear", align_corners=False)
#                 # features_stage[0] = self.resample[stage_idx](features_stage[0], confidence)

#             if depth is not None:
#                 if self.grad_method == "detach":
#                     cur_depth = depth.detach()
#                 else:
#                     cur_depth = depth
                
#                 cur_depth = F.interpolate(cur_depth.unsqueeze(1),
#                         [img.shape[2], img.shape[3]], mode='bilinear',
#                         align_corners=Align_Corners_Range).squeeze(1)
#             else:
#                 cur_depth = depth_values

#             # [B, D, H, W]
#             depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
#                     ndepth=self.ndepths[stage_idx],
#                     depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
#                     dtype=img[0].dtype,
#                     device=img[0].device,
#                     shape=[img.shape[0], img.shape[2], img.shape[3]],
#                     max_depth=depth_max,
#                     min_depth=depth_min,
#                     use_inverse_depth=Using_inverse_d)

#             if view_weights == None: # stage 1
#                 outputs_stage, view_weights = self.DepthNet(
#                         features_stage,
#                         proj_matrices_stage,
#                         depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
#                         num_depth=self.ndepths[stage_idx],
#                         normal=normal_stage,
#                         stage_intric=stage_ref_int,
#                         cost_regularization=self.cost_regularization[stage_idx], 
#                         view_weights=view_weights,
#                         )
#             else:
#                 outputs_stage = self.DepthNet(
#                         features_stage,
#                         proj_matrices_stage,
#                         depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
#                         num_depth=self.ndepths[stage_idx],
#                         normal=normal_stage,
#                         stage_intric=stage_ref_int,
#                         cost_regularization=self.cost_regularization[stage_idx], 
#                         view_weights=view_weights,
#                         )

#             # wta_index_map = torch.argmax(outputs_stage['prob_volume'], dim=1, keepdim=True).type(torch.long)
#             # depth = torch.gather(outputs_stage['depth_values'], 1, wta_index_map).squeeze(1)
#             depth = depth_regression(outputs_stage['prob_volume'], outputs_stage['depth_values'])
#             outputs_stage['depth'] = depth

#             if normal is not None:
#                 outputs_stage['normal'] = normal_stage #b 3 h w
            
#             # depth, off = self.refine_net[stage_idx](depth, features_stage[0].detach(), normal_stage, stage_ref_int,
#             #                                         self.depth_interals_ratio[stage_idx] * depth_interval)
#             # outputs_stage['depth'] = depth
#             # if self.mode == 'train':
#             #     outputs_stage['offset'] = off
#             #     outputs_stage['interval'] = self.depth_interals_ratio[stage_idx] * depth_interval
        
#             outputs["stage{}".format(stage_idx + 1)] = outputs_stage
#             outputs.update(outputs_stage)

#         return outputs

class PixelwiseNet(nn.Module):
    def __init__(self, G=1):

        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=G, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1):
        """forward.

        :param x1: [B, 1, D, H, W]
        """

        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1) # [B, D, H, W]
        output = self.output(x1)
        output = torch.max(output, dim=1, keepdim=True)[0] # [B, 1, H ,W]

        return output

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.pixel_wise_net = PixelwiseNet()

    def forward(self,
                features,
                proj_matrices,
                depth_values,
                num_depth,
                cost_regularization,
                prob_volume_init=None,
                normal=None,
                stage_intric=None,
                view_weights=None):
        """forward.
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrices: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, GCACostRegNet
        :param view_weights: pixel wise view weights for src views
        :param normal: torch.Tensor 
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shape[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]
        # similarity_prob = F.softmax(similarity.squeeze(1), dim=1)
        # similarity_depth = depth_wta(similarity_prob, depth_values=depth_values)

        #cost_reg = cost_regularization(similarity, depth_values, normal)
        cost_reg = cost_regularization(similarity, depth_values, normal, stage_intric)
        # cost_reg = cost_regularization(similarity)
        # cost_reg = cost_regularization(similarity, ref_feature)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
            
        if view_weights == None:
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}, view_weights.detach()
            # return {"depth": depth,  "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}, view_weights.detach()
        else:
            return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}
            # return {"depth": depth, "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}

class VisDepthNet(nn.Module):
    def __init__(self):
        super(VisDepthNet, self).__init__()
        self.pixel_wise_net = PixelwiseNet(G=2)

    def forward(self,
                features,
                proj_matrices,
                depth_values,
                num_depth,
                cost_regularization,
                prob_volume_init=None,
                normal=None,
                stage_intric=None,
                view_weights=None):
        """forward.
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrices: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, GCACostRegNet
        :param view_weights: pixel wise view weights for src views
        :param normal: torch.Tensor 
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shape[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []
        
            ref_R = ref_proj[:, 0, :3, :3]
            src_Rs = torch.cat([src_proj[:, 0, :3, :3].unsqueeze(0) for src_proj in src_projs], dim=0)
            nor_R = ref_R@torch.linalg.inv(ref_R)
            norv_R = nor_R[:,2,:] #Bx3
            norv_R = norv_R.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, normal.shape[2], normal.shape[3])#B 3 W H

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)


            if view_weights == None:
                src_R = src_Rs[i,:,:]
                nor_R = src_R@torch.linalg.inv(ref_R)
                norv_R = nor_R[:,2,:] #Bx3
                norv_R = norv_R.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, normal.shape[2], normal.shape[3])#B 3 W H
                
                vis_prior = torch.sum(normal * norv_R, dim=1, keepdim=True).unsqueeze(2).expand(-1, -1, num_depth, -1, -1)
                view_weight = self.pixel_wise_net(torch.cat([similarity, vis_prior], dim=1)) # [B, 1, H, W]
                # view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]
        # similarity_prob = F.softmax(similarity.squeeze(1), dim=1)
        # similarity_depth = depth_wta(similarity_prob, depth_values=depth_values)

        cost_reg = cost_regularization(similarity, depth_values, normal)
        # cost_reg = cost_regularization(similarity)
        # cost_reg = cost_regularization(similarity, ref_feature)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
            
        if view_weights == None:
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}, view_weights.detach()
            # return {"depth": depth,  "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}, view_weights.detach()
        else:
            return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}
            # return {"depth": depth, "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}

class DepthNetv2(nn.Module):
    def __init__(self):
        super(DepthNetv2, self).__init__()
        self.pixel_wise_net = PixelwiseNet()

    def forward(self,
                features,
                proj_matrices,
                depth_values,
                num_depth,
                cost_regularization,
                prob_volume_init=None,
                normal=None,
                stage_intric=None,
                view_weights=None,
                stage_idx=None):
        """forward.
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrices: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, GCACostRegNet
        :param view_weights: pixel wise view weights for src views
        :param normal: torch.Tensor 
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shape[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]
        # similarity_prob = F.softmax(similarity.squeeze(1), dim=1)
        # similarity_depth = depth_wta(similarity_prob, depth_values=depth_values)

        if stage_idx > 0:
            cost_reg = cost_regularization(similarity, depth_values, normal, stage_intric)
        else:
            cost_reg = cost_regularization(similarity)
        # cost_reg = cost_regularization(similarity, ref_feature)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
            
        if view_weights == None:
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            # return {"depth": depth,  "similarity_depth":similarity_depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}, view_weights.detach()
            return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}, view_weights.detach()
        else:
            # return {"depth": depth,  "similarity_depth":similarity_depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}
            return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}


class DepthNetv3(nn.Module):
    def __init__(self, G=1):
        super(DepthNetv3, self).__init__()
        self.G = G
        self.pixel_wise_net = PixelwiseNet(G)

    def forward(self,
                features,
                proj_matrices,
                depth_values,
                num_depth,
                cost_regularization,
                prob_volume_init=None,
                normal=None,
                stage_intric=None,
                view_weights=None):
        """forward.
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrices: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, GCACostRegNet
        :param view_weights: pixel wise view weights for src views
        :param normal: torch.Tensor 
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shape[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5
        
        G = self.G
        B, C, H, W = ref_feature.shape
        D = num_depth
        ref_feature = ref_feature.view(B, G, C // G, 1, H, W).repeat(1, 1, 1, D, 1, 1)

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            
            warped_volume = warped_volume.view(B, G, C // G, D, H, W)
            similarity = (warped_volume * ref_feature).mean(2)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]

        cost_reg = cost_regularization(similarity)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
            
        if view_weights == None:
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}, view_weights.detach()
        else:
            return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "prob_volume_pre": prob_volume_pre, "depth_values": depth_values}


class GoMVS(nn.Module):
    def __init__(self, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], grad_method="detach", cr_base_chs=[8, 8, 8], dptv2_pretrained='', mode="train"):
        super(GoMVS, self).__init__()
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.mode = mode

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
                "stage1":{
                    "scale": 4.0,
                    },
                "stage2": {
                    "scale": 2.0,
                    },
                "stage3": {
                    "scale": 1.0,
                    }
                }
        
        self.feature = FeatureNet(base_channels=8)

        self.cost_regularization = nn.ModuleList([GCACostRegNet(in_channels=1, base_channels=8),
                GCACostRegNet(in_channels=1, base_channels=8),
                GCACostRegNet(in_channels=1, base_channels=8)])

        self.DepthNet = DepthNet()

    def forward(self, imgs, proj_matrices, depth_values, normal_mono=None):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
        
        if normal_mono.shape[-1] == 3:
            normal_mono = normal_mono.permute(0, 3, 1, 2)

        if self.mode == "train":
            normal_mono = F.interpolate(normal_mono.float(),
                        [img.shape[2]//2**2, img.shape[3]//2**2], mode='bilinear',
                         align_corners=Align_Corners_Range)

        outputs = {}
        depth, cur_depth = None, None
        view_weights = None
        normal = None
        for stage_idx in range(self.num_stage):
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            Using_inverse_d = False

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                        [img.shape[2], img.shape[3]], mode='bilinear',
                        align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            # [B, D, H, W]
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                    ndepth=self.ndepths[stage_idx],
                    depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                    dtype=img[0].dtype,
                    device=img[0].device,
                    shape=[img.shape[0], img.shape[2], img.shape[3]],
                    max_depth=depth_max,
                    min_depth=depth_min,
                    use_inverse_depth=Using_inverse_d)

            if stage_idx + 1 > 1: # for stage 2 and 3
                view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")

            stage_ref_proj = torch.unbind(proj_matrices_stage, 1)[0]  # to list#b n 2 4 4
            stage_ref_int = stage_ref_proj[:, 1, :3, :3]  # b 3 3

            normal_stage = F.interpolate(normal_mono.float(),
                        [img.shape[2]//2**(2-stage_idx), img.shape[3]//2**(2-stage_idx)], mode='bilinear',
                         align_corners=Align_Corners_Range)

            if view_weights == None: # stage 1
                outputs_stage, view_weights = self.DepthNet(
                        features_stage,
                        proj_matrices_stage,
                        depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_idx],
                        normal=normal_stage,
                        stage_intric=stage_ref_int,
                        cost_regularization=self.cost_regularization[stage_idx], 
                        view_weights=view_weights)
            else:
                outputs_stage = self.DepthNet(
                        features_stage,
                        proj_matrices_stage,
                        depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_idx],
                        normal=normal_stage,
                        stage_intric=stage_ref_int,
                        cost_regularization=self.cost_regularization[stage_idx], 
                        view_weights=view_weights)

            wta_index_map = torch.argmax(outputs_stage['prob_volume'], dim=1, keepdim=True).type(torch.long)
            depth = torch.gather(outputs_stage['depth_values'], 1, wta_index_map).squeeze(1)
            outputs_stage['depth'] = depth

            if normal is not None:
                outputs_stage['normal'] = normal_stage #b 3 h w

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs

# class CasMVS(nn.Module):
#     def __init__(self, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], grad_method="detach", cr_base_chs=[8, 8, 8], mode="train"):
#         super(CasMVS, self).__init__()
#         self.ndepths = ndepths
#         self.depth_interals_ratio = depth_interals_ratio
#         self.grad_method = grad_method
#         self.cr_base_chs = cr_base_chs
#         self.num_stage = len(ndepths)
#         self.mode = mode

#         assert len(ndepths) == len(depth_interals_ratio)

#         self.stage_infos = {
#                 "stage1":{
#                     "scale": 4.0,
#                     },
#                 "stage2": {
#                     "scale": 2.0,
#                     },
#                 "stage3": {
#                     "scale": 1.0,
#                     }
#                 }

#         # self.feature = FeatureNet(base_channels=8)
#         self.feature = NormalEnhancedFPN(base_channels=8, num_stage=self.num_stage)
        
#         self.resample1 = ResampleModule(base_channels=8*2, use_conf=True)
#         self.resample2 = ResampleModule(base_channels=8, use_conf=True)

#         self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=1, base_channels=8),
#                 CostRegNet(in_channels=1, base_channels=8),
#                 CostRegNet(in_channels=1, base_channels=8)])

#         self.DepthNet = DepthNet()

#     def forward(self, imgs, proj_matrices, depth_values, normal_mono=None):
#         depth_min = float(depth_values[0, 0].cpu().numpy())
#         depth_max = float(depth_values[0, -1].cpu().numpy())
#         depth_interval = (depth_max - depth_min) / depth_values.size(1)

#         # step 1. feature extraction
#         # features = []
#         # for nview_idx in range(imgs.size(1)):
#         #     img = imgs[:, nview_idx]
#         #     features.append(self.feature(img))
            
#         features = []
#         for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
#             img = imgs[:, nview_idx]
#             normal = normal_mono[:, nview_idx]
#             if normal.shape[1] != img.shape[1] or normal.shape[2] != img.shape[2]:
#                 normal = F.interpolate(normal.float(), [img.shape[2], img.shape[3]], mode='bilinear',
#                                        align_corners=Align_Corners_Range)
#             features.append(self.feature(img, normal))
        
#         del normal_mono

#         # if self.mode == "train":
#         #     normal_mono = F.interpolate(normal_mono.float(),
#         #                 [img.shape[2]//2**2, img.shape[3]//2**2], mode='bilinear',
#         #                  align_corners=Align_Corners_Range)
#         outputs = {}
#         depth, cur_depth = None, None
#         view_weights = None
#         normal = None
#         for stage_idx in range(self.num_stage):
#             features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            
#             proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
#             stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

#             Using_inverse_d = False

#             if depth is not None:
#                 if self.grad_method == "detach":
#                     cur_depth = depth.detach()
#                 else:
#                     cur_depth = depth
#                 cur_depth = F.interpolate(cur_depth.unsqueeze(1),
#                         [img.shape[2], img.shape[3]], mode='bilinear',
#                         align_corners=Align_Corners_Range).squeeze(1)
#             else:
#                 cur_depth = depth_values

#             # [B, D, H, W]
#             depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
#                     ndepth=self.ndepths[stage_idx],
#                     depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
#                     dtype=img[0].dtype,
#                     device=img[0].device,
#                     shape=[img.shape[0], img.shape[2], img.shape[3]],
#                     max_depth=depth_max,
#                     min_depth=depth_min,
#                     use_inverse_depth=Using_inverse_d)

#             if stage_idx + 1 > 1: # for stage 2 and 3
#                 view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")
                
#             if stage_idx == 1:
#                 confidence = F.interpolate(outputs_stage['photometric_confidence'].unsqueeze(1),
#                                            size=None, scale_factor=2, mode="bilinear", align_corners=False)
#                 features_stage[0] = self.resample1(features_stage[0], confidence)
#             elif stage_idx == 2:
#                 confidence = F.interpolate(outputs_stage['photometric_confidence'].unsqueeze(1),
#                                            size=None, scale_factor=2, mode="bilinear", align_corners=False)
#                 features_stage[0] = self.resample2(features_stage[0], confidence)

#             stage_ref_proj = torch.unbind(proj_matrices_stage, 1)[0]  # to list#b n 2 4 4
#             stage_ref_int = stage_ref_proj[:, 1, :3, :3]  # b 3 3

#             # normal_stage = F.interpolate(normal_mono.float(),
#             #             [img.shape[2]//2**(2-stage_idx), img.shape[3]//2**(2-stage_idx)], mode='bilinear',
#             #              align_corners=Align_Corners_Range)

#             if view_weights == None: # stage 1
#                 outputs_stage, view_weights = self.DepthNet(
#                         features_stage,
#                         proj_matrices_stage,
#                         depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
#                         num_depth=self.ndepths[stage_idx],
#                         normal=None,
#                         stage_intric=stage_ref_int,
#                         cost_regularization=self.cost_regularization[stage_idx], 
#                         view_weights=view_weights)
#             else:
#                 outputs_stage = self.DepthNet(
#                         features_stage,
#                         proj_matrices_stage,
#                         depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
#                         num_depth=self.ndepths[stage_idx],
#                         normal=None,
#                         stage_intric=stage_ref_int,
#                         cost_regularization=self.cost_regularization[stage_idx], 
#                         view_weights=view_weights)

#             wta_index_map = torch.argmax(outputs_stage['prob_volume'], dim=1, keepdim=True).type(torch.long)
#             depth = torch.gather(outputs_stage['depth_values'], 1, wta_index_map).squeeze(1)
#             outputs_stage['depth'] = depth

#             if normal is not None:
#                 outputs_stage['normal'] = normal_stage #b 3 h w

#             outputs["stage{}".format(stage_idx + 1)] = outputs_stage
#             outputs.update(outputs_stage)

#         return outputs

class PatchDepthNet(nn.Module):
    def __init__(self, patch_neighbors=5):
        super(PatchDepthNet, self).__init__()
        self.pixel_wise_net = PixelwiseNet()
        self.patch_neighbors = patch_neighbors
    
    def get_grid(self, batch, height, width, dilation, device):
        if self.patch_neighbors == 5:
            original_offset = [[dilation, 0], [0, -dilation], [0, 0], [0, dilation], [dilation, 0]]
        elif self.patch_neighbors == 9:  # if 9 neighbors to be sampled in evaluation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
        elif self.patch_neighbors == 17:  # if 17 neighbors to be sampled in evaluation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                if offset_x != 0 or offset_y != 0:
                    original_offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError

        with torch.no_grad():
            y_grid, x_grid = torch.meshgrid(
                [
                    torch.arange(0, height, dtype=torch.float32, device=device),
                    torch.arange(0, width, dtype=torch.float32, device=device),
                ]
            )
            y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
            xy = torch.stack((x_grid, y_grid))  # [2, H*W]
            xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]

        xy_list = []
        for i in range(len(original_offset)):
            original_offset_y, original_offset_x = original_offset[i]
            # 将 int 类型转换为 tensor
            original_offset_x = torch.tensor([original_offset_x], dtype=torch.float32, device=xy.device).repeat(batch, 1)
            original_offset_y = torch.tensor([original_offset_y], dtype=torch.float32, device=xy.device).repeat(batch, 1)
            # print(xy.shape, torch.cat((original_offset_x, original_offset_y), dim=1).unsqueeze(2).shape)
            xy_list.append((xy + torch.cat([original_offset_x, original_offset_y], dim=1).unsqueeze(2)).unsqueeze(2))

        xy = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]

        del xy_list
        del x_grid
        del y_grid
        
        x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
        y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
        del xy
        grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
        del x_normalized
        del y_normalized
        return grid.view(batch, len(original_offset) * height, width, 2)
    
    def get_detph_shift(self, depth_values, normal, intri, grid):
        device = depth_values.device
        B, D, H, W = depth_values.shape
        
        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        nx, ny, nz = normal[:, 0].unsqueeze(1), normal[:, 1].unsqueeze(1), normal[:, 2].unsqueeze(1)
        
        with torch.no_grad():
            y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=device),
                                torch.arange(0, W, dtype=torch.float32, device=device)])
            xy = torch.stack((x, y))  # [2, H*W]
            xy = xy.unsqueeze(0).repeat(B, 1, 1, 1)  # B 2 H, W
        
        xy_sample = F.grid_sample(
            xy, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).view(B, 2, -1)
        del xy
        
        u_sample = (xy_sample[:, 0] - cx.unsqueeze(1)) / fx.unsqueeze(1)
        v_sample = (xy_sample[:, 1] - cy.unsqueeze(1)) / fy.unsqueeze(1)
        del xy_sample
        
        u_sample = u_sample.reshape(B, -1, H, W)
        v_sample = v_sample.reshape(B, -1, H, W)
        
        # pos - center
        u_center = u_sample[:, self.patch_neighbors // 2, :, :].unsqueeze(1)
        v_center = v_sample[:, self.patch_neighbors // 2, :, :].unsqueeze(1)

        ddw_num = nx * u_center + ny * v_center + nz  # B 1 H W
        ddw_denom = nx * u_sample + ny * v_sample + nz  # B k*k H W
        del u_sample, u_center
        del v_sample, v_center
        
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, k*k, H, W)
        del ddw_num
        del ddw_denom
        
        ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k2 h w
        
        depth_sample = ddw_weights.unsqueeze(1) * depth_values.unsqueeze(2)
        
        # print(depth_sample[..., 0, 0])
        
        return depth_sample.view(B, self.patch_neighbors, -1, H, W)
    
    def get_patch_similarity(self, x, grid, sim='exp'):
        B, C, H, W = x.shape
        # 使用零填充来处理边界情况
        # padded_input = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

        # 展平输入张量中每个点及其周围KxK范围内的点
        x_sample = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).view(B, C, -1, H, W)

        # 计算余弦相似度
        if sim == 'cos':
            similarity = F.cosine_similarity(x_sample, x.unsqueeze(2), dim=1)
        elif sim == 'dot':
            similarity = x_sample * x.unsqueeze(2)
            similarity = similarity.sum(dim=1)
        elif sim == 'exp':
            similarity = torch.sum((x_sample - x.unsqueeze(2)) ** 2, dim=1)
            similarity = torch.exp(-0.5 * torch.sqrt(similarity))
        else:
            raise NotImplementedError
        
        similarity = similarity / similarity.sum(dim=1, keepdim=True)

        return similarity

    def forward(self,
                features,
                proj_matrices,
                depth_values,
                num_depth,
                cost_regularization,
                prob_volume_init=None,
                normal=None,
                stage_intric=None,
                view_weights=None,
                stage_idx=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shape[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]
        
        k = 1
        if stage_idx > k:
            B, D, H, W = depth_values.shape
            C = ref_feature.size(1)
            device = ref_feature.device
            ref_intri = ref_proj[:, 1]

            patch_grid = self.get_grid(B, H, W, 1, device)
            depth_values_shift = self.get_detph_shift(depth_values, normal, ref_intri, patch_grid)
            
            # patch_similarity = self.get_patch_similarity(ref_feature, patch_grid, sim='cos')
            patch_similarity = self.get_patch_similarity(normal, patch_grid, sim='cos')
            # patch_similarity = patch_similarity / patch_similarity.sum(1, keepdim=True)
            
            ref_patch_feature = (F.grid_sample(
                ref_feature, patch_grid, mode="bilinear", padding_mode="border", align_corners=False
            ).view(B, C, -1, H, W) * patch_similarity.unsqueeze(1)).unsqueeze(2)

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            
            if stage_idx > k:
                warped_volume = homo_warping_shift(src_fea, src_proj_new, ref_proj_new, depth_values_shift, patch_grid)
                similarity = (warped_volume.view(B, C, D, -1, H, W) * ref_patch_feature).sum(3).mean(1, keepdim=True)
            else:
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
                similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
            
        if stage_idx > k:
            del depth_values_shift
            del ref_patch_feature
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]
        similarity_prob = F.softmax(similarity.squeeze(1), dim=1)
        similarity_depth = depth_wta(similarity_prob, depth_values=depth_values)

        # cost_reg = cost_regularization(similarity, depth_values, normal, stage_intric)
        cost_reg = cost_regularization(similarity)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
        if view_weights == None:
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            return {"depth": depth,  "similarity_depth":similarity_depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}, view_weights.detach()
        else:
            return {"depth": depth,  "similarity_depth":similarity_depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}

class ResampleDepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.pixel_wise_net = PixelwiseNet()

    def forward(self,
                features,
                proj_matrices,
                depth_values,
                num_depth,
                cost_regularization,
                resample,
                prob_volume_init=None,
                normal=None,
                stage_intric=None,
                view_weights=None,
                confidence=None):
        """forward.
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrices: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, GCACostRegNet
        :param view_weights: pixel wise view weights for src views
        :param normal: torch.Tensor 
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shape[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]
        similarity_prob = F.softmax(similarity.squeeze(1), dim=1)
        similarity_depth = depth_wta(similarity_prob, depth_values=depth_values)

        # cost_reg = cost_regularization(similarity, depth_values, normal, stage_intric)
        cost_reg = cost_regularization(similarity)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
            
        if view_weights == None:
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            return {"depth": depth,  "similarity_depth":similarity_depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}, view_weights.detach()
        else:
            return {"depth": depth,  "similarity_depth":similarity_depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}

# class CasMVS(nn.Module):
#     def __init__(self, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], grad_method="detach", cr_base_chs=[8, 8, 8], mode="train"):
#         super(CasMVS, self).__init__()
#         self.ndepths = ndepths
#         self.depth_interals_ratio = depth_interals_ratio
#         self.grad_method = grad_method
#         self.cr_base_chs = cr_base_chs
#         self.num_stage = len(ndepths)
#         self.mode = mode

#         assert len(ndepths) == len(depth_interals_ratio)

#         self.stage_infos = {
#                 "stage1":{
#                     "scale": 4.0,
#                     },
#                 "stage2": {
#                     "scale": 2.0,
#                     },
#                 "stage3": {
#                     "scale": 1.0,
#                     }
#                 }

#         self.feature = NormalEnhancedFPN_global(base_channels=8, num_stage=self.num_stage)

#         self.refine_net = nn.ModuleList([ResidualNetv4(32, 3, num_ch_dec=[72, 72, 144, 288]),
#                                          ResidualNetv4(16, 3, num_ch_dec=[72, 72, 144, 288]),
#                                          ResidualNetv4(8, 3, num_ch_dec=[72, 72, 144, 288])])
        
#         self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=1, base_channels=8), 
#                                                   CostRegNet(in_channels=1, base_channels=8), 
#                                                   CostRegNet(in_channels=1, base_channels=8)])

#         self.DepthNet = DepthNet()

#     def forward(self, imgs, proj_matrices, depth_values, normal_mono=None):
#         depth_min = float(depth_values[0, 0].cpu().numpy())
#         depth_max = float(depth_values[0, -1].cpu().numpy())
#         depth_interval = (depth_max - depth_min) / depth_values.size(1)

#         # step 1. feature extraction
#         features = []
#         for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
#             img = imgs[:, nview_idx]
#             normal = normal_mono[:, nview_idx]
#             if normal.shape[1] != img.shape[1] or normal.shape[2] != img.shape[2]:
#                 normal = F.interpolate(normal.float(), [img.shape[2], img.shape[3]], mode='bilinear',
#                                        align_corners=Align_Corners_Range)
#             features.append(self.feature(img, normal))
            
#         normal_mono = normal_mono[:, 0]
        
#         # features = self.FMT_with_pathway(features, proj_matrices)

#         if self.mode == "train":
#             normal_mono = F.interpolate(normal_mono.float(),
#                         [img.shape[2]//2**2, img.shape[3]//2**2], mode='bilinear',
#                          align_corners=Align_Corners_Range)
        
#         outputs = {}
#         depth, cur_depth = None, None
#         view_weights = None
#         normal = True
#         confidence = None
#         for stage_idx in range(self.num_stage):
#             features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            
#             proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
#             stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

#             Using_inverse_d = False
            
#             normal_stage = F.interpolate(normal_mono.float(),
#                         [img.shape[2]//2**(2-stage_idx), img.shape[3]//2**(2-stage_idx)], mode='bilinear',
#                          align_corners=Align_Corners_Range)

#             if stage_idx + 1 > 1: # for stage 2 and 3
#                 view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")
                
#             stage_ref_proj = torch.unbind(proj_matrices_stage, 1)[0]  # to list#b n 2 4 4
#             stage_ref_int = stage_ref_proj[:, 1, :3, :3]  # b 3 3
            
#             if depth is not None:
#                 if self.grad_method == "detach":
#                     cur_depth = depth.detach()
#                 else:
#                     cur_depth = depth
                
#                 cur_depth = F.interpolate(cur_depth.unsqueeze(1),
#                         [img.shape[2], img.shape[3]], mode='bilinear',
#                         align_corners=Align_Corners_Range).squeeze(1)
#             else:
#                 cur_depth = depth_values

#             # [B, D, H, W]
#             depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
#                     ndepth=self.ndepths[stage_idx],
#                     depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
#                     dtype=img[0].dtype,
#                     device=img[0].device,
#                     shape=[img.shape[0], img.shape[2], img.shape[3]],
#                     max_depth=depth_max,
#                     min_depth=depth_min,
#                     use_inverse_depth=Using_inverse_d)

#             depth_values = F.interpolate(depth_range_samples.unsqueeze(1), 
#                                          [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)],
#                                          mode='trilinear', align_corners=Align_Corners_Range).squeeze(1)
#             if view_weights == None: # stage 1
#                 outputs_stage, view_weights = self.DepthNet(
#                         features_stage,
#                         proj_matrices_stage,
#                         depth_values=depth_values,
#                         num_depth=self.ndepths[stage_idx],
#                         normal=normal_stage,
#                         stage_intric=stage_ref_int,
#                         cost_regularization=self.cost_regularization[stage_idx], 
#                         view_weights=view_weights,
#                         )
#             else:
#                 outputs_stage = self.DepthNet(
#                         features_stage,
#                         proj_matrices_stage,
#                         depth_values=depth_values,
#                         num_depth=self.ndepths[stage_idx],
#                         normal=normal_stage,
#                         stage_intric=stage_ref_int,
#                         cost_regularization=self.cost_regularization[stage_idx], 
#                         view_weights=view_weights,
#                         )

#             wta_index_map = torch.argmax(outputs_stage['prob_volume'], dim=1, keepdim=True).type(torch.long)
#             depth = torch.gather(outputs_stage['depth_values'], 1, wta_index_map).squeeze(1)
#             outputs_stage['depth'] = depth

#             if normal is not None:
#                 outputs_stage['normal'] = normal_stage #b 3 h w
            
#             if self.mode != "train" and normal is not None:
#                 outputs_stage['normal'] = depth2normal(depth, stage_ref_int) #b 3 h w
#                 outputs_stage['normal_mono'] = normal_stage #b 3 h w
                
#             depth_rf, prob_volume_rf, depth_values_rf = self.refine_net[stage_idx](depth, features_stage[0], outputs_stage['prob_volume_pre'], depth_values,
#                                                                   normal_stage, stage_ref_int)
#             outputs_stage['depth_rf'] = depth_rf
#             outputs_stage['prob_volume_rf'] = prob_volume_rf
#             outputs_stage['depth_values_rf'] = depth_values_rf
        
#             depth = depth_rf
            
#             with torch.no_grad():
#                 photometric_confidence_rf = torch.max(prob_volume_rf, dim=1)[0]
#             outputs_stage['photometric_confidence'] = (outputs_stage['photometric_confidence'] + photometric_confidence_rf * 3.) / 4.0
        
#             outputs["stage{}".format(stage_idx + 1)] = outputs_stage
#             outputs.update(outputs_stage)

#         return outputs
