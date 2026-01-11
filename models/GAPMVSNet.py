import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .compute_normal import depth2normal
from .normal_enhance import *
Align_Corners_Range = True

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
        
        self.refine_net = nn.ModuleList([ResidualNet(feat_channels=32), ResidualNet(feat_channels=16), ResidualNet(feat_channels=8)])

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
        :param cost_regularization: nn.Module
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
