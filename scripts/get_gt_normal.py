import cv2
import numpy as np
import re
from PIL import Image

import torch
import torch.nn.functional as F
import os

interval_scale = 1.06

def get_points_coordinate(depth, instrinsic_inv):
    B, height, width, C = depth.size()
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]
    depth_xyz = xyz * depth.view(B, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, height, width)

def depth2normal(depth_torch, intrinsic_torch):

    # load depth & intrinsic

    depth_torch = depth_torch.unsqueeze(-1) # (B, h, w, 1)
    B, H, W, _ = depth_torch.shape

    intrinsic_inv_torch = torch.inverse(intrinsic_torch) # (B, 3, 3)

    ## step.2 compute matrix A
    # compute 3D points xyz
    points = get_points_coordinate(depth_torch, intrinsic_inv_torch)
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # An = b
    matrix_a = point_matrix.view(B, 3, 25, H, W)  # (B, 3, 25, HxW)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, HxW, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([B, H, W, 25, 1], device=depth_torch.device)

    # dot(A.T, A)
    point_multi = torch.matmul(matrix_a_trans, matrix_a)
    matrix_deter = torch.det(point_multi)
    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
    # diag matrix to update uninverse
    diag_constant = torch.ones([3], dtype=torch.float32, device=depth_torch.device)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(1, H, W, 1, 1)
    # inversible matrix
    inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
    inv_matrix = torch.inverse(inversible_matrix)

    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm.squeeze(-1), p=2, dim=3).permute(0, 3, 1, 2) #b 3 h w

    return norm_normalize

def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color

def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1]) * interval_scale
    return intrinsics, extrinsics, depth_min, depth_interval

def prepare_img(hr_img):
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

        #downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        #crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        # #downsample
        # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img_crop

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_depth_hr(filename):
    # read pfm depth file
    #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
    depth_lr = np.array(read_pfm(filename)[0], dtype=np.float32)
    # depth_lr = prepare_img(depth_lr)
    
    h, w = depth_lr.shape
    depth_lr_ms = {
        "stage1": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
        "stage2": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
        "stage3": depth_lr,
    }

    return depth_lr_ms

def read_depth(filename):
    # read pfm depth file
    #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
    depth_lr = np.array(read_pfm(filename)[0], dtype=np.float32)

    return depth_lr

def read_mask_hr(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32)
    np_img = (np_img > 10).astype(np.float32)
    # np_img = prepare_img(np_img)

    h, w = np_img.shape
    np_img_ms = {
        "stage1": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
        "stage2": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
        "stage3": np_img,
    }

    return np_img_ms

def save_normal_map(normal_map, file_path):
    """
    将法线图保存为图像文件。
    :param normal_map: (3, H, W) 法线图，值范围在 [-1, 1]
    :param file_path: 输出文件路径（如 normal_map.png）
    """
    # 确保输入是 NumPy 数组
    normal_map = np.asarray(normal_map)

    # 转换形状为 (H, W, 3)
    normal_map = np.transpose(normal_map, (1, 2, 0))  # (3, H, W) -> (H, W, 3)

    # 将范围从 [-1, 1] 映射到 [0, 255]
    normal_map = ((normal_map + 1) / 2.0 * 255.0).astype(np.uint8)

    # 保存图像
    cv2.imwrite(file_path, normal_map)

def compute_optimal_rotation(pred_normals, gt_normals, mask):
    # 提取有效法线
    valid_pred = pred_normals[mask]
    valid_gt = gt_normals[mask]
    
    # 计算协方差矩阵 H
    H = valid_pred.T @ valid_gt
    
    # SVD 分解
    U, _, Vt = np.linalg.svd(H)
    
    # 构造旋转矩阵 R
    R = Vt.T @ U.T
    
    # 确保 R 是有效的旋转矩阵（防止反射）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    return R

def cosine_similarity(normals1, normals2):
    dot_product = np.sum(normals1 * normals2, axis=-1)
    norm1 = np.linalg.norm(normals1, axis=-1)
    norm2 = np.linalg.norm(normals2, axis=-1)
    return dot_product / (norm1 * norm2)

def read_normal(filename):
    normal = np.load(filename)#3 h w
    normal = (normal - 0.5)*2
    return normal

from kornia.filters import spatial_gradient
def estimate_normals(depth, normal_gain):
    depth = depth.unsqueeze(0).unsqueeze(0)
    xy_gradients = spatial_gradient(normal_gain*depth, mode='diff', order=1, normalized=False).squeeze(1) # B 2 H W
    normals = torch.cat([xy_gradients, torch.ones_like(xy_gradients[:,0:1])], 1) # B 3 H W
    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)
    return normals

def prepare_img(hr_img):
    #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
    #downsample
    h, w, _ = hr_img.shape
    hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    #crop
    h, w, _ = hr_img_ds.shape
    target_h, target_w = 512, 640
    start_h, start_w = (h - target_h)//2, (w - target_w)//2
    hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

    # #downsample
    # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

    return hr_img_crop


scan_list = []

list_path = 'path/to/lists/dtu/test.txt'
train_mode = False 

with open(list_path, 'r') as file:
    for line in file:
        scan_list.append(line.strip())


datapath = 'path/to/DTU/mvs_training/dtu'
output_folder = 'path/to/GT_Normal'
light_idx = 3

for scan in scan_list:
    if train_mode:
        dir_path = output_folder + f'/{scan}_train'
    else:
        dir_path = output_folder + f'/{scan}'

    os.makedirs(dir_path, exist_ok=True)

    print('Processing ' + scan)
    
    for vid in range(0, 49):
        # img_filename = os.path.join(datapath, 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
                    
        depth_filename_hr = os.path.join(datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
        proj_mat_filename = os.path.join(datapath, 'Cameras/{:0>8}_cam.txt').format(vid)

        intrinsics, extrinsics, depth_min, depth_interval = read_cam_file(proj_mat_filename)
        depth = torch.tensor(read_depth_hr(depth_filename_hr)['stage3'])
        intrinsics = torch.Tensor(intrinsics)

        normal_npy = depth2normal(depth.unsqueeze(0), intrinsics).squeeze(0)
        normal_npy = 1 - np.array(normal_npy.permute(1, 2, 0) / 2 + 0.5)

        if train_mode:
            normal_npy = prepare_img(normal_npy)
        
        output_color = Image.fromarray((normal_npy * 255).astype(np.uint8))

        save_path = dir_path + f"/{(vid):06d}_normal.npy"
        color_path = dir_path + f"/{(vid):06d}_normal.png"

        np.save(save_path, normal_npy)
        output_color.save(os.path.join(color_path))
