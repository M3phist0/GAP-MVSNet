import os
import argparse
from pathlib import Path

import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

def read_cam_file(filename, interval_scale=1.06):
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

def fov_from_intrinsics(K, width, height, degrees=True):
    fx = float(K[0, 0])
    fy = float(K[1, 1])

    fov_x = 2.0 * np.arctan(width  / (2.0 * fx))
    fov_y = 2.0 * np.arctan(height / (2.0 * fy))

    if degrees:
        fov_x = np.degrees(fov_x)
        fov_y = np.degrees(fov_y)

    return fov_x, fov_y

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_scans() -> list[str]:
    scans = [
            "advanced/Auditorium",
            "advanced/Ballroom",
            "advanced/Courtroom",
            "advanced/Museum",
            "advanced/Palace", 
            "advanced/Temple" 
        ] + [
            "intermediate/Family",
            "intermediate/Francis",
            "intermediate/Horse",
            "intermediate/Lighthouse",
            "intermediate/M60",
            "intermediate/Panther",
            "intermediate/Playground",
            "intermediate/Train",
        ]
    return scans


def read_rgb_image_to_tensor(img_path: str, device: torch.device) -> torch.Tensor:
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # HWC -> CHW, float32 in [0,1]
    tensor = torch.from_numpy(img_rgb).to(device=device, dtype=torch.float32) / 255.0
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor


def normal_to_vis(normal: np.ndarray) -> np.ndarray:
    """
    normal: float32, shape (H, W, 3) or (3, H, W), value range assumed [-1,1]
    return: uint8 RGB image (H, W, 3)
    """
    if normal.ndim != 3:
        raise ValueError(f"normal must be 3D, got shape={normal.shape}")

    if normal.shape[0] == 3 and normal.shape[2] != 3:
        # (3,H,W) -> (H,W,3)
        normal = np.transpose(normal, (1, 2, 0))

    vis = ((normal + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return vis


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA not available, running on CPU (will be slow).")

    model = MoGeModel.from_pretrained(args.pt_path).to(device)
    model.eval()

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    scans = load_scans()
    pbar = tqdm(desc="Infer normals tnt all", dynamic_ncols=True, unit="img")

    for scan in scans:
        out_scan_dir = out_root / f"{scan}"
        ensure_dir(out_scan_dir)

        img_folder = os.path.join(args.datapath, f"{scan}/blended_images")
        cam_folder = os.path.join(args.datapath, f"{scan}/cams")

        for name in os.listdir(img_folder):
            img_filename = os.path.join(img_folder, name)

            # 1. 后缀过滤（防止非图片）
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                warnings.warn(f"Skip non-image file: {img_folder}/blended_images/{name}")
                pbar.update(1)
                continue

            base = f"{int(name[:-4]):06d}"
            npy_path = out_scan_dir / f"{base}_normal.npy"
            png_path = out_scan_dir / f"{base}_normal.png"

            # 2. 若结果已存在，直接跳过
            if npy_path.exists() and (not args.visualize or png_path.exists()):
                pbar.update(1)
                continue

            cam_filename = os.path.join(cam_folder, f"{name[:-4]}_cam.txt")

            try:
                # 3. 读取图片
                input_image = read_rgb_image_to_tensor(img_filename, device=device)
                _, H, W = input_image.shape

                # 4. 读取相机参数
                intrinsics, _, _, _ = read_cam_file(cam_filename)
                fov_x, _ = fov_from_intrinsics(intrinsics, W, H)

                # 5. 模型推理
                output = model.infer(input_image, fov_x=float(fov_x))
                output_normal = output["normal"]

            except (OSError, ValueError, RuntimeError) as e:
                warnings.warn(f"Failed to process {name}: {e}")
                pbar.update(1)
                continue

            # ===== 正常后处理 =====
            normal = output_normal.detach().float().cpu().numpy()
            if normal.ndim == 3 and normal.shape[0] == 3:
                normal = np.transpose(normal, (1, 2, 0))

            normal_npy = (normal + 1.0) / 2.0  # [0,1]

            np.save(str(npy_path), normal_npy.astype(np.float32))

            if args.visualize:
                vis = (normal_npy * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(vis).save(str(png_path))

            pbar.update(1)

    pbar.close()

    print(f"Done. Outputs saved to: {out_root.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MoGeModel normal inference on DTU images with progress bar and structured outputs."
    )
    parser.add_argument("--pt_path", type=str, default="models/moge-2-vitl-normal.pt")
    parser.add_argument("--listfile", type=str, default="/root/gpufree-data/tanksandtemples")
    parser.add_argument("--datapath", type=str, default="/root/gpufree-data/tanksandtemples")
    parser.add_argument("--out_dir", "-o", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=49)
    parser.add_argument("--visualize", action="store_true", help="If set, save visualization images")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
