#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def to_uint8_vis(normal_01: np.ndarray) -> np.ndarray:
    """
    normal_01: float array in [0,1], shape (H,W,3)
    return: uint8 RGB image, shape (H,W,3)
    """
    if normal_01.ndim != 3 or normal_01.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {normal_01.shape}")

    vis = (normal_01 * 255.0).clip(0, 255).astype(np.uint8)
    return vis


def load_normal_npy(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path))
    # 兼容偶尔保存成 (3,H,W) 的情况
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{npy_path} has invalid shape: {arr.shape}")
    return arr.astype(np.float32)


def process_one(npy_path: Path, out_path: Path) -> None:
    normal_01 = load_normal_npy(npy_path)

    # 你保存时就是 (normal + 1)/2，所以这里直接乘 255 可视化即可
    vis = to_uint8_vis(normal_01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(str(out_path))

    # 生成无效区域 mask（推荐用 norm 法）
    invalid = invalid_mask_by_norm(normal_01, eps=1e-6)
    mask_path = out_path.with_name(out_path.stem + "_invalid_mask.png")
    Image.fromarray((invalid.astype(np.uint8) * 255)).save(str(mask_path))


def collect_npy_files(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        return sorted(inp.rglob("*.npy"))
    raise FileNotFoundError(f"Input path not found: {inp}")


def invalid_mask_from_normal01(normal_01: np.ndarray, atol=1e-3):
    # 无效：接近 (0.5,0.5,0.5)
    invalid = np.all(np.isfinite(normal_01), axis=2) & np.all(np.abs(normal_01 - 0.5) <= atol, axis=2)
    # 如果有 NaN/Inf，也算无效
    invalid |= ~np.all(np.isfinite(normal_01), axis=2)
    return invalid  # (H,W) bool


def invalid_mask_by_norm(normal_01: np.ndarray, eps=1e-6):
    # 还原到 [-1,1]
    n = normal_01 * 2.0 - 1.0
    # 无效：非有限 或 向量长度过小
    finite = np.all(np.isfinite(n), axis=2)
    length = np.linalg.norm(n, axis=2)
    invalid = (~finite) | (length < eps)
    return invalid


def main():
    parser = argparse.ArgumentParser(description="Load normal .npy ([0,1]) and visualize as PNG.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to a .npy file or a directory containing .npy files")
    parser.add_argument("--out_dir", "-o", type=str, default="vis_out",
                        help="Output directory for png files (default: vis_out)")
    parser.add_argument("--keep_tree", action="store_true",
                        help="If input is a directory, keep relative folder structure under out_dir")
    args = parser.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = collect_npy_files(inp)
    if len(npy_files) == 0:
        print(f"[WARN] No .npy found under: {inp}")
        return

    base_root = inp if inp.is_dir() else inp.parent

    for npy_path in npy_files:
        if args.keep_tree and inp.is_dir():
            rel = npy_path.relative_to(base_root)
            out_path = out_dir / rel.with_suffix(".png")
        else:
            out_path = out_dir / (npy_path.stem + ".png")

        try:
            process_one(npy_path, out_path)
            print(f"[OK] {npy_path} -> {out_path}")
        except Exception as e:
            print(f"[ERR] {npy_path}: {e}")


if __name__ == "__main__":
    main()
