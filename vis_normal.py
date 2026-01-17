#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def to_uint8_vis(normal_01: np.ndarray) -> np.ndarray:
    """
    normal_01: float array in [0,1], shape (H, W, 3)
    return: uint8 RGB image, shape (H, W, 3)
    """
    if normal_01.ndim != 3 or normal_01.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {normal_01.shape}")

    vis = (normal_01 * 255.0).clip(0, 255).astype(np.uint8)
    return vis


def load_normal_npy(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path))
    # Handle occasional (3, H, W) layout
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{npy_path} has invalid shape: {arr.shape}")
    return arr.astype(np.float32)


def invalid_mask_from_normal01(normal_01: np.ndarray, atol=1e-3):
    """
    Invalid pixels are those close to (0.5, 0.5, 0.5) in [0,1] space,
    or containing NaN / Inf.
    """
    invalid = (
        np.all(np.isfinite(normal_01), axis=2)
        & np.all(np.abs(normal_01 - 0.5) <= atol, axis=2)
    )
    invalid |= ~np.all(np.isfinite(normal_01), axis=2)
    return invalid  # (H, W) bool


def invalid_mask_by_norm(normal_01: np.ndarray, eps=1e-6):
    """
    Convert normals back to [-1, 1] and mark invalid pixels by:
    - non-finite values
    - vector length too small
    """
    n = normal_01 * 2.0 - 1.0
    finite = np.all(np.isfinite(n), axis=2)
    length = np.linalg.norm(n, axis=2)
    invalid = (~finite) | (length < eps)
    return invalid


def process_one(
    npy_path: Path,
    out_path: Path,
    save_mask: bool = True,
) -> None:
    normal_01 = load_normal_npy(npy_path)

    vis = to_uint8_vis(normal_01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(str(out_path))

    if save_mask:
        invalid = invalid_mask_by_norm(normal_01, eps=1e-6)
        mask_path = out_path.with_name(out_path.stem + "_invalid_mask.png")
        Image.fromarray((invalid.astype(np.uint8) * 255)).save(str(mask_path))


def collect_npy_files(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        return sorted(inp.rglob("*.npy"))
    raise FileNotFoundError(f"Input path not found: {inp}")


def visualize_normals_per_scene(
    scenes_root: Path,
    npy_suffix: str = ".npy",
    vis_suffix: str = ".png",
    recursive: bool = True,
    save_mask: bool = True,
) -> None:
    """
    scenes_root/
      ├── scene_001/
      ├── scene_002/
      └── ...
    """
    scenes_root = Path(scenes_root)
    if not scenes_root.is_dir():
        raise NotADirectoryError(f"Not a directory: {scenes_root}")

    scene_dirs = [p for p in scenes_root.iterdir() if p.is_dir()]
    if len(scene_dirs) == 0:
        print(f"[WARN] No scene directories under: {scenes_root}")
        return

    for scene_dir in sorted(scene_dirs):
        print(f"\n[Scene] {scene_dir.name}")

        if recursive:
            npy_files = sorted(scene_dir.rglob(f"*{npy_suffix}"))
        else:
            npy_files = sorted(scene_dir.glob(f"*{npy_suffix}"))

        if len(npy_files) == 0:
            print("  [WARN] No npy files found")
            continue

        for npy_path in npy_files:
            out_path = npy_path.with_suffix(vis_suffix)

            if out_path.exists():
                print(f"  [SKIP] {out_path.name}")
                continue

            try:
                process_one(npy_path, out_path, save_mask=save_mask)
                print(f"  [OK] {npy_path.name}")
            except Exception as e:
                print(f"  [ERR] {npy_path}: {e}")


def main():
    """
    Visualize a single npy file or a directory of npy files,
    and save results to a unified output directory.
    """
    parser = argparse.ArgumentParser(
        description="Visualize normal .npy ([0,1]) to PNG"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to a .npy file or directory"
    )
    parser.add_argument(
        "--out_dir", "-o", type=str, default="vis_out",
        help="Output directory"
    )
    parser.add_argument(
        "--keep_tree", action="store_true",
        help="Keep directory structure under out_dir"
    )
    parser.add_argument(
        "--save_mask", action="store_true",
        help="Save invalid normal mask"
    )

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
            process_one(
                npy_path,
                out_path,
                save_mask=args.save_mask
            )
            print(f"[OK] {npy_path} -> {out_path}")
        except Exception as e:
            print(f"[ERR] {npy_path}: {e}")


def main_all():
    """
    scenes_root/scene_xxx/*.npy → scene_xxx/*.png (in-place visualization)
    """
    parser = argparse.ArgumentParser(
        description="Visualize normal npy per scene (in-place)"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Scenes root directory"
    )
    parser.add_argument(
        "--save_mask", action="store_true",
        help="Save invalid normal mask"
    )

    args = parser.parse_args()
    inp = Path(args.input)

    visualize_normals_per_scene(
        inp,
        save_mask=args.save_mask
    )


if __name__ == "__main__":
    # main(): visualize a single file or a directory into a unified output folder
    # main_all(): generate visualization images in-place for all scenes
    main_all()
