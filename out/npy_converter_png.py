#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import cv2


def npy_depth_to_16bit_png(npy_path: Path, png_path: Path) -> None:
    """
    Converts a single .npy depth map to 16-bit PNG.

    Assumes input is float depth in meters (common).
    If your .npy is already in millimeters as integers, it will still work (it will detect large values).
    Invalid values (nan/inf/negative) -> 0.
    """
    depth = np.load(str(npy_path))

    if depth.ndim != 2:
        raise ValueError(f"{npy_path.name}: expected HxW array, got shape {depth.shape}")

    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0

    mx = float(np.max(depth)) if depth.size else 0.0
    if mx < 1000.0:
        depth_mm = np.round(depth * 1000.0)
    else:
        depth_mm = depth

    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(png_path), depth_mm)
    if not ok:
        raise RuntimeError(f"Failed to write {png_path}")


def convert_folder(depth_npy_dir: str, out_dir: str | None = None) -> None:
    in_dir = Path(depth_npy_dir).expanduser().resolve()
    if out_dir is None:
        out_dir = str(in_dir.parent / "depth_png")
    out_dir = Path(out_dir).expanduser().resolve()

    npy_files = sorted(in_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in: {in_dir}")

    print(f"Input : {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Found : {len(npy_files)} files")

    n_ok = 0
    for i, npy_path in enumerate(npy_files, 1):
        png_path = out_dir / (npy_path.stem + ".png")
        try:
            npy_depth_to_16bit_png(npy_path, png_path)
            n_ok += 1
        except Exception as e:
            print(f"[FAIL] {npy_path.name}: {e}")

        if i % 100 == 0:
            print(f"... {i}/{len(npy_files)} done")

    print(f"Done. Converted {n_ok}/{len(npy_files)} files.")


if __name__ == "__main__":
    DEPTH_NPY_DIR = "/exchange/out/run_20260129_173750/depth"

    OUT_DIR = "/exchange/out/pngs"

    convert_folder(DEPTH_NPY_DIR, OUT_DIR)