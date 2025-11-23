import os
import cv2
import numpy as np
from pathlib import Path

def load_image(image_path: Path | str) -> np.ndarray | None:
    """Load an image from the filesystem.

    Args:
        image_path (Path | str): Path to the image file.

    Returns:
        np.ndarray | None: Loaded image in BGR color order as a NumPy array,
        or None if the file does not exist or cannot be read.
    """
    path = Path(image_path)
    if not path.exists():
        print(f"Image not found at {path}.")
        return None

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image at {path}.")
        return None

    return image

def write_depth_vis(path: Path, depth: np.ndarray) -> None:
    # Mask out invalid (zero or NaN) values
    valid_mask = (depth > 0) & np.isfinite(depth)
    if not np.any(valid_mask):
        print(f"Warning: No valid depth values to visualize in {path}")
        return

    # Compute per-image min/max for normalization
    min_depth = np.min(depth[valid_mask])
    max_depth = np.max(depth[valid_mask])
    
    # Normalize to [0, 255] and convert to uint8
    depth_vis = np.clip(depth, min_depth, max_depth)
    depth_vis = ((depth_vis - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Save
    cv2.imwrite(str(path), depth_colored)