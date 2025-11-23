#!/usr/bin/env python3

"""
Minimal SAM3 test script.
Just edit the variables below and run:

    python test_sam3_minimal.py
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

from reachy2_stack.perception.segmentation_detection.sam3_segmenter import (
    Sam3Segmenter,
    Sam3Config,
)

# ================================================================
# EDIT THESE
# ================================================================
IMAGE_PATH = "/exchange/data/hloc/mlhall_processed_leica/rgb/027.jpg"       # change this
TEXT_PROMPT = "black cabinet handle"                        # change this
DEVICE = "cuda"                               # "cuda" or "cpu"
SCORE_THRESHOLD = 0.5
MASK_ALPHA = 0.5
# ================================================================


def overlay_masks_and_boxes(image: Image.Image, results, alpha: float = 0.5):
    """
    Draw semi-transparent masks + bounding boxes over an image.
    """
    image = image.convert("RGBA")
    W, H = image.size

    out = image.copy()

    # deterministic color generator
    rng = np.random.default_rng(0)

    for r in results:
        mask = r["mask"]
        bbox = r["bbox"]
        score = r["score"]
        label = r["label"]

        # random color per mask
        color_rgb = rng.integers(0, 255, size=3, dtype=np.uint8).tolist()
        color = (*color_rgb, 0)

        # convert mask to uint8
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        color_layer = Image.new("RGBA", image.size, color)

        # alpha channel proportional to mask
        alpha_mask = mask_img.point(lambda v: int(v * alpha))
        color_layer.putalpha(alpha_mask)

        out = Image.alpha_composite(out, color_layer)

        # bounding box
        draw = ImageDraw.Draw(out)
        x0, y0, x1, y1 = map(int, bbox)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=2)

        # label + score
        txt = f"{label} {score:.2f}" if label else f"{score:.2f}"
        draw.text((x0 + 4, y0 + 4), txt, fill=(255, 255, 255, 255))

    return out


def main():
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image does not exist: {img_path}")

    print(f"[INFO] Loading image: {img_path}")
    pil_img = Image.open(img_path).convert("RGB")
    rgb = np.asarray(pil_img)

    print("[INFO] Initialising SAM3...")
    cfg = Sam3Config(
        model_name="facebook/sam3",
        device=DEVICE,
        score_threshold=SCORE_THRESHOLD,
        mask_threshold=0.5,
    )
    segmenter = Sam3Segmenter(cfg)

    print(f"[INFO] Running segmentation with text prompt: '{TEXT_PROMPT}'")
    results = segmenter.segment_with_text(rgb, TEXT_PROMPT)
    print(f"[INFO] Found {len(results)} objects")

    for i, r in enumerate(results):
        print(f"  - Instance {i}: score={r['score']:.3f}, bbox={r['bbox']}")

    # visualization
    vis = overlay_masks_and_boxes(pil_img, results, alpha=MASK_ALPHA)

    out_path = img_path.with_name(
        f"{img_path.stem}_sam3_{TEXT_PROMPT.replace(' ', '_')}{img_path.suffix}"
    )

    vis.save(out_path)
    print(f"[INFO] Visualization saved to: {out_path}")


if __name__ == "__main__":
    main()
