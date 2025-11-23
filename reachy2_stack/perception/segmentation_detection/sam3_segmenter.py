from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model


@dataclass
class Sam3Config:
    """Configuration for the SAM3 segmenter."""
    model_name: str = "facebook/sam3"
    device: str = "cuda"  # "cuda" or "cpu"
    score_threshold: float = 0.5
    mask_threshold: float = 0.5


class Sam3Segmenter:
    """
    Thin wrapper around Hugging Face SAM3 (facebook/sam3).

    - Lives in perception/segmentation.
    - Operates ONLY on images (no Reachy, no world frame).
    - Main API: `segment_with_text(rgb, prompt)`.

    Returns a list of dicts:
        {
            "mask": np.ndarray[H, W] bool,
            "bbox": (x_min, y_min, x_max, y_max),
            "score": float,
            "label": str,  # the prompt
        }
    """

    def __init__(self, cfg: Sam3Config):
        self.cfg = cfg

        if cfg.device == "cuda" and not torch.cuda.is_available():
            print("[Sam3Segmenter] CUDA requested but not available, falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)
            dtype = "float16" if self.device.type == "cuda" else "float32"

        # Load HF SAM3 model + processor
        # self.model = Sam3Model.from_pretrained(cfg.model_name).to(self.device)
        # self.processor = Sam3Processor.from_pretrained(cfg.model_name)
        self.model = Sam3Model.from_pretrained(
            cfg.model_name,
            local_files_only=True,
            torch_dtype=dtype
        ).to(self.device)

        self.processor = Sam3Processor.from_pretrained(
            cfg.model_name,
            local_files_only=True,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def segment_with_text(
        self,
        rgb: np.ndarray,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Segment all instances matching a text concept in an RGB image.

        Args:
            rgb:  np.ndarray [H, W, 3], uint8 or float in [0,1].
            prompt: e.g. "handle", "switch", "drawer", ...

        Returns:
            List[dict] with keys: "mask", "bbox", "score", "label".
        """
        image = self._to_pil(rgb)

        # Build inputs (single image, single text prompt)
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to original resolution (HF helper)
        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.cfg.score_threshold,
            mask_threshold=self.cfg.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),  # [[H, W]]
        )[0]

        masks = processed["masks"]      # [N, H, W] (torch.bool / uint8)
        boxes = processed["boxes"]      # [N, 4] (xyxy)
        scores = processed["scores"]    # [N]

        results: List[Dict[str, Any]] = []
        for mask_t, box_t, score_t in zip(masks, boxes, scores):
            mask_np = mask_t.cpu().numpy().astype(bool)
            box_np = tuple(float(x) for x in box_t.cpu().tolist())
            score_f = float(score_t.cpu().item())

            results.append(
                {
                    "mask": mask_np,
                    "bbox": box_np,      # (x_min, y_min, x_max, y_max)
                    "score": score_f,
                    "label": prompt,
                }
            )

        return results

    # Optional: automatic mask generation without text
    def segment_auto(
        self,
        rgb: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Automatic mask generation (no text prompt).
        Uses SAM3's semantic head by passing no text.
        """
        image = self._to_pil(rgb)
        inputs = self.processor(
            images=image,
            text=None,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.cfg.score_threshold,
            mask_threshold=self.cfg.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = processed["masks"]
        boxes = processed["boxes"]
        scores = processed["scores"]

        results: List[Dict[str, Any]] = []
        for mask_t, box_t, score_t in zip(masks, boxes, scores):
            mask_np = mask_t.cpu().numpy().astype(bool)
            box_np = tuple(float(x) for x in box_t.cpu().tolist())
            score_f = float(score_t.cpu().item())

            results.append(
                {
                    "mask": mask_np,
                    "bbox": box_np,
                    "score": score_f,
                    "label": None,
                }
            )

        return results

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_pil(rgb: np.ndarray) -> Image.Image:
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image [H, W, 3], got shape {rgb.shape}")
        return Image.fromarray(rgb)
