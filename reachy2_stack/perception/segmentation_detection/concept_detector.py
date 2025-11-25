# reachy2_stack/perception/detection/detector_sam3_concepts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel
from reachy2_stack.perception.segmentation_detection.sam3_segmenter import Sam3Config, Sam3Segmenter
from reachy2_stack.utils.utils_dataclass import ConceptDetection

class Sam3ConceptDetector:
    """Bridge between Reachy (RGBD + poses) and SAM3 segmenter."""

    def __init__(
        self,
        sam_cfg: Sam3Config,
        camera_id: str = "depth_rgb",
    ):
        self.segmenter = Sam3Segmenter(sam_cfg)
        self.camera_id = camera_id

    def detect_concept_world(
        self,
        client: ReachyClient,
        world: WorldModel,
        prompt: str,
        max_instances: int = 10,
    ) -> List[ConceptDetection]:
        """Run SAM3 on the chosen camera and lift results to world frame.

        Returns a list of ConceptDetection, each with a 3D 'position_world'.
        """
        # 1) Grab RGBD
        if self.camera_id == "depth_rgb":
            rgb, depth = client.get_depth_rgbd()
        elif self.camera_id == "teleop_left":
            rgb = client.get_teleop_rgb_left()
            depth = None
        else:
            raise ValueError(f"Unsupported camera_id: {self.camera_id}")

        if rgb is None:
            return []

        # 2) Get intrinsics + T_world_cam
        K = world.get_intrinsics(self.camera_id)
        T_world_cam = world.get_T_world_cam(self.camera_id)
        if K is None or T_world_cam is None:
            return []

        # 3) Run SAM3 to get 2D masks
        segs = self.segmenter.segment_with_text(rgb, prompt)
        segs = sorted(segs, key=lambda s: s["score"], reverse=True)[:max_instances]

        detections: List[ConceptDetection] = []
        for seg in segs:
            mask = seg["mask"]
            score = float(seg["score"])
            bbox = tuple(int(x) for x in seg["bbox"])
            label = str(seg.get("label", prompt))

            # 4) Compute a 3D point in world coords
            pos_world = self._mask_to_world_point(mask, depth, K, T_world_cam)

            detections.append(
                ConceptDetection(
                    label=label,
                    mask=mask,
                    score=score,
                    bbox=bbox,
                    position_world=pos_world,
                )
            )

        return detections

    @staticmethod
    def _mask_to_world_point(
        mask: np.ndarray,
        depth: np.ndarray | None,
        K: np.ndarray,
        T_world_cam: np.ndarray,
    ) -> np.ndarray:
        """Compute a representative 3D point in world frame from a mask.

        Simple strategy:
          - Take mask centroid in pixels.
          - Use mean depth inside mask (if available).
          - Project to 3D in camera frame.
          - Transform to world via T_world_cam.
        """
        H, W = mask.shape
        ys, xs = np.where(mask)
        if xs.size == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=float)

        u = float(xs.mean())
        v = float(ys.mean())

        if depth is not None:
            depth_vals = depth[ys, xs]
            depth_vals = depth_vals[np.isfinite(depth_vals)]
            if depth_vals.size > 0:
                z = float(np.median(depth_vals))
            else:
                z = 1.0  # fallback
        else:
            z = 1.0  # fallback if no depth

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        p_cam = np.array([x_cam, y_cam, z, 1.0], dtype=float)

        p_world_h = T_world_cam @ p_cam
        return p_world_h[:3]
