from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

@dataclass
class Observation:
    """Standard observation that *all* policies see."""

    # --- Robot state ---
    joint_positions: np.ndarray         # shape [n_joints]
    joint_velocities: np.ndarray        # shape [n_joints]
    gripper_opening: float              # 0..1
    base_pose_world: np.ndarray         # 4x4 SE(3) homogeneous
    ee_pose_world: np.ndarray           # 4x4 SE(3)

    # --- Perception ---
    rgb: np.ndarray                     # HxWx3, uint8
    depth: Optional[np.ndarray] = None  # HxW, float32 (meters)

    # Arbitrary detection outputs, already in world coordinates where relevant
    detections: Dict[str, Any] = field(default_factory=dict)

    # --- Meta ---
    timestep: float = 0.0               # seconds since episode start
    step_idx: int = 0                   # discrete step counter






