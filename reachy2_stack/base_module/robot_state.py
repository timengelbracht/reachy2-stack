"""Unified robot state containing all sensor data."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RobotState:
    """Unified robot state containing all sensor data.

    This dataclass combines camera and odometry data into a single
    timestamped state that can be queued and processed by various
    modules (mapping, navigation, visualization).
    """

    timestamp: float

    # Odometry
    odom_x: float = 0.0
    odom_y: float = 0.0
    odom_theta: float = 0.0  # degrees

    # Camera (depth)
    rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32 meters
    intrinsics: Optional[np.ndarray] = None  # (3, 3) K matrix

    # Camera (teleop) - optional
    teleop_left: Optional[np.ndarray] = None
    teleop_right: Optional[np.ndarray] = None

    # Computed pose (if using world frame via visual localization)
    T_world_base: Optional[np.ndarray] = None  # (4, 4)

    def get_T_odom_base(self) -> np.ndarray:
        """Get 4x4 transform from odometry (SE(2) embedded in SE(3))."""
        from reachy2_stack.utils.utils_poses import xytheta_to_T

        return xytheta_to_T(self.odom_x, self.odom_y, self.odom_theta)

    def get_pose(self) -> np.ndarray:
        """Get best available pose (world frame if available, else odom)."""
        if self.T_world_base is not None:
            return self.T_world_base
        return self.get_T_odom_base()

    def has_depth(self) -> bool:
        """Check if depth camera data is available."""
        return self.rgb is not None and self.depth is not None

    def has_teleop(self) -> bool:
        """Check if teleop camera data is available."""
        return self.teleop_left is not None and self.teleop_right is not None
