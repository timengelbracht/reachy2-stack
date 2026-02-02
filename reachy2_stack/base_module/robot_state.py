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

    All camera poses in odom frame are computed as:
        T_odom_cam = T_odom_base @ T_base_cam
    """

    timestamp: float

    # Odometry
    odom_x: float = 0.0
    odom_y: float = 0.0
    odom_theta: float = 0.0  # degrees

    # Depth camera
    rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32 meters
    depth_intrinsics: Optional[np.ndarray] = None  # (3, 3) K matrix
    depth_extrinsics: Optional[np.ndarray] = None  # T_base_cam (4, 4)

    # Teleop left camera
    teleop_left: Optional[np.ndarray] = None  # (H, W, 3) uint8
    teleop_left_intrinsics: Optional[np.ndarray] = None  # (3, 3) K matrix
    teleop_left_extrinsics: Optional[np.ndarray] = None  # T_base_cam (4, 4)

    # Teleop right camera
    teleop_right: Optional[np.ndarray] = None  # (H, W, 3) uint8
    teleop_right_intrinsics: Optional[np.ndarray] = None  # (3, 3) K matrix
    teleop_right_extrinsics: Optional[np.ndarray] = None  # T_base_cam (4, 4)

    # Computed pose (if using world frame via visual localization)
    T_world_base: Optional[np.ndarray] = None  # (4, 4)
    T_world_cam: Optional[np.ndarray] = None  # (4, 4) - camera pose from localization

    # Backward compatibility alias
    @property
    def intrinsics(self) -> Optional[np.ndarray]:
        """Alias for depth_intrinsics (backward compatibility)."""
        return self.depth_intrinsics

    @intrinsics.setter
    def intrinsics(self, value: Optional[np.ndarray]) -> None:
        """Alias for depth_intrinsics (backward compatibility)."""
        self.depth_intrinsics = value

    def get_T_odom_base(self) -> np.ndarray:
        """Get 4x4 transform from odometry (SE(2) embedded in SE(3))."""
        from reachy2_stack.utils.utils_poses import xytheta_to_T

        return xytheta_to_T(self.odom_x, self.odom_y, self.odom_theta)

    def get_T_world_base(self) -> np.ndarray:
        """Get best available pose (world frame if available, else odom)."""
        if self.T_world_base is not None:
            return self.T_world_base
        return self.get_T_odom_base()

    def get_T_odom_depth_cam(self) -> Optional[np.ndarray]:
        """Get depth camera pose in odom frame.

        Returns:
            T_odom_cam = T_odom_base @ T_base_cam, or None if extrinsics unavailable
        """
        if self.depth_extrinsics is None:
            return None
        return self.get_T_world_base() @ self.depth_extrinsics

    def get_T_odom_teleop_left(self) -> Optional[np.ndarray]:
        """Get teleop left camera pose in odom frame.

        Returns:
            T_odom_cam = T_odom_base @ T_base_cam, or None if extrinsics unavailable
        """
        if self.teleop_left_extrinsics is None:
            return None
        return self.get_T_world_base() @ self.teleop_left_extrinsics

    def get_T_odom_teleop_right(self) -> Optional[np.ndarray]:
        """Get teleop right camera pose in odom frame.

        Returns:
            T_odom_cam = T_odom_base @ T_base_cam, or None if extrinsics unavailable
        """
        if self.teleop_right_extrinsics is None:
            return None
        return self.get_T_world_base() @ self.teleop_right_extrinsics

    def has_depth(self) -> bool:
        """Check if depth camera data is available."""
        return self.rgb is not None and self.depth is not None

    def has_teleop(self) -> bool:
        """Check if teleop camera data is available."""
        return self.teleop_left is not None and self.teleop_right is not None

    def has_depth_calibration(self) -> bool:
        """Check if depth camera has both intrinsics and extrinsics."""
        return self.depth_intrinsics is not None and self.depth_extrinsics is not None

    def has_teleop_left_calibration(self) -> bool:
        """Check if teleop left camera has both intrinsics and extrinsics."""
        return self.teleop_left_intrinsics is not None and self.teleop_left_extrinsics is not None

    def has_teleop_right_calibration(self) -> bool:
        """Check if teleop right camera has both intrinsics and extrinsics."""
        return self.teleop_right_intrinsics is not None and self.teleop_right_extrinsics is not None
