"""Map state dataclass for mapping results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .robot_state import RobotState


@dataclass
class MapState:
    """Map state containing mapping results from wavemap server.

    This dataclass holds the output from the mapping loop:
    - Occupied voxels (points with colors)
    - Free voxels (points without colors)
    - Reference to source RobotState for odometry/trajectory

    All points are in the odom/world frame.
    """

    timestamp: float

    # Occupied voxels (from wavemap)
    occupied_points: Optional[np.ndarray] = None  # (N, 3) float32, world frame
    occupied_colors: Optional[np.ndarray] = None  # (N, 3) uint8

    # Free voxels (from wavemap)
    free_points: Optional[np.ndarray] = None  # (M, 3) float32, world frame

    # Source robot state (for odometry/trajectory visualization)
    robot_state: Optional["RobotState"] = None

    def has_map(self) -> bool:
        """Check if map data is available."""
        return self.occupied_points is not None and len(self.occupied_points) > 0

    def has_free_space(self) -> bool:
        """Check if free space data is available."""
        return self.free_points is not None and len(self.free_points) > 0

    def num_occupied(self) -> int:
        """Get number of occupied voxels."""
        if self.occupied_points is None:
            return 0
        return len(self.occupied_points)

    def num_free(self) -> int:
        """Get number of free voxels."""
        if self.free_points is None:
            return 0
        return len(self.free_points)

    def get_odom_pose(self) -> Optional[np.ndarray]:
        """Get odometry pose from source robot state.

        Returns:
            4x4 transform T_odom_base, or None if robot_state is unavailable
        """
        if self.robot_state is None:
            return None
        return self.robot_state.get_T_odom_base()
