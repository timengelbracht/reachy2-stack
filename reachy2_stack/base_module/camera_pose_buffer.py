"""Camera pose buffer for delayed localization fusion.

This module provides a thread-safe circular buffer that stores camera poses
with timestamps, enabling lookup of historical poses for fusing delayed
visual localization results with real-time odometry.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class CameraPoseStamped:
    """Camera pose in odom frame with timestamp.

    Attributes:
        timestamp: Unix timestamp when this pose was captured
        T_odom_cam: 4x4 transform (odom <- cam), camera pose in odom frame
        T_odom_base: 4x4 transform (odom <- base), robot base pose in odom frame
    """

    timestamp: float
    T_odom_cam: np.ndarray  # (4, 4)
    T_odom_base: np.ndarray  # (4, 4)

    def __post_init__(self) -> None:
        """Ensure transforms are numpy arrays."""
        self.T_odom_cam = np.asarray(self.T_odom_cam, dtype=np.float64)
        self.T_odom_base = np.asarray(self.T_odom_base, dtype=np.float64)


def _interpolate_se3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate between two SE(3) transforms.

    Uses linear interpolation for translation and SLERP for rotation.

    Args:
        T0: Starting transform (4x4)
        T1: Ending transform (4x4)
        alpha: Interpolation factor in [0, 1]

    Returns:
        Interpolated transform (4x4)
    """
    # Extract translations
    t0 = T0[:3, 3]
    t1 = T1[:3, 3]

    # Linear interpolation for translation
    t_interp = (1.0 - alpha) * t0 + alpha * t1

    # Extract rotations
    R0 = T0[:3, :3]
    R1 = T1[:3, :3]

    # SLERP for rotation using scipy
    rotations = Rotation.from_matrix([R0, R1])
    slerp = Slerp([0.0, 1.0], rotations)
    R_interp = slerp(alpha).as_matrix()

    # Compose result
    T_interp = np.eye(4, dtype=np.float64)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = t_interp

    return T_interp


class CameraPoseBuffer:
    """Thread-safe circular buffer for camera pose history.

    Enables lookup of camera poses at past timestamps for fusing delayed
    visual localization results. This is essential because localization
    returns poses for images captured in the past, not the current time.

    The buffer stores both camera pose (T_odom_cam) and base pose (T_odom_base)
    so that both can be corrected using the computed world-odom offset.

    Thread Safety:
        All methods are thread-safe. Multiple threads can safely push and
        lookup poses concurrently.

    Example:
        >>> buffer = CameraPoseBuffer(max_age_seconds=5.0, expected_hz=30.0)
        >>> # In robot_state_loop thread:
        >>> buffer.push(timestamp, T_odom_cam, T_odom_base)
        >>> # In localization_loop thread:
        >>> pose = buffer.lookup(keyframe_timestamp)
        >>> if pose is not None:
        ...     T_world_odom = T_world_cam_keyframe @ np.linalg.inv(pose.T_odom_cam)
    """

    def __init__(
        self,
        max_age_seconds: float = 5.0,
        expected_hz: float = 30.0,
    ) -> None:
        """Initialize the buffer.

        Args:
            max_age_seconds: Maximum age of poses to keep in buffer.
                Older poses are automatically evicted.
            expected_hz: Expected rate of pose updates. Used to size the buffer.
        """
        self._max_size = int(max_age_seconds * expected_hz) + 10  # margin
        self._buffer: deque[CameraPoseStamped] = deque(maxlen=self._max_size)
        self._timestamps: list[float] = []  # Parallel list for bisect lookup
        self._lock = Lock()

    def push(
        self,
        timestamp: float,
        T_odom_cam: np.ndarray,
        T_odom_base: np.ndarray,
    ) -> None:
        """Add a camera pose to the buffer.

        Args:
            timestamp: Unix timestamp when this pose was captured
            T_odom_cam: 4x4 transform (odom <- cam)
            T_odom_base: 4x4 transform (odom <- base)
        """
        pose = CameraPoseStamped(
            timestamp=timestamp,
            T_odom_cam=T_odom_cam.copy(),
            T_odom_base=T_odom_base.copy(),
        )

        with self._lock:
            self._buffer.append(pose)
            self._timestamps.append(timestamp)

            # Trim parallel list if buffer rotated (deque auto-trims, list doesn't)
            if len(self._timestamps) > len(self._buffer):
                excess = len(self._timestamps) - len(self._buffer)
                self._timestamps = self._timestamps[excess:]

    def lookup(
        self,
        timestamp: float,
        tolerance_sec: float = 0.1,
    ) -> Optional[CameraPoseStamped]:
        """Find the camera pose closest to the given timestamp.

        Args:
            timestamp: Unix timestamp to look up
            tolerance_sec: Maximum time difference to accept. If the closest
                pose is further than this, returns None.

        Returns:
            CameraPoseStamped if found within tolerance, None otherwise.
        """
        with self._lock:
            if not self._timestamps:
                return None

            # Binary search for closest timestamp
            idx = bisect.bisect_left(self._timestamps, timestamp)

            # Check neighbors to find closest
            candidates = []
            if idx > 0:
                candidates.append((abs(self._timestamps[idx - 1] - timestamp), idx - 1))
            if idx < len(self._timestamps):
                candidates.append((abs(self._timestamps[idx] - timestamp), idx))

            if not candidates:
                return None

            best_delta, best_idx = min(candidates, key=lambda x: x[0])

            if best_delta > tolerance_sec:
                return None

            # Return a copy to prevent external modification
            pose = self._buffer[best_idx]
            return CameraPoseStamped(
                timestamp=pose.timestamp,
                T_odom_cam=pose.T_odom_cam.copy(),
                T_odom_base=pose.T_odom_base.copy(),
            )

    def interpolate(self, timestamp: float) -> Optional[CameraPoseStamped]:
        """Interpolate camera pose at exact timestamp.

        Uses linear interpolation for translation and SLERP for rotation
        to compute a pose at the exact requested timestamp.

        Args:
            timestamp: Unix timestamp to interpolate at

        Returns:
            Interpolated CameraPoseStamped if timestamp is within buffer range,
            None otherwise.
        """
        with self._lock:
            if len(self._timestamps) < 2:
                return None

            # Check bounds
            if timestamp < self._timestamps[0] or timestamp > self._timestamps[-1]:
                return None

            # Find bracketing indices
            idx = bisect.bisect_left(self._timestamps, timestamp)
            if idx == 0:
                idx = 1
            if idx >= len(self._timestamps):
                idx = len(self._timestamps) - 1

            t0 = self._timestamps[idx - 1]
            t1 = self._timestamps[idx]

            # Compute interpolation factor
            alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

            pose0 = self._buffer[idx - 1]
            pose1 = self._buffer[idx]

            # Interpolate both transforms
            T_odom_cam_interp = _interpolate_se3(
                pose0.T_odom_cam, pose1.T_odom_cam, alpha
            )
            T_odom_base_interp = _interpolate_se3(
                pose0.T_odom_base, pose1.T_odom_base, alpha
            )

            return CameraPoseStamped(
                timestamp=timestamp,
                T_odom_cam=T_odom_cam_interp,
                T_odom_base=T_odom_base_interp,
            )

    def get_latest(self) -> Optional[CameraPoseStamped]:
        """Get the most recent camera pose.

        Returns:
            Most recent CameraPoseStamped, or None if buffer is empty.
        """
        with self._lock:
            if not self._buffer:
                return None

            pose = self._buffer[-1]
            return CameraPoseStamped(
                timestamp=pose.timestamp,
                T_odom_cam=pose.T_odom_cam.copy(),
                T_odom_base=pose.T_odom_base.copy(),
            )

    def __len__(self) -> int:
        """Return number of poses in buffer."""
        with self._lock:
            return len(self._buffer)

    def get_time_range(self) -> Optional[tuple[float, float]]:
        """Get the time range covered by the buffer.

        Returns:
            Tuple of (oldest_timestamp, newest_timestamp), or None if empty.
        """
        with self._lock:
            if not self._timestamps:
                return None
            return (self._timestamps[0], self._timestamps[-1])
