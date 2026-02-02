"""Localization fusion for delayed visual localization with real-time odometry.

This module provides the LocalizationFusion class that fuses delayed visual
localization results with real-time odometry using an offset correction approach.
This is the standard approach used in SLAM systems for handling delayed measurements.

The key insight is that visual localization returns poses for images captured
in the past. Instead of directly using these poses (which causes jumps), we:
1. Look up the odometry pose at the keyframe time
2. Compute the offset: T_world_odom = T_world_cam @ inv(T_odom_cam)
3. Apply this offset to current odometry to get current world pose
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np

from .camera_pose_buffer import CameraPoseBuffer


@dataclass
class LocalizationResult:
    """Result from visual localization server.

    Attributes:
        timestamp_keyframe: Unix timestamp when the image was captured
        timestamp_received: Unix timestamp when the result was received
        T_world_cam_keyframe: 4x4 transform (world <- cam), localized camera pose
        camera_id: Identifier for which camera was localized
        confidence: Optional confidence score from localization
    """

    timestamp_keyframe: float
    timestamp_received: float
    T_world_cam_keyframe: np.ndarray  # (4, 4)
    camera_id: str = "depth"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Ensure transform is numpy array."""
        self.T_world_cam_keyframe = np.asarray(
            self.T_world_cam_keyframe, dtype=np.float64
        )

    @property
    def latency(self) -> float:
        """Compute localization latency in seconds."""
        return self.timestamp_received - self.timestamp_keyframe


@dataclass
class FusionStats:
    """Statistics from the fusion process."""

    update_count: int = 0
    rejected_outliers: int = 0
    rejected_rate_limit: int = 0
    rejected_not_in_buffer: int = 0
    last_update_time: float = 0.0
    last_latency: float = 0.0
    initialized: bool = False


class LocalizationFusion:
    """Fuses delayed visual localization with real-time odometry.

    This class maintains a world-odom offset (T_world_odom) that corrects
    odometry drift. When visual localization results arrive (which are for
    past timestamps), the offset is updated and can be applied to current
    odometry to get current world frame poses.

    The approach:
    1. Visual localization returns T_world_cam for a past keyframe
    2. Look up T_odom_cam at that keyframe time from the camera buffer
    3. Compute: T_world_odom = T_world_cam @ inv(T_odom_cam)
    4. Apply: T_world_base_now = T_world_odom @ T_odom_base_now

    Thread Safety:
        All methods are thread-safe. Multiple threads can safely call
        update_from_localization() and get_T_world_base() concurrently.

    Example:
        >>> buffer = CameraPoseBuffer()
        >>> T_base_cam = client.get_depth_extrinsics()
        >>> fusion = LocalizationFusion(buffer, T_base_cam)
        >>>
        >>> # When localization result arrives:
        >>> result = LocalizationResult(
        ...     timestamp_keyframe=keyframe_time,
        ...     timestamp_received=time.time(),
        ...     T_world_cam_keyframe=T_world_cam,
        ... )
        >>> fusion.update_from_localization(result)
        >>>
        >>> # Get current world pose:
        >>> T_world_base = fusion.get_T_world_base(current_T_odom_base)
    """

    def __init__(
        self,
        camera_buffer: CameraPoseBuffer,
        T_base_cam: np.ndarray,
        smooth_alpha: float = 0.3,
        max_translation_jump: float = 0.5,
        max_rotation_jump: float = 30.0,
        min_update_interval: float = 0.05,
    ) -> None:
        """Initialize the fusion manager.

        Args:
            camera_buffer: Buffer containing historical camera poses
            T_base_cam: Static camera extrinsics (base <- cam), 4x4 transform
            smooth_alpha: Blend factor for offset updates. 0 = ignore new,
                1 = use new directly. Default 0.3 for smooth transitions.
            max_translation_jump: Maximum allowed translation change in meters.
                Updates exceeding this are rejected as outliers.
            max_rotation_jump: Maximum allowed rotation change in degrees.
                Updates exceeding this are rejected as outliers.
            min_update_interval: Minimum time between offset updates in seconds.
                Prevents oscillation from rapid updates.
        """
        self._camera_buffer = camera_buffer
        self._T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
        self._T_cam_base = np.linalg.inv(self._T_base_cam)

        self._smooth_alpha = smooth_alpha
        self._max_translation_jump = max_translation_jump
        self._max_rotation_jump_rad = np.deg2rad(max_rotation_jump)
        self._min_update_interval = min_update_interval

        self._lock = Lock()
        self._T_world_odom: np.ndarray = np.eye(4, dtype=np.float64)
        self._initialized: bool = False
        self._last_update_time: float = 0.0

        self._stats = FusionStats()

    def update_from_localization(self, result: LocalizationResult) -> bool:
        """Process a localization result and update the offset.

        This method:
        1. Looks up the camera pose at keyframe time from the buffer
        2. Computes the new world-odom offset
        3. Optionally rejects outliers and applies smoothing
        4. Updates the internal offset

        Args:
            result: Localization result containing keyframe timestamp and pose

        Returns:
            True if the offset was updated, False if rejected (outlier,
            rate limit, or keyframe not in buffer).
        """
        # Look up camera pose at keyframe time
        pose_at_keyframe = self._camera_buffer.interpolate(result.timestamp_keyframe)
        if pose_at_keyframe is None:
            # Try lookup with tolerance
            pose_at_keyframe = self._camera_buffer.lookup(
                result.timestamp_keyframe, tolerance_sec=0.2
            )

        if pose_at_keyframe is None:
            with self._lock:
                self._stats.rejected_not_in_buffer += 1
            time_range = self._camera_buffer.get_time_range()
            if time_range:
                print(
                    f"[FUSION] Keyframe {result.timestamp_keyframe:.3f} not in buffer "
                    f"(range: {time_range[0]:.3f} - {time_range[1]:.3f})"
                )
            return False

        # Compute new offset: T_world_odom = T_world_cam @ inv(T_odom_cam)
        T_world_odom_new = result.T_world_cam_keyframe @ np.linalg.inv(
            pose_at_keyframe.T_odom_cam
        )

        with self._lock:
            now = time.time()

            # Rate limiting
            if now - self._last_update_time < self._min_update_interval:
                self._stats.rejected_rate_limit += 1
                return False

            # Outlier rejection (only after initialization)
            if self._initialized:
                delta_T = np.linalg.inv(self._T_world_odom) @ T_world_odom_new

                # Translation jump
                delta_trans = np.linalg.norm(delta_T[:3, 3])

                # Rotation jump (axis-angle magnitude)
                R = delta_T[:3, :3]
                trace = np.clip(np.trace(R), -1.0, 3.0)
                angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

                if (
                    delta_trans > self._max_translation_jump
                    or angle_rad > self._max_rotation_jump_rad
                ):
                    self._stats.rejected_outliers += 1
                    print(
                        f"[FUSION] Rejected outlier: trans={delta_trans:.3f}m, "
                        f"rot={np.rad2deg(angle_rad):.1f}deg"
                    )
                    return False

            # Apply update with smoothing
            if self._initialized:
                # Smooth translation
                t_old = self._T_world_odom[:3, 3]
                t_new = T_world_odom_new[:3, 3]
                t_blend = (1.0 - self._smooth_alpha) * t_old + self._smooth_alpha * t_new

                # Smooth rotation (blend and re-orthogonalize via SVD)
                R_old = self._T_world_odom[:3, :3]
                R_new = T_world_odom_new[:3, :3]
                R_blend = (1.0 - self._smooth_alpha) * R_old + self._smooth_alpha * R_new

                # Re-orthogonalize: find closest SO(3) matrix
                U, _, Vt = np.linalg.svd(R_blend)
                R_blend = U @ Vt

                # Ensure proper rotation (det = +1)
                if np.linalg.det(R_blend) < 0:
                    U[:, -1] *= -1
                    R_blend = U @ Vt

                self._T_world_odom[:3, :3] = R_blend
                self._T_world_odom[:3, 3] = t_blend
            else:
                # First update: use directly
                self._T_world_odom = T_world_odom_new.copy()
                self._initialized = True
                self._stats.initialized = True

            self._last_update_time = now
            self._stats.update_count += 1
            self._stats.last_update_time = now
            self._stats.last_latency = result.latency

        return True

    def get_T_world_cam(self, T_odom_cam: np.ndarray) -> np.ndarray:
        """Get camera pose in world frame from camera pose in odom frame.

        Args:
            T_odom_cam: Current camera pose in odom frame (4x4)

        Returns:
            Camera pose in world frame (4x4): T_world_odom @ T_odom_cam
        """
        with self._lock:
            return self._T_world_odom @ T_odom_cam

    def get_T_world_base(self, T_odom_base: np.ndarray) -> np.ndarray:
        """Get robot base pose in world frame from base pose in odom frame.

        Args:
            T_odom_base: Current robot base pose in odom frame (4x4)

        Returns:
            Robot base pose in world frame (4x4): T_world_odom @ T_odom_base
        """
        with self._lock:
            return self._T_world_odom @ T_odom_base

    def get_T_world_odom(self) -> np.ndarray:
        """Get the current world-odom offset transform.

        Returns:
            Copy of the current T_world_odom (4x4)
        """
        with self._lock:
            return self._T_world_odom.copy()

    @property
    def is_initialized(self) -> bool:
        """Whether at least one localization update has been received."""
        with self._lock:
            return self._initialized

    def get_stats(self) -> FusionStats:
        """Get fusion statistics.

        Returns:
            Copy of current statistics.
        """
        with self._lock:
            return FusionStats(
                update_count=self._stats.update_count,
                rejected_outliers=self._stats.rejected_outliers,
                rejected_rate_limit=self._stats.rejected_rate_limit,
                rejected_not_in_buffer=self._stats.rejected_not_in_buffer,
                last_update_time=self._stats.last_update_time,
                last_latency=self._stats.last_latency,
                initialized=self._stats.initialized,
            )

    def reset(self) -> None:
        """Reset the fusion state.

        Clears the offset and statistics. Useful for re-initialization
        after localization failure or relocalization.
        """
        with self._lock:
            self._T_world_odom = np.eye(4, dtype=np.float64)
            self._initialized = False
            self._last_update_time = 0.0
            self._stats = FusionStats()
