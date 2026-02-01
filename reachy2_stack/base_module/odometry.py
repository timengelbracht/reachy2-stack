"""Odometry polling and state management."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import open3d as o3d

from reachy2_stack.core.client import ReachyClient


@dataclass
class OdometryState:
    """Thread-safe container for odometry data and camera pose tracking."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # in degrees
    timestamp: float = 0.0
    trajectory: List[List[float]] = field(default_factory=list)
    max_trail_points: int = 500

    # Camera extrinsics (base -> camera, static)
    T_base_cam: np.ndarray = field(default_factory=lambda: np.eye(4))

    # Camera pose in world frame (updated with odometry)
    T_world_cam: np.ndarray = field(default_factory=lambda: np.eye(4))

    # Point cloud in world frame
    pointcloud_world: Optional[o3d.geometry.PointCloud] = None
    pointcloud_timestamp: float = 0.0

    def init_camera_extrinsics(self, client: ReachyClient):
        """Initialize camera extrinsics from robot.

        Args:
            client: ReachyClient instance to query camera extrinsics
        """
        try:
            # Get depth camera extrinsics (T_base_cam)
            T_base_cam = client.get_depth_extrinsics()
            with self.lock:
                self.T_base_cam = np.array(T_base_cam, dtype=float)
                print(f"[ODOM] Initialized camera extrinsics (base -> cam)")
        except Exception as e:
            print(f"[ODOM] Warning: Could not get camera extrinsics: {e}")
            with self.lock:
                self.T_base_cam = np.eye(4)

    def update(self, x: float, y: float, theta: float):
        """Update odometry state, trajectory, and camera pose in world frame.

        Args:
            x: X position in meters
            y: Y position in meters
            theta: Orientation in degrees
        """
        with self.lock:
            self.x = x
            self.y = y
            self.theta = theta
            self.timestamp = time.time()

            # Add to trajectory (xy plane, z=0)
            self.trajectory.append([x, y, 0.0])
            # Limit trajectory length
            if len(self.trajectory) > self.max_trail_points:
                self.trajectory.pop(0)

            # Update camera pose in world frame
            # T_world_cam = T_world_base @ T_base_cam
            T_world_base = self._pose_to_transform(x, y, theta)
            self.T_world_cam = T_world_base @ self.T_base_cam

    def _pose_to_transform(self, x: float, y: float, theta_deg: float) -> np.ndarray:
        """Convert 2D pose to 4x4 transform (internal helper).

        Args:
            x: X position in meters
            y: Y position in meters
            theta_deg: Rotation in degrees

        Returns:
            4x4 homogeneous transformation matrix
        """
        theta = np.deg2rad(theta_deg)
        T = np.eye(4)
        T[0, 0] = np.cos(theta)
        T[0, 1] = -np.sin(theta)
        T[1, 0] = np.sin(theta)
        T[1, 1] = np.cos(theta)
        T[0, 3] = x
        T[1, 3] = y
        return T

    def get_pose(self):
        """Get current pose (thread-safe).

        Returns:
            Tuple of (x, y, theta)
        """
        with self.lock:
            return self.x, self.y, self.theta

    def get_trajectory(self):
        """Get trajectory history (thread-safe).

        Returns:
            Numpy array of shape (N, 3) with trajectory points
        """
        with self.lock:
            return np.array(self.trajectory) if self.trajectory else np.zeros((0, 3))

    def get_camera_pose(self):
        """Get camera pose in world frame (thread-safe).

        Returns:
            4x4 transformation matrix (world -> camera)
        """
        with self.lock:
            return self.T_world_cam.copy()

    def get_camera_position(self):
        """Get camera position in world frame (thread-safe).

        Returns:
            Numpy array [x, y, z] of camera position in world frame
        """
        with self.lock:
            return self.T_world_cam[:3, 3].copy()

    def update_pointcloud(self, pcd_camera: o3d.geometry.PointCloud):
        """Update point cloud, transforming from camera frame to world frame.

        Args:
            pcd_camera: Point cloud in camera frame
        """
        with self.lock:
            # Transform point cloud to world frame
            pcd_world = o3d.geometry.PointCloud(pcd_camera)
            pcd_world.transform(self.T_world_cam)
            self.pointcloud_world = pcd_world
            self.pointcloud_timestamp = time.time()

    def get_pointcloud(self) -> Optional[o3d.geometry.PointCloud]:
        """Get current point cloud in world frame (thread-safe).

        Returns:
            Point cloud in world frame, or None if no point cloud available
        """
        with self.lock:
            if self.pointcloud_world is not None:
                return o3d.geometry.PointCloud(self.pointcloud_world)
            return None



def odometry_loop(client: ReachyClient, odom_state: OdometryState, stop_evt: threading.Event) -> None:
    """Poll odometry from robot and update shared state.

    Args:
        client: ReachyClient instance
        odom_state: Shared OdometryState object
        stop_evt: Threading event to signal loop termination
    """
    # Initialize camera extrinsics from robot
    odom_state.init_camera_extrinsics(client)

    dt = 1.0 / 30.0  # Poll at 30 Hz

    while not stop_evt.is_set():
        try:
            odom = client.get_mobile_odometry()
            x = odom.get("x", 0.0)
            y = odom.get("y", 0.0)
            theta = odom.get("theta", 0.0)
            odom_state.update(x, y, theta)
        except Exception as e:
            print(f"[ODOM] error: {e}")

        time.sleep(dt)
