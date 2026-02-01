"""3D mapping from RGBD camera data."""

import time
import threading
from typing import Optional

import numpy as np
import open3d as o3d

from .camera import CameraState
from .odometry import OdometryState
from .utils import rgbd_to_pointcloud


def mapping_loop(
    camera_state: CameraState,
    odom_state: OdometryState,
    stop_evt: threading.Event,
    client,
    mapping_hz: float = 2.0,
    depth_scale: float = 1.0,
    depth_trunc: float = 3.0,
) -> None:
    """Generate point clouds from RGBD frames and transform to world frame.

    This loop:
    1. Gets latest RGB/depth frames from camera_state
    2. Converts RGBD to point cloud in camera frame
    3. Transforms point cloud to world frame using odometry
    4. Updates point cloud in odom_state for visualization

    Args:
        camera_state: CameraState instance for getting RGB/depth frames
        odom_state: OdometryState instance for getting pose and storing point clouds
        stop_evt: Threading event to signal loop termination
        client: ReachyClient instance for getting camera intrinsics
        mapping_hz: Point cloud generation rate in Hz
        depth_scale: Scale factor for depth (1.0 if depth is in meters, 0.001 if in mm)
        depth_trunc: Maximum depth value to include in point cloud (in meters)
    """
    dt = 1.0 / max(1e-6, mapping_hz)

    # Get camera intrinsics
    intrinsics = None
    try:
        intrinsics = client.get_depth_intrinsics()
        print(f"[MAPPING] Camera intrinsics loaded")
        print(f"[MAPPING] Point cloud generation at {mapping_hz} Hz")
        print(f"[MAPPING] Depth scale: {depth_scale}, truncation: {depth_trunc}m")
    except Exception as e:
        print(f"[MAPPING] ERROR: Could not get camera intrinsics: {e}")
        print(f"[MAPPING] Mapping disabled")
        return

    print("[MAPPING] Mapping loop started")

    frame_count = 0

    while not stop_evt.is_set():
        t0 = time.time()

        try:
            # Get latest frames from camera state
            frames = camera_state.get_frames()

            if frames is None:
                # No frames available yet, wait
                time.sleep(dt)
                continue

            rgb_frame, depth_frame, frame_timestamp = frames
            frame_count += 1

            # Convert RGBD to point cloud in camera frame
            pcd_camera = rgbd_to_pointcloud(
                rgb_frame,
                depth_frame,
                intrinsics,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
            )

            num_points = len(pcd_camera.points)

            if num_points > 0:
                # Update point cloud in odometry state (transforms to world frame internally)
                odom_state.update_pointcloud(pcd_camera)

                if frame_count % 10 == 0:  # Print status every 10 frames
                    print(f"[MAPPING] Frame {frame_count}: {num_points} points generated")
            else:
                if frame_count % 10 == 0:
                    print(f"[MAPPING] Frame {frame_count}: 0 points (check depth values)")

        except Exception as e:
            print(f"[MAPPING] Error: {e}")
            import traceback
            traceback.print_exc()

        # Pace the loop
        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    print("[MAPPING] Mapping loop stopped")


def mapping_loop_client(