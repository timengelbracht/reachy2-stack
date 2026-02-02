#!/usr/bin/env python3
"""Test script for client-server mapping architecture.

This demonstrates the client-server setup where:
1. Client runs on the robot (this script)
2. Server runs separately (wavemap_server.py) for volumetric mapping

Usage:
    # Terminal 1: Start the wavemap server
    python -m reachy2_stack.base_module.wavemap_server --port 5555

    # Terminal 2: Run the client (this script)
    python reachy2_stack/tests/module_test_client_server.py
"""

import time
import threading
import cv2

from reachy2_stack.tests.module_test import DEPTH_SCALE, DEPTH_TRUNC
from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.base_module import (
    camera_loop,
    CameraState,
    teleop_loop,
    odometry_loop,
    OdometryState,
    open3d_vis_loop,
    mapping_loop,
    mapping_loop_client,  # Client-server mapping
)

# ---------------- CONFIG ----------------
HOST = "192.168.1.71"

# Teleop
CMD_HZ = 30
VX = 0.6
VY = 0.6
WZ = 110.0

# Camera rendering (using OpenCV - compatible with Open3D)
RENDER_FPS = 20
GRAB_EVERY_N = 1
SHOW_RGB = True  # Show RGB camera feed from depth camera
SHOW_DEPTH = True  # Show depth camera feed
SHOW_TELEOP = True  # Show teleop stereo cameras (disabled for server mode)
GRAB_TELEOP_EVERY_N = 1  # Disabled

# Depth display
DEPTH_COLORMAP = cv2.COLORMAP_JET
DEPTH_MINMAX = None
DEPTH_NORMALIZE_PERCENTILE = True
DEPTH_PERCENTILE_RANGE = (15, 85)

# Open3D visualization
VIS_UPDATE_HZ = 10  # Update rate for 3D visualization
SHOW_TRAJECTORY = True  # Show odometry trail
SHOW_CAMERA = True  # Show camera coordinate frame
SHOW_POINTCLOUD = True  # Show point cloud from wavemap server
MAX_TRAIL_POINTS = 500  # Maximum trajectory points to keep
COORD_FRAME_SIZE = 1  # Size of coordinate frame axes

# Client-server mapping
MAPPING_SERVER_HOST = "localhost"  # Change to server IP if on different machine
MAPPING_SERVER_PORT = 5555
MAPPING_HZ = 2.0  # Request rate in Hz
MAPPING_TIMEOUT_MS = 5000  # ZeroMQ timeout in milliseconds
MAPPING_DEPTH_SCALE = 0.001  # Depth scale (0.001 = mm to meters)
# --------------------------------------


def main() -> None:
    """Main entry point."""
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    if reachy.mobile_base is None:
        print("[BASE] No mobile base.")
        return

    client.turn_on_all()

    # Shared state
    stop_evt = threading.Event()
    odom_state = OdometryState(max_trail_points=MAX_TRAIL_POINTS)

    cam_state = CameraState()

    # Start threads
    cam_thread = threading.Thread(
        target=camera_loop,
        args=(reachy, cam_state, stop_evt),
        kwargs={
            "show_rgb": SHOW_RGB,
            "show_depth": SHOW_DEPTH,
            "show_teleop": SHOW_TELEOP,
            "render_fps": RENDER_FPS,
            "grab_every_n": GRAB_EVERY_N,
            "grab_teleop_every_n": GRAB_TELEOP_EVERY_N,
            "depth_colormap": DEPTH_COLORMAP,
            "depth_minmax": DEPTH_MINMAX,
            "depth_normalize_percentile": DEPTH_NORMALIZE_PERCENTILE,
            "depth_percentile_range": DEPTH_PERCENTILE_RANGE,
            "client": client,
        },
        daemon=True,
    )

    # Client-server mapping thread (intrinsics come from cam_state, set by camera_loop)
    mapping_thread = threading.Thread(
        target=mapping_loop_client,
        args=(cam_state, odom_state, stop_evt),
        kwargs={
            "server_host": MAPPING_SERVER_HOST,
            "server_port": MAPPING_SERVER_PORT,
            "mapping_hz": MAPPING_HZ,
            "timeout_ms": MAPPING_TIMEOUT_MS,
            "depth_scale": MAPPING_DEPTH_SCALE,
        },
        daemon=True,
    )


    teleop_thread = threading.Thread(
        target=teleop_loop,
        args=(client, stop_evt),
        kwargs={
            "cmd_hz": CMD_HZ,
            "vx": VX,
            "vy": VY,
            "wz": WZ,
        },
        daemon=True,
    )

    odom_thread = threading.Thread(
        target=odometry_loop,
        args=(client, odom_state, stop_evt),
        daemon=True,
    )

    try:
        print("\n" + "=" * 60)
        print("CLIENT-SERVER MAPPING TEST")
        print("=" * 60)
        print(f"Wavemap server: {MAPPING_SERVER_HOST}:{MAPPING_SERVER_PORT}")
        print(f"Mapping rate: {MAPPING_HZ} Hz")
        print(f"Timeout: {MAPPING_TIMEOUT_MS} ms")
        print("=" * 60 + "\n")

        cam_thread.start()
        teleop_thread.start()
        odom_thread.start()
        mapping_thread.start()

        # Run Open3D in main thread (blocking until window closes or ESC pressed)
        open3d_vis_loop(
            odom_state,
            stop_evt,
            vis_update_hz=VIS_UPDATE_HZ,
            show_trajectory=SHOW_TRAJECTORY,
            show_camera=SHOW_CAMERA,
            show_pointcloud=SHOW_POINTCLOUD,
            coord_frame_size=COORD_FRAME_SIZE,
        )

    finally:
        stop_evt.set()
        cam_thread.join(timeout=2.0)
        teleop_thread.join(timeout=2.0)
        odom_thread.join(timeout=2.0)
        mapping_thread.join(timeout=2.0)
        try:
            client.goto_base_defined_speed(0.0, 0.0, 0.0)
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
