#!/usr/bin/env python3
"""Clean test script for camera, base movement, and odometry visualization."""

import time
import threading
import cv2

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.base_module import (
    camera_loop,
    teleop_loop,
    odometry_loop,
    open3d_vis_loop,
    OdometryState,
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
SHOW_RGB = True  # Show RGB camera feed
SHOW_DEPTH = True  # Show depth camera feed

# Depth display
DEPTH_COLORMAP = cv2.COLORMAP_JET  # Options: COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT, etc.
DEPTH_MINMAX = None  # e.g., (300, 3000) for depth in mm; None = auto-scale
DEPTH_NORMALIZE_PERCENTILE = True  # If True, use percentile normalization (clips outliers for better visibility)
DEPTH_PERCENTILE_RANGE = (1, 99)  # Percentile range for normalization (1st to 99th percentile)

# Open3D visualization
VIS_UPDATE_HZ = 10  # Update rate for 3D visualization
SHOW_TRAJECTORY = True  # Show odometry trail
SHOW_CAMERA = True  # Show camera coordinate frame
MAX_TRAIL_POINTS = 500  # Maximum trajectory points to keep
COORD_FRAME_SIZE = 1  # Size of coordinate frame axes
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

    # Start threads
    cam_thread = threading.Thread(
        target=camera_loop,
        args=(reachy, stop_evt),
        kwargs={
            "show_rgb": SHOW_RGB,
            "show_depth": SHOW_DEPTH,
            "render_fps": RENDER_FPS,
            "grab_every_n": GRAB_EVERY_N,
            "depth_colormap": DEPTH_COLORMAP,
            "depth_minmax": DEPTH_MINMAX,
            "depth_normalize_percentile": DEPTH_NORMALIZE_PERCENTILE,
            "depth_percentile_range": DEPTH_PERCENTILE_RANGE,
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
        cam_thread.start()
        teleop_thread.start()
        odom_thread.start()

        # Run Open3D in main thread (blocking until window closes or ESC pressed)
        open3d_vis_loop(
            odom_state,
            stop_evt,
            vis_update_hz=VIS_UPDATE_HZ,
            show_trajectory=SHOW_TRAJECTORY,
            show_camera=SHOW_CAMERA,
            coord_frame_size=COORD_FRAME_SIZE,
        )

    finally:
        stop_evt.set()
        cam_thread.join(timeout=2.0)
        teleop_thread.join(timeout=2.0)
        odom_thread.join(timeout=2.0)
        try:
            client.goto_base_defined_speed(0.0, 0.0, 0.0)
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
