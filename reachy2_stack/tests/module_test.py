#!/usr/bin/env python3
"""Clean test script for camera, base movement, and odometry visualization."""

import time
import threading
import cv2
import numpy as np
from pydualsense import pydualsense

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
SHOW_TELEOP = True  # Show teleop stereo cameras (left + right)
GRAB_TELEOP_EVERY_N = 1  # Grab teleop frames every N iterations (0 = disabled)

# Depth display
DEPTH_COLORMAP = cv2.COLORMAP_JET  # Options: COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT, etc.
DEPTH_MINMAX = None  # e.g., (300, 3000) for depth in mm; None = auto-scale
DEPTH_NORMALIZE_PERCENTILE = True  # If True, use percentile normalization (clips outliers for better visibility)
DEPTH_PERCENTILE_RANGE = (15, 85)  # Percentile range for normalization (1st to 99th percentile)

# Open3D visualization
VIS_UPDATE_HZ = 10  # Update rate for 3D visualization
SHOW_TRAJECTORY = True  # Show odometry trail
SHOW_CAMERA = True  # Show camera coordinate frame
SHOW_POINTCLOUD = True  # Show point cloud from RGBD
MAX_TRAIL_POINTS = 500  # Maximum trajectory points to keep
COORD_FRAME_SIZE = 1  # Size of coordinate frame axes

# 3D Mapping (RGBD to point cloud)
ENABLE_MAPPING = True  # Enable 3D mapping from RGBD
MAPPING_HZ = 2.0  # Point cloud generation rate in Hz (lower = less CPU)
DEPTH_SCALE = 0.001  # Scale factor for depth (1.0 if already in meters, 0.001 if in mm)
DEPTH_TRUNC = 3.5  # Maximum depth in meters to include in point cloud

# DualSense Controller
USE_DUALSENSE = True  # Use DualSense controller instead of keyboard
CALIB_CENTER_SECONDS = 1.0  # Keep sticks untouched during calibration
CALIB_RANGE_SECONDS = 2.0  # Move sticks to corners/circles during calibration
RAW_DEADZONE = 10  # In raw units (0..255)
EXPO = 1.0  # Exponential curve for stick response
SMOOTH_ALPHA = 0.15  # Smoothing factor for commands
REQUIRE_R1_ENABLE = True  # Only move while holding R1 (recommended for safety)
STOP_BTN_CROSS = True  # X button stops movement
QUIT_BTN_OPTIONS = True  # Options button quits
# --------------------------------------


def calibrate_center(ds: pydualsense, seconds: float) -> dict[str, float]:
    """Calibrate the center position of controller sticks."""
    t_end = time.time() + max(0.1, seconds)
    xs = {"LX": [], "LY": [], "RX": [], "RY": []}
    while time.time() < t_end:
        st = ds.state
        xs["LX"].append(st.LX)
        xs["LY"].append(st.LY)
        xs["RX"].append(st.RX)
        xs["RY"].append(st.RY)
        time.sleep(0.01)
    return {k: float(np.mean(v)) for k, v in xs.items()}


def calibrate_range(ds: pydualsense, seconds: float) -> tuple[dict[str, float], dict[str, float]]:
    """Calibrate the min/max range of controller sticks."""
    mins = {"LX": 1e9, "LY": 1e9, "RX": 1e9, "RY": 1e9}
    maxs = {"LX": -1e9, "LY": -1e9, "RX": -1e9, "RY": -1e9}
    t_end = time.time() + max(0.1, seconds)
    while time.time() < t_end:
        st = ds.state
        vals = {"LX": st.LX, "LY": st.LY, "RX": st.RX, "RY": st.RY}
        for k, v in vals.items():
            mins[k] = min(mins[k], float(v))
            maxs[k] = max(maxs[k], float(v))
        time.sleep(0.01)
    return mins, maxs


def stick_u8_to_unit(v: int, *, vmin: float, vcenter: float, vmax: float, deadzone_raw: float, expo: float) -> float:
    """
    Symmetric normalization using calibrated min/center/max.
    Guarantees: left extreme ~ -1, center ~ 0, right extreme ~ +1
    (within the range you actually reach).
    """
    v = float(v)

    if abs(v - vcenter) <= deadzone_raw:
        return 0.0

    if v > vcenter:
        denom = max(1e-6, (vmax - vcenter) - deadzone_raw)
        x = (v - vcenter - deadzone_raw) / denom
    else:
        denom = max(1e-6, (vcenter - vmin) - deadzone_raw)
        x = (v - vcenter + deadzone_raw) / denom

    x = float(np.clip(x, -1.0, 1.0))
    s = 1.0 if x >= 0 else -1.0
    x = s * (abs(x) ** expo)
    return float(np.clip(x, -1.0, 1.0))


def teleop_dualsense_loop(client: ReachyClient, stop_evt: threading.Event) -> None:
    """Teleop loop using DualSense controller."""
    ds = pydualsense()
    ds.init()
    dt = 1.0 / CMD_HZ

    print("\n[DUALSENSE] Center calibration: DO NOT touch sticks...")
    centers = calibrate_center(ds, CALIB_CENTER_SECONDS)
    print("[DUALSENSE] centers:", {k: round(v, 2) for k, v in centers.items()})

    print(f"[DUALSENSE] Range calibration: MOVE sticks to corners for {CALIB_RANGE_SECONDS:.1f}s...")
    mins, maxs = calibrate_range(ds, CALIB_RANGE_SECONDS)
    print("[DUALSENSE] mins:", mins)
    print("[DUALSENSE] maxs:", maxs)
    print(f"[DUALSENSE] deadzone=±{RAW_DEADZONE} raw, expo={EXPO}")

    vx_f = vy_f = wz_f = 0.0

    print(
        "\n[DUALSENSE TELEOP]\n"
        "Left stick: move (forward/back/left/right)\n"
        "Right stick X: rotate\n"
        "R1: hold-to-enable (safety)\n"
        "X (cross): stop\n"
        "Options: quit\n"
    )

    try:
        while not stop_evt.is_set():
            st = ds.state

            if QUIT_BTN_OPTIONS and st.options:
                stop_evt.set()
                break

            enabled = (not REQUIRE_R1_ENABLE) or bool(st.R1)

            lx = stick_u8_to_unit(
                st.LX, vmin=mins["LX"], vcenter=centers["LX"], vmax=maxs["LX"],
                deadzone_raw=RAW_DEADZONE, expo=EXPO
            )
            ly = stick_u8_to_unit(
                st.LY, vmin=mins["LY"], vcenter=centers["LY"], vmax=maxs["LY"],
                deadzone_raw=RAW_DEADZONE, expo=EXPO
            )
            rx = stick_u8_to_unit(
                st.RX, vmin=mins["RX"], vcenter=centers["RX"], vmax=maxs["RX"],
                deadzone_raw=RAW_DEADZONE, expo=EXPO
            )

            # NOTE: DualSense LY: 255 is "up". We want up => forward => +vx
            vx = (-ly) * VX
            vy = (-lx) * VY
            wz = (-rx) * WZ

            if not enabled:
                vx = vy = wz = 0.0

            if STOP_BTN_CROSS and st.cross:
                vx = vy = wz = 0.0

            vx_f = (1.0 - SMOOTH_ALPHA) * vx_f + SMOOTH_ALPHA * vx
            vy_f = (1.0 - SMOOTH_ALPHA) * vy_f + SMOOTH_ALPHA * vy
            wz_f = (1.0 - SMOOTH_ALPHA) * wz_f + SMOOTH_ALPHA * wz

            client.goto_base_defined_speed(vx_f, vy_f, wz_f)
            time.sleep(dt)

    finally:
        client.goto_base_defined_speed(0.0, 0.0, 0.0)
        try:
            ds.close()
        except Exception:
            pass


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

    mapping_thread = threading.Thread(
        target=mapping_loop,
        args=(cam_state, odom_state, stop_evt, client),
        kwargs={
            "mapping_hz": MAPPING_HZ,
            "depth_scale": DEPTH_SCALE,
            "depth_trunc": DEPTH_TRUNC,
        },
        daemon=True,
    ) if ENABLE_MAPPING else None

    # Choose teleop method based on configuration
    if USE_DUALSENSE:
        teleop_thread = threading.Thread(
            target=teleop_dualsense_loop,
            args=(client, stop_evt),
            daemon=True,
        )
    else:
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
        if mapping_thread is not None:
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
        if mapping_thread is not None:
            mapping_thread.join(timeout=2.0)
        try:
            client.goto_base_defined_speed(0.0, 0.0, 0.0)
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
