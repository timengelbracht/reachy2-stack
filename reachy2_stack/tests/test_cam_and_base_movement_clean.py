#!/usr/bin/env python3
"""Clean test script for camera, base movement, and odometry visualization.

Adds logging:
- rgb frames (.png) in <OUT_DIR>/rgb
- depth frames (16-bit .png, in mm if possible) in <OUT_DIR>/depth
- odometry to <OUT_DIR>/odom.csv
- camera intrinsics to <OUT_DIR>/camera_intrinsics.txt
- camera extrinsics to <OUT_DIR>/camera_extrinsics.txt
"""

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

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
SHOW_RGB = True
SHOW_DEPTH = True

# Depth display
DEPTH_COLORMAP = cv2.COLORMAP_JET
DEPTH_MINMAX = None
DEPTH_NORMALIZE_PERCENTILE = True
DEPTH_PERCENTILE_RANGE = (1, 99)

# Open3D visualization
VIS_UPDATE_HZ = 10
SHOW_TRAJECTORY = True
SHOW_CAMERA = True
MAX_TRAIL_POINTS = 500
COORD_FRAME_SIZE = 1

# --------- NEW: Logging / output ---------
ENABLE_LOGGING = True
OUT_DIR = "/exchange/out/run_demo"  # change per run if you want
LOG_RGB = True
LOG_DEPTH = True
LOG_ODOM = True
LOG_INTRINSICS = True
LOG_EXTRINSICS = True

# Save rates (independent of rendering)
SAVE_IMAGE_HZ = 10.0         # save rgb/depth at this rate
SAVE_ODOM_HZ = 10.0          # odometry save rate
# ----------------------------------------


# ---------------- Logging helpers ----------------
@dataclass
class Recorder:
    out_dir: str
    rgb_dir: str
    depth_dir: str
    odom_csv: str
    intrinsics_txt: str
    extrinsics_txt: str

    # For periodic saving
    last_img_save_t: float = 0.0
    last_odom_save_t: float = 0.0

    # CSV header guard
    odom_header_written: bool = False


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp_ns() -> str:
    # Same style as your TSDF pipeline expects (float-ish filename also okay)
    return f"{time.time_ns():.6f}"


def _write_matrix_txt(path: str, M: np.ndarray) -> None:
    np.savetxt(path, M, fmt="%.10f")


def _write_intrinsics_txt(path: str, width: int, height: int, K: np.ndarray) -> None:
    # Matches your TSDF parser expectations (camera_matrix_K then 3 rows)
    with open(path, "w") as f:
        f.write(f"image_width {width}\n")
        f.write(f"image_height {height}\n")
        f.write("camera_matrix_K\n")
        for r in range(3):
            f.write(f"{K[r,0]:.10f} {K[r,1]:.10f} {K[r,2]:.10f}\n")


def _append_odom_csv(path: str, row: Tuple[float, float, float, float], write_header_if_needed: bool) -> None:
    # row = (timestamp, x, y, theta)
    header = "timestamp,x,y,theta\n"
    line = f"{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}\n"
    file_exists = os.path.exists(path)
    with open(path, "a") as f:
        if write_header_if_needed and (not file_exists):
            f.write(header)
        f.write(line)


# -------------- NEW: grab + save loops --------------
def save_static_calibration_once(client: ReachyClient, out: Recorder) -> None:
    """Save intrinsics/extrinsics once at start."""
    # ---- EXTRINSICS (base <- cam, same as your OdometryState expects) ----
    if LOG_EXTRINSICS:
        try:
            T_base_cam = np.array(client.get_depth_extrinsics(), dtype=float)
            _write_matrix_txt(out.extrinsics_txt, T_base_cam)
            print(f"[LOG] Saved camera extrinsics -> {out.extrinsics_txt}")
        except Exception as e:
            print(f"[LOG] Could not save extrinsics: {e}")

    # ---- INTRINSICS ----
    if LOG_INTRINSICS:
        try:
            # Try common patterns; adjust if your API differs.
            # Prefer a direct call if available.
            intr = None
            if hasattr(client, "get_depth_intrinsics"):
                intr = client.get_depth_intrinsics()
            elif hasattr(client, "get_camera_intrinsics"):
                intr = client.get_camera_intrinsics()

            if intr is None:
                raise RuntimeError("No intrinsics getter found on client.")

            # Expect either dict-like or object with fields
            # We'll try to parse fx, fy, cx, cy, width, height
            if isinstance(intr, dict):
                width = int(intr.get("width") or intr.get("image_width") or intr.get("w"))
                height = int(intr.get("height") or intr.get("image_height") or intr.get("h"))
                fx = float(intr.get("fx"))
                fy = float(intr.get("fy"))
                cx = float(intr.get("cx"))
                cy = float(intr.get("cy"))
            else:
                width = int(getattr(intr, "width"))
                height = int(getattr(intr, "height"))
                fx = float(getattr(intr, "fx"))
                fy = float(getattr(intr, "fy"))
                cx = float(getattr(intr, "cx"))
                cy = float(getattr(intr, "cy"))

            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)

            _write_intrinsics_txt(out.intrinsics_txt, width, height, K)
            print(f"[LOG] Saved camera intrinsics -> {out.intrinsics_txt}")

        except Exception as e:
            print(f"[LOG] Could not save intrinsics: {e}")


def image_save_loop(
    reachy,
    stop_evt: threading.Event,
    out: Recorder,
    save_hz: float,
) -> None:
    """
    Periodically grab the latest RGB + Depth frames and save them.
    This is intentionally separate from camera_loop (which is display-focused).

    NOTE: You may need to adjust the accessors depending on Reachy SDK:
      - reachy.depth_camera.get_frame() / get_rgb_frame() / etc.
      - reachy.head.depth_camera / reachy.depth_camera
    """
    period = 1.0 / max(save_hz, 1e-6)

    while not stop_evt.is_set():
        t0 = time.time()

        # Rate limit
        if (t0 - out.last_img_save_t) < period:
            time.sleep(0.001)
            continue

        try:
            ts = _timestamp_ns()

            # --------- RGB ---------
            if LOG_RGB:
                rgb = None
                # Try common patterns; adapt if your SDK differs.
                if hasattr(reachy, "rgb_camera") and hasattr(reachy.rgb_camera, "get_frame"):
                    rgb = reachy.rgb_camera.get_frame()
                elif hasattr(reachy, "head") and hasattr(reachy.head, "camera") and hasattr(reachy.head.camera, "get_frame"):
                    rgb = reachy.head.camera.get_frame()

                if rgb is not None:
                    # Ensure uint8 BGR or RGB; OpenCV expects BGR. If it looks wrong, swap channels.
                    rgb_path = os.path.join(out.rgb_dir, f"{ts}.png")
                    cv2.imwrite(rgb_path, rgb)
                else:
                    # If you can’t grab here, you can still rely on your camera_loop display only.
                    pass

            # --------- Depth ---------
            if LOG_DEPTH:
                depth = None
                if hasattr(reachy, "depth_camera") and hasattr(reachy.depth_camera, "get_frame"):
                    depth = reachy.depth_camera.get_frame()
                elif hasattr(reachy, "head") and hasattr(reachy.head, "depth_camera") and hasattr(reachy.head.depth_camera, "get_frame"):
                    depth = reachy.head.depth_camera.get_frame()

                if depth is not None:
                    # Expect depth in millimeters or meters.
                    # Save as 16-bit PNG in millimeters for your TSDF pipeline (DEPTH_SCALE=1000).
                    depth_u16 = None
                    if depth.dtype == np.uint16:
                        depth_u16 = depth
                    elif depth.dtype in (np.float32, np.float64):
                        # Assume meters -> convert to mm
                        depth_u16 = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
                    else:
                        # Try best-effort conversion
                        depth_u16 = depth.astype(np.uint16)

                    depth_path = os.path.join(out.depth_dir, f"{ts}.png")
                    cv2.imwrite(depth_path, depth_u16)
                else:
                    pass

            out.last_img_save_t = t0

        except Exception as e:
            print(f"[LOG] image_save_loop error: {e}")

        # Be nice to CPU
        dt = time.time() - t0
        if dt < period:
            time.sleep(max(0.0, period - dt))


def odom_save_loop(
    odom_state: OdometryState,
    stop_evt: threading.Event,
    out: Recorder,
    save_hz: float,
) -> None:
    """Periodically write odom state to CSV."""
    period = 1.0 / max(save_hz, 1e-6)

    while not stop_evt.is_set():
        t0 = time.time()

        if (t0 - out.last_odom_save_t) < period:
            time.sleep(0.001)
            continue

        try:
            x, y, theta = odom_state.get_pose()
            # IMPORTANT: use the same timestamp base as images if possible.
            # Here we use time.time_ns() converted to float-ish seconds in ns-space is awkward,
            # so we store as float nanoseconds (consistent with your TSDF filenames).
            ts = float(time.time_ns())

            _append_odom_csv(
                out.odom_csv,
                (ts, x, y, theta),
                write_header_if_needed=True,
            )
            out.last_odom_save_t = t0

        except Exception as e:
            print(f"[LOG] odom_save_loop error: {e}")

        dt = time.time() - t0
        if dt < period:
            time.sleep(max(0.0, period - dt))


# ---------------- Main ----------------
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

    # ---- NEW: setup output dirs ----
    rec: Optional[Recorder] = None
    if ENABLE_LOGGING:
        rgb_dir = os.path.join(OUT_DIR, "rgb")
        depth_dir = os.path.join(OUT_DIR, "depth")  # keep separate from your old /exchange/out/pngs
        _mkdir(OUT_DIR)
        _mkdir(rgb_dir)
        _mkdir(depth_dir)

        rec = Recorder(
            out_dir=OUT_DIR,
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            odom_csv=os.path.join(OUT_DIR, "odom.csv"),
            intrinsics_txt=os.path.join(OUT_DIR, "camera_intrinsics.txt"),
            extrinsics_txt=os.path.join(OUT_DIR, "camera_extrinsics.txt"),
        )

        # Save calibration once
        save_static_calibration_once(client, rec)

    # Start threads (existing)
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

    # ---- NEW: logger threads ----
    img_save_thread = None
    odom_save_thread = None
    if ENABLE_LOGGING and rec is not None:
        if LOG_RGB or LOG_DEPTH:
            img_save_thread = threading.Thread(
                target=image_save_loop,
                args=(reachy, stop_evt, rec, SAVE_IMAGE_HZ),
                daemon=True,
            )
        if LOG_ODOM:
            odom_save_thread = threading.Thread(
                target=odom_save_loop,
                args=(odom_state, stop_evt, rec, SAVE_ODOM_HZ),
                daemon=True,
            )

    try:
        cam_thread.start()
        teleop_thread.start()
        odom_thread.start()

        if img_save_thread is not None:
            img_save_thread.start()
        if odom_save_thread is not None:
            odom_save_thread.start()

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
        if img_save_thread is not None:
            img_save_thread.join(timeout=2.0)
        if odom_save_thread is not None:
            odom_save_thread.join(timeout=2.0)

        try:
            client.goto_base_defined_speed(0.0, 0.0, 0.0)
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
