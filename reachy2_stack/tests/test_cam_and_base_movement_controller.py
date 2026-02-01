#!/usr/bin/env python3
from __future__ import annotations

import time
import threading

import numpy as np
import matplotlib.pyplot as plt

from pydualsense import pydualsense

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient

# ---------------- CONFIG ----------------
HOST = "192.168.1.71"

# Teleop
CMD_HZ = 30
VX = 0.6
VY = 0.6
WZ = 110.0

# Controller shaping
CALIB_CENTER_SECONDS = 1.0   # keep sticks untouched
CALIB_RANGE_SECONDS = 2.0    # move sticks to corners/circles
RAW_DEADZONE = 10            # in raw units (0..255)
EXPO = 1.0
SMOOTH_ALPHA = 0.15

# Safety
REQUIRE_R1_ENABLE = True     # only move while holding R1 (recommended)

# Buttons
STOP_BTN_CROSS = True        # X stops
QUIT_BTN_OPTIONS = True      # Options quits

# Camera rendering
RENDER_FPS = 20
GRAB_EVERY_N = 1
SHOW_RGB = True
SHOW_DEPTH = True

DEPTH_CMAP = "viridis"
DEPTH_MINMAX = None
# --------------------------------------


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1]


def camera_loop(reachy, stop_evt: threading.Event) -> None:
    if not (SHOW_RGB or SHOW_DEPTH):
        return

    dt = 1.0 / max(1e-6, RENDER_FPS)

    plt.ion()
    ncols = int(SHOW_RGB) + int(SHOW_DEPTH)
    fig, axes = plt.subplots(1, ncols, num="Reachy Live (FAST)", figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    ax_i = 0
    ax_rgb = axes[ax_i] if SHOW_RGB else None
    if SHOW_RGB:
        ax_rgb.set_title("Frame")
        ax_rgb.axis("off")
        ax_i += 1

    ax_depth = axes[ax_i] if SHOW_DEPTH else None
    if SHOW_DEPTH:
        ax_depth.set_title("Depth")
        ax_depth.axis("off")

    rgb_artist = None
    depth_artist = None

    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    grab_count = 0
    last_rgb = None
    last_depth = None

    while not stop_evt.is_set():
        t0 = time.time()
        grab_count += 1

        if grab_count % GRAB_EVERY_N == 0:
            try:
                if SHOW_RGB:
                    frame, _ = reachy.cameras.depth.get_frame()
                    frame = np.asarray(frame)
                    last_rgb = _bgr_to_rgb(frame) if (frame.ndim == 3 and frame.shape[2] == 3) else frame

                if SHOW_DEPTH:
                    d, _ = reachy.cameras.depth.get_depth_frame()
                    d = np.asarray(d)
                    if d.ndim == 3 and d.shape[2] == 3:
                        last_depth = _bgr_to_rgb(d)
                    else:
                        dd = np.squeeze(d)
                        last_depth = dd.astype(np.float32) if dd.ndim == 2 else None
            except Exception as e:
                print("[CAM] grab error:", e)

        if rgb_artist is None and SHOW_RGB:
            if last_rgb is None:
                last_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb_artist = ax_rgb.imshow(last_rgb)

        if depth_artist is None and SHOW_DEPTH:
            if last_depth is None:
                last_depth = np.zeros((480, 640), dtype=np.float32)
            depth_artist = ax_depth.imshow(last_depth, cmap=DEPTH_CMAP) if (
                isinstance(last_depth, np.ndarray) and last_depth.ndim == 2
            ) else ax_depth.imshow(last_depth)

        if SHOW_RGB and last_rgb is not None:
            rgb_artist.set_data(last_rgb)

        if SHOW_DEPTH and last_depth is not None:
            if isinstance(last_depth, np.ndarray) and last_depth.ndim == 2:
                dshow = last_depth
                if DEPTH_MINMAX is not None:
                    lo, hi = DEPTH_MINMAX
                    dshow = np.clip(dshow, lo, hi)
                if getattr(depth_artist, "cmap", None) is None:
                    ax_depth.clear()
                    ax_depth.set_title("Depth")
                    ax_depth.axis("off")
                    depth_artist = ax_depth.imshow(dshow, cmap=DEPTH_CMAP)
                    fig.canvas.draw()
                    bg = fig.canvas.copy_from_bbox(fig.bbox)
                else:
                    depth_artist.set_data(dshow)
            else:
                if getattr(depth_artist, "cmap", None) is not None:
                    ax_depth.clear()
                    ax_depth.set_title("Depth")
                    ax_depth.axis("off")
                    depth_artist = ax_depth.imshow(last_depth)
                    fig.canvas.draw()
                    bg = fig.canvas.copy_from_bbox(fig.bbox)
                else:
                    depth_artist.set_data(last_depth)

        fig.canvas.restore_region(bg)
        if SHOW_RGB:
            ax_rgb.draw_artist(rgb_artist)
        if SHOW_DEPTH:
            ax_depth.draw_artist(depth_artist)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    plt.close(fig)


def calibrate_center(ds: pydualsense, seconds: float) -> dict[str, float]:
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
            # print(wz_f)
            client.goto_base_defined_speed(vx_f, vy_f, wz_f)
            time.sleep(dt)

    finally:
        client.goto_base_defined_speed(0.0, 0.0, 0.0)
        try:
            ds.close()
        except Exception:
            pass


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    if reachy.mobile_base is None:
        print("[BASE] No mobile base.")
        return

    client.turn_on_all()
    stop_evt = threading.Event()

    cam_thread = threading.Thread(target=camera_loop, args=(reachy, stop_evt), daemon=True)
    teleop_thread = threading.Thread(target=teleop_dualsense_loop, args=(client, stop_evt), daemon=True)

    try:
        cam_thread.start()
        teleop_thread.start()
        while not stop_evt.is_set():
            time.sleep(0.2)
    finally:
        stop_evt.set()
        cam_thread.join(timeout=2.0)
        teleop_thread.join(timeout=2.0)
        try:
            client.goto_base_defined_speed(0.0, 0.0, 0.0)
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.turn_off_all()
        client.close()


if __name__ == "__main__":
    main()


