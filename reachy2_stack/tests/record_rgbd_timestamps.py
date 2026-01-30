#!/usr/bin/env python3
from __future__ import annotations

import time
import threading
import queue
import csv
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pynput import keyboard
from pydualsense import pydualsense

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient

# ================= CONFIG =================
HOST = "192.168.1.71"

CMD_HZ = 30
VX = 0.4
VY = 0.4
WZ = 60.0

RENDER_FPS = 30
SHOW_RGB = True
SHOW_DEPTH = True

OUT_ROOT = Path("./out")
MAX_QUEUE = 300
# =========================================


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1]


# =============== RECORDER =================
class RawRecorder:
    def __init__(self, out_dir: Path, max_queue: int):
        self.out_dir = out_dir
        self.rgb_dir = out_dir / "rgb"
        self.depth_dir = out_dir / "depth"

        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._loop, daemon=True)

        self.rgb_csv = open(out_dir / "rgb.csv", "w", newline="", buffering=1)
        self.depth_csv = open(out_dir / "depth.csv", "w", newline="", buffering=1)
        self.odom_csv = open(out_dir / "odom.csv", "w", newline="", buffering=1)

        self.rgb_writer = csv.writer(self.rgb_csv)
        self.depth_writer = csv.writer(self.depth_csv)
        self.odom_writer = csv.writer(self.odom_csv)

        self.rgb_writer.writerow(["timestamp", "filename"])
        self.depth_writer.writerow(["timestamp", "filename"])
        self.odom_writer.writerow(["timestamp", "x", "y", "theta", "vx", "vy", "vtheta"])

        self.count = 0

    def start(self):
        self.thread.start()

    def push(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(item)
            except Exception:
                pass

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=5.0)
        self.rgb_csv.close()
        self.depth_csv.close()
        self.odom_csv.close()
        print(f"[REC] Closed. Frames written: {self.count}")

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                break

            ts, rgb, depth, odom = item
            ts_str = f"{ts:.6f}"

            # ---- RGB ----
            if rgb is not None:
                rgb_path = self.rgb_dir / f"{ts_str}.png"
                cv2.imwrite(str(rgb_path), rgb)
                self.rgb_writer.writerow([ts_str, f"rgb/{ts_str}.png"])

            # ---- DEPTH (RAW) ----
            if depth is not None:
                depth_path = self.depth_dir / f"{ts_str}.npy"
                np.save(depth_path, depth)
                self.depth_writer.writerow([
                    ts_str,
                    f"depth/{ts_str}.npy"
                ])

            # ---- ODOM (VERBATIM) ----
            if odom is not None:
                self.odom_writer.writerow([
                    ts_str,
                    odom["x"],
                    odom["y"],
                    odom["theta"],
                    odom["vx"],
                    odom["vy"],
                    odom["vtheta"],
                ])

            self.count += 1
            if self.count % 50 == 0:
                print(f"[REC] written {self.count} frames")


# =============== CAMERA LOOP =================
def camera_loop(reachy, client, stop_evt, recorder: RawRecorder):
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    img_artist = None

    dt = 1.0 / RENDER_FPS

    while not stop_evt.is_set():
        t0 = time.time()

        rgb = depth = None
        ts = None

        try:
            rgb_bgr, ts = reachy.cameras.depth.get_frame()
            rgb = np.asarray(rgb_bgr)

            depth_raw, _ = reachy.cameras.depth.get_depth_frame()
            depth = np.asarray(depth_raw)

        except Exception as e:
            print("[CAM] error:", e)

        odom = None
        try:
            odom = reachy.mobile_base.get_current_odometry()
        except Exception as e:
            print("[ODOM] error:", e)

        if ts is not None:
            recorder.push((ts, rgb, depth, odom))

        # ---- DISPLAY ----
        if SHOW_RGB and rgb is not None:
            rgb_disp = _bgr_to_rgb(rgb)
            if img_artist is None:
                img_artist = ax.imshow(rgb_disp)
                ax.axis("off")
            else:
                img_artist.set_data(rgb_disp)
            fig.canvas.draw()
            fig.canvas.flush_events()

        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    plt.close(fig)


# =============== TELEOP =================
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
            print(wz_f)
            client.goto_base_defined_speed(vx_f, vy_f, wz_f)
            time.sleep(dt)

    finally:
        client.goto_base_defined_speed(0.0, 0.0, 0.0)
        try:
            ds.close()
        except Exception:
            pass

def teleop_loop(client, stop_evt):
    pressed = set()

    def on_press(key):
        if key == keyboard.Key.esc:
            stop_evt.set()
            return False
        if key in (keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right):
            pressed.add(key)
        try:
            if key.char in [",", "."]:
                pressed.add(key.char)
        except Exception:
            pass

    def on_release(key):
        pressed.discard(key)
        try:
            pressed.discard(key.char)
        except Exception:
            pass

    keyboard.Listener(on_press=on_press, on_release=on_release).start()
    dt = 1.0 / CMD_HZ

    while not stop_evt.is_set():
        vx = vy = wz = 0.0

        if keyboard.Key.up in pressed:
            vx += VX
        if keyboard.Key.down in pressed:
            vx -= VX
        if keyboard.Key.left in pressed:
            vy += VY
        if keyboard.Key.right in pressed:
            vy -= VY
        if "," in pressed:
            wz += WZ
        if "." in pressed:
            wz -= WZ

        client.goto_base_defined_speed(vx, vy, wz)
        time.sleep(dt)

    client.goto_base_defined_speed(0.0, 0.0, 0.0)

def save_camera_extrinsics_txt(reachy, out_dir: Path) -> None:
    """
    Save camera extrinsics as a plain text 4x4 matrix.
    """
    try:
        T = reachy.cameras.depth.get_extrinsics()
        T = np.asarray(T, dtype=float)

        path = out_dir / "camera_extrinsics.txt"
        with open(path, "w") as f:
            for row in T:
                f.write(" ".join(f"{v:.9f}" for v in row) + "\n")

        print(f"[REC] Camera extrinsics saved to {path}")

    except Exception as e:
        print("[REC] Failed to save camera extrinsics:", e)

# =============== MAIN =================
def main():
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    client.turn_on_all()

    stop_evt = threading.Event()

    run_dir = OUT_ROOT / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_camera_extrinsics_txt(reachy, run_dir)

    recorder = RawRecorder(run_dir, MAX_QUEUE)
    recorder.start()

    print(f"[REC] logging to {run_dir.resolve()}")

    cam_thread = threading.Thread(
        target=camera_loop,
        args=(reachy, client, stop_evt, recorder),
        daemon=True,
    )

    # teleop_thread = threading.Thread(
    #     target=teleop_loop,
    #     args=(client, stop_evt),
    #     daemon=True,
    # )
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
        recorder.close()
        client.turn_off_all()
        client.close()
        print("[DONE]")


if __name__ == "__main__":
    main()
