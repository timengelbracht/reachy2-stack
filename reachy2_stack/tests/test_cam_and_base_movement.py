#!/usr/bin/env python3
from __future__ import annotations

import time
import threading

import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient

# ---------------- CONFIG ----------------
HOST = "192.168.1.71"

# Teleop
CMD_HZ = 30
VX = 0.6
VY = 0.6
WZ = 110.0

# Camera rendering (this is the key knob)
RENDER_FPS = 20        # try 15–25 for smooth + low CPU
GRAB_EVERY_N = 1       # grab every loop; set 2 to halve camera calls
SHOW_RGB = True
SHOW_DEPTH = False

# Depth display
DEPTH_CMAP = "viridis"
DEPTH_MINMAX = None    # e.g. (300, 3000) if depth is mm; leave None to autoscale
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

    # Init artists
    rgb_artist = None
    depth_artist = None

    # Create a background for blitting
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    grab_count = 0
    last_rgb = None
    last_depth = None

    while not stop_evt.is_set():
        t0 = time.time()
        grab_count += 1

        # Grab frames only every N-th render iteration (reduces network / decode load)
        if grab_count % GRAB_EVERY_N == 0:
            try:
                if SHOW_RGB:
                    frame, _ = reachy.cameras.depth.get_frame()
                    frame = np.asarray(frame)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        last_rgb = _bgr_to_rgb(frame)
                    else:
                        last_rgb = frame
                if SHOW_DEPTH:
                    d, _ = reachy.cameras.depth.get_depth_frame()
                    d = np.asarray(d)
                    # If depth is colorized (H,W,3), show it as RGB. If single-channel, show with cmap.
                    if d.ndim == 3 and d.shape[2] == 3:
                        last_depth = _bgr_to_rgb(d)
                    else:
                        dd = np.squeeze(d)
                        last_depth = dd.astype(np.float32) if dd.ndim == 2 else None
            except Exception as e:
                print("[CAM] grab error:", e)

        # First-time create artists
        if rgb_artist is None and SHOW_RGB:
            if last_rgb is None:
                last_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb_artist = ax_rgb.imshow(last_rgb)

        if depth_artist is None and SHOW_DEPTH:
            if last_depth is None:
                last_depth = np.zeros((480, 640), dtype=np.float32)
            if isinstance(last_depth, np.ndarray) and last_depth.ndim == 2:
                depth_artist = ax_depth.imshow(last_depth, cmap=DEPTH_CMAP)
            else:
                depth_artist = ax_depth.imshow(last_depth)

        # Update artist data (no full redraw)
        if SHOW_RGB and last_rgb is not None:
            rgb_artist.set_data(last_rgb)

        if SHOW_DEPTH and last_depth is not None:
            if isinstance(last_depth, np.ndarray) and last_depth.ndim == 2:
                dshow = last_depth
                if DEPTH_MINMAX is not None:
                    lo, hi = DEPTH_MINMAX
                    dshow = np.clip(dshow, lo, hi)
                # if artist was RGB previously, recreate once
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

        # Blit
        fig.canvas.restore_region(bg)
        if SHOW_RGB:
            ax_rgb.draw_artist(rgb_artist)
        if SHOW_DEPTH:
            ax_depth.draw_artist(depth_artist)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

        # Pace rendering
        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    plt.close(fig)


def teleop_loop(client: ReachyClient, stop_evt: threading.Event) -> None:
    pressed = set()

    def on_press(key):
        if key == keyboard.Key.esc:
            stop_evt.set()
            return False
        if key in (keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right):
            pressed.add(key)
            return
        if key == keyboard.Key.space:
            pressed.add("space")
            return
        try:
            ch = key.char
        except Exception:
            ch = None
        if ch == ",":
            pressed.add(",")
        elif ch == ".":
            pressed.add(".")

    def on_release(key):
        if key in (keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right):
            pressed.discard(key)
            return
        if key == keyboard.Key.space:
            pressed.discard("space")
            return
        try:
            ch = key.char
        except Exception:
            ch = None
        if ch == ",":
            pressed.discard(",")
        elif ch == ".":
            pressed.discard(".")

    keyboard.Listener(on_press=on_press, on_release=on_release).start()

    dt = 1.0 / CMD_HZ
    print(
        "\n[TELEOP]\n"
        "↑/↓ : forward/back\n"
        "←/→ : left/right\n"
        ",/. : rotate left/right\n"
        "SPACE: stop\n"
        "ESC: quit\n"
    )

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

        if "space" in pressed:
            vx = vy = wz = 0.0

        client.goto_base_defined_speed(vx, vy, wz)
        time.sleep(dt)

    client.goto_base_defined_speed(0.0, 0.0, 0.0)


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
    teleop_thread = threading.Thread(target=teleop_loop, args=(client, stop_evt), daemon=True)

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
        client.close()


if __name__ == "__main__":
    main()
