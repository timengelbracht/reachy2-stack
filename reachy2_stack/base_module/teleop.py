"""Keyboard teleoperation control."""

import time
import threading

from pynput import keyboard

from reachy2_stack.core.client import ReachyClient


def teleop_loop(
    client: ReachyClient,
    stop_evt: threading.Event,
    cmd_hz: float = 30.0,
    vx: float = 0.6,
    vy: float = 0.6,
    wz: float = 110.0,
) -> None:
    """Keyboard teleoperation control loop.

    Args:
        client: ReachyClient instance
        stop_evt: Threading event to signal loop termination
        cmd_hz: Command rate in Hz
        vx: Forward/backward velocity (m/s)
        vy: Left/right velocity (m/s)
        wz: Rotation velocity (deg/s)
    """
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

    dt = 1.0 / cmd_hz
    print(
        "\n[TELEOP]\n"
        "↑/↓ : forward/back\n"
        "←/→ : left/right\n"
        ",/. : rotate left/right\n"
        "SPACE: stop\n"
        "ESC: quit\n"
    )

    while not stop_evt.is_set():
        vel_x = vel_y = vel_wz = 0.0

        if keyboard.Key.up in pressed:
            vel_x += vx
        if keyboard.Key.down in pressed:
            vel_x -= vx
        if keyboard.Key.left in pressed:
            vel_y += vy
        if keyboard.Key.right in pressed:
            vel_y -= vy

        if "," in pressed:
            vel_wz += wz
        if "." in pressed:
            vel_wz -= wz

        if "space" in pressed:
            vel_x = vel_y = vel_wz = 0.0

        client.goto_base_defined_speed(vel_x, vel_y, vel_wz)
        time.sleep(dt)

    client.goto_base_defined_speed(0.0, 0.0, 0.0)
