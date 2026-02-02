#!/usr/bin/env python3
from __future__ import annotations

import curses
import time

from pydualsense import pydualsense


def safe_add(stdscr, y: int, x: int, s: str) -> int:
    """Safely write a line to curses without crashing on small terminals."""
    h, w = stdscr.getmaxyx()
    if y >= h:
        return y

    s = s.replace("\t", "    ")
    max_len = max(0, w - x - 1)
    if len(s) > max_len:
        s = s[:max_len]

    try:
        stdscr.addstr(y, x, s)
    except curses.error:
        pass
    return y + 1


def print_states(stdscr, dualsense: pydualsense) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    stdscr.nodelay(1)

    while True:
        stdscr.erase()
        y = 0

        states = dualsense.states
        if states is None:
            y = safe_add(stdscr, y, 0, "Waiting until connection is established...")
            y = safe_add(stdscr, y, 0, f"epoch: {time.time():.2f}")
            stdscr.refresh()
            time.sleep(0.1)
            if stdscr.getch() == ord("q"):
                break
            continue

        pretty_states = [f"{v:03}" for v in states]

        y = safe_add(stdscr, y, 0, f"epoch: {time.time():.2f}")
        y = safe_add(stdscr, y, 0, f"states[0:10]:  {pretty_states[0:10]}")
        y = safe_add(stdscr, y, 0, f"states[10:20]: {pretty_states[10:20]}")
        y = safe_add(stdscr, y, 0, f"states[20:30]: {pretty_states[20:30]}")
        y = safe_add(stdscr, y, 0, f"states[30:40]: {pretty_states[30:40]}")
        y = safe_add(stdscr, y, 0, f"states[40:50]: {pretty_states[40:50]}")
        y = safe_add(stdscr, y, 0, f"states[50:60]: {pretty_states[50:60]}")
        y = safe_add(stdscr, y, 0, f"states[60:70]: {pretty_states[60:70]}")
        y = safe_add(stdscr, y, 0, f"states[70:78]: {pretty_states[70:78]}")
        y = safe_add(stdscr, y, 0, "")

        st = dualsense.state

        y = safe_add(
            stdscr,
            y,
            0,
            f"square:{st.square!s:>5}  triangle:{st.triangle!s:>5}  circle:{st.circle!s:>5}  cross:{st.cross!s:>5}",
        )
        y = safe_add(
            stdscr,
            y,
            0,
            f"DpadUp:{st.DpadUp!s:>5}  DpadDown:{st.DpadDown!s:>5}  DpadLeft:{st.DpadLeft!s:>5}  DpadRight:{st.DpadRight!s:>5}",
        )
        y = safe_add(
            stdscr,
            y,
            0,
            f"L1:{st.L1!s:>5}  L2:{st.L2:3}  L2Btn:{st.L2Btn!s:>5}  L3:{st.L3!s:>5}  "
            f"R1:{st.R1!s:>5}  R2:{st.R2:3d}  R2Btn:{st.R2Btn!s:>5}  R3:{st.R3!s:>5}",
        )
        y = safe_add(
            stdscr,
            y,
            0,
            f"share:{st.share!s:>5}  options:{st.options!s:>5}  ps:{st.ps!s:>5}  micBtn:{st.micBtn!s:>5}",
        )
        y = safe_add(
            stdscr,
            y,
            0,
            f"touch1:{st.touch1!s:>5}  touch2:{st.touch2!s:>5}  touchBtn:{st.touchBtn!s:>5}  touchLeft:{st.touchLeft!s:>5}  touchRight:{st.touchRight!s:>5}",
        )
        y = safe_add(stdscr, y, 0, f"touchFinger1:{st.touchFinger1}  touchFinger2:{st.touchFinger2}")
        y = safe_add(stdscr, y, 0, f"RX:{st.RX:4}  RY:{st.RY:4}  LX:{st.LX:4}  LY:{st.LY:4}")
        y = safe_add(
            stdscr,
            y,
            0,
            f"trackPad0: ID:{st.trackPadTouch0.ID} active:{st.trackPadTouch0.isActive!s:>5} X:{st.trackPadTouch0.X:4d} Y:{st.trackPadTouch0.Y:4d}",
        )
        y = safe_add(
            stdscr,
            y,
            0,
            f"trackPad1: ID:{st.trackPadTouch1.ID} active:{st.trackPadTouch1.isActive!s:>5} X:{st.trackPadTouch1.X:4d} Y:{st.trackPadTouch1.Y:4d}",
        )
        y = safe_add(stdscr, y, 0, f"gyro: roll:{st.gyro.Roll:6}  pitch:{st.gyro.Pitch:6}  yaw:{st.gyro.Yaw:6}")
        y = safe_add(
            stdscr,
            y,
            0,
            f"acc: X:{st.accelerometer.X:6}  Y:{st.accelerometer.Y:6}  Z:{st.accelerometer.Z:6}",
        )
        y = safe_add(
            stdscr,
            y,
            0,
            f"battery: Level:{dualsense.battery.Level}  State:{dualsense.battery.State}",
        )
        y = safe_add(stdscr, y, 0, "")
        y = safe_add(stdscr, y, 0, "Exit script with 'q'")

        stdscr.refresh()

        if stdscr.getch() == ord("q"):
            break

        time.sleep(0.02)  # ~50 FPS UI refresh


def main() -> None:
    dualsense = pydualsense()
    dualsense.init()

    try:
        curses.wrapper(lambda stdscr: print_states(stdscr, dualsense))
    finally:
        try:
            dualsense.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
