#!/usr/bin/env python3
from __future__ import annotations

import time
import math
import numpy as np

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.base import BaseController  # adjust import if needed


# ---------------- CONFIG ----------------
HOST = "192.168.1.71"

SLEEP_BETWEEN = 0.15

# Reachy limitation workaround:
MAX_STEP_M = 0.90        # <= 1.0m, use a bit less for robustness

# Optional: skip tiny moves
MIN_SEG_DIST = 0.02      # meters
MIN_ROT_DEG  = 1.0       # degrees

ROTATE_IN_DEGREES = True
# --------------------------------------


# ---------------- PATH (planner output) ----------------
PLANNER_PATH = [
    {"pos": [3.0, 4.0, -0.85], "yaw_deg": 0.0},
    {"pos": [6.2709397998760865, 3.9723887366213413, -0.85], "yaw_deg": -10.6},
    {"pos": [6.670852133741838, 3.897303253410672, -0.85], "yaw_deg": -66.8},
    {"pos": [7.0850847693540615, 2.931177549247635, -0.85], "yaw_deg": -77.0},
    {"pos": [7.121664382956576, 2.7727246616973154, -0.85], "yaw_deg": -81.4},
    {"pos": [7.182387995608586, 2.3717823850502975, -0.85], "yaw_deg": -126.6},
    {"pos": [5.2, -0.3, -0.85], "yaw_deg": 90.0},
]
# ------------------------------------------------------


def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def segment_heading_deg(p0_xy: np.ndarray, p1_xy: np.ndarray) -> float:
    dx = float(p1_xy[0] - p0_xy[0])
    dy = float(p1_xy[1] - p0_xy[1])
    return math.degrees(math.atan2(dy, dx))


def chunk_distance(total: float, max_step: float) -> list[float]:
    """
    Split 'total' into steps each <= max_step.
    Example: total=2.3, max_step=0.9 -> [0.9, 0.9, 0.5]
    """
    if total <= 0:
        return []
    n_full = int(total // max_step)
    rem = total - n_full * max_step
    steps = [max_step] * n_full
    if rem > 1e-9:
        steps.append(rem)
    return steps


def translate_chunked(base: BaseController, distance_m: float) -> None:
    """
    Translate forward by distance_m, but in multiple translate_by calls
    each <= MAX_STEP_M.
    """
    steps = chunk_distance(distance_m, MAX_STEP_M)
    for j, d in enumerate(steps):
        print(f"    [MOVE] translate step {j+1}/{len(steps)}: {d:.3f} m")
        base.translate_by(x=float(d), y=0.0, wait=True)
        time.sleep(SLEEP_BETWEEN)


def execute_path_relative_chunked(
    base: BaseController,
    path: list[dict],
    *,
    start_yaw_deg: float,
    goal_yaw_deg: float,
) -> None:
    if len(path) < 2:
        print("[EXEC] Path too short.")
        return

    pts_xy = np.array([[s["pos"][0], s["pos"][1]] for s in path], dtype=np.float64)

    current_yaw_deg = float(start_yaw_deg)

    print(f"[EXEC] Relative execution start yaw = {current_yaw_deg:.1f} deg")
    print(f"[EXEC] Goal yaw = {goal_yaw_deg:.1f} deg")
    print(f"[EXEC] MAX_STEP_M = {MAX_STEP_M:.2f} m")
    print(f"[EXEC] Segments = {len(pts_xy) - 1}")

    for i in range(len(pts_xy) - 1):
        p0 = pts_xy[i]
        p1 = pts_xy[i + 1]

        desired_heading_deg = segment_heading_deg(p0, p1)
        dist = float(np.linalg.norm(p1 - p0))

        d_yaw = wrap_to_180(desired_heading_deg - current_yaw_deg)

        print(
            f"[EXEC] seg {i}->{i+1}: "
            f"dist={dist:.3f} m, "
            f"seg_heading={desired_heading_deg:.1f} deg, "
            f"rotate_by={d_yaw:+.1f} deg"
        )

        # Rotate to face segment
        if abs(d_yaw) >= MIN_ROT_DEG:
            base.rotate_by(theta=float(d_yaw), wait=True, degrees=ROTATE_IN_DEGREES)
            time.sleep(SLEEP_BETWEEN)
            current_yaw_deg = wrap_to_180(current_yaw_deg + d_yaw)

        # Translate along segment in chunks
        if dist >= MIN_SEG_DIST:
            translate_chunked(base, dist)

    # Final yaw
    final_turn = wrap_to_180(goal_yaw_deg - current_yaw_deg)
    print(f"[EXEC] Final yaw align: rotate_by={final_turn:+.1f} deg")
    if abs(final_turn) >= MIN_ROT_DEG:
        base.rotate_by(theta=float(final_turn), wait=True, degrees=ROTATE_IN_DEGREES)

    print("[EXEC] Relative path execution DONE ✅")


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    if not hasattr(reachy, "mobile_base") or reachy.mobile_base is None:
        print("[BASE] No mobile_base available on this Reachy. Exiting.")
        client.close()
        return

    base = BaseController(client=client, world=None)
    client.turn_on_all()

    start_yaw_deg = float(PLANNER_PATH[0]["yaw_deg"])
    goal_yaw_deg = float(PLANNER_PATH[-1]["yaw_deg"])

    try:
        execute_path_relative_chunked(
            base,
            PLANNER_PATH,
            start_yaw_deg=start_yaw_deg,
            goal_yaw_deg=goal_yaw_deg,
        )
    finally:
        try:
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
