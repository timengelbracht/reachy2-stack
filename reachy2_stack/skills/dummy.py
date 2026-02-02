#!/usr/bin/env python3
from __future__ import annotations

import time
import numpy as np

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.base import BaseController

# ============================== CONFIG =======================================
HOST = "192.168.1.71"
SIDE = "right"

# Rotate the TARGET itself around Z in the base frame (optional)
TARGET_YAW_DEG = 180.0

# Target pose (base <- ee)
A = np.array(
    [
        [0, 0, -1, 0.4],
        [0, 1,  0, -0.4],
        [1, 0,  0,  0.1],
        [0, 0,  0,  1.0],
    ],
    dtype=float,
)


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def Rz(deg: float) -> np.ndarray:
    th = np.deg2rad(float(deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    R = np.eye(4, dtype=float)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


# ------------------------- Base motion models -------------------------
def T_translate_then_rotate(dx: float, dy: float, yaw_rad: float) -> np.ndarray:
    """
    Model for Strategy-1: base executes translate_by(dx,dy) THEN rotate_by(yaw)
    => T = Trans(dx,dy) @ Rot(yaw)
    """
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))

    Tt = np.eye(4, dtype=float)
    Tt[0, 3] = float(dx)
    Tt[1, 3] = float(dy)

    R = np.eye(4, dtype=float)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c

    return Tt @ R


def T_rotate_then_translate(dx: float, dy: float, yaw_rad: float) -> np.ndarray:
    """
    Model for Strategy-2: base executes rotate_by(yaw) THEN translate_by(dx,dy)
    => T = Rot(yaw) @ Trans(dx,dy)
    (This matches the typical "rotate then move" execution order.)
    """
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))

    R = np.eye(4, dtype=float)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c

    Tt = np.eye(4, dtype=float)
    Tt[0, 3] = float(dx)
    Tt[1, 3] = float(dy)

    return R @ Tt


def apply_base_se2(A_cur: np.ndarray, dx: float, dy: float, yaw_rad: float, order: str) -> np.ndarray:
    """
    Update target pose expressed in NEW base frame:
        A_new = inv(T) @ A_old
    where T is the base motion in world/odom.
    order: "tr" (translate then rotate) or "rt" (rotate then translate)
    """
    if order == "tr":
        T = T_translate_then_rotate(dx, dy, yaw_rad)
    elif order == "rt":
        T = T_rotate_then_translate(dx, dy, yaw_rad)
    else:
        raise ValueError(f"Unknown order={order!r}")
    return np.linalg.inv(T) @ A_cur


# ------------------------- Arm + base helpers -------------------------
def try_arm(arm, A_try: np.ndarray) -> bool:
    ARM_DURATION = 4.0
    resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
    gid = getattr(resp, "id", None)
    ok = gid is not None and gid != -1
    print("[ARM] goto id:", gid, "=>", "OK" if ok else "NO")
    return ok


def execute_translation_in_steps(base: BaseController, dx_goal: float, dy_goal: float) -> float:
    remaining_dx = float(dx_goal)
    remaining_dy = float(dy_goal)
    total = 0.0
    MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
    MAX_STEP_TRANS = 0.30             # clamp per translate_by (m)

    while True:
        dist = float(np.hypot(remaining_dx, remaining_dy))
        if dist < 1e-6:
            break

        s = 1.0
        if dist > MAX_STEP_TRANS:
            s = MAX_STEP_TRANS / dist

        sx = remaining_dx * s
        sy = remaining_dy * s

        print(f"[BASE] translate_by(x={sx:+.3f}, y={sy:+.3f})")
        base.translate_by(x=float(sx), y=float(sy), wait=True)

        step_dist = float(np.hypot(sx, sy))
        total += step_dist
        if total > MAX_BASE_TOTAL_TRANS:
            raise RuntimeError("Exceeded MAX_BASE_TOTAL_TRANS while translating base.")

        remaining_dx -= sx
        remaining_dy -= sy

    return total


def execute_yaw_in_steps(base: BaseController, yaw_deg_goal: float) -> float:
    remaining = float(yaw_deg_goal)
    total_abs = 0.0
    MAX_BASE_TOTAL_YAW_DEG = 180.0    # max total yaw (deg)
    MAX_STEP_YAW_DEG = 30.0           # clamp per rotate_by (deg)

    while abs(remaining) > 1e-3:
        step = float(np.clip(remaining, -MAX_STEP_YAW_DEG, MAX_STEP_YAW_DEG))
        print(f"[BASE] rotate_by(theta={step:.1f} deg)")
        base.rotate_by(theta=step, wait=True, degrees=True)

        total_abs += abs(step)
        if total_abs > MAX_BASE_TOTAL_YAW_DEG:
            raise RuntimeError("Exceeded MAX_BASE_TOTAL_YAW_DEG while rotating base.")

        remaining -= step

    return total_abs


# ========================== STRATEGY 1 ======================================
def translation_priority_candidates(A_cur: np.ndarray) -> list[tuple[float, float]]:
    p = A_cur[:2, 3].astype(float)
    n = float(np.linalg.norm(p))
    if n < 1e-9:
        u = np.array([1.0, 0.0], dtype=float)
    else:
        u = p / n

    cands: list[tuple[float, float]] = []
    MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
    LINE_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50]
    AXIS_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
    DIAG_FACTOR = 0.7

    # 1) along target direction
    for s in LINE_STEP_TRIES:
        dx = float(u[0] * s)
        dy = float(u[1] * s)
        cands.append((dx, dy))

    # 1b) opposite direction
    for s in LINE_STEP_TRIES[:6]:
        dx = float(-u[0] * s)
        dy = float(-u[1] * s)
        cands.append((dx, dy))

    # 2) axis
    for s in AXIS_STEP_TRIES:
        for dx, dy in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
            cands.append((float(dx), float(dy)))

    # 3) diagonals
    for s in AXIS_STEP_TRIES:
        a = float(s)
        b = float(DIAG_FACTOR * s)
        for dx, dy in [(a, b), (a, -b), (-a, b), (-a, -b)]:
            cands.append((dx, dy))

    # filter + dedup
    seen = set()
    uniq: list[tuple[float, float]] = []
    for dx, dy in cands:
        if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS + 1e-9:
            continue
        key = (round(dx, 4), round(dy, 4))
        if key not in seen:
            seen.add(key)
            uniq.append((dx, dy))
    return uniq


def coarse_ring_candidates(r_step: float, r_max: float, angle_priority_deg: list[int]) -> list[tuple[float, float]]:
    radii = np.arange(r_step, r_max + 1e-9, r_step)
    cands: list[tuple[float, float]] = []
    MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
    for r in radii:
        for ang_deg in angle_priority_deg:
            th = np.deg2rad(float(ang_deg))
            dx = float(r * np.cos(th))
            dy = float(r * np.sin(th))
            if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
                cands.append((dx, dy))
    return cands


def find_base_assist_strategy1(arm, A_cur: np.ndarray) -> tuple[float, float, float] | None:
    """
    Strategy-1 (translation-first):
      A) translation-only (yaw=0) prioritized
      B) translation-only coarse rings
      C) small yaw last (optionally with tiny translations)
      D) final yaw fallback (90/-90/180)
    Returns (dx, dy, yaw_deg) or None.
    NOTE: Feasibility test uses apply_base_se2 with order="tr".
    """
    print("\n[STRAT-1 / A] translation-only (yaw=0) prioritized")
    for dx, dy in translation_priority_candidates(A_cur):
        print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
        if try_arm(arm, apply_base_se2(A_cur, dx, dy, 0.0, order="tr")):
            return dx, dy, 0.0
        
    R_STEP_COARSE = 0.20
    R_MAX = 2.0
    ANGLE_PRIORITY_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180]
    print("\n[STRAT-1 / B] translation-only coarse rings (yaw=0)")
    for dx, dy in coarse_ring_candidates(R_STEP_COARSE, R_MAX, ANGLE_PRIORITY_DEG):
        print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
        if try_arm(arm, apply_base_se2(A_cur, dx, dy, 0.0, order="tr")):
            return dx, dy, 0.0
        
    YAW_SMALL_DEG = [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0]
    print("\n[STRAT-1 / C] rotation as last resort (small yaws first)")
    for yaw_deg in YAW_SMALL_DEG:
        yaw_rad = float(np.deg2rad(yaw_deg))

        print(f"\n[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
        if try_arm(arm, apply_base_se2(A_cur, 0.0, 0.0, yaw_rad, order="tr")):
            return 0.0, 0.0, yaw_deg

        for s in [0.05, 0.10, 0.20]:
            for dx, dy in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
                print(f"[TRY] yaw={yaw_deg:+.1f} dx={dx:+.3f} dy={dy:+.3f}")
                if try_arm(arm, apply_base_se2(A_cur, float(dx), float(dy), yaw_rad, order="tr")):
                    return float(dx), float(dy), yaw_deg

    YAW_FALLBACK_DEG = [90.0, -90.0, 180.0]
    print("\n[STRAT-1 / D] final yaw fallback (90/-90/180)")
    for yaw_deg in YAW_FALLBACK_DEG:
        yaw_rad = float(np.deg2rad(yaw_deg))
        print(f"[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
        if try_arm(arm, apply_base_se2(A_cur, 0.0, 0.0, yaw_rad, order="tr")):
            return 0.0, 0.0, yaw_deg

    return None


# ========================== STRATEGY 2 ======================================
def find_base_assist_strategy2(arm, A_cur: np.ndarray) -> tuple[float, float, float] | None:
    """
    Strategy-2 (fallback): SE(2) scan over yaw + ring translations (your older commented approach).
    Returns (dx, dy, yaw_deg) or None.
    NOTE: Feasibility test uses apply_base_se2 with order="rt" (rotate then translate).
    """
    print("\n[STRAT-2] SE(2) scan over (yaw, r, angle)")
    SE2_R_STEP = 0.05
    SE2_R_MAX = 2.0
    radii = np.arange(SE2_R_STEP, SE2_R_MAX + 1e-9, SE2_R_STEP)
    MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
    SE2_ANGLES = 16
    SE2_YAW_LIST_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180]

    for yaw_deg in SE2_YAW_LIST_DEG:
        yaw_rad = float(np.deg2rad(yaw_deg))

        for r in radii:
            for i in range(SE2_ANGLES):
                ang = 2.0 * np.pi * (i / SE2_ANGLES)
                dx = float(r * np.cos(ang))
                dy = float(r * np.sin(ang))

                if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS + 1e-9:
                    continue

                print(f"[TRY] yaw={yaw_deg:>4.0f} dx={dx:+.3f} dy={dy:+.3f}")
                A_try = apply_base_se2(A_cur, dx, dy, yaw_rad, order="rt")
                if try_arm(arm, A_try):
                    return dx, dy, float(yaw_deg)

    return None


# ============================== MAIN =========================================
def main() -> int:
    A_target = (Rz(TARGET_YAW_DEG) @ A).astype(float)
    A_cur = A_target.copy()

    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()

    arm = client._get_arm(SIDE)
    base = BaseController(client=client, world=None)

    print("[BASE] (optional) Resetting odometry once at start...")
    base.reset_odometry()
    time.sleep(0.5)

    try:
        print("\n[0] Try arm without moving base")
        if try_arm(arm, A_cur):
            print("[SUCCESS] Reached without base move.")
            return 0

        # ---------- Strategy 1 ----------
        print("\n[1] Strategy-1: translation-first search")
        best = find_base_assist_strategy1(arm, A_cur)

        # If strategy-1 fails to find ANY candidate, fall back to strategy-2
        if best is None:
            print("\n[1->2] Strategy-1 found no solution. Falling back to Strategy-2 (yaw+translation scan).")
            best = find_base_assist_strategy2(arm, A_cur)
            if best is None:
                print("[FAIL] No feasible base offset found in either strategy.")
                return 1

            dx_goal, dy_goal, yaw_deg_goal = best
            print(f"\n[BASE] (Strategy-2) Best motion: dx={dx_goal:+.3f}, dy={dy_goal:+.3f}, yaw={yaw_deg_goal:+.1f} deg")

            # Execute Strategy-2 order: rotate THEN translate (matches order="rt")
            yaw_abs = execute_yaw_in_steps(base, yaw_deg_goal)
            A_cur = apply_base_se2(A_cur, 0.0, 0.0, np.deg2rad(yaw_deg_goal), order="rt")

            trans_abs = execute_translation_in_steps(base, dx_goal, dy_goal)
            A_cur = apply_base_se2(A_cur, dx_goal, dy_goal, 0.0, order="rt")

            print(f"[BASE] Executed yaw_abs={yaw_abs:.1f}deg, trans_abs={trans_abs:.3f}m")

        else:
            dx_goal, dy_goal, yaw_deg_goal = best
            print(f"\n[BASE] (Strategy-1) Best motion: dx={dx_goal:+.3f}, dy={dy_goal:+.3f}, yaw={yaw_deg_goal:+.1f} deg")

            # Execute Strategy-1 order: translate THEN rotate (matches order="tr")
            trans_abs = execute_translation_in_steps(base, dx_goal, dy_goal)
            A_cur = apply_base_se2(A_cur, dx_goal, dy_goal, 0.0, order="tr")

            yaw_abs = execute_yaw_in_steps(base, yaw_deg_goal)
            A_cur = apply_base_se2(A_cur, 0.0, 0.0, np.deg2rad(yaw_deg_goal), order="tr")

            print(f"[BASE] Executed trans_abs={trans_abs:.3f}m, yaw_abs={yaw_abs:.1f}deg")

        # ---------- Final retry ----------
        print("\n[2] Retrying arm.goto after base assist")
        if try_arm(arm, A_cur):
            print("[SUCCESS] Reached after base assist.")
            return 0

        print("[FAIL] Feasible in search but failed after base motion (drift/slip/frames).")
        return 1

    finally:
        try:
            arm.turn_off_smoothly()
        except Exception:
            pass
        try:
            reachy = client.connect_reachy
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()
        print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
