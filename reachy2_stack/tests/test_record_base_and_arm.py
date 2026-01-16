#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import select
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.base import BaseController


# --- CONFIG -------------------------------------------------------------------
HOST = "192.168.1.71"

ARM_HZ = 100.0
BASE_HZ = 10.0

# Base playback: goto_odom is a "goal setter", not a perfect trajectory controller.
# Keeping wait=False and short timeout makes it stream-ish.
BASE_GOTO_TIMEOUT = 10.0
BASE_DIST_TOL = 0.05
BASE_ANG_TOL_DEG = 5.0
# ------------------------------------------------------------------------------


@dataclass
class BaseSample:
    t: float
    x: float
    y: float
    theta_deg: float


@dataclass
class ArmSample:
    t: float
    q: np.ndarray  # shape (14,)


def _stdin_has_key() -> bool:
    return select.select([sys.stdin], [], [], 0.0)[0] != []


def _read_key_nonblocking() -> Optional[str]:
    if not _stdin_has_key():
        return None
    ch = sys.stdin.read(1)
    return ch


def _sleep_to_rate(last_t: float, hz: float) -> float:
    """Sleep enough to maintain hz. Returns new time."""
    dt = 1.0 / hz
    now = time.time()
    to_sleep = (last_t + dt) - now
    if to_sleep > 0:
        time.sleep(to_sleep)
    return time.time()


def main() -> None:
    # NOTE: This expects to be run in a real terminal (not VSCode "Python output" pane).
    print(
        "\nControls:\n"
        "  b : start/stop+save BASE recording\n"
        "  a : start/stop+save ARM recording\n"
        "  p : play BASE then ARMS\n"
        "  q : quit\n"
    )
    print("Make sure this terminal has focus when pressing keys.\n")

    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    # --- Arms handles (same as your script) ---
    right_arm = client._get_arm("right")
    left_arm = client._get_arm("left")

    recorded_joints = [
        right_arm._shoulder.pitch,
        right_arm._shoulder.roll,
        right_arm._elbow.yaw,
        right_arm._elbow.pitch,
        right_arm._wrist.roll,
        right_arm._wrist.pitch,
        right_arm._wrist.yaw,
        left_arm._shoulder.pitch,
        left_arm._shoulder.roll,
        left_arm._elbow.yaw,
        left_arm._elbow.pitch,
        left_arm._wrist.roll,
        left_arm._wrist.pitch,
        left_arm._wrist.yaw,
    ]

    # --- Base controller ---
    base: Optional[BaseController] = None
    has_base = hasattr(reachy, "mobile_base") and reachy.mobile_base is not None
    if has_base:
        base = BaseController(client=client, world=None)
    else:
        print("[WARN] No mobile_base detected. Base record/play will be disabled.\n")

    base_traj: List[BaseSample] = []
    arm_traj: List[ArmSample] = []

    recording_base = False
    recording_arm = False

    last_base_tick = time.time()
    last_arm_tick = time.time()

    try:
        # If you prefer, you can avoid turning on everything and only turn on when playing.
        # client.turn_on_all()

        # Put stdin into cbreak mode so we can read single keys without Enter (POSIX).
        # If this fails (rare), you can fall back to "press Enter" input approach.
        import termios, tty

        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        reachy.mobile_base.turn_off()

        print("[READY] Waiting for key input...")

        while True:
            key = _read_key_nonblocking()
            if key is not None:
                key = key.lower()

                if key == "q":
                    print("\n[QUIT] Exiting.")
                    break

                if key == "b":
                    if not has_base:
                        print("[BASE] Not available on this robot.")
                    else:
                        recording_base = not recording_base
                        if recording_base:
                            base_traj.clear()
                            assert base is not None
                            print("[BASE] Resetting odometry then recording... (press 'b' again to stop/save)")
                            base.reset_odometry()
                            time.sleep(0.2)
                            last_base_tick = time.time()
                        else:
                            print(f"[BASE] Saved {len(base_traj)} samples.")

                if key == "a":
                    recording_arm = not recording_arm
                    if recording_arm:
                        arm_traj.clear()
                        print("[ARM] Recording... (press 'a' again to stop/save)")
                        last_arm_tick = time.time()
                    else:
                        print(f"[ARM] Saved {len(arm_traj)} samples.")

                if key == "p":
                    if not has_base:
                        print("[PLAY] Base not available -> will only play arms (if recorded).")
                    if len(base_traj) == 0 and len(arm_traj) == 0:
                        print("[PLAY] Nothing recorded yet.")
                        continue

                    print("[PLAY] Starting playback: base then arms...")

                    # ---- PLAY BASE FIRST ----
                    if has_base and len(base_traj) > 0:
                        assert base is not None
                        print("[PLAY][BASE] Resetting odometry...")
                        client.turn_on_all()
                        base.reset_odometry()
                        time.sleep(0.2)

                        t0 = base_traj[0].t
                        play_start = time.time()

                        for s in base_traj:
                            # replay at recorded timestamps
                            target_time = play_start + (s.t - t0)
                            while time.time() < target_time:
                                time.sleep(0.001)

                            base.goto_odom(
                                x=float(s.x),
                                y=float(s.y),
                                theta=float(s.theta_deg),
                                wait=False,
                                distance_tolerance=BASE_DIST_TOL,
                                angle_tolerance=BASE_ANG_TOL_DEG,
                                timeout=BASE_GOTO_TIMEOUT,
                            )

                        # give it a moment to settle
                        time.sleep(0.5)
                        print("[PLAY][BASE] Done.")

                    # ---- THEN PLAY ARMS ----
                    if len(arm_traj) > 0:
                        print("[PLAY][ARM] Turning arms on...")
                        right_arm.turn_on()
                        left_arm.turn_on()

                        # Smooth ramp to first recorded pose (safer than jumping)
                        first = arm_traj[0].q.astype(float)
                        q0 = np.array([j.present_position for j in recorded_joints], dtype=float)

                        ramp_T = 2.0
                        ramp_N = max(10, int(ramp_T * ARM_HZ))
                        for a in np.linspace(0.0, 1.0, ramp_N):
                            q = (1 - a) * q0 + a * first
                            for joint, pos in zip(recorded_joints, q):
                                joint.goal_position = float(pos)
                            client.send_goal_positions(check_positions=False)
                            time.sleep(1.0 / ARM_HZ)

                        t0 = arm_traj[0].t
                        play_start = time.time()
                        for s in arm_traj:
                            target_time = play_start + (s.t - t0)
                            while time.time() < target_time:
                                time.sleep(0.001)

                            for joint, pos in zip(recorded_joints, s.q):
                                joint.goal_position = float(pos)
                            client.send_goal_positions(check_positions=False)

                        time.sleep(0.2)
                        right_arm.turn_off_smoothly()
                        left_arm.turn_off_smoothly()
                        print("[PLAY][ARM] Done.")

                    print("[PLAY] Playback complete.")

            # --- RECORDING LOOPS (non-blocking) ---
            now = time.time()

            if recording_base and has_base:
                if now - last_base_tick >= (1.0 / BASE_HZ):
                    odo: Dict[str, float] = client.get_mobile_odometry()  # theta in degrees by default
                    base_traj.append(
                        BaseSample(
                            t=now,
                            x=float(odo["x"]),
                            y=float(odo["y"]),
                            theta_deg=float(odo["theta"]),
                        )
                    )
                    last_base_tick = now

            if recording_arm:
                if now - last_arm_tick >= (1.0 / ARM_HZ):
                    q = np.array([j.present_position for j in recorded_joints], dtype=float)
                    arm_traj.append(ArmSample(t=now, q=q))
                    last_arm_tick = now

            time.sleep(0.001)

        # restore terminal
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)

    finally:
        print("[CLEANUP] Closing client.")
        try:
            if hasattr(reachy, "mobile_base") and reachy.mobile_base is not None:
                reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
