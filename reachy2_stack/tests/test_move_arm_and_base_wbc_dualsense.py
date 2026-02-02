#!/usr/bin/env python3
from __future__ import annotations

import sys
sys.path.insert(0, "/exchange")

import time
import numpy as np

from pydualsense import pydualsense
from pydualsense.enums import PlayerID
from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.arm import ArmController


# ---------------- CONFIG ----------------
HOST = "192.168.1.71"
SIDE = "right"

# Calibration / feel
CALIB_CENTER_SECONDS = 1.0
CALIB_RANGE_SECONDS = 2.0
RAW_DEADZONE = 6
EXPO = 1.15

# Loop
CMD_HZ = 30
DT = 1.0 / CMD_HZ

# Speeds (per second at full stick deflection)
LIN_SPEED = 0.25   # m/s
ANG_SPEED = 1.5    # rad/s

# Optional Z buttons (NOT R1)
ENABLE_Z = True
Z_SPEED_BTN = 0.15  # m/s
Z_DOWN_BTN = "L2Btn"
Z_UP_BTN = "R2Btn"  # click R2

QUIT_BTN_OPTIONS = True

# Optional clamps
CLAMP_XYZ = False
XYZ_MIN = np.array([-2.0, -2.0, -2.0])
XYZ_MAX = np.array([ 2.0,  2.0,  2.0])
# --------------------------------------


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


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)
def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def apply_delta_to_T(T: np.ndarray, dp: np.ndarray, droll: float, dyaw: float, dpitch: float) -> np.ndarray:
    """
    BASE-frame translation. Orientation update is body-fixed:
      R <- R * Rz(dyaw) * Rx(droll)
    """
    Tn = T.copy()
    R = Tn[:3, :3]
    p = Tn[:3, 3]

    p = p + dp
    # R = R @ rot_z(dyaw) @ rot_y(dpitch) @ rot_x(droll)
    R = rot_z(dyaw) @ rot_y(dpitch) @ rot_x(droll) @ R


    Tn[:3, :3] = R
    Tn[:3, 3] = p
    return Tn


def main() -> int:
    # Initial pose (example)
    theta = np.deg2rad(0.0)
    Rz0 = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, 0.0],
        [np.sin(theta),  np.cos(theta), 0.0, 0.0],
        [0.0,            0.0,           1.0, 0.0],
        [0.0,            0.0,           0.0, 1.0],
    ])
    A = np.array(
        [
            [0, 0, -1,  0.4],
            [0, 1,  0, -0.4],
            [1, 0,  0,  0.1],
            [0, 0,  0,  1.0],
        ],
        dtype=float,
    )
    T_base_ee = Rz0 @ A

    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()
    # client._get_arm(SIDE).turn_off_smoothly()

    # You use reachy.goto directly
    reachy_arm = client.connect_reachy._r_arm
    _ = ArmController(client=client, side=SIDE, world=None)  # keep if you need it elsewhere
    ok = reachy_arm.goto(T_base_ee, duration= 2.0, wait=True)

    ds = pydualsense()
    ds.init()

    try:
        print("\n[DUALSENSE] Center calibration: DO NOT touch sticks...")
        centers = calibrate_center(ds, CALIB_CENTER_SECONDS)
        print("[DUALSENSE] centers:", {k: round(v, 2) for k, v in centers.items()})

        print(f"[DUALSENSE] Range calibration: MOVE sticks to corners for {CALIB_RANGE_SECONDS:.1f}s...")
        mins, maxs = calibrate_range(ds, CALIB_RANGE_SECONDS)
        print("[DUALSENSE] mins:", mins)
        print("[DUALSENSE] maxs:", maxs)

        print(
            "\n[EE TELEOP]\n"
            "Hold R1 to update+SEND (dead-man)\n"
            "Left stick: translate (LY->X forward/back, LX->Y left/right)\n"
            "Right stick: RX->yaw, RY->roll\n"
            + (f"{Z_DOWN_BTN}: Z down, {Z_UP_BTN}: Z up\n" if ENABLE_Z else "")
            + "Options: quit\n"
        )

        while True:
            st = ds.state

            if QUIT_BTN_OPTIONS and st.options:
                break

            # ONLY act while R1 is pressed
            if not st.R1:
                time.sleep(DT)
                continue

            # Normalize sticks
            lx = stick_u8_to_unit(st.LX, vmin=mins["LX"], vcenter=centers["LX"], vmax=maxs["LX"],
                                  deadzone_raw=RAW_DEADZONE, expo=EXPO)
            ly = stick_u8_to_unit(st.LY, vmin=mins["LY"], vcenter=centers["LY"], vmax=maxs["LY"],
                                  deadzone_raw=RAW_DEADZONE, expo=EXPO)


            # Translation (base frame)
            dx = (-ly) * LIN_SPEED * DT   # 
            dy = (-lx) * LIN_SPEED * DT   #

            dz = 0.0
            if ENABLE_Z:
                if getattr(st, Z_DOWN_BTN):
                    dz -= Z_SPEED_BTN * DT
                if getattr(st, Z_UP_BTN):
                    dz += Z_SPEED_BTN * DT

            dp = np.array([dx, dy, dz], dtype=float)


            yaw_dir = 0.0
            roll_dir = 0.0
            # LED feedback (pick ONE color each loop)
            if st.DpadLeft:
                yaw_dir += 1.0
                ds.light.setColorI(255, 0, 0)   # red
            elif st.DpadRight:
                yaw_dir -= 1.0
                ds.light.setColorI(0, 255, 0)   # green
            else:
                ds.light.setColorI(0, 0, 255)   # blue


            roll_dir = 0.0
            if st.DpadUp:
                roll_dir -= 1.0
                ds.light.setColorI(0, 255, 0)   # green
            elif st.DpadDown:
                roll_dir += 1.0
                ds.light.setColorI(255, 0, 0)   # red
            else:
                ds.light.setColorI(0, 0, 255)   # blue
                
            
            pitch_dir = 0.0
            if st.circle:
                pitch_dir += 1.0
                ds.light.setColorI(0, 255, 0)   # green
            elif st.square:
                pitch_dir -= 1.0
                ds.light.setColorI(255, 0, 0)   # green
            else:
                ds.light.setColorI(0, 0, 255)   # blue

    

            dyaw = yaw_dir * ANG_SPEED * DT
            droll = roll_dir* ANG_SPEED * DT 
            dpitch = pitch_dir * ANG_SPEED * DT


            # Apply & clamp
            T_base_ee = apply_delta_to_T(T_base_ee, dp, droll, dyaw, dpitch)

            if CLAMP_XYZ:
                p = T_base_ee[:3, 3]
                p = np.minimum(np.maximum(p, XYZ_MIN), XYZ_MAX)
                T_base_ee[:3, 3] = p

            # SEND (only while R1 held)
            ok = reachy_arm.goto(T_base_ee, duration= 0.01, wait=False)
            if not ok:
                print("[EE] goto failed for this target (keeping last target).")

            time.sleep(DT)

        return 0

    finally:
        try:
            ds.close()
        except Exception:
            pass
        try:
            client._get_arm(SIDE).turn_off_smoothly()
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
