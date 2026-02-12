#!/usr/bin/env python3
from __future__ import annotations

import sys
sys.path.insert(0, "/exchange")  # or your repo root

import os
import time
import numpy as np

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.arm import ArmController
from reachy2_stack.control.base import BaseController
from reachy2_stack.control.gripper import GripperController


# -----------------------------
# Loaders
# -----------------------------
def load_traj_T(path: str) -> np.ndarray:
    """
    Loads a trajectory of homogeneous transforms.

    Supported:
      - *.npy: (T,4,4) or (T,3)
      - *.npz: expects key 'transforms' or 'positions'
    Returns:
      traj_T: (T,4,4) float64
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}\nCWD: {os.getcwd()}")

    if path.endswith(".npy"):
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(float)

        if arr.ndim == 2 and arr.shape[1] == 3:
            # positions-only -> identity rotations
            T = arr.shape[0]
            out = np.tile(np.eye(4, dtype=float), (T, 1, 1))
            out[:, :3, 3] = arr.astype(float)
            return out

        raise ValueError(f"Unsupported .npy shape: {arr.shape}")

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        if "transforms" in z.files:
            traj_T = np.asarray(z["transforms"], dtype=float)
            if traj_T.ndim != 3 or traj_T.shape[1:] != (4, 4):
                raise ValueError(f"Bad 'transforms' shape in {path}: {traj_T.shape}")
            return traj_T
        if "positions" in z.files:
            pos = np.asarray(z["positions"], dtype=float)
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError(f"Bad 'positions' shape in {path}: {pos.shape}")
            T = pos.shape[0]
            out = np.tile(np.eye(4, dtype=float), (T, 1, 1))
            out[:, :3, 3] = pos
            return out
        raise ValueError(f"No 'transforms' or 'positions' in {path}. keys={z.files}")

    raise ValueError(f"Unsupported file type: {path}")


def is_valid_T(T: np.ndarray, tol: float = 1e-3) -> bool:
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], np.array([0, 0, 0, 1], dtype=float), atol=tol):
        return False
    R = T[:3, :3]
    if not np.isfinite(R).all() or not np.isfinite(T[:3, 3]).all():
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=5e-2):
        return False
    det = np.linalg.det(R)
    return 0.5 < det < 1.5


def subsample(traj_T: np.ndarray, K: int) -> np.ndarray:
    if K <= 1:
        return traj_T
    return traj_T[::K]


def max_step_distance(traj_T: np.ndarray) -> float:
    if traj_T.shape[0] < 2:
        return 0.0
    p = traj_T[:, :3, 3]
    return float(np.max(np.linalg.norm(np.diff(p, axis=0), axis=1)))


# -----------------------------
# Execution
# -----------------------------
def execute_trajectory(
    arm: ArmController,
    traj_T: np.ndarray,
    step_sleep_sec: float,
    stop_on_fail: bool = True,
    label: str = "traj",
    gripper = None,
) -> bool:
    if traj_T.shape[0] == 0:
        print(f"[{label}] Empty trajectory, skipping.")
        return True

    # Validate first/last
    for idx in [0, min(1, len(traj_T) - 1), len(traj_T) - 1]:
        if not is_valid_T(traj_T[idx]):
            raise ValueError(f"[{label}] Invalid transform at idx {idx}:\n{traj_T[idx]}")

    print(f"\n--- Executing {label} ---")
    print(f"[{label}] length: {len(traj_T)}")
    print(f"[{label}] first xyz: {traj_T[0][:3, 3]}")
    print(f"[{label}] last  xyz: {traj_T[-1][:3, 3]}")
    print(f"[{label}] max step distance (m): {max_step_distance(traj_T):.4f}")

    for i, T_base_ee in enumerate(traj_T):
        if i >=1 and label=="post_grasp":
            print("CLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOSSSSSSSSSSEEEEEEEEE")
            # gripper.open()
            time.sleep(1.0)
        ok = arm.goto_pose_base_with_base_assist(T_base_ee=T_base_ee)
        if not ok:
            print(f"[{label}] FAIL at waypoint {i}/{len(traj_T)-1}")
            if stop_on_fail:
                return False
        else:
            print(f"[{label}] OK   {i}/{len(traj_T)-1}")

        time.sleep(step_sleep_sec)

    print(f"--- Done {label} ---")
    return True


def main() -> int:
    HOST = "192.168.1.71"
    SIDE = "right"

    # Update these paths to wherever you saved them
    PRE_PATH  = "/exchange/reachy2_stack/manipulation/pre_grasp_T_base_ee.npy"
    POST_PATH = "/exchange/reachy2_stack/manipulation/post_grasp_T_base_ee.npy"
    # Alternatively you can load the .npz:
    # PRE_PATH  = "/exchange/reachy2_stack/manipulation/pre_grasp_trajectory.npz"
    # POST_PATH = "/exchange/reachy2_stack/manipulation/post_grasp_trajectory.npz"

    # Timing/subsampling per segment
    PRE_SLEEP  = 0.5
    POST_SLEEP = 0.5
    PRE_K  = 10     # keep 1 out of every K poses
    POST_K = 10

    STOP_ON_FAIL = False
    # --- Load both segments ---
    pre_T  = subsample(load_traj_T(PRE_PATH),  PRE_K)
    post_T = subsample(load_traj_T(POST_PATH), POST_K)
    print(post_T.size)

    # --- Connect ---
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()

    arm = ArmController(client=client, side=SIDE, world=None)
    base = BaseController(client=client, world=None)
    gripper = GripperController(client=client, side=SIDE)
    reachy = client.connect_reachy
    # client._get_arm(SIDE).turn_off_smoothly()
    try:
        print("Pre_T.")
        print(pre_T)
        print("Post_T.")
        print(post_T)
        ok = execute_trajectory(arm, pre_T, step_sleep_sec=PRE_SLEEP, stop_on_fail=STOP_ON_FAIL, label="pre_grasp",gripper=gripper)
        if not ok:
            return 1

        # -----------------------------
        # GRASP ACTION HERE
        # -----------------------------
        # time.sleep(10)
        # gripper.close()


        ok = execute_trajectory(arm, post_T, step_sleep_sec=POST_SLEEP, stop_on_fail=STOP_ON_FAIL, label="post_grasp",gripper=gripper)
        return 0 if ok else 1

    finally:
        try:
            print("Turning off.")
            gripper.open()
            client._get_arm(SIDE).turn_off_smoothly()
        except Exception:
            pass
        try:
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()
        print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
