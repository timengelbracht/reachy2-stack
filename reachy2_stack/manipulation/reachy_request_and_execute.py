#!/usr/bin/env python3
from __future__ import annotations

import sys
sys.path.insert(0, "/exchange")

import os
import time
import io
import binascii
import argparse
import numpy as np
import requests

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.arm import ArmController
from reachy2_stack.control.base import BaseController
from reachy2_stack.control.gripper import GripperController



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


def exec_traj(arm: ArmController, traj_T: np.ndarray, step_sleep: float, label: str, gripper = None) -> bool:
    if traj_T.ndim != 3 or traj_T.shape[1:] != (4, 4):
        raise ValueError(f"{label}: expected (T,4,4), got {traj_T.shape}")

    for idx in [0, min(1, len(traj_T)-1), len(traj_T)-1]:
        if not is_valid_T(traj_T[idx]):
            raise ValueError(f"{label}: invalid T at idx {idx}\n{traj_T[idx]}")

    print(f"\n--- Executing {label} --- len={len(traj_T)}")
    for i, T_base_ee in enumerate(traj_T):
        if i ==1 and label=="post_grasp":
            gripper.close()
            time.sleep(1.0)
        if not label=="post_grasp":
            ok = arm.goto_pose_base(T_base_ee=T_base_ee)
        else:
            ok = arm.goto_pose_base_with_base_assist(T_base_ee=T_base_ee)
        if not ok:
            print(f"{label}: FAIL at {i}/{len(traj_T)-1}")
            return False
        time.sleep(step_sleep)
    return True

def exec_traj_final(arm: ArmController, traj_T: np.ndarray, step_sleep: float, label: str, gripper = None) -> bool:
    if traj_T.ndim != 3 or traj_T.shape[1:] != (4, 4):
        raise ValueError(f"{label}: expected (T,4,4), got {traj_T.shape}")

    for idx in [0, min(1, len(traj_T)-1), len(traj_T)-1]:
        if not is_valid_T(traj_T[idx]):
            raise ValueError(f"{label}: invalid T at idx {idx}\n{traj_T[idx]}")

    print(f"\n--- Executing final move")
    for i, T_base_ee in enumerate(traj_T):
        if i ==0:
            ok = arm.goto_pose_base(T_base_ee=T_base_ee)
        else:
            break
    return True

import cv2
import numpy as np
import tempfile
import os

def capture_rgb_depth_png_bytes(reachy) -> tuple[bytes, bytes]:
    """
    Capture RGB + depth from Reachy and return PNG bytes.

    RGB:
      - uint8
      - saved as PNG (BGR->RGB handled if needed by vidbot)

    Depth:
      - uint16
      - saved as 16-bit PNG
      - NO processing, NO scaling
    """

    # -----------------------------
    # Capture from Reachy
    # -----------------------------
    rgb_bgr, ts = reachy.cameras.depth.get_frame()
    rgb = np.asarray(rgb_bgr, dtype=np.uint8)      # (H,W,3), uint8

    depth_raw, _ = reachy.cameras.depth.get_depth_frame()
    depth = np.asarray(depth_raw, dtype=np.uint16) # (H,W), uint16

    # -----------------------------
    # Save exactly as required
    # -----------------------------
    # Use temp files to avoid polluting FS
    with tempfile.TemporaryDirectory() as tmp:
        rgb_path = os.path.join(tmp, "000000_rgb.png")
        depth_path = os.path.join(tmp, "000000_depth.png")

        ok = cv2.imwrite(rgb_path, rgb)          # uint8
        if not ok:
            raise RuntimeError("Failed to write RGB PNG")

        ok = cv2.imwrite(depth_path, depth)      # uint16, raw
        if not ok:
            raise RuntimeError("Failed to write depth PNG")

        with open(rgb_path, "rb") as f:
            rgb_bytes = f.read()
        with open(depth_path, "rb") as f:
            depth_bytes = f.read()

    return rgb_bytes, depth_bytes



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reachy-host", type=str, default="192.168.1.71")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"])

    # USER INPUTS YOU ASKED FOR:
    parser.add_argument("-o", "--object", dest="object_name", type=str, required=True,
                        help="Object name (passed to vidbot -o)")
    parser.add_argument("-i", "--action", dest="action", type=str, required=True,
                        help="Action verb (passed to vidbot -i)")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="If set, vidbot runs inference with -v")

    # Server endpoint
    parser.add_argument("--vidbot-url", type=str, required=True,
                        help="Vidbot server URL, e.g. http://GPU_MACHINE_IP:9000/infer_and_return_trajectories")

    # Timing
    parser.add_argument("--pre-sleep", type=float, default=0.5)
    parser.add_argument("--post-sleep", type=float, default=0.5)

    args = parser.parse_args()

    HOST = args.reachy_host
    SIDE = args.side
    object_name = args.object_name
    action = args.action
    visualize = args.visualize
    VIDBOT_URL = args.vidbot_url

    print("Prompt:", {"object_name": object_name, "action": action, "visualize": visualize})
    # 0) Connect Reachy + execute
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()

    arm = ArmController(client=client, side=SIDE, world=None)
    base = BaseController(client=client, world=None)
    gripper = GripperController(client=client, side=SIDE)
    reachy = client.connect_reachy
    gripper.open()

    # 1) Capture RGBD on Reachy
    rgb_bytes, depth_bytes = capture_rgb_depth_png_bytes(reachy)


    # 2) Send to vidbot server
    files = {
        "rgb_png": ("000000.png", rgb_bytes, "image/png"),
        "depth_png": ("000000.png", depth_bytes, "image/png"),
    }
    data = {
        "object_name": object_name,
        "action": action,
        "visualize": "true" if visualize else "false",
    }

    print("Requesting inference from:", VIDBOT_URL)
    r = requests.post(VIDBOT_URL, files=files, data=data, timeout=600)
    r.raise_for_status()
    resp = r.json()

    # 3) Decode returned payload
    # Server returns trajectory payload as hex string (see vidbot_server.py)
    payload_hex = resp["payload_npz_bytes"]
    payload = binascii.unhexlify(payload_hex)

    z = np.load(io.BytesIO(payload), allow_pickle=True)
    pre_T = np.asarray(z["pre_T"], dtype=float)
    post_T = np.asarray(z["post_T"], dtype=float)
    K_PRE = 3
    K_POST = 10

    pre_T = pre_T[::K_PRE]
    post_T = post_T[::K_POST]

    print("Received:")
    print(" pre_T :", pre_T.shape)
    print(" post_T:", post_T.shape)
    if "best_idx" in z.files:
        print(" best_idx:", int(z["best_idx"]))
    if "best_loss" in z.files:
        print(" best_loss:", float(z["best_loss"]))



    try:
        ok = exec_traj(arm, pre_T, args.pre_sleep, "pre_grasp",gripper=gripper)
        if not ok:
            return 1

        # gripper.close()
        # time.sleep(1.0)

        ok = exec_traj(arm, post_T, args.post_sleep, "post_grasp",gripper=gripper)

        time.sleep(1.0)
        ok = exec_traj_final(arm, pre_T, args.pre_sleep, "pre_grasp",gripper=gripper)
        return 0 if ok else 1


    finally:
        try:
            pass
            # gripper.open()
            
        except Exception:
            pass
        # try:
        #     client._get_arm(SIDE).turn_off_smoothly()
        # except Exception:
        #     pass
        try:
            reachy.mobile_base.turn_off()
        except Exception:
            pass

        try:
            client.close()
        except Exception:
            pass
       


if __name__ == "__main__":
    raise SystemExit(main())

# python reachy2_stack/manipulation/reachy_request_and_execute.py   -o toy -i pickup    --vidbot-url http://192.168.1.223:9000/infer_and_return_trajectories   -v