#!/usr/bin/env python3
from __future__ import annotations

import sys
sys.path.insert(0, "/exchange")  # or your repo root


import time
import numpy as np

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.arm import ArmController
from reachy2_stack.control.base import BaseController


# If you want to test world-frame too (optional):
# from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig


def main() -> int:
    HOST = "192.168.1.71"
    SIDE = "right"

    # A target pose in BASE frame (base <- ee)
    # Replace with your real pose(s)

    theta = np.deg2rad(0.0)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, 0.0],
        [np.sin(theta),  np.cos(theta), 0.0, 0.0],
        [0.0,            0.0,           1.0, 0.0],
        [0.0,            0.0,           0.0, 1.0],
    ])
    A = np.array(
        [
            [0, 0, -1, -1.4],
            [0, 1,  0, -0.4],
            [1, 0,  0,  0.1],
            [0, 0,  0,  1.0],
        ],
        dtype=float,
    )
    T_base_ee = Rz @ A



    # --- Connect client ---
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()
   
    

    # --- Create controller (no world model needed for base-frame tests) ---
    arm = ArmController(client=client, side=SIDE, world=None)
    
    base = BaseController(client=client, world=None)
    # base.reset_odometry()

    try:

        print("\n==============================")
        print("start of the test")
        print("==============================")
        ok = arm.goto_pose_base_with_base_assist(
            T_base_ee=T_base_ee
        )
        print("Result:", "SUCCESS" if ok else "FAIL")
        time.sleep(1.0)


        return 0 if (ok) else 1

    finally:
        try:
            print("turning off.")
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



