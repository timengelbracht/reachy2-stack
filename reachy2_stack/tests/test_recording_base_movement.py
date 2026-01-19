#!/usr/bin/env python
from __future__ import annotations

import time
import json

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.base import BaseController  # adjust import if needed


# --- CONFIG -------------------------------------------------------------------
HOST = "192.168.1.71"   # <- change to your Reachy IP
SAVE_PATH = "/exchange/reposition_base_odom.json"
# ------------------------------------------------------------------------------


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

    try:
        print("[BASE] Resetting odometry...")
        base.reset_odometry()
        time.sleep(0.5)

        trajectories = []

        sampling_frequency = 10  # in Hz
        record_duration = 30  # in sec.

        start = time.time()
        reachy.mobile_base.turn_off()
        print("started recording")
        while (time.time() - start) < record_duration:
        
            current_point = client.get_mobile_odometry()
            
            trajectories.append(current_point)

            time.sleep(1 / sampling_frequency)
        
        print("Done recording")
        time.sleep(20)
        time.sleep(0.5)
        print("[BASE] Resetting odometry...")
        base.reset_odometry()
        client.turn_on_all()
        time.sleep(0.5)
        print("started playing")
       
        pose = trajectories[-1]
        with open(SAVE_PATH, "w") as f:
            json.dump(pose, f, indent=2)

        print(f"[BASE] Final odometry saved to {SAVE_PATH}")
        # print(pose)
        # print(client.get_mobile_odometry())
                
        # base.goto_odom(
        #                         x=float(pose["x"]),
        #                         y=float(pose["y"]),
        #                         theta=float(pose["theta"]),
        #                         wait=True,
        #                         distance_tolerance=0.01,
        #                         angle_tolerance=0.5,
        #                         timeout=30.0,
        #         )
        
        # print("ended playing")
        # print(client.get_mobile_odometry())



    finally:
        print("[BASE] Turning mobile base off and closing client.")
        try:
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()