#!/usr/bin/env python
from __future__ import annotations

import time
import numpy as np
from matplotlib import pyplot as plt
import json

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig
from reachy2_stack.control.arm import ArmController  # adjust import if needed
from reachy2_stack.control.base import BaseController 


# --- CONFIG -------------------------------------------------------------------
HOST = "192.168.1.71"
SAVE_PATH = "/exchange/arm_traj.npz"
SAVE_PATH_2 = "/exchange/closing_arm_traj.npz"


# ------------------------------------------------------------------------------


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy
    base = BaseController(client=client, world=None)

    client.turn_on_all()

    
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

    
  
    base.reset_odometry()
    with open("/exchange/first_base_odom.json", "r") as f:
        pose = json.load(f)

    base.goto_odom(
                                x=float(pose["x"]),
                                y=float(pose["y"]),
                                theta=float(pose["theta"]),
                                wait=True,
                                distance_tolerance=0.01,
                                angle_tolerance=0.5,
                                timeout=6.0,
                )
        
    time.sleep(5)

    data = np.load(SAVE_PATH)
    traj = data["traj"]              # (T, 14)
    hz = float(data["hz"])           # original sampling frequency
    dt = 1.0 / hz


    for q in traj:
        for joint, pos in zip(recorded_joints, q):
            joint.goal_position = float(pos)
        client.send_goal_positions(check_positions=False)
        time.sleep(dt)

    

    

    time.sleep(5)
    base.reset_odometry()
    with open("/exchange/mid_base_odom.json", "r") as f:
        pose = json.load(f)
    base.goto_odom(
                                x=float(pose["x"]),
                                y=float(pose["y"]),
                                theta=float(pose["theta"]),
                                wait=True,
                                distance_tolerance=0.01,
                                angle_tolerance=0.5,
                                timeout=5.0,
                )
    
    time.sleep(5)




    with open("/exchange/final_base_odom.json", "r") as f:
        pose = json.load(f)
    base.goto_odom(
                                x=float(pose["x"]),
                                y=float(pose["y"]),
                                theta=float(pose["theta"]),
                                wait=True,
                                distance_tolerance=0.01,
                                angle_tolerance=0.5,
                                timeout=5.0,
                )
    

   

    for q in traj[::-1]:
        for joint, pos in zip(recorded_joints, q):
            joint.goal_position = float(pos)
        client.send_goal_positions(check_positions=False)
        time.sleep(dt)

    
    right_arm.turn_off_smoothly()
    left_arm.turn_off_smoothly()
    reachy.mobile_base.turn_off()

if __name__ == "__main__":
    main()
