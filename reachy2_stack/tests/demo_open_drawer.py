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
# ------------------------------------------------------------------------------


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    #initialize the base controller
    base = BaseController(client=client, world=None)
    right_arm = client._get_arm("right")
    left_arm = client._get_arm("left")
    right_arm.turn_off_smoothly()
    left_arm.turn_off_smoothly()
    reachy.mobile_base.turn_off()
    #turn on the drivers. Warning, do not be close to the robot or try to move it/ its arms
    client.turn_on_all()

    client.goto_head([0.0,0.0,-45.0],2,False,interpolation_mode = 'minimum_jerk',degrees=True)

    # Arm joints that can be actuated
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

    # we reset the odometry to always work in base frame
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
        
    time.sleep(1)

    #load arm movements
    data = np.load(SAVE_PATH)
    traj = data["traj"]             
    hz = float(data["hz"])          
    dt = 1.0 / hz

    # move the arm
    for q in traj:
        for joint, pos in zip(recorded_joints, q):
            joint.goal_position = float(pos)
        client.send_goal_positions(check_positions=False)
        time.sleep(dt)

    time.sleep(1)

    # reset the base and move it back a bit to open the drawer

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
    
    time.sleep(1)

    # go back to the original place

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
    
    # remove the arm
    for q in traj[::-1]:
        for joint, pos in zip(recorded_joints, q):
            joint.goal_position = float(pos)
        client.send_goal_positions(check_positions=False)
        time.sleep(dt)

    time.sleep(1)

    client.goto_head([0.0,0.0,0.0],2,False,interpolation_mode = 'minimum_jerk',degrees=True)

    time.sleep(1)

    base.reset_odometry()
    with open("/exchange/reposition_base_odom.json", "r") as f:
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
    

    # turn off everything

    right_arm.turn_off_smoothly()
    left_arm.turn_off_smoothly()
    reachy.mobile_base.turn_off()

if __name__ == "__main__":
    main()
