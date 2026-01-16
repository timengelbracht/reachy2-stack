#!/usr/bin/env python
from __future__ import annotations

import time
import numpy as np
from matplotlib import pyplot as plt

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig
from reachy2_stack.control.arm import ArmController  # adjust import if needed


# --- CONFIG -------------------------------------------------------------------
HOST = "192.168.1.71"   # <- change to your Reachy IP
SIDE = "right"          # "left" or "right"
# ------------------------------------------------------------------------------


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()

    
    right_arm = client._get_arm("right")
    left_arm = client._get_arm("left")
    right_gripper = client._get_gripper("right")

    recorded_joints = [
    right_arm._shoulder.pitch,
    right_arm._shoulder.roll,
    right_arm._elbow.yaw,
    right_arm._elbow.pitch,
    right_arm._wrist.roll,
    right_arm._wrist.pitch,
    right_arm._wrist.yaw,

    # left_arm._shoulder.pitch,
    # left_arm._shoulder.roll,
    # left_arm._elbow.yaw,
    # left_arm._elbow.pitch,
    # left_arm._wrist.roll,
    # left_arm._wrist.pitch,
    # left_arm._wrist.yaw,
    ]

    sampling_frequency = 100  # in Hz
    record_duration = 5  # in sec.

    trajectories = []

    start = time.time()
    while (time.time() - start) < record_duration:
        
        current_point = [joint.present_position for joint in recorded_joints]
        
        trajectories.append(current_point)

        time.sleep(1 / sampling_frequency)

    # print (trajectories)
    # plt.figure()
    # plt.plot(trajectories)
    # plt.show()


    plt.figure()
    plt.plot(trajectories)         
    plt.title("Joint trajectories")
    plt.xlabel("Sample")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("/exchange/trajectories.png", dpi=200)
    print("done. will sleep now")


    time.sleep(5)
    right_arm.turn_on()

   
    for joints_positions in trajectories:
        for joint, pos in zip(recorded_joints, joints_positions):
            joint.goal_position = pos
        client.send_goal_positions(check_positions=False)
        time.sleep(1 / sampling_frequency)

    right_arm.turn_off_smoothly()
    client.close()


if __name__ == "__main__":
    main()
