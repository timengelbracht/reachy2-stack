#!/usr/bin/env python
from __future__ import annotations

import time
import numpy as np

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

    # Minimal world model (only really needed if you later use world-frame APIs)
    world = WorldModel(WorldModelConfig(location_name="lab", localization_mode="odom"))

    arm = ArmController(client=client, side=SIDE, world=world)
    reachy = client.connect_reachy

    try:
        client.turn_on_all()

        print("[ARM] Turning robot on and going to default posture...")
        client.goto_posture("default", wait=True)

        # 1) Joint-space goto -------------------------------------------------
        print("[ARM] Reading current joint state...")
        if SIDE == "right":
            q_curr, _ = client.get_joint_state_right()
        else:
            q_curr, _ = client.get_joint_state_left()
        print(f"[ARM] Current joints ({SIDE}): {np.round(q_curr, 2)}")

        q_target = q_curr.copy()
        q_target[3] += -15.0  # small elbow bend
        print(f"[ARM] Joint-space goto -> {np.round(q_target, 2)}")
        arm.goto_joints(q_target, duration=3.0, wait=True)

        time.sleep(0.5)

        # 2) Forward / inverse kinematics ------------------------------------
        print("[ARM] Forward kinematics at current configuration...")
        T_fk = arm.forward_kinematics()
        print("[ARM] T_base_ee:\n", np.array2string(T_fk, precision=3))

        print("[ARM] Inverse kinematics on that pose...")
        q_ik = arm.inverse_kinematics(T_fk)
        print(f"[ARM] IK joints: {np.round(q_ik, 2)}")
        print(f"[ARM] IK - current: {np.round(q_ik - q_curr, 3)}")

        # 3) Cartesian goto in base frame ------------------------------------
        print("[ARM] Cartesian goto: +5cm forward in base frame...")
        T_target = T_fk.copy()
        T_target[2, 3] += 0.05  # +5 cm in +X

        arm.goto_pose_base(
            T_target,
            duration=3.0,
            wait=True,
            interpolation_mode="minimum_jerk",
            interpolation_space="joint_space",
        )

        time.sleep(0.5)

        # 4) Return to original joint configuration --------------------------
        print("[ARM] Returning to original joint configuration...")
        arm.goto_joints(q_curr, duration=3.0, wait=True)

        print("[ARM] Arm test DONE ✅")

    finally:
        print("[ARM] Turning arm off smoothly and closing client.")
        if SIDE == "right":
            reachy.r_arm.turn_off_smoothly()
        else:
            reachy.l_arm.turn_off_smoothly()
        client.close()


if __name__ == "__main__":
    main()
