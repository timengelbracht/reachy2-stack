#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from reachy2_stack.core.client import ReachyClient, ReachyConfig
from reachy2_stack.control.arm import ArmController


HOST = "192.168.1.71"   # your Reachy IP
SIDE = "left"          # or "left"


def main() -> None:
    # --- connect --------------------------------------------------------------
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect_reachy


    client.turn_on_all()
    arm = ArmController(client=client, side=SIDE, world=None)

    # --- your apex joint pose (in degrees) ------------------------------------
    # q_apex = np.array([
    #     0.0,   # shoulder pitch
    #     50.0,   # shoulder roll
    #     90.0,   # arm yaw
    #     -120.0,  # elbow pitch -100,
    #     0.0,   # wrist yaw
    #     0.0,   # wrist pitch
    #     -90.0,   # wrist roll
    # ], dtype=float)

    q_apex = np.array([
        -20.0,   # shoulder pitch
        0.0,   # shoulder roll
        0.0,   # arm yaw
        -135.0,  # elbow pitch -120,
        0.0,   # wrist yaw
        0.0,   # wrist pitch
        -30.0,   # wrist roll
    ], dtype=float)

    # move to apex first
    arm.goto_joints(q_apex, duration=2.0, wait=True)

    # --- get EE pose at apex --------------------------------------------------
    # T_base_ee: 4x4 homogeneous transform in base frame
    T0 = arm.forward_kinematics(q_apex)
    T0 = np.asarray(T0, dtype=float)
    p0 = T0[:3, 3].copy()   # position
    R0 = T0[:3, :3].copy()  # orientation (kept fixed)

    # --- define circular arc around apex --------------------------------------
    radius = 0.2          # [m] radius of the partial circle
    theta_max_deg = 25.0   # max angle on each side
    n_steps = 1           # resolution per half-wave
    move_duration = 0.3    # seconds per small move

    theta_max = np.deg2rad(theta_max_deg)

    # We build a symmetric sequence:
    #   0 -> +theta_max -> 0 -> -theta_max -> 0
    thetas = np.concatenate([
        np.linspace(0.0,  theta_max, n_steps, endpoint=True),
        np.linspace(theta_max, 0.0,  n_steps, endpoint=True),
        np.linspace(0.0, -theta_max, n_steps, endpoint=True),
        np.linspace(-theta_max, 0.0, n_steps, endpoint=True),
    ])

    # Circle is in the base YZ-plane, with center below the hand:
    #   center = p0 + [0, 0, -radius]
    #   offset(theta) = [0,
    #                    radius * sin(theta),
    #                    radius * (cos(theta) - 1)]
    # so theta=0 => offset = 0 (apex pose)
    for theta in thetas:
        offset = np.array([
            0.0,
            radius * np.sin(theta),
            radius * (np.cos(theta) - 1.0),
        ])

        p = p0 + offset
        T = np.eye(4)
        T[:3, :3] = R0
        T[:3, 3] = p

        arm.goto_pose_base(T, duration=move_duration, wait=True)
        # optional small sleep if you want a pause
        # time.sleep(0.05)
        a = 2

    print("Finished circular wave arc.")


if __name__ == "__main__":
    main()