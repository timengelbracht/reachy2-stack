#!/usr/bin/env python3
"""
Interactive debugging script for verifying WorldModel + SE(3) plumbing.

Run with:
    python tests/test_world_model.py
"""

import pprint
import numpy as np

from reachy2_stack.core.client import ReachyClient, ReachyConfig
from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    # ------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------
    print_header("1. Loading config")
    cfg = ReachyConfig.from_yaml("config/config.yaml")
    pprint.pprint(cfg)

    # ------------------------------------------------------
    # 2. Connect to Reachy
    # ------------------------------------------------------
    print_header("2. Connecting to Reachy")
    client = ReachyClient(cfg)
    client.connect()
    print("Connected.")

    # ------------------------------------------------------
    # 3. Init WorldModel from client calib
    # ------------------------------------------------------
    print_header("3. Initializing WorldModel from client calibration")

    wm_cfg = WorldModelConfig(
        location_name="default",
        localization_mode="odom",  # for now we trust odom only
    )
    world = WorldModel(wm_cfg)

    world.init_from_client_calib(client)
    print("WorldModel camera calibration initialized.")

    # ------------------------------------------------------
    # 4. Update base pose from odometry
    # ------------------------------------------------------
    print_header("4. Updating T_world_base from mobile odometry")

    odom = client.get_mobile_odometry()
    print("Raw odometry dict:", odom)

    if odom is not None:
        world.update_from_odom(odom)
    else:
        print("No mobile_base present -> T_world_base stays identity.")

    T_world_base = world.get_T_world_base()
    print("T_world_base:\n", T_world_base)

    # ------------------------------------------------------
    # 5. Set EE poses in base frame (left / right)
    # ------------------------------------------------------
    print_header("5. Setting EE poses from Reachy forward_kinematics")

    # direct SDK access is fine for now – later we can wrap this in ReachyClient
    reachy = client.reachy
    assert reachy is not None, "Client should hold a connected ReachySDK instance."

    # Right arm
    try:
        T_base_ee_right = np.array(reachy.r_arm.forward_kinematics(), dtype=float)
        world.set_T_base_ee_right(T_base_ee_right)
        print("T_base_ee_right:\n", T_base_ee_right)
    except Exception as e:
        print("Could not get right arm FK:", e)
        T_base_ee_right = None

    # Left arm
    try:
        T_base_ee_left = np.array(reachy.l_arm.forward_kinematics(), dtype=float)
        world.set_T_base_ee_left(T_base_ee_left)
        print("T_base_ee_left:\n", T_base_ee_left)
    except Exception as e:
        print("Could not get left arm FK:", e)
        T_base_ee_left = None

    # ------------------------------------------------------
    # 6. Query world-frame EE poses
    # ------------------------------------------------------
    print_header("6. Querying T_world_ee_left / T_world_ee_right")

    T_world_ee_right = world.get_T_world_ee_right()
    print("T_world_ee_right:\n", T_world_ee_right)

    T_world_ee_left = world.get_T_world_ee_left()
    print("T_world_ee_left:\n", T_world_ee_left)

    # Also show generic helper for reference if we have FK
    if T_base_ee_right is not None:
        T_world_ee_right_generic = world.get_T_world_ee(T_base_ee_right)
        print("T_world_ee_right (via generic helper):\n", T_world_ee_right_generic)

    # ------------------------------------------------------
    # 7. Camera poses in world-frame
    # ------------------------------------------------------
    print_header("7. Camera poses in world frame")

    for cam_id in ["teleop_left", "teleop_right", "depth_rgb"]:
        T_world_cam = world.get_T_world_cam(cam_id)
        print(f"{cam_id} -> T_world_cam:\n{T_world_cam}")

    # ------------------------------------------------------
    # 8. Camera intrinsics + image sizes
    # ------------------------------------------------------
    print_header("8. Camera intrinsics + image sizes")

    for cam_id in ["teleop_left", "teleop_right", "depth_rgb"]:
        K = world.get_intrinsics(cam_id)
        hw = world.get_image_size(cam_id)
        print(f"{cam_id}:")
        print("  K:\n", K)
        print("  (height, width):", hw)

    # ------------------------------------------------------
    # 9. Done
    # ------------------------------------------------------
    print_header("DONE - WorldModel SE(3) + calibration pipeline looks wired up.")


if __name__ == "__main__":
    main()
