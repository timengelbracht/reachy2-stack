#!/usr/bin/env python3
"""
Interactive debugging script for verifying ReachyClient sensor access.

Run with:
    python tests/test_sensors.py
"""

import pprint
import matplotlib.pyplot as plt
import numpy as np

from reachy2_sdk.media.camera import CameraView
from reachy2_stack.core.client import ReachyClient, ReachyConfig


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def show_image(img: np.ndarray | None, title: str) -> None:
    if img is None:
        print(f"{title}: None (not displaying)")
        return

    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


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
    # 3. Joint states
    # ------------------------------------------------------
    print_header("3. Joint States")

    q_r, dq_r = client.get_joint_state_right()
    print("Right arm positions (deg):", q_r)
    print("Right arm velocities (deg/s):", dq_r)

    q_l, dq_l = client.get_joint_state_left()
    print("Left arm positions (deg):", q_l)
    print("Left arm velocities (deg/s):", dq_l)

    # ------------------------------------------------------
    # 4. Gripper openings
    # ------------------------------------------------------
    print_header("4. Gripper Openings")

    g_r = client.get_gripper_opening_right()
    print("Right gripper opening [0..1]:", g_r)

    g_l = client.get_gripper_opening_left()
    print("Left gripper opening [0..1]:", g_l)

    # ------------------------------------------------------
    # 5. Mobile base odometry
    # ------------------------------------------------------
    print_header("5. Mobile Base Odometry")

    odo = client.get_mobile_odometry()
    print("Odometry dict:", odo)

    # ------------------------------------------------------
    # 6. Teleop RGB cameras
    # ------------------------------------------------------
    print_header("6. Teleop RGB Cameras")

    rgb_left = client.get_teleop_rgb_left()
    print("Teleop LEFT RGB shape:", rgb_left.shape if rgb_left is not None else None)
    show_image(rgb_left, "Teleop RGB Left")

    rgb_right = client.get_teleop_rgb_right()
    print("Teleop RIGHT RGB shape:", rgb_right.shape if rgb_right is not None else None)
    show_image(rgb_right, "Teleop RGB Right")

    # ------------------------------------------------------
    # 7. Depth camera RGB + depth
    # ------------------------------------------------------
    print_header("7. Depth Camera RGB + Depth")

    rgb_depth, depth_map = client.get_depth_rgbd()
    print("Depth RGB shape:", rgb_depth.shape if rgb_depth is not None else None)
    print("Depth map shape:", depth_map.shape if depth_map is not None else None)

    show_image(rgb_depth, "Depth RGB")
    show_image(depth_map, "Depth Map")

    # ------------------------------------------------------
    # 8. Teleop intrinsics
    # ------------------------------------------------------
    print_header("8. Teleop Intrinsics")

    intr_left = client.get_teleop_intrinsics_left()
    print("Teleop LEFT intrinsics:")
    pprint.pprint(intr_left)

    intr_right = client.get_teleop_intrinsics_right()
    print("Teleop RIGHT intrinsics:")
    pprint.pprint(intr_right)

    # ------------------------------------------------------
    # 9. Teleop extrinsics
    # ------------------------------------------------------
    print_header("9. Teleop Extrinsics")

    T_left = client.get_teleop_extrinsics_left()
    print("Teleop LEFT extrinsics T:\n", T_left)

    T_right = client.get_teleop_extrinsics_right()
    print("Teleop RIGHT extrinsics T:\n", T_right)
    # ------------------------------------------------------
    # 10. Depth intrinsics (RGB / DEPTH views)
    # ------------------------------------------------------
    print_header("10. Depth Intrinsics")

    # default (no view argument)
    intr_depth_default = client.get_depth_intrinsics()
    print("Depth intrinsics (default):")
    pprint.pprint(intr_depth_default)

    # explicitly for LEFT and RIGHT views (if supported by SDK)
    intr_depth_left = client.get_depth_intrinsics(CameraView.LEFT)
    print("Depth intrinsics (LEFT view):")
    pprint.pprint(intr_depth_left)

    intr_depth_right = client.get_depth_intrinsics(CameraView.RIGHT)
    print("Depth intrinsics (RIGHT view):")
    pprint.pprint(intr_depth_right)

    # ------------------------------------------------------
    # 11. Depth extrinsics
    # ------------------------------------------------------
    print_header("11. Depth Extrinsics")

    T_default = client.get_depth_extrinsics()
    print("Depth extrinsics (default) R:\n", T_default)

    T_left = client.get_depth_extrinsics(CameraView.LEFT)
    print("Depth extrinsics (LEFT view) R:\n", T_left)

    T_right = client.get_depth_extrinsics(CameraView.RIGHT)
    print("Depth extrinsics (RIGHT view) R:\n", T_right)

    # ------------------------------------------------------
    # 12. Done
    # ------------------------------------------------------
    print_header("DONE - All ReachyClient getters tested.")


if __name__ == "__main__":
    main()
