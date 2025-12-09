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
import cv2
import time 
import argparse
import os

def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    argparser = argparse.ArgumentParser(description="Record Reachy sensors")
    argparser.add_argument("--output_dir", type=str, default="/home/cvg/reachy_record", help="Output directory to save recordings")

    rgb_depth_dir = f"{argparser.parse_args().output_dir}/rgb"
    depth_dir = f"{argparser.parse_args().output_dir}/depth"
    teleop_left_dir = f"{argparser.parse_args().output_dir}/teleop_left"
    teleop_right_dir = f"{argparser.parse_args().output_dir}/teleop_right"

    os.makedirs(rgb_depth_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(teleop_left_dir, exist_ok=True)
    os.makedirs(teleop_right_dir, exist_ok=True)
    
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



    frame_id = 0
    max_frames = 1000
    while frame_id < max_frames:
        start  = time.time()
        print_header("")

        rgb_depth, depth_map = client.get_depth_rgbd()
        # print("Depth RGB shape:", rgb_depth.shape if rgb_depth is not None else None)
        # print("Depth map shape:", depth_map.shape if depth_map is not None else None)
        end = time.time()
        # print("Frame rate: ", 1/(end - start))

        np.save(rgb_depth_dir + f"/{frame_id}.npy", rgb_depth)
        np.save(depth_dir + f"/{frame_id}.npy", depth_map)
        # also save a jpg
        cv2.imwrite(rgb_depth_dir + f"/{frame_id}.jpg", rgb_depth)


        rgb_left = client.get_teleop_rgb_left()
        rgb_right = client.get_teleop_rgb_right()
        np.save(teleop_left_dir + f"/{frame_id}.npy", rgb_left)
        np.save(teleop_right_dir + f"/{frame_id}.npy", rgb_right)
        cv2.imwrite(teleop_left_dir + f"/{frame_id}.jpg", rgb_left)
        cv2.imwrite(teleop_right_dir + f"/{frame_id}.jpg", rgb_right)

        frame_id += 1
        time.sleep(0.05)

    
    print_header("Teleop Intrinsics")
    intr_left = client.get_teleop_intrinsics_left()
    print("Teleop LEFT intrinsics:")
    pprint.pprint(intr_left)

    intr_right = client.get_teleop_intrinsics_right()
    print("Teleop RIGHT intrinsics:")
    pprint.pprint(intr_right)


    print_header("Depth Intrinsics")
    intr_depth_default = client.get_depth_intrinsics()
    print("Depth intrinsics (default):")
    pprint.pprint(intr_depth_default)

    print_header("Depth Extrinsics")

    T_default = client.get_depth_extrinsics()
    print("Depth extrinsics (default) R:\n", T_default)


    print_header("Done recording.")


if __name__ == "__main__":
    main()
