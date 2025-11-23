

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

from reachy2_stack.perception.localization.hloc_localizer import HLocLocalizer
from reachy2_stack.utils.utils_mapping import HLocConfig


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    # ------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------
    print_header("1. Loading config")
    cfg_reachy = ReachyConfig.from_yaml("config/config.yaml")
    pprint.pprint(cfg_reachy)

    cfg_hloc = HLocConfig.from_yaml("config/config_hloc.yaml")
    pprint.pprint(cfg_hloc)

    # ------------------------------------------------------
    # 2. Connect to Reachy
    # ------------------------------------------------------
    print_header("2. Connecting to Reachy")
    client = ReachyClient(cfg_reachy)
    client.connect()
    print("Connected.")


    # ------------------------------------------------------
    # 7. Depth camera RGB + depth
    # ------------------------------------------------------
    print_header("7. Depth Camera RGB + Depth")

    rgb_depth, depth_map = client.get_depth_rgbd()
    print("Depth RGB shape:", rgb_depth.shape if rgb_depth is not None else None)
    print("Depth map shape:", depth_map.shape if depth_map is not None else None)

    # ------------------------------------------------------
    # 10. Depth intrinsics (RGB / DEPTH views)
    # ------------------------------------------------------
    print_header("10. Depth Intrinsics")

    # todo




if __name__ == "__main__":
    main()
