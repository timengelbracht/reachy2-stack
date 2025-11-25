

#!/usr/bin/env python3
"""
Interactive debugging script for verifying ReachyClient sensor access.

Run with:
    python tests/test_sensors.py
"""
import os
from pathlib import Path
import torch

# 1. Prevent PyTorch DataLoader from forking
torch.set_num_threads(1) 

# 2. Force MKL/OMP to single-thread (prevents CPU thrashing)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 3. Your existing gRPC flags (Keep these!)
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

import pprint
import matplotlib.pyplot as plt
import numpy as np

from reachy2_sdk.media.camera import CameraView
from reachy2_stack.core.client import ReachyClient, ReachyConfig

from reachy2_stack.perception.localization.hloc_localizer import HLocLocalizer
from reachy2_stack.utils.utils_dataclass import HLocConfig
from reachy2_stack.utils.utils_visual_localization import visualize_localization_in_mesh


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
    # get camera data
    # ------------------------------------------------------
    cam_data = client.get_all_camera_data()
    client.close()

    # ------------------------------------------------------
    # 4. Initialize HLoc localizer
    # ------------------------------------------------------
    print_header("4. Initializing HLoc localizer")
    localizer = HLocLocalizer(cfg_hloc)
    localizer.setup_database_structure()
    localizer.setup_query_structure_from_from_reachy_camera_dataclass(cam_data)
    print("HLoc localizer initialized.")
    loc_results = localizer.localize_from_reachy_camera_dataclass(cam_data)

    # ------------------------------------------------------
    # Visualize results
    # ------------------------------------------------------
    print_header("5. Visualizing results")
    mesh_path = Path(cfg_hloc.mesh_path)
    visualize_localization_in_mesh(
        mesh_path=mesh_path,
        loc_results=loc_results,
    )





if __name__ == "__main__":
    main()
