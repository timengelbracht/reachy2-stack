#!/usr/bin/env python3
"""
Interactive debugging script for verifying WorldModel + ReachyClient + HLoc + visualization.

Run with:
    python tests/test_world_model.py
"""

import os
from pathlib import Path
import pprint

import numpy as np

# (Optional) keep things sane on CPU-heavy machines
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import open3d as o3d  # noqa: F401  (used indirectly via visualizer)
from open3d.visualization import gui, rendering  # noqa: F401

from reachy2_stack.core.client import ReachyClient, ReachyConfig
from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig
from reachy2_stack.perception.localization.hloc_localizer import HLocLocalizer
from reachy2_stack.utils.utils_dataclass import HLocConfig
from reachy2_stack.utils.utils_world_model import (
    print_header,
    pretty_mat,
    visualize_world_model_in_mesh,
)


def main() -> None:
    # ------------------------------------------------------
    # 1. Load configs
    # ------------------------------------------------------
    print_header("1. Loading configs")

    cfg_reachy = ReachyConfig.from_yaml("config/config.yaml")
    pprint.pprint(cfg_reachy)

    cfg_hloc = HLocConfig.from_yaml("config/config_hloc.yaml")
    pprint.pprint(cfg_hloc)

    # Use fused so that visual localization actually influences T_world_base
    wm_cfg = WorldModelConfig(
        location_name=cfg_hloc.location_name,
        localization_mode="fused",  # "odom" / "vision" / "fused"
    )
    pprint.pprint(wm_cfg)

    # ------------------------------------------------------
    # 2. Connect to Reachy
    # ------------------------------------------------------
    print_header("2. Connecting to Reachy")
    client = ReachyClient(cfg_reachy)
    client.connect()
    print("Reachy connected.")

    # ------------------------------------------------------
    # 3. Initialize WorldModel from Reachy calibration
    # ------------------------------------------------------
    print_header("3. Initializing WorldModel from Reachy calibration")
    world_model = WorldModel(wm_cfg)
    world_model.init_from_client_calib(client)
    print("WorldModel camera calibration loaded.")

    # ------------------------------------------------------
    # 4. Update base pose from odometry (if available)
    # ------------------------------------------------------
    print_header("4. Updating base pose from mobile_base.odometry")
    odom = client.get_mobile_odometry()
    if odom is None:
        print("[WARN] No mobile_base present on Reachy – leaving T_world_base as identity.")
    else:
        print("Raw odometry:", odom)
        world_model.update_from_odom(odom)

    pretty_mat("T_world_base (world ← base) after odom", world_model.get_T_world_base())

    # ------------------------------------------------------
    # 5. Run HLoc visual localization (depth-only for WorldModel)
    # ------------------------------------------------------
    print_header("5. Running HLoc visual localization (depth only for WorldModel)")

    # Grab all camera streams once (teleop_left/right + depth)
    # HLoc will localize all three, but WorldModel will only read 'depth'.
    from reachy2_stack.utils.utils_dataclass import ReachyCameraData  # noqa: F401

    cam_data = client.get_all_camera_data()

    # Initialize HLoc localizer
    localizer = HLocLocalizer(cfg_hloc)

    # Build / reuse database structure
    localizer.setup_database_structure()

    # Run localization from the current Reachy images
    loc_results = localizer.localize_from_reachy_camera_dataclass(cam_data)
    print("\n[HLOC] Localization results keys:", list(loc_results.keys()))

    # Feed ONLY the depth camera result into the world model
    world_model.update_from_visual_localization_depth(
        loc_results=loc_results,
        alpha=1.0,  # 1.0 = trust vision fully, 0<alpha<1 = blend with odom
    )

    pretty_mat("T_world_base (world ← base) after depth visual fix",
               world_model.get_T_world_base())

    # ------------------------------------------------------
    # 6. (Optional) Set EE poses from FK
    # ------------------------------------------------------
    print_header("6. Setting EE poses from Reachy FK (if available)")
    try:
        reachy = client.connect_reachy  # underlying SDK

        # Right arm
        if hasattr(reachy, "r_arm") and hasattr(reachy.r_arm, "forward_kinematics"):
            q_r, _ = client.get_joint_state_right()
            T_base_ee_r = np.array(reachy.r_arm.forward_kinematics(q_r), dtype=float)
            world_model.set_T_base_ee("right", T_base_ee_r)
            pretty_mat("T_base_ee_right (base ← ee)", T_base_ee_r)
        else:
            print("[INFO] Right arm FK not available, skipping.")

        # Left arm
        if hasattr(reachy, "l_arm") and hasattr(reachy.l_arm, "forward_kinematics"):
            q_l, _ = client.get_joint_state_left()
            T_base_ee_l = np.array(reachy.l_arm.forward_kinematics(q_l), dtype=float)
            world_model.set_T_base_ee("left", T_base_ee_l)
            pretty_mat("T_base_ee_left (base ← ee)", T_base_ee_l)
        else:
            print("[INFO] Left arm FK not available, skipping.")

    except Exception as e:
        print(f"[WARN] Failed to set EE poses from FK: {e}")

    # ------------------------------------------------------
    # 7. Visualize in mesh
    # ------------------------------------------------------
    print_header("7. Launching visualizer")
    mesh_path = Path(cfg_hloc.mesh_path)
    visualize_world_model_in_mesh(mesh_path=mesh_path, world_model=world_model)

    # ------------------------------------------------------
    # 8. Clean shutdown (after GUI closes)
    # ------------------------------------------------------
    print_header("8. Closing Reachy client")
    client.close()
    print("Done.")


if __name__ == "__main__":
    main()
