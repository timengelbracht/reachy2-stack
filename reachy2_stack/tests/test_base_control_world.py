#!/usr/bin/env python3
from __future__ import annotations

import time
import numpy as np

from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig
from reachy2_stack.control.base import BaseController
from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.utils.utils_poses import T_to_xytheta

from reachy2_stack.utils.utils_world_model import print_header
from reachy2_stack.perception.localization.hloc_localizer import HLocLocalizer
from pathlib import Path
from reachy2_stack.utils.utils_dataclass import HLocConfig

POSE_DRAWER = np.array([
    [-0.99975390, -0.01689299,  0.01437935, -0.80394776],
    [ 0.01703849, -0.99980425,  0.01005712, -0.36883071],
    [ 0.01420664,  0.01029965,  0.99984603,  0.01699635],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=float)

def main() -> None:
    # --- Connect client -------------------------------------------------
    # Adjust host if needed
    cfg = ReachyConfig(host="192.168.1.71")
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()
    client.turn_off_all()

    cfg_hloc = HLocConfig.from_yaml("config/config_hloc.yaml")
    print(cfg_hloc)
    print_header("4. Initializing HLoc localizer")
    localizer = HLocLocalizer(cfg_hloc)
    localizer.setup_database_structure()

    # --- World model init ----------------------------------------------
    wm_cfg = WorldModelConfig(location_name="lab", localization_mode="vision")
    world = WorldModel(wm_cfg)
    world.init_from_client_calib(client)

    # Initialize world pose from current odom (world ≈ odom for now)
    odom = client.get_mobile_odometry()
    if odom is None:
        raise RuntimeError("No mobile_base odometry available.")
    
    # In a vision-based setup, you would instead call:
    #   world.update_from_visual_localization_depth(loc_results, alpha=1.0)
    # Here we just keep using odom for consistency in the test.
    #  Read camera
    # ------------------------------------------------------
    # visual localization and update world model
    # ------------------------------------------------------
    cam_data = client.get_all_camera_data()
    localizer.setup_query_structure_from_from_reachy_camera_dataclass(cam_data)
    print("HLoc localizer initialized.")
    loc_results = localizer.localize_from_reachy_camera_dataclass(cam_data)
    world.update_from_visual_localization_depth(
        loc_results=loc_results,
        alpha=1.0,  # fully trust visual localization
    )
    print(world.get_T_world_base())

    # --- Controllers ----------------------------------------------------
    base = BaseController(client=client, world=world)

    # --- Print initial poses -------------------------------------------
    print("\n[BASE] Initial odom:", odom)

    T_world_base = world.get_T_world_base()
    x_w, y_w, th_w_deg = T_to_xytheta(T_world_base)
    print(
        f"[BASE] Initial world pose: "
        f"x={x_w:.3f} m, y={y_w:.3f} m, theta={th_w_deg:.1f} deg"
    )

    # --- Define world-frame target -------------------------------------
    # Simple test: move +0.3 m in world X, keep same yaw
    # x_target = 0.0
    # y_target = 0.0
    # th_target = 0.0
    x_target, y_target, th_target = T_to_xytheta(POSE_DRAWER)

    print(
        f"\n[BASE] Commanding goto_world → "
        f"x={x_target:.3f} m, y={y_target:.3f} m, theta={th_target:.1f} deg"
    )

    base.goto_world(
        x_world=x_target,
        y_world=y_target,
        theta_world=th_target,
        wait=True,
        distance_tolerance=0.05,
        angle_tolerance=5.0,
        timeout=30.0,
        degrees=True,
    )

    # Small extra wait to let motion fully settle
    time.sleep(0.5)

    # --- Read back odom and update world -------------------------------
    odom_new = client.get_mobile_odometry()
    print("\n[BASE] New odom:", odom_new)



    # ------------------------------------------------------
    # 4. Initialize HLoc localizer
    # ------------------------------------------------------
    cam_data = client.get_all_camera_data()
    localizer.setup_query_structure_from_from_reachy_camera_dataclass(cam_data)
    print("HLoc localizer initialized.")
    loc_results = localizer.localize_from_reachy_camera_dataclass(cam_data)
    world.update_from_visual_localization_depth(
        loc_results=loc_results,
        alpha=1.0,  # fully trust visual localization
    )


    T_world_base_new = world.get_T_world_base()
    x_w2, y_w2, th_w2_deg = T_to_xytheta(T_world_base_new)
    print(
        f"[BASE] New world pose: "
        f"x={x_w2:.3f} m, y={y_w2:.3f} m, theta={th_w2_deg:.1f} deg\n"
    )

    client.turn_off_all()


if __name__ == "__main__":
    main()
