from __future__ import annotations

from pathlib import Path

import numpy as np

from reachy2_stack.core.client import ReachyClient, ReachyConfig

from reachy2_stack.perception.semantic_map.semantic_mapping import SemanticMapBuilder


from reachy2_stack.perception.localization.hloc_localizer import HLocLocalizer
from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig
from reachy2_stack.control.arm import ArmController
from reachy2_stack.control.gripper import GripperController
from reachy2_stack.skills.drawer import DrawerOpenFixedBaseSkill
from reachy2_stack.utils.utils_dataclass import ReachyConfig, Sam3Config, HLocConfig, SemanticMapConfig, ArticulatedObjectInstance


def print_header(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80 + "\n")


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Load semantic map and pick articulated instance id=3
    # ------------------------------------------------------------------
    LOCATION_NAME = "mlhall_1"
    MAPS_ROOT = Path("data/hloc/maps")
    MESH_PATH = Path("data/leica/mesh.ply")  # currently unused here, but kept for context
    DB_NAME = "semantic_map.npz"


    # ------------------------------------------------------------------
    # 2) Connect to Reachy and run HLoc to set T_world_base
    # ------------------------------------------------------------------
    print_header("2. Connecting to Reachy")

    cfg = ReachyConfig().from_yaml("config/config.yaml")
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()
    client.goto_posture("default")
    
    db_path = MAPS_ROOT / LOCATION_NAME / DB_NAME

    print_header("1. Loading semantic map")

    sam_cfg = Sam3Config(device="cuda")
    map_cfg = SemanticMapConfig(
        maps_root=MAPS_ROOT,
        location_name=LOCATION_NAME,
        db_name=DB_NAME,
        overwrite=False,
        instance_merge_radius=0.1,
        min_combined_score=0.7,
    )

    builder = SemanticMapBuilder(sam_cfg, map_cfg)
    instances = builder.load_semantic_map(db_path=db_path)

    if not instances:
        raise RuntimeError("No instances loaded from semantic map.")

    drawer: ArticulatedObjectInstance | None = None
    for inst in instances:
        if inst.id == 3:
            drawer = inst
            break

    if drawer is None:
        raise RuntimeError("No ArticulatedObjectInstance with id=3 found.")
    if drawer.articulation_type != "prismatic":
        raise RuntimeError(
            f"Instance id=3 is not prismatic (got {drawer.articulation_type!r})."
        )

    print(f"Selected instance id={drawer.id}, class={drawer.class_name!r}")
    print(f"  articulation_type: {drawer.articulation_type}")
    print(f"  trajectory shape : {drawer.trajectory.shape}")
    print(f"  normal_closed    : {drawer.normal_closed}")
    print(f"  handle_longest   : {drawer.handle_longest_axis}")




    print_header("3. Initializing HLoc localizer")

    cfg_hloc = HLocConfig.from_yaml("config/config_hloc.yaml")
    print(cfg_hloc)

    localizer = HLocLocalizer(cfg_hloc)
    localizer.setup_database_structure()

    wm_cfg = WorldModelConfig(location_name=LOCATION_NAME, localization_mode="vision")
    world = WorldModel(wm_cfg)
    world.init_from_client_calib(client)

    cam_data = client.get_all_camera_data()
    localizer.setup_query_structure_from_from_reachy_camera_dataclass(cam_data)

    print("Running visual localization (HLoc)...")
    loc_results = localizer.localize_from_reachy_camera_dataclass(cam_data)

    world.update_from_visual_localization_depth(
        loc_results=loc_results,
        alpha=1.0,  # fully trust visual localization
    )

    print("Updated T_world_base:")
    print(world.get_T_world_base())

    # ------------------------------------------------------------------
    # 3) Arm + Gripper + Drawer-open skill (fixed base)
    # ------------------------------------------------------------------
    print_header("4. Executing DrawerOpenFixedBaseSkill")

    arm = ArmController(client=client, side="right", world=world)
    gripper = GripperController(client=client, side="right")

    skill = DrawerOpenFixedBaseSkill(
        arm=arm,
        gripper=gripper,
        approach_dist=0.12,
        pregrasp_duration=2.0,
        grasp_duration=1.2,
        waypoint_duration=1.5,
        grasp_offset=0.06,
    )

    # This will:
    #  - go to pre-grasp near trajectory[0]
    #  - move to grasp, close gripper
    #  - follow drawer.trajectory in WORLD frame
    #  - open gripper at the end
    skill.open_drawer(drawer)

    print("Drawer opening skill finished.")


if __name__ == "__main__":
    main()
