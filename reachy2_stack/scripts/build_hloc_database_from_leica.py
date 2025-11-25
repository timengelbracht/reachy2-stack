from reachy2_stack.perception.localization.hloc_mapping import (
    MappingInputLeica,
    HLocMapConfig,
    build_hloc_map_from_leica,
)

from pathlib import Path



if __name__ == "__main__":

    pano_path = Path("/exchange/data/leica/pano.jpg")
    pano_pose_path = Path("/exchange/data/leica/pano_pose.txt")
    mesh_path = Path("/exchange/data/leica/mesh.ply")

    mapping_input = MappingInputLeica(
        pano_path=pano_path,
        pano_pose_path=pano_pose_path,
        mesh_path=mesh_path,
    )

    cfg = HLocMapConfig(
        maps_root=Path("/exchange/data/hloc"),
        location_name="mlhall",
        overwrite=False,
    )

    map_dir = build_hloc_map_from_leica(mapping_input, cfg)
    print(f"HLoc map created at: {map_dir}")