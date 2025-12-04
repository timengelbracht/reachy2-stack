from reachy2_stack.perception.semantic_map.semantic_mapping import (
    SemanticMapBuilder,
    SemanticMapConfig,
)
from reachy2_stack.perception.segmentation_detection.sam3_segmenter import Sam3Config
from pathlib import Path

LOCATION_NAME = "mlhall_1"
MAPS_ROOT = Path("data/hloc/maps")
MESH_PATH = Path("data/leica/mesh.ply")
DB_NAME = "semantic_map.npz"


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


db_path = builder.build_from_mat_folder(
    mat_dir=Path("/exchange/data/hloc/mlhall_hloc_registration/inloc/database/cutouts/")  # your folder
)
instances= builder.load_semantic_map(db_path=db_path)

print("Semantic map saved at:", db_path)

builder.run_postprocessing()

# visualize
from reachy2_stack.utils.utils_semantic_mapping import visualize_semantic_map_in_mesh
visualize_semantic_map_in_mesh(MESH_PATH, db_path)

