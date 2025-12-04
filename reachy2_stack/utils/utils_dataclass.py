from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from dataclasses import field

# Rechy client
@dataclass
class TeleopCameraData:
    rgb: np.ndarray
    intrinsics: Dict[str, Any]
    extrinsics: Dict[str, Any]


@dataclass
class DepthCameraData:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: Dict[str, Any]
    extrinsics: Dict[str, Any]


@dataclass
class ReachyCameraData:
    teleop_left: TeleopCameraData
    teleop_right: TeleopCameraData
    depth: DepthCameraData

@dataclass
class ReachyConfig:
    host: str = "10.0.0.201"
    use_sim: bool = False
    default_speed: float = 0.5
    # extend with joint names, mapping, etc. later

    @classmethod
    def from_yaml(cls, path: str) -> "ReachyConfig":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Detection, Segmentation
@dataclass
class ConceptDetection:
    label: str
    mask: np.ndarray             # HxW bool
    score: float
    bbox: tuple[int, int, int, int]
    position_world: np.ndarray   # 3D point in world frame (e.g., mask centroid)

@dataclass
class Sam3Config:
    """Configuration for the SAM3 segmenter."""
    model_name: str = "facebook/sam3"
    device: str = "cuda"  # "cuda" or "cpu"
    score_threshold: float = 0.5
    mask_threshold: float = 0.5

@dataclass
class ArticulatedObjectInstance:
    """
    An articulated object instance in the world, currently defined as:
    - a cabinet front (drawer/door)
    - at least one associated handle

    In the future you can generalize to other articulated classes.
    """
    id: int
    class_name: str  # e.g. "cabinet"
    front_position_world: np.ndarray   # (3,)
    handle_position_world: np.ndarray  # (3,) - representative (e.g. main handle)
    front_points3d: np.ndarray         # (N, 3) merged 3D points on front
    handle_points3d: np.ndarray        # (M, 3) merged 3D points on handle
    score: float                       # combined confidence
    num_observations: int              # how many per-image detections merged
    image_name: str                    # one representative image
    front_bbox: Tuple[int, int, int, int]
    handle_bbox: Tuple[int, int, int, int]
    normal_closed: np.ndarray        # (3,) - estimated normal vector when closed
    articulation_type: str        # e.g. "revolute", "prismatic"
    articulation_axis: np.ndarray     # (3,) - estimated axis of rotation/translation
    handle_longest_axis: np.ndarray  # (3,) - estimated longest axis of handle
    trajectory: np.ndarray           # (K, 3) - recorded trajectory of handle positions

    # state variables
    open_fraction: float = 0.0               # 0=closed, 1=open
    is_open: bool = False
    last_handle_position_world: Optional[np.ndarray] = None
    last_update_time: float = field(default_factory=lambda: 0.0)

@dataclass
class ArticulatedClassConfig:
    """
    Configuration for one articulated object type.

    For now we design it around the cabinet example:
    - 'front' = drawer/door face
    - 'handle' = handle/knob/pull

    Later you can add more types (e.g. 'fridge_door', etc.).
    """
    class_name: str
    front_prompt: str
    handle_prompt: str

@dataclass
class PartCluster:
    obj_class: str
    part_type: str          # "front" or "handle"
    points: np.ndarray      # merged (M, 3)
    score: float            # e.g. max score of members
    mat_names: List[str]    # source mat names
    bboxes_2d: List[Tuple[int, int, int, int]]


@dataclass
class PartDetection3D:
    """
    A raw SAM3 detection lifted into 3D, before any global association.

    part_type: "front" or "handle"
    obj_class: e.g. "cabinet"
    """
    part_type: str                 # "front" or "handle"
    obj_class: str                 # e.g. "cabinet"
    mat_name: str                  # source .mat (for debugging)
    score: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in image
    points: np.ndarray             # (N, 3) world-coord points on this mask


@dataclass
class SemanticMapConfig:
    maps_root: Path
    location_name: str
    db_name: str = "semantic_map.npz"
    overwrite: bool = False
    min_combined_score: float = 0.9

    # Articulated object classes to detect.
    articulated_classes: List[ArticulatedClassConfig] = field(
        default_factory=lambda: [
            ArticulatedClassConfig(
                class_name="cabinet",
                front_prompt="cabinet drawer or cabinet door front",
                handle_prompt="cabinet handle",
            )
        ]
    )

    # radius [m] used to merge per-image detections into the same physical instance
    instance_merge_radius: float = 0.05  # ~5 cm


@dataclass
class MappingInputLeica:
    """Minimal representation for map building from leica."""
    pano_path: Path        # RGB panorama
    pano_pose_path: Path   # pose .txt of the panorama
    mesh_path: Path        # mesh of the scan used for depth rendering
    num_database_images: int = 20  # number of database images to sample from the pano

    @classmethod
    def from_yaml(cls, path: str) -> "MappingInputLeica":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

@dataclass
class HLocConfig:
    maps_root: Path         # root folder for all maps
    location_name: str      # e.g. "kitchen_1"
    overwrite: bool = False
    mesh_path: Path = None    # optional, for visualization

    @classmethod
    def from_yaml(cls, path: str) -> "HLocConfig":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
@dataclass
class WorldCameraState:
    """
    Pose + intrinsics for one camera.

    Conventions:
      - T_world_cam : 4x4 transform from camera to world   (world ← cam)
      - T_base_cam  : 4x4 transform from camera to base    (base  ← cam)
    """
    camera_id: str
    T_world_cam: np.ndarray
    T_base_cam: np.ndarray
    K: np.ndarray
    image_size: Tuple[int, int]  # (H, W)


@dataclass
class WorldEEState:
    """
    Pose for one end-effector.

    Conventions:
      - T_world_ee : 4x4 transform from EE to world   (world ← ee)
      - T_base_ee  : 4x4 transform from EE to base    (base  ← ee)
    """
    side: str  # "left" or "right"
    T_world_ee: Optional[np.ndarray]
    T_base_ee: Optional[np.ndarray]


@dataclass
class WorldState:
    """
    Read-only snapshot of the kinematic state the WorldModel maintains.
    No semantics, no articulation states – just geometry.
    """
    location_name: str

    # Base pose
    T_world_base: np.ndarray       # world ← base (current estimate)
    T_world_base_odom: np.ndarray  # world ← base (pure odom)

    # Cameras & end-effectors
    cameras: Dict[str, WorldCameraState]
    end_effectors: Dict[str, WorldEEState]
    
