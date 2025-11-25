import pickle
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    triangulation
)
from scipy.spatial.transform import Rotation as R
import pycolmap
from hloc import localize_inloc
from typing import Any, Dict
from contextlib import contextmanager
import numpy as np
import torch
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm
import os
from dataclasses import dataclass

from reachy2_stack.utils.utils_dataclass import HLocConfig, ReachyCameraData
from reachy2_stack.utils.utils_visual_localization import colmap_camera_cfg_from_reachy_camera_intrinsics, generate_dummy_transformations_files

# Mokey patches for pycolmap compatibility and missing features
import torch.utils.data
class SafeDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        # 1. Force workers to 0 (Main Thread Only)
        if 'num_workers' in kwargs and kwargs['num_workers'] > 0:
            print(f"[PATCH] Intercepted DataLoader requesting {kwargs['num_workers']} workers. Forcing to 0.")
        kwargs['num_workers'] = 0
        
        # 2. Disable memory pinning (requires a background thread)
        kwargs['pin_memory'] = False
        
        # 3. Disable multiprocessing context
        kwargs['multiprocessing_context'] = None
        
        super().__init__(*args, **kwargs)

# Overwrite the class globally
torch.utils.data.DataLoader = SafeDataLoader

CFG_PER_QUERY: Dict[str, Dict] = {}

if not hasattr(pycolmap, "absolute_pose_estimation"):
    def absolute_pose_estimation(points2D, points3D, camera,
                                 estimation_options=None,
                                 refinement_options=None):
        return pycolmap.estimate_and_refine_absolute_pose(
            points2D, points3D, camera,
            estimation_options or {},
            refinement_options or {},
        )
    
    pycolmap.absolute_pose_estimation = absolute_pose_estimation

if not hasattr(pycolmap.Rigid3d, "essential_matrix"):

    def _skew(t):
        tx, ty, tz = t
        return np.array([[ 0, -tz,  ty],
                         [ tz,   0, -tx],
                         [-ty,  tx,   0]])

    def essential_matrix(self: pycolmap.Rigid3d) -> np.ndarray:
        """
        Return E = [t]_x R  (world→left, right pose = self)
        Equivalent to the old C++ helper that existed in pycolmap < 0.7.
        """
        R = self.rotation.matrix()      
        t = self.translation            
        return _skew(t) @ R            

    pycolmap.Rigid3d.essential_matrix = essential_matrix

# Monkey patch for hloc localize_inloc to accept additional focal length parameter
def set_intrinsics(*, fx: float, fy: float):
    global FX, FY
    FX, FY = float(fx), float(fy)

# def set_camera_model(config: Dict):
#     global CFG
#     CFG = config
def set_camera_models(config_per_query: Dict[str, Dict]):
    """
    Store a mapping from query image (relative path used by HLoc)
    to its individual COLMAP / pycolmap camera config.
    """
    global CFG_PER_QUERY
    CFG_PER_QUERY = config_per_query

def pose_from_cluster_patched(
        dataset_dir, q, retrieved,
        feature_file, match_file,
        skip=None
    ):
    """Drop‑in replacement for hloc.localize_inloc.pose_from_cluster
    that takes focal_length as an optional argument."""

    all_mkpq, all_mkpr, all_mkp3d, all_indices = [], [], [], []
    kpq = feature_file[q]["keypoints"].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        kpr = feature_file[r]["keypoints"].__array__()
        pair = localize_inloc.names_to_pair(q, r)
        m = match_file[pair]["matches0"].__array__()
        v = m > -1

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        scan_r = localize_inloc.loadmat(Path(dataset_dir, r + ".mat"))["XYZcut"]
        mkp3d, valid = localize_inloc.interpolate_scan(scan_r, mkpr)
        Tr = localize_inloc.get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    # Guard: nothing survived matching/skip -> return safe failure with identity pose
    if len(all_mkpq) == 0:
        ret = {
            "success": False,
            "error": "no_correspondences",
            "message": "No valid 2D-3D correspondences after matching/interpolation.",
            "cfg": CFG_PER_QUERY.get(q),
            "cam_from_world": pycolmap.Rigid3d()
        }
        empty_2d = np.empty((0, 2), dtype=np.float32)
        empty_3d = np.empty((0, 3), dtype=np.float32)
        empty_i  = np.empty((0,), dtype=int)
        return ret, empty_2d, empty_2d, empty_3d, empty_i, num_matches

    all_mkpq  = np.concatenate(all_mkpq,  0)
    all_mkpr  = np.concatenate(all_mkpr,  0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    # cfg = CFG
    try:
        cfg = CFG_PER_QUERY[q]   # q is the HLoc query image key, e.g. "query/iphone7/000_query_iphone7_teleop_left.png"
    except KeyError:
        raise KeyError(
            f"No camera config found for query image {q!r} in CFG_PER_QUERY. "
            f"Make sure you pass a dict mapping each query path to its camera config."
        )

    opts = pycolmap.AbsolutePoseEstimationOptions()
    opts.ransac.max_error = 48
    ret = pycolmap.estimate_and_refine_absolute_pose(
        all_mkpq, all_mkp3d, cfg, opts
    )

    # Guard: nothing survived matching/skip -> return safe failure with identity pose
    if ret == None:
        ret = {
            "success": False,
            "error": "no_correspondences",
            "message": "No valid 2D-3D correspondences after matching/interpolation.",
            "cfg": CFG_PER_QUERY.get(q),
            "cam_from_world": pycolmap.Rigid3d()
        }
        empty_2d = np.empty((0, 2), dtype=np.float32)
        empty_3d = np.empty((0, 3), dtype=np.float32)
        empty_i  = np.empty((0,), dtype=int)
        return ret, empty_2d, empty_2d, empty_3d, empty_i, num_matches
    
    ret["cfg"] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches
# Patch the original function
localize_inloc.pose_from_cluster = pose_from_cluster_patched

def interpolate_scan_patched(scan, kp):
    h, w, c = scan.shape
    kp = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    kp = kp.astype(np.float32)
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(scan, kp, align_corners=True, mode="bilinear")[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode="nearest"
    )[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    kp3d = interp.T.numpy()
    valid = valid.numpy()
    return kp3d, valid
# Patch the original function
localize_inloc.interpolate_scan = interpolate_scan_patched

@contextmanager
def pass_focal_length(focal_length: float):                  # focal is a scalar, px
    """
    Temporarily monkey‑patch localize_inloc.pose_from_cluster so it uses
    the given focal length. 
    """
    original = localize_inloc.pose_from_cluster      # keep reference

    def patched(dataset_dir, q, retrieved,
                feature_file, match_file,
                skip=None, *, focal_length=focal):
        # call the original but override the kwarg
        return original(dataset_dir, q, retrieved,
                        feature_file, match_file,
                        skip=skip, focal_length=focal_length)

    localize_inloc.pose_from_cluster = patched       # <‑‑ patch
    try:
        yield
    finally:                                         # <‑‑ restore
        localize_inloc.pose_from_cluster = original


class HLocLocalizer:
    def __init__(self, cfg: HLocConfig):
        self.cfg = cfg

        self.visual_registration_output_path = Path(cfg.maps_root) / f"{cfg.location_name}_hloc_registration"

        self.retrieval_conf = extract_features.confs["netvlad"]
        self.feature_conf = extract_features.confs["superpoint_inloc"]
        self.matcher_conf = match_features.confs["superglue"]
        self.matcher_conf["num_workers"] = 0

        self.images = self.visual_registration_output_path / "inloc"
        self.references = [p.relative_to(self.images).as_posix() for p in (self.images / "database" / "cutouts").glob("*.jpg")]
        self.features = self.visual_registration_output_path / "outputs" / f"{self.feature_conf['output']}.h5"
        self.query = [p.relative_to(self.images).as_posix() for p in (self.images / "query" / "iphone7").glob("*.png")]
        self.features_retrieval = self.visual_registration_output_path / "outputs" / f"{self.retrieval_conf['output']}.h5"
        self.loc_pairs = self.visual_registration_output_path / "outputs" / "pairs-loc.txt"       

        self.results_dir = self.visual_registration_output_path / "outputs" / "results.txt"
        self.matches = self.visual_registration_output_path / "outputs" / f"{self.matcher_conf['output']}.h5"
        self.image_path_map = Path(cfg.maps_root) / f"{cfg.location_name}_processed_leica" / "rgb"
        self.pose_path_map = Path(cfg.maps_root) / f"{cfg.location_name}_processed_leica" / "poses"
        self.depth_path_map = Path(cfg.maps_root) / f"{cfg.location_name}_processed_leica" / "depth"
        self.xyz_path_map = Path(cfg.maps_root) / f"{cfg.location_name}_processed_leica" / "xyz"

    def setup_database_structure(self):
        """Set up the directory structure for HLoc visual localization.
        Creates necessary directories and copies images and .mat files
        to the appropriate locations.
        """

        out_dir = self.visual_registration_output_path / "outputs"
        image_dir = self.visual_registration_output_path / "inloc" 
        image_dir_mapping = image_dir / "database" / "cutouts"
        image_dir_query = image_dir / "query" / "iphone7"
        transform_dir = image_dir / "database" / "alignments" / "database" / "transformations"

        if self.images.exists():
            print(f"[REGISTRATION] Outputs directory already exists: {out_dir}. Use force=True to overwrite.")
            # return

        out_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        image_dir_mapping.mkdir(parents=True, exist_ok=True)
        image_dir_query.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)

        # copy xyz to mapping directory
        for xyz in tqdm(self.xyz_path_map.glob("*"), desc="Copying xyz to mapping", total=len(list(self.xyz_path_map.glob("*")))):
            if xyz.suffix.lower() in [".mat"]:
                dest_path = image_dir_mapping / f"000_database_cutouts_{xyz.stem}.mat"
                dest_path_transform = transform_dir / f"000_trans_cutouts.txt"
                try:
                    shutil.copy(xyz, dest_path)
                    generate_dummy_transformations_files(dest_path_transform)
                except Exception as e:
                    print(f"Failed to copy {xyz}: {e}")

        # Copy images to the mapping and query directories
        for image in tqdm(self.image_path_map.glob("*"), desc="Copying images to mapping", total=len(list(self.image_path_map.glob("*")))):
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dest_path = image_dir_mapping / f"000_database_cutouts_{image.stem}.jpg"
            try:
                img = Image.open(image)
                img.convert("RGB").save(dest_path, "JPEG")
            except Exception as e:
                print(f"Failed to convert {image}: {e}")

        # copy xyz to mapping directory
        for xyz in tqdm(self.xyz_path_map.glob("*"), desc="Copying xyz to mapping", total=len(list(self.xyz_path_map.glob("*")))):
            if xyz.suffix.lower() in [".mat"]:
                dest_path = image_dir_mapping / f"000_database_cutouts_{xyz.stem}.mat"
                dest_path_transform = transform_dir / f"000_trans_cutouts.txt"
                try:
                    shutil.copy(xyz, dest_path)
                    generate_dummy_transformations_files(dest_path_transform)
                except Exception as e:
                    print(f"Failed to copy {xyz}: {e}")

        print("[REGISTRATION] Pre-computing Database Features...")
        
        # 1. Extract Local Features for Database (Once)
        extract_features.main(
            conf=self.feature_conf,
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features,
            overwrite=False, # Only compute if missing
        )

        # 2. Extract Global Descriptors for Database (Once)
        extract_features.main(
            conf=self.retrieval_conf,
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features_retrieval,
            overwrite=False, # Only compute if missing
        )

    def setup_query_structure_from_single_image(self, query_image: Image.Image):
        image_dir_query = self.images / "query" / "iphone7"
        image_dir_query.mkdir(parents=True, exist_ok=True)

        dest_path = image_dir_query / f"000_query_iphone7_image.png"
        try:
            query_image.convert("RGB").save(dest_path, "PNG")
        except Exception as e:
            print(f"Failed to save query image: {e}")

    def setup_query_structure_from_from_reachy_camera_dataclass(self, cam_data: ReachyCameraData):
        """
        Set up the directory structure for HLoc visual localization using ReachyCameraData dataclass.
        (3 camera images: teleop left, teleop right, depth)
        """
        image_dir_query = self.images / "query" / "iphone7"
        image_dir_query.mkdir(parents=True, exist_ok=True)

        # Save teleop left image
        dest_path_left = image_dir_query / "000_query_iphone7_teleop_left.png"
        try:
            img_left = cam_data.teleop_left.rgb  # OpenCV BGR
            img_left = Image.fromarray(img_left[..., ::-1])  # BGR→RGB
            img_left.convert("RGB").save(dest_path_left, "PNG")
        except Exception as e:
            print(f"Failed to save teleop left image: {e}")

        # Save teleop right image
        dest_path_right = image_dir_query / "000_query_iphone7_teleop_right.png"
        try:
            img_right = cam_data.teleop_right.rgb
            img_right = Image.fromarray(img_right[..., ::-1])
            img_right.convert("RGB").save(dest_path_right, "PNG")
        except Exception as e:
            print(f"Failed to save teleop right image: {e}")

        # Save depth RGB (visualization) image
        dest_path_depth = image_dir_query / "000_query_iphone7_depth.png"
        try:
            img_depth = cam_data.depth.rgb
            img_depth = Image.fromarray(img_depth[..., ::-1])
            img_depth.convert("RGB").save(dest_path_depth, "PNG")
        except Exception as e:
            print(f"Failed to save depth image: {e}")

    def localize_from_reachy_camera_dataclass(self, cam_data: ReachyCameraData) -> None:
        """
        Perform visual localization using HLoc from a ReachyCameraData instance.

        Steps:
          1) Write the three query images (teleop_left/right, depth) to disk.
          2) Build per-query COLMAP camera configs from the intrinsics in cam_data.
          3) Run HLoc feature extraction, retrieval, matching, and localization.
        """
        # 1) Write query images (teleop_left/right, depth) to disk
        self.setup_query_structure_from_from_reachy_camera_dataclass(cam_data)

        # 2) Refresh self.query after writing the files
        self.query = [
            p.relative_to(self.images).as_posix()
            for p in (self.images / "query" / "iphone7").glob("*.png")
        ]

        # 3) Build per-query COLMAP camera configs using the Reachy helper
        colmap_camera_cfg_per_query: Dict[str, Dict] = {}

        # keys must match what HLoc uses as `q` (relative to self.images)
        q_left = "query/iphone7/000_query_iphone7_teleop_left.png"
        q_right = "query/iphone7/000_query_iphone7_teleop_right.png"
        q_depth = "query/iphone7/000_query_iphone7_depth.png"

        colmap_camera_cfg_per_query[q_left] = colmap_camera_cfg_from_reachy_camera_intrinsics(
            cam_data.teleop_left
        )
        colmap_camera_cfg_per_query[q_right] = colmap_camera_cfg_from_reachy_camera_intrinsics(
            cam_data.teleop_right
        )
        colmap_camera_cfg_per_query[q_depth] = colmap_camera_cfg_from_reachy_camera_intrinsics(
            cam_data.depth
        )

        # Features for query images
        extract_features.main(
            conf=self.feature_conf,
            image_dir=self.images,
            image_list=self.query,
            feature_path=self.features,
            overwrite=True,
        )

        # Global descriptors for query images
        extract_features.main(
            conf=self.retrieval_conf,
            image_dir=self.images,
            image_list=self.query,
            feature_path=self.features_retrieval,
            overwrite=True,
        )

        # Build query->db pairs from retrieval
        pairs_from_retrieval.main(
            descriptors=self.features_retrieval,
            output=self.loc_pairs,
            num_matched=20,
            query_list=self.query,
            db_list=self.references,
        )

        # Local feature matching
        match_features.main(
            conf=self.matcher_conf,
            pairs=self.loc_pairs,
            features=self.features,
            matches=self.matches,
            overwrite=True,
        )

        # Register per-query camera configs for the monkey-patched pose_from_cluster
        set_camera_models(config_per_query=colmap_camera_cfg_per_query)

        # Run HLoc / InLoc localization
        localize_inloc.main(
            dataset_dir=self.images,
            retrieval=self.loc_pairs,
            features=self.features,
            matches=self.matches,
            results=self.results_dir,
            skip_matches=5,
        )

        print(f"[REGISTRATION] Visual registration completed. Results saved to {self.results_dir}")

        return self.load_localization_results(cam_data)

    def load_localization_results(
            self, 
            cam_data: ReachyCameraData, 
            min_matches: int = 40, 
            min_inliers: int = 15   
        ) -> Dict[str, Dict[str, Any]]:
            """
            Parses the HLoc result logs and maps them back to the specific Reachy cameras.
            
            Args:
                cam_data: The data object containing the intrinsics (K) for each camera.
                min_matches: Minimum keypoint matches required to consider the result valid.
                min_inliers: Minimum RANSAC inliers required to consider the pose valid.

            Returns:
                Dict containing the pose data for 'teleop_left', 'teleop_right', and 'depth'.
                Format:
                {
                    'teleop_left': {
                        'pose_w_T_c': np.ndarray (4x4), # Camera in World Frame
                        'num_inliers': int,
                        'intrinsics': dict
                    },
                    ...
                }
            """
            
            # 1. Define the mapping from HLoc filename stems to DataClass attributes
            # These match the filenames defined in 'setup_query_structure_from_from_reachy_camera_dataclass'
            filename_map = {
                "000_query_iphone7_teleop_left": "teleop_left",
                "000_query_iphone7_teleop_right": "teleop_right",
                "000_query_iphone7_depth": "depth"
            }

            # 2. Load the log file
            log_path = Path(str(self.results_dir) + "_logs.pkl")
            if not log_path.exists():
                print(f"[!] Log file not found at {log_path}. Localization likely failed completely.")
                return {}

            with open(log_path, "rb") as f:
                logs = pickle.load(f)

            results = {}

            # 3. Iterate through expected cameras
            for hloc_name, camera_key in filename_map.items():
                
                # Use the full path key as stored in logs['loc']
                # HLoc usually stores keys as relative paths, e.g., "query/iphone7/000_..."
                # We search for the key that ends with our filename
                log_key = next((k for k in logs['loc'].keys() if hloc_name in k), None)

                if not log_key:
                    print(f"[!] No log entry found for {camera_key}")
                    continue

                log_entry = logs['loc'][log_key]
                
                # 4. Filter by quality
                n_matches = log_entry.get('num_matches', 0)
                n_inliers = log_entry['PnP_ret'].get('num_inliers', 0)

                if n_matches < min_matches or n_inliers < min_inliers:
                    print(f"[!] {camera_key}: Low quality (Matches: {n_matches}, Inliers: {n_inliers}). Skipping.")
                    continue

                # 5. Extract Pose (Camera from World) -> T_cw
                # HLoc/Colmap returns: The transform that moves a point from World to Camera
                rigid3d = log_entry['PnP_ret']['cam_from_world']
                R_cw = rigid3d.rotation.matrix()
                t_cw = rigid3d.translation

                # 6. Invert to get Robot Pose (World from Camera) -> T_wc
                # We want: The position of the camera in the world
                R_wc = R_cw.T
                t_wc = -R_wc @ t_cw

                # Build 4x4 Matrix
                T_wc = np.eye(4)
                T_wc[:3, :3] = R_wc
                T_wc[:3, 3] = t_wc

                # 7. Get Intrinsics from the Input DataClass (not the loader)
                # We fetch the specific K matrix for this specific camera
                camera_obj = getattr(cam_data, camera_key) # e.g. cam_data.teleop_left
                K = camera_obj.intrinsics['K']
                
                results[camera_key] = {
                    "T_wc": T_wc,             # The matrix you likely want for the robot
                    "T_cw": np.linalg.inv(T_wc), # The raw colmap pose (inverse)
                    "q_wc": R.from_matrix(R_wc).as_quat(), # Quaternion (x,y,z,w)
                    "num_matches": n_matches,
                    "num_inliers": n_inliers,
                    "K": K
                }
                
                print(f"[SUCCESS] Localized {camera_key} with {n_inliers} inliers.")

            return results


