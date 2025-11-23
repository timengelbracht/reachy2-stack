from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    triangulation
)
import pycolmap
from hloc import localize_inloc
from typing import Dict
from contextlib import contextmanager
import numpy as np
import torch
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm
import os
from dataclasses import dataclass

from reachy2_stack.utils.utils_mapping import HLocMapConfig, MappingInputLeica

# Mokey patches for pycolmap compatibility and missing features
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

def set_camera_model(config: Dict):
    global CFG
    CFG = config

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
            "cfg": CFG,
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

    cfg = CFG

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
            "cfg": CFG,
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
    def __init__(self, cfg: HLocMapConfig):
        self.cfg = cfg

        self.visual_registration_output_path = cfg.maps_root / f"{cfg.location_name}_hloc_registration"

        self.retrieval_conf = extract_features.confs["netvlad"]
        self.feature_conf = extract_features.confs["superpoint_inloc"]
        self.matcher_conf = match_features.confs["superglue"]

        self.images = self.visual_registration_output_path / "inloc"
        self.references = [p.relative_to(self.images).as_posix() for p in (self.images / "database" / "cutouts").glob("*.jpg")]
        self.features = self.visual_registration_output_path / "outputs" / f"{self.feature_conf['output']}.h5"
        self.query = [p.relative_to(self.images).as_posix() for p in (self.images / "query" / "iphone7").glob("*.png")]
        self.features_retrieval = self.visual_registration_output_path / "outputs" / f"{self.retrieval_conf['output']}.h5"
        self.loc_pairs = self.visual_registration_output_path / "outputs" / "pairs-loc.txt"       

        self.results_dir = self.visual_registration_output_path / "outputs" / "results.txt"
        self.image_path_map = cfg.maps_root / f"{cfg.location_name}_processed_leica" / "rgb"
        self.pose_path_map = cfg.maps_root / f"{cfg.location_name}_processed_leica" / "poses"
        self.depth_path_map = cfg.maps_root / f"{cfg.location_name}_processed_leica" / "depth"
        self.xyz_path_map = cfg.maps_root / f"{cfg.location_name}_processed_leica" / "xyz"

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

        # if self.images.exists():
        #     print(f"[REGISTRATION] Outputs directory already exists: {out_dir}. Use force=True to overwrite.")
        #     return

        out_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        image_dir_mapping.mkdir(parents=True, exist_ok=True)
        image_dir_query.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)

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
                    self._generate_dummy_transformations_files(dest_path_transform)
                except Exception as e:
                    print(f"Failed to copy {xyz}: {e}")

    def setup_query_structure_from_single_image(self, query_image: Image.Image):
        image_dir_query = self.images / "query" / "iphone7"
        image_dir_query.mkdir(parents=True, exist_ok=True)

        dest_path = image_dir_query / f"000_query_iphone7_image.png"
        try:
            query_image.convert("RGB").save(dest_path, "PNG")
        except Exception as e:
            print(f"Failed to save query image: {e}")

    def localize(self, query_image: Image.Image, colmap_camera_cfg: dict) -> None:

        self.setup_query_structure_from_single_image(query_image)

        extract_features.main(
            conf=self.feature_conf,
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features,
            overwrite=True)
        
        extract_features.main(
            conf=self.feature_conf, 
            image_dir=self.images,
            image_list=self.query,
            feature_path=self.features,
            overwrite=False)

        extract_features.main(                       
            conf=self.retrieval_conf, 
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features_retrieval, 
            overwrite=True)

        extract_features.main(                       
            conf=self.retrieval_conf, 
            image_dir = self.images,
            image_list=self.query,
            feature_path=self.features_retrieval, 
            overwrite=False)
        
        pairs_from_retrieval.main(
            descriptors = self.features_retrieval,
            output      = self.loc_pairs,
            num_matched = 20,
            query_list  = self.query,        
            db_list     = self.references,  
        )

        match_features.main(
            conf      = self.matcher_conf,
            pairs     = self.loc_pairs,
            features  = self.features,
            matches   = self.matches,        
            overwrite = True
        )

        set_camera_model(config=colmap_camera_cfg)

        localize_inloc.main(
            dataset_dir=self.images,
            retrieval=self.loc_pairs,
            features=self.features,
            matches=self.matches,
            results=self.results_dir,
            skip_matches=5)

        print(f"[REGISTRATION] Visual registration completed. Results saved to {self.results_dir}")


