from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import scipy.io as sio
from tqdm import tqdm


from reachy2_stack.perception.segmentation_detection.sam3_segmenter import Sam3Config, Sam3Segmenter
from reachy2_stack.perception.semantic_map.articulation_map import _ArticulatedInstanceAccumulator
from reachy2_stack.utils.utils_semantic_mapping import _mask_to_points_from_xyz, _iou3d, _bbox_from_points, estimate_plane_normal, estimate_handle_longest_axis, compute_articulation_geometry

from reachy2_stack.utils.utils_dataclass import SemanticMapConfig, ArticulatedClassConfig, ArticulatedObjectInstance, PartCluster, PartDetection3D

class SemanticMapBuilder:
    """
    Offline builder that:
      - runs SAM3 on dense RGB+XYZ DB images
      - detects articulated objects (front+handle) per image
      - merges them across views into a global semantic map
    """

    def __init__(self, sam_cfg: Sam3Config, cfg: SemanticMapConfig):
        self.segmenter = Sam3Segmenter(sam_cfg)
        self.cfg = cfg
        self.instances: List[ArticulatedObjectInstance] = []

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    def build_from_mat_folder(self, mat_dir: Path) -> Path:
        mat_paths = sorted(mat_dir.glob("*.mat"))
        if not mat_paths:
            raise RuntimeError(f"No .mat files found in {mat_dir}")

        if not self.cfg.overwrite and (self.cfg.maps_root / self.cfg.location_name / self.cfg.db_name).exists():
            print(f"[SemanticMapBuilder] Semantic map already exists, skipping: {self.cfg.maps_root / self.cfg.location_name / self.cfg.db_name}")
            return self.cfg.maps_root / self.cfg.location_name / self.cfg.db_name

        out_dir = self.cfg.maps_root / self.cfg.location_name
        out_dir.mkdir(parents=True, exist_ok=True)
        db_path = out_dir / self.cfg.db_name

        all_parts: List[PartDetection3D] = []

        # ----------------------
        # 1) Global detection pass
        # ----------------------
        for mat_path in tqdm(mat_paths, desc="Global detection pass"):
            data = sio.loadmat(str(mat_path))
            if "RGBcut" not in data or "XYZcut" not in data:
                raise KeyError(f"{mat_path} missing RGBcut or XYZcut")

            rgb = data["RGBcut"]   # (H,W,3)
            xyz = data["XYZcut"]   # (H,W,3)
            mat_name = mat_path.stem

            for class_cfg in self.cfg.articulated_classes:
                # fronts
                front_segs = self.segmenter.segment_with_text(rgb, class_cfg.front_prompt)
                for seg in front_segs:
                    mask = seg["mask"].astype(bool)
                    score = float(seg["score"])
                    bbox = tuple(int(x) for x in seg["bbox"])
                    pts = _mask_to_points_from_xyz(mask, xyz)
                    if pts.shape[0] == 0:
                        continue

                    all_parts.append(
                        PartDetection3D(
                            part_type="front",
                            obj_class=class_cfg.class_name,
                            mat_name=mat_name,
                            score=score,
                            bbox=bbox,
                            points=pts,
                        )
                    )

                # handles
                handle_segs = self.segmenter.segment_with_text(rgb, class_cfg.handle_prompt)
                for seg in handle_segs:
                    mask = seg["mask"].astype(bool)
                    score = float(seg["score"])
                    bbox = tuple(int(x) for x in seg["bbox"])
                    pts = _mask_to_points_from_xyz(mask, xyz)
                    if pts.shape[0] == 0:
                        continue

                    all_parts.append(
                        PartDetection3D(
                            part_type="handle",
                            obj_class=class_cfg.class_name,
                            mat_name=mat_name,
                            score=score,
                            bbox=bbox,
                            points=pts,
                        )
                    )

        # ----------------------
        # 2) Global clustering of fronts and handles in 3D
        # ----------------------
        fronts = [p for p in all_parts if p.part_type == "front"]
        handles = [p for p in all_parts if p.part_type == "handle"]

        front_clusters = _cluster_parts_3d(fronts, dist_thresh=0.15, iou_thresh=0.05)
        handle_clusters = _cluster_parts_3d(handles, dist_thresh=0.30, iou_thresh=0.05)

        # ----------------------
        # 3) Pair front clusters with handle clusters, then register
        # ----------------------
        acc = _ArticulatedInstanceAccumulator(self.cfg.instance_merge_radius)

        # group handle clusters by obj_class
        by_class_handles: Dict[str, List[PartCluster]] = {}
        for hc in handle_clusters:
            by_class_handles.setdefault(hc.obj_class, []).append(hc)

        used_front_indices = []
        used_handle_ids = set()  # (class_name, local_idx)

        for fc_idx, fc in enumerate(front_clusters):
            cls = fc.obj_class
            if cls not in by_class_handles:
                continue

            front_center = fc.points.mean(axis=0)

            best_handle_idx = None
            best_dist = float("inf")

            for hc_idx, hc in enumerate(by_class_handles[cls]):
                if (cls, hc_idx) in used_handle_ids:
                    continue  # enforce 1 handle cluster per front (optional)

                handle_center = hc.points.mean(axis=0)
                dist = np.linalg.norm(front_center - handle_center)
                if dist < best_dist:
                    best_dist = dist
                    best_handle_idx = hc_idx

            if best_handle_idx is None:
                continue

            # ensure front really has a nearby handle; otherwise ignore this front
            if best_dist > 0.3:  # 30 cm, tune as you like
                continue

            hc = by_class_handles[cls][best_handle_idx]
            used_front_indices.append(fc_idx)
            used_handle_ids.add((cls, best_handle_idx))

            handle_center = hc.points.mean(axis=0)

            # representative 2D bboxes: take first (or later best by score)
            front_bbox_2d = fc.bboxes_2d[0]
            handle_bbox_2d = hc.bboxes_2d[0]

            # combined score
            combined_score = 0.5 * (fc.score + hc.score)

            if combined_score < self.cfg.min_combined_score:
                continue

            # register ONLY fronts that have a paired handle
            acc.add_detection(
                class_name=cls,
                image_name=fc.mat_names[0],
                front_position_world=front_center,
                handle_position_world=handle_center,
                score=combined_score,
                front_bbox=front_bbox_2d,
                handle_bbox=handle_bbox_2d,
                front_points3d=fc.points,
                handle_points3d=hc.points,
            )

        print(f"[SemanticMapBuilder] front_clusters:  {len(front_clusters)}")
        print(f"[SemanticMapBuilder] handle_clusters: {len(handle_clusters)}")
        print(f"[SemanticMapBuilder] paired (front+handle) objects: {len(used_front_indices)}")

        # ----------------------
        # 4) Save DB
        # ----------------------
        npz_dict = acc.to_npz_dict()
        np.savez(db_path, **npz_dict)
        print(f"[SemanticMapBuilder] Wrote semantic map to {db_path}")

        self.instances = self.load_semantic_map(db_path)
        return db_path

    def run_postprocessing(
        self,
        db_path: Path | None = None,
        out_path: Path | None = None,
    ) -> Path:
        """
        Postprocess the semantic map NPZ:

          - estimate normal_closed from front_points3d
          - estimate handle_longest_axis from handle_points3d
          - classify articulation type (prismatic / revolute)
          - estimate articulation axis + origin
          - generate a 20-step opening trajectory
          - compute handle eccentricity

        Writes all results back into an updated NPZ.
        """

        if db_path is None:
            db_path = self.cfg.maps_root / self.cfg.location_name / self.cfg.db_name

        if out_path is None:
            out_path = db_path

        if not db_path.exists():
            raise FileNotFoundError(f"[run_postprocessing] Semantic map not found at {db_path}")

        data = np.load(db_path, allow_pickle=True)

        # Base fields (must exist)
        class_names = data["class_names"]
        front_positions = data["front_positions"]
        handle_positions = data["handle_positions"]
        scores = data["scores"]

        # Geometry arrays (must exist for proper postproc)
        front_points3d_arr = data.get("front_points3d", None)
        handle_points3d_arr = data.get("handle_points3d", None)

        if front_points3d_arr is None or handle_points3d_arr is None:
            raise KeyError(
                "[run_postprocessing] front_points3d and handle_points3d must be present "
                "in the NPZ to run articulation postprocessing."
            )

        N = len(class_names)
        assert len(handle_positions) == N

        # Prepare new arrays
        normal_closed_list: list[np.ndarray] = []
        handle_longest_axis_list: list[np.ndarray] = []
        articulation_type_list: list[str] = []
        articulation_axis_list: list[np.ndarray] = []
        articulation_origin_list: list[np.ndarray] = []
        articulation_traj_list: list[np.ndarray] = []
        eccentricity_list: list[float] = []

        print(f"[run_postprocessing] Processing {N} articulated objects in {db_path.name}")

        for i in range(N):
            fp3d = np.asarray(front_points3d_arr[i])
            hp3d = np.asarray(handle_points3d_arr[i])
            handle_centroid = np.asarray(handle_positions[i], dtype=float)

            # Default fallbacks
            normal_closed = np.array([np.nan, np.nan, np.nan], dtype=float)
            handle_longest_axis = np.array([np.nan, np.nan, np.nan], dtype=float)
            arti_type = "unknown"
            arti_axis = np.array([np.nan, np.nan, np.nan], dtype=float)
            arti_origin = np.array([np.nan, np.nan, np.nan], dtype=float)
            traj = np.full((20, 3), np.nan, dtype=float)
            eccentricity = np.nan

            if fp3d.size >= 9 and hp3d.size >= 9:
                # 1) plane normal from front points
                normal_closed = estimate_plane_normal(fp3d)

                # 2) handle longest axis from handle points
                handle_longest_axis = estimate_handle_longest_axis(hp3d)

                # 3) articulation geometry (type, axis, origin, trajectory, ecc)
                geom = compute_articulation_geometry(
                    front_points=fp3d,
                    handle_centroid=handle_centroid,
                    front_normal=normal_closed,
                )

                arti_type = geom.get("type", "unknown")
                arti_axis = np.asarray(geom.get("axis", arti_axis), dtype=float)
                arti_origin = np.asarray(geom.get("origin", arti_origin), dtype=float)
                traj = np.asarray(geom.get("trajectory", traj), dtype=float)
                eccentricity = float(geom.get("eccentricity", np.nan))

                # Make sure trajectory has consistent shape [T,3]
                if traj.ndim != 2 or traj.shape[1] != 3:
                    raise ValueError(
                        f"[run_postprocessing] trajectory for index {i} has invalid shape {traj.shape}"
                    )

            normal_closed_list.append(normal_closed)
            handle_longest_axis_list.append(handle_longest_axis)
            articulation_type_list.append(arti_type)
            articulation_axis_list.append(arti_axis)
            articulation_origin_list.append(arti_origin)
            articulation_traj_list.append(traj)
            eccentricity_list.append(eccentricity)

        # Stack / pack into arrays
        normal_closed_arr = np.stack(normal_closed_list, axis=0)              # (N, 3)
        handle_longest_axis_arr = np.stack(handle_longest_axis_list, axis=0)  # (N, 3)
        articulation_axis_arr = np.stack(articulation_axis_list, axis=0)      # (N, 3)
        articulation_origin_arr = np.stack(articulation_origin_list, axis=0)  # (N, 3)
        articulation_traj_arr = np.stack(articulation_traj_list, axis=0)      # (N, T, 3)
        articulation_type_arr = np.array(articulation_type_list, dtype=object)
        eccentricity_arr = np.array(eccentricity_list, dtype=float)

        # Build new NPZ dict: keep all existing keys, add/overwrite new ones
        new_data: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
        new_data["normal_closed"] = normal_closed_arr
        new_data["handle_longest_axis"] = handle_longest_axis_arr
        new_data["articulation_type"] = articulation_type_arr
        new_data["articulation_axis"] = articulation_axis_arr
        new_data["articulation_origin"] = articulation_origin_arr
        new_data["articulation_trajectory"] = articulation_traj_arr
        new_data["eccentricity"] = eccentricity_arr

        # Save updated NPZ
        np.savez(out_path, **new_data)
        print(f"[run_postprocessing] Wrote postprocessed semantic map to {out_path}")

        # Optionally refresh in-memory instances (if you want them)
        try:
            self.instances = self.load_semantic_map(out_path)
        except Exception as e:
            print(f"[run_postprocessing] Warning: failed to reload instances: {e}")

        return out_path
    
    def load_semantic_map(
        self,
        db_path: Path | None = None,
    ) -> List[ArticulatedObjectInstance]:
        """
        Load the semantic map NPZ and return a list of ArticulatedObjectInstance.

        If articulation-related fields (normal_closed, articulation_type,
        articulation_axis) are not yet present in the NPZ, they are filled
        with default placeholders and can be refined in a post-processing
        step.
        """
        if db_path is None:
            db_path = self.cfg.maps_root / self.cfg.location_name / self.cfg.db_name

        if not db_path.exists():
            raise FileNotFoundError(f"Semantic map not found at {db_path}")

        data = np.load(db_path, allow_pickle=True)

        class_names = data["class_names"]
        front_positions = data["front_positions"]
        handle_positions = data["handle_positions"]
        scores = data["scores"]
        num_obs = data.get("num_observations", np.ones(len(class_names), dtype=int))
        image_names = data.get("image_names", np.array([""] * len(class_names), dtype=object))
        front_bboxes = data.get("front_bboxes", np.zeros((len(class_names), 4), dtype=int))
        handle_bboxes = data.get("handle_bboxes", np.zeros((len(class_names), 4), dtype=int))
        front_points3d_arr = data.get("front_points3d", None)
        handle_points3d_arr = data.get("handle_points3d", None)

        # articulation-related fields are optional for now
        normal_closed_arr = data.get("normal_closed", None)
        articulation_type_arr = data.get("articulation_type", None)
        articulation_axis_arr = data.get("articulation_axis", None)
        handle_longest_axis_arr = data.get("handle_longest_axis", None)
        trajectory_arr = data.get("articulation_trajectory", None)

        # synamic states
        open_fraction_arr = data.get("open_fraction", None)
        is_open_arr = data.get("is_open", None)
        last_handle_pos_arr = data.get("last_handle_position_world", None)
        last_update_time_arr = data.get("last_update_time", None)

        instances: List[ArticulatedObjectInstance] = []
        N = len(class_names)

        for i in range(N):
            cls = str(class_names[i])
            fp = np.asarray(front_positions[i], dtype=float)
            hp = np.asarray(handle_positions[i], dtype=float)
            sc = float(scores[i])
            nobs = int(num_obs[i])
            img_name = str(image_names[i])
            fb = tuple(int(x) for x in front_bboxes[i])
            hb = tuple(int(x) for x in handle_bboxes[i])

            # defaults if arrays are missing
            if normal_closed_arr is not None:
                normal_closed = np.asarray(normal_closed_arr[i], dtype=float)
            else:
                normal_closed = np.array([np.nan, np.nan, np.nan], dtype=float)

            if articulation_type_arr is not None:
                articulation_type = str(articulation_type_arr[i])
            else:
                articulation_type = "unknown"

            if articulation_axis_arr is not None:
                articulation_axis = np.asarray(articulation_axis_arr[i], dtype=float)
            else:
                articulation_axis = np.array([np.nan, np.nan, np.nan], dtype=float)

            if handle_longest_axis_arr is not None:
                handle_longest_axis = np.asarray(handle_longest_axis_arr[i], dtype=float)
            else:
                handle_longest_axis = np.array([np.nan, np.nan, np.nan], dtype=float)
                
            if front_points3d_arr is not None:
                front_pts = np.asarray(front_points3d_arr[i])
            else:
                front_pts = np.zeros((0, 3), dtype=float)

            if handle_points3d_arr is not None:
                handle_pts = np.asarray(handle_points3d_arr[i])
            else:
                handle_pts = np.zeros((0, 3), dtype=float)
            if open_fraction_arr is not None:
                open_fraction = float(open_fraction_arr[i])
            else:
                open_fraction = 0.0
            if is_open_arr is not None:
                is_open = bool(is_open_arr[i])
            else:
                is_open = False
            if last_handle_pos_arr is not None:
                last_handle_pos = np.asarray(last_handle_pos_arr[i], dtype=float)
            else:
                last_handle_pos = None
            if last_update_time_arr is not None:
                last_update_time = float(last_update_time_arr[i])
            else:
                last_update_time = 0.0
            if trajectory_arr is not None:
                trajectory = np.asarray(trajectory_arr[i], dtype=float)
            else:
                trajectory = np.full((20, 3), np.nan, dtype=float)
                
            instances.append(
                ArticulatedObjectInstance(
                    id=i,
                    class_name=cls,
                    front_position_world=fp,
                    handle_position_world=hp,
                    front_points3d=front_pts,
                    handle_points3d=handle_pts,
                    score=sc,
                    num_observations=nobs,
                    image_name=img_name,
                    front_bbox=fb,
                    handle_bbox=hb,
                    normal_closed=normal_closed,
                    articulation_type=articulation_type,
                    articulation_axis=articulation_axis,
                    handle_longest_axis=handle_longest_axis,
                    open_fraction=open_fraction,
                    is_open=is_open,
                    last_handle_position_world=last_handle_pos,
                    last_update_time=last_update_time,
                    trajectory=trajectory,
                )
            )

        self.instances = instances

        return instances

    def update_instance_state(
        self,
        instance_id: int,
        *,
        open_fraction: float | None = None,
        is_open: bool | None = None,
        last_handle_position_world: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Update dynamic state for a single instance (in memory only)."""
        if timestamp is None:
            timestamp = time.time()

        # find instance by id
        matches = [inst for inst in self.instances if inst.id == instance_id]
        if not matches:
            raise KeyError(f"No ArticulatedObjectInstance with id={instance_id}")
        inst = matches[0]

        if open_fraction is not None:
            inst.open_fraction = float(open_fraction)
        if is_open is not None:
            inst.is_open = bool(is_open)
        if last_handle_position_world is not None:
            inst.last_handle_position_world = np.asarray(
                last_handle_position_world, dtype=float
            )
        inst.last_update_time = float(timestamp)

    def save_dynamic_state(
        self,
        db_path: Path | None = None,
    ) -> Path:
        """
        Save dynamic state fields (open_fraction, is_open, last_handle_position_world,
        last_update_time) from self.instances back into the NPZ.

        - If the NPZ does not have these arrays yet, they are added.
        - If it does, they are overwritten.
        """
        if db_path is None:
            db_path = self.cfg.maps_root / self.cfg.location_name / self.cfg.db_name

        if not db_path.exists():
            raise FileNotFoundError(f"[save_dynamic_state] Semantic map not found at {db_path}")

        data = np.load(db_path, allow_pickle=True)
        N = len(self.instances)
        if N == 0:
            raise RuntimeError("[save_dynamic_state] No instances in memory to save.")

        # Sanity-check N vs NPZ size
        class_names = data["class_names"]
        if len(class_names) != N:
            raise ValueError(
                f"[save_dynamic_state] Mismatch: NPZ has {len(class_names)} instances, "
                f"but builder has {N} in memory."
            )

        # Build arrays from current instances
        open_fraction_arr = np.zeros((N,), dtype=float)
        is_open_arr = np.zeros((N,), dtype=bool)
        last_handle_pos_arr = np.full((N, 3), np.nan, dtype=float)
        last_update_time_arr = np.zeros((N,), dtype=float)

        for i, inst in enumerate(self.instances):
            open_fraction_arr[i] = float(inst.open_fraction)
            is_open_arr[i] = bool(inst.is_open)

            if inst.last_handle_position_world is not None:
                last_handle_pos_arr[i, :] = np.asarray(
                    inst.last_handle_position_world, dtype=float
                )

            last_update_time_arr[i] = float(inst.last_update_time)

        # Copy existing data and insert/overwrite dynamic fields
        new_data: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
        new_data["open_fraction"] = open_fraction_arr
        new_data["is_open"] = is_open_arr
        new_data["last_handle_position_world"] = last_handle_pos_arr
        new_data["last_update_time"] = last_update_time_arr

        np.savez(db_path, **new_data)
        print(f"[save_dynamic_state] Wrote dynamic state to {db_path}")

        return db_path
    
    def reset_dynamic_state(self) -> None:
        """
        Reset dynamic state (open_fraction, is_open, last_handle_position_world,
        last_update_time) for all loaded instances *in memory only*.
        """
        if not self.instances:
            return

        for inst in self.instances:
            inst.open_fraction = 0.0
            inst.is_open = False
            inst.last_handle_position_world = None
            inst.last_update_time = 0.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_xyz(mat_path: Path, xyz_key: str) -> np.ndarray:
        data = sio.loadmat(str(mat_path))
        if xyz_key not in data:
            # simple fallback heuristics
            for alt in ("xyz_world", "xyz", "XYZ"):
                if alt in data:
                    xyz_key = alt
                    break
            else:
                raise KeyError(f"Could not find '{xyz_key}' (or xyz_world/xyz/XYZ) in {mat_path}")

        xyz = data[xyz_key]
        if xyz.ndim == 3 and xyz.shape[2] == 3:
            return xyz.astype(np.float32)
        raise ValueError(f"Expected (H,W,3) XYZ in {mat_path}, got {xyz.shape}")

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return float("nan"), float("nan")
        return float(xs.mean()), float(ys.mean())

    @staticmethod
    def _mask_to_world_point_from_xyz(
        mask: np.ndarray,
        xyz: np.ndarray,
    ) -> np.ndarray:
        H, W, _ = xyz.shape
        assert mask.shape == (H, W), "Mask and XYZ shape mismatch"

        ys, xs = np.where(mask)
        if xs.size == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=float)

        pts = xyz[ys, xs, :]  # (N, 3)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.size == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=float)

        return np.median(pts, axis=0).astype(float)

    # ------------------------------------------------------------------
    # skeleton: articulation prediction (future work)
    # ------------------------------------------------------------------

    def predict_articulation(
        self,
        obj: ArticulatedObjectInstance,
    ) -> Dict[str, float]:
        """
        Placeholder for later:

        Given an articulated object instance (front & handle positions, class name,
        maybe normals / map context), predict articulation parameters, e.g.:

            {
                "type": "revolute" or "prismatic",
                "axis_x": ...,
                "axis_y": ...,
                "axis_z": ...,
                "origin_x": ...,
                ...
            }

        For now, this is intentionally left empty.
        """
        raise NotImplementedError("Articulation prediction not implemented yet.")

def _cluster_parts_3d(
    parts: List[PartDetection3D],
    dist_thresh: float = 0.15,   # meters
    iou_thresh: float = 0.05,
) -> List[PartCluster]:
    """
    Greedy clustering of 3D part detections based on:
      - centroid distance in 3D
      - 3D AABB IoU (via _bbox_from_points + _iou3d)

    Clusters are separated per (obj_class, part_type).
    """
    clusters: List[PartCluster] = []

    for det in parts:
        c_det = det.points.mean(axis=0)
        det_min, det_max = _bbox_from_points(det.points)

        best_idx = -1
        best_iou = 0.0

        for idx, cl in enumerate(clusters):
            if cl.obj_class != det.obj_class or cl.part_type != det.part_type:
                continue

            c_cl = cl.points.mean(axis=0)
            dist = np.linalg.norm(c_det - c_cl)
            if dist > dist_thresh:
                continue

            cl_min, cl_max = _bbox_from_points(cl.points)
            iou = _iou3d(det_min, det_max, cl_min, cl_max)

            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx < 0:
            # new cluster
            clusters.append(
                PartCluster(
                    obj_class=det.obj_class,
                    part_type=det.part_type,
                    points=det.points.copy(),
                    score=det.score,
                    mat_names=[det.mat_name],
                    bboxes_2d=[det.bbox],
                )
            )
        else:
            # merge into existing cluster
            cl = clusters[best_idx]
            cl.points = np.concatenate([cl.points, det.points], axis=0)
            cl.score = max(cl.score, det.score)
            cl.mat_names.append(det.mat_name)
            cl.bboxes_2d.append(det.bbox)

    return clusters
