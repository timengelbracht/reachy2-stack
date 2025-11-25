import numpy as np
from typing import Dict, List, Tuple



class _ArticulatedInstanceAccumulator:
    """Clusters articulated objects (front+handle) and stores merged 3D shapes."""

    def __init__(self, merge_radius: float):
        self.merge_radius = float(merge_radius)

        self._class_names: List[str] = []
        self._front_positions: List[np.ndarray] = []
        self._handle_positions: List[np.ndarray] = []
        self._scores: List[float] = []
        self._image_names: List[str] = []
        self._front_bboxes: List[Tuple[int, int, int, int]] = []
        self._handle_bboxes: List[Tuple[int, int, int, int]] = []
        self._num_obs: List[int] = []

        # new: merged 3D points per instance (ragged lists)
        self._front_points3d: List[np.ndarray] = []
        self._handle_points3d: List[np.ndarray] = []

    def add_detection(
        self,
        class_name: str,
        image_name: str,
        front_position_world: np.ndarray,
        handle_position_world: np.ndarray,
        score: float,
        front_bbox: Tuple[int, int, int, int],
        handle_bbox: Tuple[int, int, int, int],
        front_points3d: np.ndarray,
        handle_points3d: np.ndarray,
    ) -> None:
        """
        Merge with nearest instance of same class if centroid distance <= merge_radius,
        otherwise create a new instance.

        front_points3d / handle_points3d: (Ni, 3) and (Nj, 3) arrays; will be concatenated
        across detections for the same instance.
        """

        if not np.all(np.isfinite(front_position_world)) or not np.all(
            np.isfinite(handle_position_world)
        ):
            return

        # filter by class
        candidate_idxs = [i for i, c in enumerate(self._class_names) if c == class_name]

        if not candidate_idxs:
            # new instance
            self._class_names.append(class_name)
            self._front_positions.append(front_position_world.copy())
            self._handle_positions.append(handle_position_world.copy())
            self._scores.append(float(score))
            self._image_names.append(image_name)
            self._front_bboxes.append(front_bbox)
            self._handle_bboxes.append(handle_bbox)
            self._num_obs.append(1)
            self._front_points3d.append(front_points3d.copy())
            self._handle_points3d.append(handle_points3d.copy())
            return

        # choose nearest existing instance in 3D (based on front centroids)
        candidate_fronts = np.stack(
            [self._front_positions[i] for i in candidate_idxs], axis=0
        )
        dists = np.linalg.norm(candidate_fronts - front_position_world[None, :], axis=1)
        min_local = int(np.argmin(dists))
        min_dist = float(dists[min_local])
        inst_idx = candidate_idxs[min_local]

        if min_dist > self.merge_radius:
            # create new instance
            self._class_names.append(class_name)
            self._front_positions.append(front_position_world.copy())
            self._handle_positions.append(handle_position_world.copy())
            self._scores.append(float(score))
            self._image_names.append(image_name)
            self._front_bboxes.append(front_bbox)
            self._handle_bboxes.append(handle_bbox)
            self._num_obs.append(1)
            self._front_points3d.append(front_points3d.copy())
            self._handle_points3d.append(handle_points3d.copy())
        else:
            # merge into existing instance
            n = self._num_obs[inst_idx]

            old_front = self._front_positions[inst_idx]
            old_handle = self._handle_positions[inst_idx]

            self._front_positions[inst_idx] = (old_front * n + front_position_world) / (n + 1)
            self._handle_positions[inst_idx] = (old_handle * n + handle_position_world) / (n + 1)

            self._num_obs[inst_idx] = n + 1

            # scores: keep max
            if score >= self._scores[inst_idx]:
                self._scores[inst_idx] = float(score)
                self._image_names[inst_idx] = image_name
                self._front_bboxes[inst_idx] = front_bbox
                self._handle_bboxes[inst_idx] = handle_bbox

            # concatenate 3D points
            self._front_points3d[inst_idx] = np.concatenate(
                [self._front_points3d[inst_idx], front_points3d], axis=0
            )
            self._handle_points3d[inst_idx] = np.concatenate(
                [self._handle_points3d[inst_idx], handle_points3d], axis=0
            )

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        if not self._front_positions:
            return dict(
                class_names=np.array([], dtype=object),
                front_positions=np.zeros((0, 3), dtype=float),
                handle_positions=np.zeros((0, 3), dtype=float),
                scores=np.zeros((0,), dtype=float),
                image_names=np.array([], dtype=object),
                front_bboxes=np.zeros((0, 4), dtype=int),
                handle_bboxes=np.zeros((0, 4), dtype=int),
                num_observations=np.zeros((0,), dtype=int),
                front_points3d=np.array([], dtype=object),
                handle_points3d=np.array([], dtype=object),
            )

        class_names = np.asarray(self._class_names, dtype=object)
        front_positions = np.stack(self._front_positions, axis=0)
        handle_positions = np.stack(self._handle_positions, axis=0)
        scores = np.asarray(self._scores, dtype=float)
        image_names = np.asarray(self._image_names, dtype=object)
        front_bboxes = np.asarray(self._front_bboxes, dtype=int)
        handle_bboxes = np.asarray(self._handle_bboxes, dtype=int)
        num_obs = np.asarray(self._num_obs, dtype=int)

        # ragged lists as object arrays
        front_points3d = np.asarray(self._front_points3d, dtype=object)
        handle_points3d = np.asarray(self._handle_points3d, dtype=object)

        return dict(
            class_names=class_names,
            front_positions=front_positions,
            handle_positions=handle_positions,
            scores=scores,
            image_names=image_names,
            front_bboxes=front_bboxes,
            handle_bboxes=handle_bboxes,
            num_observations=num_obs,
            front_points3d=front_points3d,
            handle_points3d=handle_points3d,
        )