from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from reachy2_stack.core.client import ReachyClient


@dataclass
class WorldModelConfig:
    """Config for how we track global poses."""
    location_name: str = "default"
    # "odom"  : trust wheel odom only
    # "vision": trust visual localization only (future)
    # "fused" : combine both (future)
    localization_mode: str = "odom"


class WorldModel:
    """Central place for transforms & map.

    - Maintains T_world_base (world←base).
    - Stores static camera extrinsics (base←cam) and intrinsics.
    - Stores current EE poses in base frame for left/right arms.
    - Provides helpers to get T_world_* for EE and cameras.
    """

    def __init__(self, cfg: WorldModelConfig):
        self.cfg = cfg

        # Base pose
        self._T_world_base = np.eye(4, dtype=float)
        self._T_world_base_odom = np.eye(4, dtype=float)

        # Static camera calibration (base frame)
        #   key examples: "teleop_left", "teleop_right", "depth_rgb"
        self._T_base_cam: Dict[str, np.ndarray] = {}
        self._K: Dict[str, np.ndarray] = {}
        self._image_size: Dict[str, Tuple[int, int]] = {}

        # Current EE poses in base frame
        # keys: "left", "right"
        self._T_base_ee: Dict[str, np.ndarray] = {}

    # --------------------------------------------------------------------------
    # Initialization from ReachyClient / SDK
    # --------------------------------------------------------------------------

    def init_from_client_calib(self, client: ReachyClient) -> None:
        """Query SDK once via client and cache all camera calibration.

        This sets:
          - base→cam extrinsics (4x4 T) in self._T_base_cam
          - intrinsics K and image size in self._K, self._image_size
        """
        # Make sure we are connected
        client.connect()

        # --- Teleop LEFT ---
        intr_L = client.get_teleop_intrinsics_left()
        T_base_teleop_left = np.array(client.get_teleop_extrinsics_left(), dtype=float)

        self._T_base_cam["teleop_left"] = T_base_teleop_left
        self._K["teleop_left"] = np.array(intr_L["K"], dtype=float)
        self._image_size["teleop_left"] = (
            int(intr_L["height"]),
            int(intr_L["width"]),
        )

        # --- Teleop RIGHT ---
        intr_R = client.get_teleop_intrinsics_right()
        T_base_teleop_right = np.array(client.get_teleop_extrinsics_right(), dtype=float)

        self._T_base_cam["teleop_right"] = T_base_teleop_right
        self._K["teleop_right"] = np.array(intr_R["K"], dtype=float)
        self._image_size["teleop_right"] = (
            int(intr_R["height"]),
            int(intr_R["width"]),
        )

        # --- Depth RGB view (simplest: default view) ---
        intr_D = client.get_depth_intrinsics()
        T_base_depth_rgb = np.array(client.get_depth_extrinsics(), dtype=float)

        self._T_base_cam["depth_rgb"] = T_base_depth_rgb
        self._K["depth_rgb"] = np.array(intr_D["K"], dtype=float)
        self._image_size["depth_rgb"] = (
            int(intr_D["height"]),
            int(intr_D["width"]),
        )

        # If you later want a dedicated depth view:
        # from reachy2_sdk.media.camera import CameraView
        # intr_Dd = client.get_depth_intrinsics(view=CameraView.DEPTH)
        # T_base_depth_depth = np.array(
        #     client.get_depth_extrinsics(view=CameraView.DEPTH), dtype=float
        # )
        # self._T_base_cam["depth_depth"] = T_base_depth_depth
        # self._K["depth_depth"] = np.array(intr_Dd["K"], dtype=float)
        # self._image_size["depth_depth"] = (
        #     int(intr_Dd["height"]),
        #     int(intr_Dd["width"]),
        # )

    # --------------------------------------------------------------------------
    # Odometry + (future) visual localization
    # --------------------------------------------------------------------------

    def update_from_odom(self, odom: dict) -> None:
        """Update T_world_base from Reachy mobile_base.odometry.

        odom: {'x': meters, 'y': meters, 'theta': degrees}
        """
        x = float(odom["x"])
        y = float(odom["y"])
        theta_deg = float(odom["theta"])
        theta = np.deg2rad(theta_deg)

        c, s = np.cos(theta), np.sin(theta)
        T = np.eye(4, dtype=float)
        T[0, 0] = c
        T[0, 1] = -s
        T[1, 0] = s
        T[1, 1] = c
        T[0, 3] = x
        T[1, 3] = y

        self._T_world_base_odom = T

        if self.cfg.localization_mode in ("odom", "fused"):
            self._T_world_base = T

    def apply_visual_fix(
        self, T_world_base_meas: np.ndarray, alpha: float = 1.0
    ) -> None:
        """(Future) Incorporate a visual localization measurement.

        T_world_base_meas: 4x4 pose from HLoc / SLAM.
        alpha: 1.0 = trust vision fully, 0.0 = ignore vision,
               in between = simple interpolation on SE(3) surrogate.
        """
        T_world_base_meas = np.asarray(T_world_base_meas, dtype=float)
        if T_world_base_meas.shape != (4, 4):
            raise ValueError("T_world_base_meas must be 4x4")

        if self.cfg.localization_mode == "vision":
            self._T_world_base = T_world_base_meas
            return

        if self.cfg.localization_mode == "fused":
            # Very simple fusion: interpolate translation, slerp-ish rotation.
            T_odom = self._T_world_base_odom

            # translation
            t_odom = T_odom[:3, 3]
            t_meas = T_world_base_meas[:3, 3]
            t_new = (1.0 - alpha) * t_odom + alpha * t_meas

            # rotation (naive: blend rotation matrices and re-orthogonalize)
            R_odom = T_odom[:3, :3]
            R_meas = T_world_base_meas[:3, :3]
            R_blend = (1.0 - alpha) * R_odom + alpha * R_meas
            # project to SO(3) via SVD
            U, _, Vt = np.linalg.svd(R_blend)
            R_new = U @ Vt

            T_new = np.eye(4, dtype=float)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = t_new

            self._T_world_base = T_new

    # --------------------------------------------------------------------------
    # EE pose setters (base frame)
    # --------------------------------------------------------------------------

    def set_T_base_ee(self, side: str, T_base_ee: np.ndarray) -> None:
        """Set base→EE pose for a given side ('left' or 'right')."""
        T_base_ee = np.asarray(T_base_ee, dtype=float)
        if T_base_ee.shape != (4, 4):
            raise ValueError("T_base_ee must be 4x4")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self._T_base_ee[side] = T_base_ee

    def set_T_base_ee_left(self, T_base_ee: np.ndarray) -> None:
        """Set base→EE for the left arm."""
        self.set_T_base_ee("left", T_base_ee)

    def set_T_base_ee_right(self, T_base_ee: np.ndarray) -> None:
        """Set base→EE for the right arm."""
        self.set_T_base_ee("right", T_base_ee)

    # --------------------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------------------

    def get_T_world_base(self) -> np.ndarray:
        """Return current T_world_base (copy)."""
        return self._T_world_base.copy()

    def get_T_world_ee(self, T_base_ee: np.ndarray) -> np.ndarray:
        """Lift a base-frame EE pose to world frame.

        T_base_ee: 4x4 from e.g. reachy.{l,r}_arm.forward_kinematics()
        """
        T_base_ee = np.asarray(T_base_ee, dtype=float)
        if T_base_ee.shape != (4, 4):
            raise ValueError("T_base_ee must be 4x4")
        return self._T_world_base @ T_base_ee

    def get_T_world_ee_left(self) -> Optional[np.ndarray]:
        """Return T_world_ee for left arm, if set."""
        T_base_ee = self._T_base_ee.get("left")
        if T_base_ee is None:
            return None
        return self.get_T_world_ee(T_base_ee)

    def get_T_world_ee_right(self) -> Optional[np.ndarray]:
        """Return T_world_ee for right arm, if set."""
        T_base_ee = self._T_base_ee.get("right")
        if T_base_ee is None:
            return None
        return self.get_T_world_ee(T_base_ee)

    def get_T_world_cam(self, camera_id: str) -> Optional[np.ndarray]:
        """T_world_cam for a named camera: 'teleop_left', 'teleop_right', 'depth_rgb'."""
        T_base_cam = self._T_base_cam.get(camera_id)
        if T_base_cam is None:
            return None
        return self._T_world_base @ T_base_cam

    def get_intrinsics(self, camera_id: str) -> Optional[np.ndarray]:
        """Return 3x3 K for the given camera_id."""
        K = self._K.get(camera_id)
        if K is None:
            return None
        return np.asarray(K, dtype=float)

    def get_image_size(self, camera_id: str) -> Optional[Tuple[int, int]]:
        """Return (height, width) for the given camera_id."""
        return self._image_size.get(camera_id)
