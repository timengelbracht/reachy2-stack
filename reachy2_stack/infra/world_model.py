from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np

from reachy2_stack.core.client import ReachyClient
from reachy2_stack.utils.utils_dataclass import WorldCameraState, WorldEEState, WorldState


@dataclass
class WorldModelConfig:
    """Config for how we track global poses."""
    location_name: str = "default"
    # "odom"  : trust wheel odom only
    # "vision": trust visual localization only
    # "fused" : simple odom + vision fusion
    localization_mode: str = "odom"


class WorldModel:
    """Central place for transforms & map-free kinematics.

    Conventions:
      - T_world_base : 4x4 transform from base to world    (world ← base)
      - T_base_cam   : 4x4 transform from camera to base   (base  ← cam)
      - T_base_ee    : 4x4 transform from EE to base       (base  ← ee)
    """

    def __init__(self, cfg: WorldModelConfig):
        self.cfg = cfg

        # Base pose
        self._T_world_base = np.eye(4, dtype=float)
        self._T_world_base_odom = np.eye(4, dtype=float)

        # Static camera calibration (base frame)
        # keys: "teleop_left", "teleop_right", "depth_rgb"
        self._T_base_cam: Dict[str, np.ndarray] = {}
        self._K: Dict[str, np.ndarray] = {}
        self._image_size: Dict[str, Tuple[int, int]] = {}

        # Current EE poses in base frame
        # keys: "left", "right"
        self._T_base_ee: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset dynamic pose state (does not touch camera calibration)."""
        self._T_world_base = np.eye(4, dtype=float)
        self._T_world_base_odom = np.eye(4, dtype=float)
        self._T_base_ee.clear()

    # ------------------------------------------------------------------
    # Initialization from ReachyClient / SDK
    # ------------------------------------------------------------------

    def init_from_client_calib(self, client: ReachyClient) -> None:
        """Query SDK once via client and cache all camera calibration.

        This sets:
          - T_base_cam (base ← cam) in self._T_base_cam
          - intrinsics K and image size in self._K, self._image_size
        """
        client.connect()

        # Teleop LEFT
        intr_L = client.get_teleop_intrinsics_left()
        T_base_teleop_left = client.get_teleop_extrinsics_left()  # base ← cam

        self._T_base_cam["teleop_left"] = np.asarray(T_base_teleop_left, dtype=float)
        self._K["teleop_left"] = np.asarray(intr_L["K"], dtype=float)
        self._image_size["teleop_left"] = (int(intr_L["height"]), int(intr_L["width"]))

        # Teleop RIGHT
        intr_R = client.get_teleop_intrinsics_right()
        T_base_teleop_right = client.get_teleop_extrinsics_right()  # base ← cam

        self._T_base_cam["teleop_right"] = np.asarray(T_base_teleop_right, dtype=float)
        self._K["teleop_right"] = np.asarray(intr_R["K"], dtype=float)
        self._image_size["teleop_right"] = (int(intr_R["height"]), int(intr_R["width"]))

        # Depth RGB (default RGB stream of depth cam)
        intr_D = client.get_depth_intrinsics()
        T_base_depth_rgb = client.get_depth_extrinsics()  # base ← cam

        self._T_base_cam["depth_rgb"] = np.asarray(T_base_depth_rgb, dtype=float)
        self._K["depth_rgb"] = np.asarray(intr_D["K"], dtype=float)
        self._image_size["depth_rgb"] = (int(intr_D["height"]), int(intr_D["width"]))

    # ------------------------------------------------------------------
    # Odometry + visual localization fusion
    # ------------------------------------------------------------------

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
        self,
        T_world_base_meas: np.ndarray,
        alpha: float = 1.0,
    ) -> None:
        """Incorporate a visual localization measurement.

        Args:
            T_world_base_meas: 4x4 matrix (world ← base) from vision.
            alpha: 1.0 = trust vision fully,
                   0.0 = ignore vision,
                   (0,1) = blend with odom (simple fusion).
        """
        T_world_base_meas = np.asarray(T_world_base_meas, dtype=float)
        if T_world_base_meas.shape != (4, 4):
            raise ValueError("T_world_base_meas must be 4x4")

        if self.cfg.localization_mode == "vision":
            self._T_world_base = T_world_base_meas
            return

        if self.cfg.localization_mode == "fused":
            T_odom = self._T_world_base_odom

            # translation
            t_odom = T_odom[:3, 3]
            t_meas = T_world_base_meas[:3, 3]
            t_new = (1.0 - alpha) * t_odom + alpha * t_meas

            # rotation: blend rotation matrices and re-orthogonalize
            R_odom = T_odom[:3, :3]
            R_meas = T_world_base_meas[:3, :3]
            R_blend = (1.0 - alpha) * R_odom + alpha * R_meas

            U, _, Vt = np.linalg.svd(R_blend)
            R_new = U @ Vt

            T_new = np.eye(4, dtype=float)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = t_new

            self._T_world_base = T_new

        # mode "odom": ignore visual fix by design

    def update_from_visual_localization_depth(
        self,
        loc_results: Dict[str, Dict[str, Any]],
        alpha: float = 1.0,
    ) -> None:
        """Update world pose using ONLY the depth camera HLoc result.

        Assumes:
          - loc_results["depth"]["T_wc"] is a 4x4 (world ← cam) pose
            as returned by HLocLocalizer.load_localization_results().
          - self._T_base_cam["depth_rgb"] is base ← cam for the depth camera
            (Reachy convention).
        """
        if "depth" not in loc_results:
            print("[WorldModel] No 'depth' entry in loc_results; skipping visual update.")
            return

        entry = loc_results["depth"]
        if "T_wc" not in entry:
            print("[WorldModel] 'depth' entry has no T_wc; skipping visual update.")
            return

        # 1) HLoc gives world ← cam for the depth camera
        T_world_cam = np.asarray(entry["T_wc"], dtype=float)  # world ← cam

        # 2) Reachy calibration: base ← cam
        T_base_cam = self.get_T_base_cam("depth_rgb")
        if T_base_cam is None:
            raise RuntimeError(
                "[WorldModel] Missing T_base_cam['depth_rgb'] calibration. "
                "Call init_from_client_calib() first."
            )

        # 3) cam ← base
        T_cam_base = np.linalg.inv(T_base_cam)

        # 4) world ← base = (world ← cam) ∘ (cam ← base)
        T_world_base_meas = T_world_cam @ T_cam_base

        # 5) Apply via fusion logic
        self.apply_visual_fix(T_world_base_meas, alpha=alpha)

    # ------------------------------------------------------------------
    # EE pose setters (base frame)
    # ------------------------------------------------------------------

    def set_T_base_ee(self, side: str, T_base_ee: np.ndarray) -> None:
        """Set base ← EE pose for a given side ('left' or 'right')."""
        T_base_ee = np.asarray(T_base_ee, dtype=float)
        if T_base_ee.shape != (4, 4):
            raise ValueError("T_base_ee must be 4x4")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self._T_base_ee[side] = T_base_ee

    def set_T_base_ee_left(self, T_base_ee: np.ndarray) -> None:
        self.set_T_base_ee("left", T_base_ee)

    def set_T_base_ee_right(self, T_base_ee: np.ndarray) -> None:
        self.set_T_base_ee("right", T_base_ee)

    # ------------------------------------------------------------------
    # Queries & transforms
    # ------------------------------------------------------------------

    def get_T_world_base(self) -> np.ndarray:
        """world ← base"""
        return self._T_world_base.copy()

    def get_T_world_ee(self, T_base_ee: np.ndarray) -> np.ndarray:
        """world ← ee = (world ← base) ∘ (base ← ee)"""
        T_base_ee = np.asarray(T_base_ee, dtype=float)
        if T_base_ee.shape != (4, 4):
            raise ValueError("T_base_ee must be 4x4")
        return self._T_world_base @ T_base_ee

    def get_T_world_ee_left(self) -> Optional[np.ndarray]:
        T_base_ee = self._T_base_ee.get("left")
        if T_base_ee is None:
            return None
        return self.get_T_world_ee(T_base_ee)

    def get_T_world_ee_right(self) -> Optional[np.ndarray]:
        T_base_ee = self._T_base_ee.get("right")
        if T_base_ee is None:
            return None
        return self.get_T_world_ee(T_base_ee)

    def get_T_base_cam(self, camera_id: str) -> Optional[np.ndarray]:
        """Return base ← cam for a named camera."""
        T = self._T_base_cam.get(camera_id)
        if T is None:
            return None
        return np.asarray(T, dtype=float)

    def get_T_world_cam(self, camera_id: str) -> Optional[np.ndarray]:
        """world ← cam = (world ← base) ∘ (base ← cam)."""
        T_base_cam = self.get_T_base_cam(camera_id)
        if T_base_cam is None:
            return None
        return self._T_world_base @ T_base_cam

    def get_intrinsics(self, camera_id: str) -> Optional[np.ndarray]:
        K = self._K.get(camera_id)
        if K is None:
            return None
        return np.asarray(K, dtype=float)

    def get_image_size(self, camera_id: str) -> Optional[Tuple[int, int]]:
        return self._image_size.get(camera_id)

    # ------------------------------------------------------------------
    # World snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> WorldState:
        """
        Build a read-only snapshot of all geometric state:
        base pose, camera poses, EE poses.
        """
        # Cameras
        cameras: Dict[str, WorldCameraState] = {}
        for cam_id, T_base_cam in self._T_base_cam.items():
            T_base_cam = np.asarray(T_base_cam, dtype=float)
            T_world_cam = self.get_T_world_cam(cam_id)
            if T_world_cam is None:
                continue
            K = self._K.get(cam_id)
            img_size = self._image_size.get(cam_id)
            if K is None or img_size is None:
                continue

            cameras[cam_id] = WorldCameraState(
                camera_id=cam_id,
                T_world_cam=T_world_cam,
                T_base_cam=T_base_cam,
                K=np.asarray(K, dtype=float),
                image_size=img_size,
            )

        # End-effectors
        ee: Dict[str, WorldEEState] = {}
        for side, T_base_ee in self._T_base_ee.items():
            T_base_ee = np.asarray(T_base_ee, dtype=float)
            T_world_ee = self.get_T_world_ee(T_base_ee)
            ee[side] = WorldEEState(
                side=side,
                T_world_ee=T_world_ee,
                T_base_ee=T_base_ee,
            )

        return WorldState(
            location_name=self.cfg.location_name,
            T_world_base=self._T_world_base.copy(),
            T_world_base_odom=self._T_world_base_odom.copy(),
            cameras=cameras,
            end_effectors=ee,
        )
