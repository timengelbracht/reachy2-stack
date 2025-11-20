from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from reachy2_sdk import ReachySDK
from reachy2_sdk.media.camera import CameraView


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


class ReachyClient:
    """Thin wrapper around ReachySDK."""

    def __init__(self, cfg: ReachyConfig):
        self.cfg = cfg
        self.reachy: Optional[ReachySDK] = None

    # --- connection lifecycle ---

    @property
    def connect_reachy(self) -> ReachySDK:
        """Lazily construct and return the ReachySDK client."""
        if self.reachy is None:
            # TODO: pass sim flag / different port if needed
            self.reachy = ReachySDK(host=self.cfg.host)

            if self.reachy.is_connected():
                print("Reachy SDK connected successfully.")
            else:
                # reset to None for clarity
                self.reachy = None
                raise ConnectionError("Failed to connect to Reachy SDK.")

        return self.reachy

    def connect(self) -> None:
        """Connect to the robot/sim. Idempotent."""
        _ = self.connect_reachy

    # --- joint state ---

    def get_joint_state_right(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities) for the right arm joints.

        For now:
        - positions = reachy.r_arm.get_current_positions()  # 7-DoF, degrees
        """
        self.connect()
        assert self.reachy is not None

        q_r = np.array(self.reachy.r_arm.get_current_positions(), dtype=float)
        dq_r = np.zeros_like(q_r)
        return q_r, dq_r

    def get_joint_state_left(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities) for the left arm joints.

        For now:
        - positions = reachy.l_arm.get_current_positions()  # 7-DoF, degrees
        """
        self.connect()
        assert self.reachy is not None

        q_l = np.array(self.reachy.l_arm.get_current_positions(), dtype=float)
        dq_l = np.zeros_like(q_l)
        return q_l, dq_l

    # --- grippers ---

    def get_gripper_opening_right(self) -> float:
        """Return normalized right gripper opening in [0, 1].

        SDK side:
            reachy.r_arm.gripper.get_current_opening()  # 0..100
        We just normalize it to 0..1.
        """
        self.connect()
        assert self.reachy is not None

        g = self.reachy.r_arm.gripper
        opening_raw = float(g.get_current_opening())  # 0..100
        return opening_raw / 100.0

    def get_gripper_opening_left(self) -> float:
        """Return normalized left gripper opening in [0, 1].

        SDK side:
            reachy.l_arm.gripper.get_current_opening()  # 0..100
        We just normalize it to 0..1.
        """
        self.connect()
        assert self.reachy is not None

        g = self.reachy.l_arm.gripper
        opening_raw = float(g.get_current_opening())  # 0..100
        return opening_raw / 100.0

    # --- base / odometry ---

    def get_mobile_odometry(self) -> Optional[dict]:
        """Return raw odometry dict from mobile_base, or None if no base."""
        self.connect()
        assert self.reachy is not None

        mb = getattr(self.reachy, "mobile_base", None)
        if mb is None:
            return None
        # {'x': m, 'y': m, 'theta': deg}
        return dict(mb.odometry)

    # --- cameras: teleop stereo ---

    def get_teleop_rgb_left(self) -> Optional[np.ndarray]:
        """Return LEFT teleop RGB frame, or None if missing."""
        self.connect()
        assert self.reachy is not None

        cams = getattr(self.reachy, "cameras", None)
        if cams is None or cams.teleop is None:
            return None
        frame, ts = cams.teleop.get_frame(CameraView.LEFT)
        return frame

    def get_teleop_rgb_right(self) -> Optional[np.ndarray]:
        """Return RIGHT teleop RGB frame, or None if missing."""
        self.connect()
        assert self.reachy is not None

        cams = getattr(self.reachy, "cameras", None)
        if cams is None or cams.teleop is None:
            return None
        frame, ts = cams.teleop.get_frame(CameraView.RIGHT)
        return frame

    # --- cameras: depth RGBD ---

    def get_depth_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (rgb, depth) from depth camera, or (None, None) if missing."""
        self.connect()
        assert self.reachy is not None

        cams = getattr(self.reachy, "cameras", None)
        if cams is None or cams.depth is None:
            return None, None

        rgb, ts_rgb = cams.depth.get_frame()
        depth, ts_d = cams.depth.get_depth_frame()
        return rgb, depth

    # --- intrinsics and extrinsics ---

    def get_teleop_intrinsics_left(self) -> Dict[str, Any]:
        """Return raw intrinsics for teleop camera in a dict."""
        self.connect()
        assert self.reachy is not None

        h, w, distortion_model, D, K, R, P = self.reachy.cameras.teleop.get_parameters(
            CameraView.LEFT
        )
        return {
            "height": h,
            "width": w,
            "distortion_model": distortion_model,
            "D": D,
            "K": K,
            "R": R,
            "P": P,
        }

    def get_teleop_intrinsics_right(self) -> Dict[str, Any]:
        """Return raw intrinsics for teleop camera in a dict."""
        self.connect()
        assert self.reachy is not None

        h, w, distortion_model, D, K, R, P = self.reachy.cameras.teleop.get_parameters(
            CameraView.RIGHT
        )
        return {
            "height": h,
            "width": w,
            "distortion_model": distortion_model,
            "D": D,
            "K": K,
            "R": R,
            "P": P,
        }

    def get_teleop_extrinsics_left(self) -> np.ndarray:
        """Return 4x4 T_base_cam for LEFT teleop camera in robot/base frame."""
        self.connect()
        assert self.reachy is not None
        T = self.reachy.cameras.teleop.get_extrinsics(CameraView.LEFT)
        return np.asarray(T, dtype=float)

    def get_teleop_extrinsics_right(self) -> np.ndarray:
        """Return 4x4 T_base_cam for RIGHT teleop camera in robot/base frame."""
        self.connect()
        assert self.reachy is not None
        T = self.reachy.cameras.teleop.get_extrinsics(CameraView.RIGHT)
        return np.asarray(T, dtype=float)

    def get_depth_intrinsics(self, view: Optional[CameraView] = None) -> Dict[str, Any]:
        """Return intrinsics for depth camera (RGB or DEPTH view)."""
        self.connect()
        assert self.reachy is not None

        if view is None:
            args = ()
        else:
            args = (view,)

        h, w, distortion_model, D, K, R, P = self.reachy.cameras.depth.get_parameters(
            *args
        )
        return {
            "height": h,
            "width": w,
            "distortion_model": distortion_model,
            "D": D,
            "K": K,
            "R": R,
            "P": P,
        }

    def get_depth_extrinsics(
        self, view: Optional[CameraView] = None
    ) -> np.ndarray:
        """Return 4x4 T_base_cam for depth camera in robot/base frame."""
        self.connect()
        assert self.reachy is not None

        if view is None:
            T = self.reachy.cameras.depth.get_extrinsics()
        else:
            T = self.reachy.cameras.depth.get_extrinsics(view)
        return np.asarray(T, dtype=float)

    # --- generic action interfaces (to be filled in later) ---

    def send_joint_delta(self, joint_delta: np.ndarray) -> None:
        """Apply a small delta in joint space."""
        # TODO: implement when joint-space control design is decided
        pass

    def send_ee_delta(self, ee_delta: np.ndarray) -> None:
        """Apply a small delta in task space (end-effector)."""
        # TODO: implement when Cartesian control design is decided
        pass

    def execute_skill(self, name: str, **kwargs) -> None:
        """Call a named skill implemented in reachy2_stack.skills."""
        from reachy2_stack.skills import skill_registry

        skill_fn = skill_registry.get(name)
        if skill_fn is None:
            raise ValueError(f"Unknown skill: {name}")
        skill_fn(self, **kwargs)
