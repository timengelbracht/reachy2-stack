from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel  


@dataclass
class ArmController:
    """High-level arm controller on top of ReachyClient + WorldModel.

    - Uses Reachy base frame for low-level motion (SDK-native).
    - Optionally exposes world-frame APIs when a WorldModel is provided.
    """

    client: ReachyClient
    side: str  # "left" or "right"
    world: Optional[WorldModel] = None

    # --- base-frame APIs (always available) ---

    def goto_joints(
        self,
        q: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        degrees: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ):
        return self.client.goto_arm_joints(
            side=self.side,
            q=q,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            degrees=degrees,
        )

    def goto_pose_base(
        self,
        T_base_ee: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        interpolation_space: str = "joint_space",
        **cartesian_kwargs: Any,
    ):
        return self.client.goto_ee_pose_base(
            side=self.side,
            T_base_ee=T_base_ee,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            interpolation_space=interpolation_space,
            **cartesian_kwargs,
        )

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """EE pose in base frame."""
        return self.client.arm_forward_kinematics(self.side, q=q)

    def inverse_kinematics(self, T_base_ee: np.ndarray) -> np.ndarray:
        """Joint solution for given pose in base frame."""
        return self.client.arm_inverse_kinematics(self.side, T_base_ee)

    # --- world-frame APIs (require WorldModel) ---

    def goto_pose_world(
        self,
        T_world_ee: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        interpolation_space: str = "joint_space",
        **cartesian_kwargs: Any,
    ):
        """Goto EE pose expressed in world frame.

        world <- ee  = (world <- base) ∘ (base <- ee)
        => base <- ee = (world <- base)^-1 ∘ (world <- ee)
        """
        if self.world is None:
            raise RuntimeError("ArmController.goto_pose_world requires a WorldModel.")

        T_world_base = self.world.get_T_world_base()
        T_world_ee = np.asarray(T_world_ee, dtype=float)
        T_base_ee = np.linalg.inv(T_world_base) @ T_world_ee
        return self.goto_pose_base(
            T_base_ee,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            interpolation_space=interpolation_space,
            **cartesian_kwargs,
        )

    def goto_handle(
        self,
        handle_pose_world: np.ndarray,
        duration: float = 2.0,
        wait: bool = True,
        **kwargs: Any,
    ):
        """Semantic sugar: goto a handle pose in world frame."""
        return self.goto_pose_world(
            handle_pose_world,
            duration=duration,
            wait=wait,
            **kwargs,
        )
