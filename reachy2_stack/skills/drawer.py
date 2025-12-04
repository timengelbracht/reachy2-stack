from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from reachy2_stack.infra.world_model import WorldModel
from reachy2_stack.control.arm import ArmController
from reachy2_stack.control.gripper import GripperController
from reachy2_stack.utils.utils_skills import pose_from_normal_and_handle
from reachy2_stack.utils.utils_dataclass import ArticulatedObjectInstance


@dataclass
class DrawerOpenFixedBaseSkill:
    """Open a drawer using a known handle trajectory (WORLD frame).
    
    Requires:
      - arm.world is a valid WorldModel
      - drawer.trajectory: (K,3) array of handle center positions in WORLD frame
      - drawer.normal_closed: (3,)
      - drawer.handle_longest_axis: (3,)
      - drawer.articulation_type == "prismatic"
    """

    arm: ArmController
    gripper: GripperController

    # Tunable parameters
    approach_dist: float = 0.12
    grasp_offset: float = 0.06
    pregrasp_duration: float = 2.0
    grasp_duration: float = 1.2
    waypoint_duration: float = 1.5
    post_retreat_dist: float = 0.10  # retreat after releasing handle

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _compute_pregrasp_and_grasp(
        self,
        normal_world: np.ndarray,
        handle_axis_world: np.ndarray,
        start_point_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Orientation at the handle
        T_world_ee_start = pose_from_normal_and_handle(
            normal_world=normal_world,
            handle_axis_world=handle_axis_world,
            handle_center_world=start_point_world,
        )

        # Extract +Z axis of EE (world frame)
        z_w = T_world_ee_start[:3, 2]

        # Grasp pose (slightly offset before touching)
        T_grasp = T_world_ee_start.copy()
        T_grasp[:3, 3] = start_point_world + self.grasp_offset * z_w

        # Pre-grasp pose (further away)
        T_pre = T_world_ee_start.copy()
        T_pre[:3, 3] = start_point_world + self.approach_dist * z_w

        return T_pre, T_grasp

    def _poses_for_trajectory(
        self,
        normal_world: np.ndarray,
        handle_axis_world: np.ndarray,
        traj_world: np.ndarray,
    ) -> List[np.ndarray]:

        poses: List[np.ndarray] = []
        for p in traj_world:
            T = pose_from_normal_and_handle(
                normal_world=normal_world,
                handle_axis_world=handle_axis_world,
                handle_center_world=p,
            )
            T[:3, 3] = p  # ensure exact translation
            poses.append(T)
        return poses

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def open_drawer(self, drawer: ArticulatedObjectInstance, wait: bool = True) -> None:
        """Open a prismatic drawer along a known WORLD-frame trajectory."""

        if drawer.articulation_type != "prismatic":
            raise ValueError(
                f"DrawerOpenFixedBaseSkill only supports prismatic drawers, got {drawer.articulation_type!r}"
            )

        if self.arm.world is None:
            raise RuntimeError("ArmController.world must be set (WorldModel).")

        traj = np.asarray(drawer.trajectory, dtype=float)
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError("drawer.trajectory must be (K,3).")

        if traj.shape[0] < 2:
            raise ValueError("Trajectory must contain at least two points.")

        n_w = np.asarray(drawer.normal_closed, dtype=float)
        a_w = np.asarray(drawer.handle_longest_axis, dtype=float)
        p0 = traj[0]

        # ---- pre-grasp + grasp ----
        T_pre, T_grasp = self._compute_pregrasp_and_grasp(n_w, a_w, p0)

        # Move into pre-grasp (blocks via ArmController.wait)
        self.arm.goto_pose_world(
            T_pre,
            duration=self.pregrasp_duration,
            wait=True,
            interpolation_mode="minimum_jerk",
        )

        # Move into grasp (blocks)
        self.arm.goto_pose_world(
            T_grasp,
            duration=self.grasp_duration,
            wait=True,
            interpolation_mode="minimum_jerk",
        )

        # Close gripper *after* arm is done (no explicit wait here)
        self.gripper.close()

        # ---- go directly to final pose of trajectory ----
        poses = self._poses_for_trajectory(n_w, a_w, traj)
        T_final = poses[-1]

        # Pull to final pose (blocks)
        self.arm.goto_pose_world(
            T_final,
            duration=self.waypoint_duration,
            wait=True,  # we want to be sure we're at the final pose
            interpolation_mode="minimum_jerk",
        )

        # Open gripper *after* arm is done (no explicit wait here)
        self.gripper.open()

        # ---- post-release retreat away from handle ----
        z_w_final = T_final[:3, 2]
        T_post = T_final.copy()
        T_post[:3, 3] = T_final[:3, 3] + self.post_retreat_dist * z_w_final

        self.arm.goto_pose_world(
            T_post,
            duration=self.waypoint_duration,
            wait=wait,
            interpolation_mode="minimum_jerk",
        )

        drawer.is_open = True
        drawer.open_fraction = 1.0
