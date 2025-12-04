from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel  

from reachy2_stack.utils.utils_poses import T_to_xytheta, xytheta_to_T



@dataclass
class BaseController:
    """Controller for mobile base (odom / robot frame).

    For now this is odom + robot-frame only.
    If you later define a stable transform between world and odom,
    you can add world-frame helpers here.
    """

    client: ReachyClient
    world: Optional[WorldModel] = None  # kept for future world<->odom logic

    def goto_odom(
        self,
        x: float,
        y: float,
        theta: float,
        wait: bool = False,
        distance_tolerance: float = 0.05,
        angle_tolerance: float = 5.0,
        timeout: float = 100.0,
        degrees: bool = True,
    ):
        """Goto a pose in odom frame (SDK-native)."""
        return self.client.base_goto(
            x=x,
            y=y,
            theta=theta,
            wait=wait,
            distance_tolerance=distance_tolerance,
            angle_tolerance=angle_tolerance,
            timeout=timeout,
            degrees=degrees,
        )

    def reset_odometry(self) -> None:
        self.client.base_reset_odometry()

    def translate_by(
        self,
        x: float = 0.0,
        y: float = 0.0,
        wait: bool = False,
        **kwargs: Any,
    ):
        """Translate in robot frame."""
        return self.client.base_translate_by(x=x, y=y, wait=wait, **kwargs)

    def rotate_by(
        self,
        theta: float,
        wait: bool = False,
        degrees: bool = True,
        **kwargs: Any,
    ):
        """Rotate in robot frame."""
        return self.client.base_rotate_by(theta=theta, wait=wait, degrees=degrees, **kwargs)

    def goto_world(
        self,
        x_world: float,
        y_world: float,
        theta_world: float,
        wait: bool = False,
        distance_tolerance: float = 0.05,
        angle_tolerance: float = 5.0,
        timeout: float = 100.0,
        degrees: bool = True,
    ):
        """
        Goto a pose in WORLD frame, using WorldModel's T_world_base.

        WORLD frame = the map / HLoc frame used by WorldModel:
          - world ← base is tracked in self.world._T_world_base
            via odom + visual localization fusion.

        We:
          1) Read current odom pose (odom ← base) from SDK via client.
          2) Use WorldModel.get_T_world_base() for world ← base.
          3) Derive world ← odom and odom ← world.
          4) Convert the WORLD target pose into an ODOM target.
          5) Call standard goto_odom(...) which uses client.base_goto().
        """

        if self.world is None:
            raise RuntimeError(
                "[BaseController] WorldModel is None; cannot use goto_world()."
            )


        # interpret theta_world according to 'degrees' flag
        if not degrees:
            theta_world_deg = float(np.rad2deg(theta_world))
        else:
            theta_world_deg = float(theta_world)

        # 1) Current ODOM pose: odom ← base
        odom = self.client.get_mobile_odometry()
        if odom is None:
            raise RuntimeError(
                "[BaseController] No mobile_base odometry available; cannot goto_world."
            )

        T_odom_base = xytheta_to_T(
            float(odom["x"]), float(odom["y"]), float(odom["theta"])
        )  # odom ← base

        # 2) WORLD pose from WorldModel: world ← base
        T_world_base = self.world.get_T_world_base()  # world ← base

        # 3) Derive world ← odom
        #    world ← base = (world ← odom) ∘ (odom ← base)
        #    => T_world_odom = T_world_base ∘ (odom ← base)^{-1}
        T_world_odom = T_world_base @ np.linalg.inv(T_odom_base)
        T_odom_world = np.linalg.inv(T_world_odom)

        # 4) Build WORLD target pose: world ← base_target
        T_world_base_target = xytheta_to_T(x_world, y_world, theta_world_deg)

        # 5) Convert WORLD target → ODOM target:
        #    world ← base_target = (world ← odom) ∘ (odom ← base_target)
        #    => T_odom_base_target = (odom ← world) ∘ (world ← base_target)
        T_odom_base_target = T_odom_world @ T_world_base_target

        x_odom_target, y_odom_target, theta_odom_target_deg = T_to_xytheta(
            T_odom_base_target
        )

        # 6) Call the existing odom-based goto
        return self.goto_odom(
            x=x_odom_target,
            y=y_odom_target,
            theta=theta_odom_target_deg,
            wait=wait,
            distance_tolerance=distance_tolerance,
            angle_tolerance=angle_tolerance,
            timeout=timeout,
            degrees=True,  # we explicitly converted to degrees
        )