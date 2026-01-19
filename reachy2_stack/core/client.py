from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from reachy2_sdk import ReachySDK
from reachy2_sdk.media.camera import CameraView

from reachy2_stack.utils.utils_dataclass import ReachyConfig, ReachyCameraData, TeleopCameraData, DepthCameraData


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

    def close(self) -> None:
        """Cleanly close the underlying ReachySDK client."""
        if self.reachy is not None:
            # ReachySDK supports context-manager style, so it should have close().
            try:
                self.reachy.close()
            except AttributeError:
                # Fallback: just drop the reference; background thread will die when process ends.
                pass
            self.reachy = None


    # ------------------------------------------------------------------
    # Internal helpers (NEW)
    # ------------------------------------------------------------------

    def _get_arm(self, side: str):
        """Return SDK arm object for 'left' or 'right'."""
        assert self.reachy is not None
        if side == "left":
            arm = getattr(self.reachy, "l_arm", None)
        elif side == "right":
            arm = getattr(self.reachy, "r_arm", None)
        else:
            raise ValueError(f"Unknown arm side: {side!r}, expected 'left' or 'right'.")
        if arm is None:
            raise RuntimeError(f"Reachy SDK has no '{side}_arm' attribute.")
        return arm

    def _get_gripper(self, side: str):
        """Return SDK gripper object for 'left' or 'right'."""
        arm = self._get_arm(side)
        g = getattr(arm, "gripper", None)
        if g is None:
            raise RuntimeError(f"Arm '{side}' has no gripper attached.")
        return g

    def _get_mobile_base(self):
        """Return SDK mobile_base object, or raise if not present."""
        assert self.reachy is not None
        mb = getattr(self.reachy, "mobile_base", None)
        if mb is None:
            raise RuntimeError("This Reachy has no mobile_base attached.")
        return mb

    # --- joint state ---

    def get_joint_state_right(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities) for the right arm joints.

        For now:
        - positions = reachy.r_arm.get_current_positions()  # 7-DoF, degrees
        """
        # self.connect()
        assert self.reachy is not None

        q_r = np.array(self.reachy.r_arm.get_current_positions(), dtype=float)
        dq_r = np.zeros_like(q_r)
        return q_r, dq_r

    def get_joint_state_left(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities) for the left arm joints.

        For now:
        - positions = reachy.l_arm.get_current_positions()  # 7-DoF, degrees
        """
        # self.connect()
        assert self.reachy is not None

        q_l = np.array(self.reachy.l_arm.get_current_positions(), dtype=float)
        dq_l = np.zeros_like(q_l)
        return q_l, dq_l


    # ------------------------------------------------------------------
    # power
    # ------------------------------------------------------------------
    def turn_on_all(self) -> None:
        """Turn on all actuators (arms, head, base, etc.) if supported."""
        assert self.reachy is not None
        if hasattr(self.reachy, "turn_on"):
            self.reachy.turn_on()

    def turn_off_all(self) -> None:
        """Turn off all actuators smoothly if supported."""
        assert self.reachy is not None
        if hasattr(self.reachy, "turn_off_smoothly"):
            self.reachy.turn_off_smoothly()

    # ------------------------------------------------------------------
    # Arm motion: goto (NEW)
    # ------------------------------------------------------------------

    def goto_arm_joints(
        self,
        side: str,
        q: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ):
        """Goto a joint configuration for given arm.

        Args:
            side: "left" or "right".
            q: 7-D joint array.
            duration: movement duration in seconds (must be > 0).
            wait: if True, block until the movement is complete.
            interpolation_mode: 'minimum_jerk', 'linear'.
            degrees: if False, `q` is interpreted in radians.
        """
        assert self.reachy is not None
        arm = self._get_arm(side)
        q_list = np.asarray(q, dtype=float).tolist()
        goto_obj = arm.goto(
            q_list,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            degrees=degrees,
        )
        return goto_obj

    def goto_right_arm_joints(
        self,
        q: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ):
        return self.goto_arm_joints(
            side="right",
            q=q,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            degrees=degrees,
        )

    def goto_left_arm_joints(
        self,
        q: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
    ):
        return self.goto_arm_joints(
            side="left",
            q=q,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            degrees=degrees,
        )
    
    def goto_posture(self, name: str = "default", wait: bool = True) -> Any:
        """
        Wrapper around reachy.goto_posture().
        Example posture names (from Reachy SDK):
            - "default"
            - "elbow_90"
        """
        assert self.reachy is not None

        if not hasattr(self.reachy, "goto_posture"):
            raise AttributeError("This ReachySDK version does not support goto_posture().")

        return self.reachy.goto_posture(common_posture=name, wait=wait)

    def goto_ee_pose_base(
        self,
        side: str,
        T_base_ee: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        interpolation_space: str = "joint_space",
        **cartesian_kwargs: Any,
    ):
        """Goto EE pose in Reachy base frame.

        Args:
            side: "left" or "right".
            T_base_ee: 4x4 homogeneous transform (base <- ee).
            duration: movement duration in seconds.
            wait: if True, block until the movement is complete.
            interpolation_mode: 'minimum_jerk', 'linear', 'elliptical' (cartesian only).
            interpolation_space: 'joint_space' (default) or 'cartesian_space'.
            cartesian_kwargs: forwarded directly to SDK
                (e.g. arc_direction, secondary_radius for elliptical).
        """
        assert self.reachy is not None
        arm = self._get_arm(side)
        T = np.asarray(T_base_ee, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"T_base_ee must be 4x4, got {T.shape}.")

        goto_obj = arm.goto(
            T,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            interpolation_space=interpolation_space,
            **cartesian_kwargs,
        )
        return goto_obj

    def goto_ee_pose_base_right(
        self,
        T_base_ee: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        interpolation_space: str = "joint_space",
        **cartesian_kwargs: Any,
    ):
        return self.goto_ee_pose_base(
            side="right",
            T_base_ee=T_base_ee,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            interpolation_space=interpolation_space,
            **cartesian_kwargs,
        )

    def goto_ee_pose_base_left(
        self,
        T_base_ee: np.ndarray,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        interpolation_space: str = "joint_space",
        **cartesian_kwargs: Any,
    ):
        return self.goto_ee_pose_base(
            side="left",
            T_base_ee=T_base_ee,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            interpolation_space=interpolation_space,
            **cartesian_kwargs,
        )
    def goto_base_defined_speed(self,
                                vx: float | int = 0,	
                                vy: float | int = 0,	
                                vtheta: float | int = 0
                                ):
        """sets a target base speed and maitains that speed for 200 ms. A sleep time of 0.1 second is necessary
        between two consecutive commands.
        
        Args:
            vx: velocity on x in m/s.
            vy: velocity on y in m/s.
            vtheta: yaw angular velocity in deg/s.
        """
        
        self.reachy._mobile_base.set_goal_speed(vx=vx, vy=vy, vtheta=vtheta)
        return self.reachy._mobile_base.send_speed_command() 

    
    def goto_head(
            self,	
            target: Any,	
            duration: float = 2.0,	
            wait: bool = False,	
            interpolation_mode: str = 'minimum_jerk',
            degrees: bool = True):
        """go to a certain head pose.

        Args:
            target: a list [roll, pitch, yaw].
            duration: the movement duration in seconds.
            duration: movement duration in seconds.
            wait: if True, block until the movement is complete.
            interpolation_mode: 'minimum_jerk', 'linear', 'elliptical' (cartesian only).
            degrees: if True, reads target values in degrees, otherwise in radians.
            
        """
        return self.reachy._head.goto(target=target,	
            duration=duration,	
            wait=wait,	
            interpolation_mode = interpolation_mode,
            degrees=degrees)


    # --- arm kinematics (NEW) ---

    def arm_forward_kinematics(
        self,
        side: str,
        q: Optional[np.ndarray] = None,
        degrees: bool = True,
    ) -> np.ndarray:
        """Compute EE pose in base frame using SDK forward_kinematics().

        If q is None: uses current joint positions.
        """
        assert self.reachy is not None
        arm = self._get_arm(side)
        if q is None:
            return np.asarray(arm.forward_kinematics(), dtype=float)
        q_list = np.asarray(q, dtype=float).tolist()
        return np.asarray(arm.forward_kinematics(q_list, degrees=degrees), dtype=float)

    def arm_inverse_kinematics(
        self,
        side: str,
        T_base_ee: np.ndarray,
        degrees: bool = True,
    ) -> np.ndarray:
        """Compute joints for a given EE pose in base frame."""
        assert self.reachy is not None
        arm = self._get_arm(side)
        T = np.asarray(T_base_ee, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"T_base_ee must be 4x4, got {T.shape}.")
        q = arm.inverse_kinematics(T, degrees=degrees)
        return np.asarray(q, dtype=float)
    

    # --- grippers ---

    def get_gripper_opening_right(self) -> float:
        """Return normalized right gripper opening in [0, 1].

        SDK side:
            reachy.r_arm.gripper.get_current_opening()  # 0..100
        We just normalize it to 0..1.
        """
        # self.connect()
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
        # self.connect()
        assert self.reachy is not None

        g = self.reachy.l_arm.gripper
        opening_raw = float(g.get_current_opening())  # 0..100
        return opening_raw / 100.0

    # --- gripper motion (NEW) ---

    def gripper_goto(
        self,
        side: str,
        target: float,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
        percentage: bool = False,
    ):
        """Goto gripper position.

        Args:
            side: 'left' or 'right'.
            target: position (deg, rad or %) depending on flags.
            duration, wait, interpolation_mode: as in SDK.
            degrees: interpret `target` as degrees (if not percentage).
            percentage: if True, interpret `target` as 0..100 opening.
        """
        assert self.reachy is not None
        g = self._get_gripper(side)
        goto_obj = g.goto(
            target,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            degrees=degrees,
            percentage=percentage,
        )
        return goto_obj

    def gripper_open(self, side: str) -> None:
        """Fully open gripper (non-goto, immediate)."""
        assert self.reachy is not None
        g = self._get_gripper(side)
        g.open()

    def gripper_close(self, side: str) -> None:
        """Close gripper with smart stopping (non-goto, immediate)."""
        assert self.reachy is not None
        g = self._get_gripper(side)
        g.close()

    def gripper_set_opening(self, side: str, opening_percent: float) -> None:
        """Set gripper opening [0, 100] (non-goto)."""
        assert self.reachy is not None
        g = self._get_gripper(side)
        g.set_opening(float(opening_percent))


    # --- base / odometry ---

    def get_mobile_odometry(self) -> Optional[dict]:
        """Return raw odometry dict from mobile_base, or None if no base."""
        # self.connect()
        assert self.reachy is not None

        mb = getattr(self.reachy, "mobile_base", None)
        if mb is None:
            return None
        # {'x': m, 'y': m, 'theta': deg}
        return dict(mb.odometry)
    

    def base_reset_odometry(self) -> None:
        """Reset mobile base odometry to current pose."""
        assert self.reachy is not None
        mb = self._get_mobile_base()
        mb.reset_odometry()

    def base_goto(
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
        """Goto a pose in odom frame.

        Args:
            x, y: meters.
            theta: heading (deg by default, rad if degrees=False).
            wait: if True, block until goto finished.
            distance_tolerance: in meters.
            angle_tolerance: in deg or rad (matches `degrees` flag).
            timeout: max duration before abort [s].
            degrees: interpret theta & angle_tolerance in degrees.
        """
        assert self.reachy is not None
        mb = self._get_mobile_base()
        goto_obj = mb.goto(
            x=x,
            y=y,
            theta=theta,
            distance_tolerance=distance_tolerance,
            angle_tolerance=angle_tolerance,
            timeout=timeout,
            wait=wait,
            degrees=degrees,
        )
        return goto_obj

    def base_translate_by(
        self,
        x: float = 0.0,
        y: float = 0.0,
        wait: bool = False,
        **kwargs: Any,
    ):
        """Translate base in robot frame using translate_by()."""
        assert self.reachy is not None
        mb = self._get_mobile_base()
        goto_obj = mb.translate_by(x=x, y=y, wait=wait, **kwargs)
        return goto_obj

    def base_rotate_by(
        self,
        theta: float,
        wait: bool = False,
        degrees: bool = True,
        **kwargs: Any,
    ):
        """Rotate base in robot frame using rotate_by()."""
        assert self.reachy is not None
        mb = self._get_mobile_base()
        goto_obj = mb.rotate_by(theta=theta, wait=wait, degrees=degrees, **kwargs)
        return goto_obj


    # --- cameras: teleop stereo ---

    def get_teleop_rgb_left(self) -> Optional[np.ndarray]:
        """Return LEFT teleop RGB frame, or None if missing."""
        # self.connect()
        assert self.reachy is not None

        cams = getattr(self.reachy, "cameras", None)
        if cams is None or cams.teleop is None:
            return None
        frame, ts = cams.teleop.get_frame(CameraView.LEFT)
        return frame

    def get_teleop_rgb_right(self) -> Optional[np.ndarray]:
        """Return RIGHT teleop RGB frame, or None if missing."""
        # self.connect()
        assert self.reachy is not None

        cams = getattr(self.reachy, "cameras", None)
        if cams is None or cams.teleop is None:
            return None
        frame, ts = cams.teleop.get_frame(CameraView.RIGHT)
        return frame

    # --- cameras: depth RGBD ---

    def get_depth_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (rgb, depth) from depth camera, or (None, None) if missing."""
        # self.connect()
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
        # self.connect()
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
        # self.connect()
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
        # self.connect()
        assert self.reachy is not None
        T = self.reachy.cameras.teleop.get_extrinsics(CameraView.LEFT)
        return np.linalg.inv(np.asarray(T, dtype=float))

    def get_teleop_extrinsics_right(self) -> np.ndarray:
        """Return 4x4 T_base_cam for RIGHT teleop camera in robot/base frame."""
        # self.connect()
        assert self.reachy is not None
        T = self.reachy.cameras.teleop.get_extrinsics(CameraView.RIGHT)
        return np.linalg.inv(np.asarray(T, dtype=float))

    def get_depth_intrinsics(self, view: Optional[CameraView] = None) -> Dict[str, Any]:
        """Return intrinsics for depth camera (RGB or DEPTH view)."""
        # self.connect()
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
        # self.connect()
        assert self.reachy is not None

        if view is None:
            T = self.reachy.cameras.depth.get_extrinsics()
        else:
            T = self.reachy.cameras.depth.get_extrinsics(view)
        return np.linalg.inv(np.asarray(T, dtype=float))
    
    def get_all_camera_data(self) -> ReachyCameraData:
        # self.connect()
        assert self.reachy is not None

        # Teleop left
        teleop_left = TeleopCameraData(
            rgb=self.get_teleop_rgb_left(),
            intrinsics=self.get_teleop_intrinsics_left(),
            extrinsics=self.get_teleop_extrinsics_left(),
        )

        # Teleop right
        teleop_right = TeleopCameraData(
            rgb=self.get_teleop_rgb_right(),
            intrinsics=self.get_teleop_intrinsics_right(),
            extrinsics=self.get_teleop_extrinsics_right(),
        )

        # Depth camera
        depth_rgb, depth_map = self.get_depth_rgbd()
        depth = DepthCameraData(
            rgb=depth_rgb,
            depth=depth_map,
            intrinsics=self.get_depth_intrinsics(),
            extrinsics=self.get_depth_extrinsics(),
        )

        return ReachyCameraData(
            teleop_left=teleop_left,
            teleop_right=teleop_right,
            depth=depth,
        )
    
    def get_teleop_left_camera_data(self) -> TeleopCameraData:
        # self.connect()
        assert self.reachy is not None

        return TeleopCameraData(
            rgb=self.get_teleop_rgb_left(),
            intrinsics=self.get_teleop_intrinsics_left(),
            extrinsics=self.get_teleop_extrinsics_left(),
        )
    
    def get_teleop_right_camera_data(self) -> TeleopCameraData:
        # self.connect()
        assert self.reachy is not None

        return TeleopCameraData(
            rgb=self.get_teleop_rgb_right(),
            intrinsics=self.get_teleop_intrinsics_right(),
            extrinsics=self.get_teleop_extrinsics_right(),
        )
    
    def get_depth_camera_data(self) -> DepthCameraData:
        # self.connect()
        assert self.reachy is not None

        rgb, depth_map = self.get_depth_rgbd()
        return DepthCameraData(
            rgb=rgb,
            depth=depth_map,
            intrinsics=self.get_depth_intrinsics(),
            extrinsics=self.get_depth_extrinsics(),
        )


    def is_goto_finished(self, goto_handle: Any) -> bool:
        """Return True if the given goto is finished (or cancelled)."""
        assert self.reachy is not None
        return bool(self.reachy.is_goto_finished(goto_handle))

    def get_goto_request(self, goto_handle: Any):
        """Return SDK goto request object for inspection."""
        assert self.reachy is not None
        return self.reachy.get_goto_request(goto_handle)

    def cancel_goto_by_id(self, goto_id: int) -> None:
        """Cancel a specific goto by its id."""
        assert self.reachy is not None
        self.reachy.cancel_goto_by_id(goto_id)

    def cancel_all_goto(self) -> None:
        """Cancel all gotos on all parts."""
        assert self.reachy is not None
        self.reachy.cancel_all_goto()

    # --- generic action interfaces (to be filled in later) ---
# --- generic action interfaces (wired up lightly) ---

    def send_joint_delta(self, joint_delta: np.ndarray) -> None:
        """Apply a small delta in joint space on the RIGHT arm.

        For teleop-style nudges. Expects 7-D delta in degrees.
        """
        assert self.reachy is not None

        dq = np.asarray(joint_delta, dtype=float)
        if dq.shape != (7,):
            raise ValueError(f"joint_delta must be shape (7,), got {dq.shape}.")

        q_curr, _ = self.get_joint_state_right()
        q_target = q_curr + dq

        # Short, smooth move, non-blocking
        self.goto_right_arm_joints(q_target, duration=0.5, wait=False)

    def send_ee_delta(self, ee_delta: np.ndarray) -> None:
        """Apply a small delta in EE pose (right arm, base frame).

        ee_delta = [dx, dy, dz, droll, dpitch, dyaw] in base frame.
        Rotations in radians.

        For serious use, prefer ArmController with a WorldModel.
        """
        raise NotImplementedError(
            "send_ee_delta requires composing with current EE pose. "
            "Use ArmController.goto_pose_base/world instead."
        )
    def send_goal_positions(self, check_positions: bool = False) -> None:
        """Send the goal positions to the robot.

        If goal positions have been specified for any joint of the robot, sends them to the robot.

        Args :
            check_positions: A boolean indicating whether to check the positions after sending the command.
                Defaults to True.
        """
        self.reachy.send_goal_positions(check_positions=check_positions)

    def execute_skill(self, name: str, **kwargs) -> None:
        """Call a named skill implemented in reachy2_stack.skills."""
        from reachy2_stack.skills import skill_registry

        skill_fn = skill_registry.get(name)
        if skill_fn is None:
            raise ValueError(f"Unknown skill: {name}")
        skill_fn(self, **kwargs)