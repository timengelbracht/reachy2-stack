from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import time
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
    
    # ------------------------------------------------------------------
    # Internal mathematical base transformation helpers
    # ------------------------------------------------------------------
    
    def _wrap180(self, deg: float) -> float:
        return (deg + 180.0) % 360.0 - 180.0

    def _Rz4(self, deg: float) -> np.ndarray:
        """Helper function to get a rotation on z transformation matrix"""
        th = np.deg2rad(float(deg))
        c, s = float(np.cos(th)), float(np.sin(th))
        R = np.eye(4, dtype=float)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
        return R
    
    def _T_translate_then_rotate(self, dx: float, dy: float, yaw_rad: float) -> np.ndarray:
        """Model: translate_by(dx,dy) then rotate_by(yaw)."""
        c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))

        Tt = np.eye(4, dtype=float)
        Tt[0, 3] = float(dx)
        Tt[1, 3] = float(dy)

        R = np.eye(4, dtype=float)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c

        return Tt @ R
    
    def _T_rotate_then_translate(self, dx: float, dy: float, yaw_rad: float) -> np.ndarray:
        """Model: rotate_by(yaw) then translate_by(dx,dy)."""
        c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))

        R = np.eye(4, dtype=float)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c

        Tt = np.eye(4, dtype=float)
        Tt[0, 3] = float(dx)
        Tt[1, 3] = float(dy)

        return R @ Tt
    
    def _apply_base_se2(self, A_cur: np.ndarray, dx: float, dy: float, yaw_rad: float, order: str) -> np.ndarray:
        """
        Update target pose expressed in NEW base frame:
            A_new = inv(T) @ A_old
        order:
        - "tr": translate then rotate
        - "rt": rotate then translate
        """
        if order == "tr":
            T = self._T_translate_then_rotate(dx, dy, yaw_rad)
        elif order == "rt":
            T = self._T_rotate_then_translate(dx, dy, yaw_rad)
        else:
            raise ValueError(f"Unknown order={order!r}")
        return np.linalg.inv(T) @ A_cur

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
    
    # ------------------------------------------------------------------
    # arm-base coupled movement
    # ------------------------------------------------------------------

    def _try_arm(self, arm, A_try: np.ndarray) -> bool:
        """
        Send EE pose goto and return True if accepted.
        This matches your earlier logic: gid None/-1 => failure.
        """
        ARM_DURATION = 4.0
        resp = arm.goto(A_try, duration=ARM_DURATION, wait=True)
        gid = getattr(resp, "id", None)
        ok = gid is not None and gid != -1
        print("[ARM] goto id:", gid, "=>", "OK" if ok else "NO")
        return ok


    def _execute_translation_in_steps(
        self,
        dx_goal: float,
        dy_goal: float,
    ) -> float:
        
        remaining_dx = float(dx_goal)
        remaining_dy = float(dy_goal)
        total = 0.0
        MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
        MAX_STEP_TRANS = 0.30             # clamp per translate_by (m)

        while True:
            dist = float(np.hypot(remaining_dx, remaining_dy))
            if dist < 1e-6:
                break

            s = 1.0
            if dist > MAX_STEP_TRANS:
                s = MAX_STEP_TRANS / dist

            sx = remaining_dx * s
            sy = remaining_dy * s

            print(f"[BASE] translate_by(x={sx:+.3f}, y={sy:+.3f})")
            self.base_translate_by(x=float(sx), y=float(sy), wait=True)

            step_dist = float(np.hypot(sx, sy))
            total += step_dist
            if total > MAX_BASE_TOTAL_TRANS:
                raise RuntimeError("Exceeded MAX_BASE_TOTAL_TRANS while translating base.")

            remaining_dx -= sx
            remaining_dy -= sy

        return total

    def _execute_yaw_in_steps(
        self,
        yaw_deg_goal: float,
    ) -> float:
        remaining = float(yaw_deg_goal)
        total_abs = 0.0
        MAX_BASE_TOTAL_YAW_DEG = 180.0    # max total yaw (deg)
        MAX_STEP_YAW_DEG = 30.0           # clamp per rotate_by (deg)

        while abs(remaining) > 1e-3:
            step = float(np.clip(remaining, -MAX_STEP_YAW_DEG, MAX_STEP_YAW_DEG))
            print(f"[BASE] rotate_by(theta={step:.1f} deg)")
            self.base_rotate_by(theta=step, wait=True, degrees=True)

            total_abs += abs(step)
            if total_abs > MAX_BASE_TOTAL_YAW_DEG:
                raise RuntimeError("Exceeded MAX_BASE_TOTAL_YAW_DEG while rotating base.")

            remaining -= step

        return total_abs


    def _translation_priority_candidates(
        self,
        A_cur: np.ndarray
    ) -> list[tuple[float, float]]:
        
        p = A_cur[:2, 3].astype(float)
        n = float(np.linalg.norm(p))
        if n < 1e-9:
            u = np.array([1.0, 0.0], dtype=float)
        else:
            u = p / n

        cands: list[tuple[float, float]] = []
        MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
        LINE_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50]
        AXIS_STEP_TRIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
        DIAG_FACTOR = 0.7

        # 1) along target direction
        for s in LINE_STEP_TRIES:
            dx = float(u[0] * s)
            dy = float(u[1] * s)
            cands.append((dx, dy))

        # 1b) opposite direction
        for s in LINE_STEP_TRIES[:6]:
            dx = float(-u[0] * s)
            dy = float(-u[1] * s)
            cands.append((dx, dy))

        # 2) axis
        for s in AXIS_STEP_TRIES:
            for dx, dy in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
                cands.append((float(dx), float(dy)))

        # 3) diagonals
        for s in AXIS_STEP_TRIES:
            a = float(s)
            b = float(DIAG_FACTOR * s)
            for dx, dy in [(a, b), (a, -b), (-a, b), (-a, -b)]:
                cands.append((dx, dy))

        # filter + dedup
        seen = set()
        uniq: list[tuple[float, float]] = []
        for dx, dy in cands:
            if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS + 1e-9:
                continue
            key = (round(dx, 4), round(dy, 4))
            if key not in seen:
                seen.add(key)
                uniq.append((dx, dy))
        return uniq

    def _coarse_ring_candidates(
        self,
        r_step: float,
        r_max: float,
        angle_priority_deg: list[int]
    ) -> list[tuple[float, float]]:
        
        radii = np.arange(r_step, r_max + 1e-9, r_step)
        cands: list[tuple[float, float]] = []
        MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
        for r in radii:
            for ang_deg in angle_priority_deg:
                th = np.deg2rad(float(ang_deg))
                dx = float(r * np.cos(th))
                dy = float(r * np.sin(th))
                if np.hypot(dx, dy) <= MAX_BASE_TOTAL_TRANS + 1e-9:
                    cands.append((dx, dy))
        return cands


    def _find_base_assist_strategy1(self, arm, A_cur: np.ndarray) -> tuple[float, float, float] | None:
        """
        Strategy-1 (translation-first):
        A) translation-only (yaw=0) prioritized
        B) translation-only coarse rings
        C) small yaw last (optionally with tiny translations)
        D) final yaw fallback (90/-90/180)
        Returns (dx, dy, yaw_deg) or None.
        NOTE: Feasibility test uses apply_base_se2 with order="tr".
        """
        print("\n[STRAT-1 / A] translation-only (yaw=0) prioritized")
        for dx, dy in self._translation_priority_candidates(A_cur):
            print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
            if self._try_arm(arm, self._apply_base_se2(A_cur, dx, dy, 0.0, order="tr")):
                return dx, dy, 0.0
            
        R_STEP_COARSE = 0.20
        R_MAX = 2.0
        ANGLE_PRIORITY_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180]
        print("\n[STRAT-1 / B] translation-only coarse rings (yaw=0)")
        for dx, dy in self._coarse_ring_candidates(R_STEP_COARSE, R_MAX, ANGLE_PRIORITY_DEG):
            print(f"[TRY] yaw=0.0 dx={dx:+.3f} dy={dy:+.3f}")
            if self._try_arm(arm, self._apply_base_se2(A_cur, dx, dy, 0.0, order="tr")):
                return dx, dy, 0.0
            
        YAW_SMALL_DEG = [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0]
        print("\n[STRAT-1 / C] rotation as last resort (small yaws first)")
        for yaw_deg in YAW_SMALL_DEG:
            yaw_rad = float(np.deg2rad(yaw_deg))

            print(f"\n[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
            if self._try_arm(arm, self._apply_base_se2(A_cur, 0.0, 0.0, yaw_rad, order="tr")):
                return 0.0, 0.0, yaw_deg

            for s in [0.05, 0.10, 0.20]:
                for dx, dy in [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]:
                    print(f"[TRY] yaw={yaw_deg:+.1f} dx={dx:+.3f} dy={dy:+.3f}")
                    if self._try_arm(arm, self._apply_base_se2(A_cur, float(dx), float(dy), yaw_rad, order="tr")):
                        return float(dx), float(dy), yaw_deg

        YAW_FALLBACK_DEG = [90.0, -90.0, 180.0]
        print("\n[STRAT-1 / D] final yaw fallback (90/-90/180)")
        for yaw_deg in YAW_FALLBACK_DEG:
            yaw_rad = float(np.deg2rad(yaw_deg))
            print(f"[TRY-YAW] yaw={yaw_deg:+.1f} (no translation)")
            if self._try_arm(arm, self._apply_base_se2(A_cur, 0.0, 0.0, yaw_rad, order="tr")):
                return 0.0, 0.0, yaw_deg

        return None



    def _find_base_assist_strategy2(self, arm, A_cur: np.ndarray) -> tuple[float, float, float] | None:
        """
        Strategy-2 (fallback): SE(2) scan over yaw + ring translations (your older commented approach).
        Returns (dx, dy, yaw_deg) or None.
        NOTE: Feasibility test uses apply_base_se2 with order="rt" (rotate then translate).
        """
        print("\n[STRAT-2] SE(2) scan over (yaw, r, angle)")
        SE2_R_STEP = 0.05
        SE2_R_MAX = 2.0
        radii = np.arange(SE2_R_STEP, SE2_R_MAX + 1e-9, SE2_R_STEP)
        MAX_BASE_TOTAL_TRANS = 3.0        # max total translation (m)
        SE2_ANGLES = 16
        SE2_YAW_LIST_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180]

        for yaw_deg in SE2_YAW_LIST_DEG:
            yaw_rad = float(np.deg2rad(yaw_deg))

            for r in radii:
                for i in range(SE2_ANGLES):
                    ang = 2.0 * np.pi * (i / SE2_ANGLES)
                    dx = float(r * np.cos(ang))
                    dy = float(r * np.sin(ang))

                    if np.hypot(dx, dy) > MAX_BASE_TOTAL_TRANS + 1e-9:
                        continue

                    print(f"[TRY] yaw={yaw_deg:>4.0f} dx={dx:+.3f} dy={dy:+.3f}")
                    A_try = self._apply_base_se2(A_cur, dx, dy, yaw_rad, order="rt")
                    if self._try_arm(arm, A_try):
                        return dx, dy, float(yaw_deg)

        return None

    def goto_ee_pose_base_with_base_assist(self, side, T_base_ee: np.ndarray,) -> bool:
        """
        Public API:
        - First tries direct arm goto
        - Then Strategy-1 (translation-first, rotation last)
        - If Strategy-1 finds nothing, fallback to Strategy-2 (yaw+translation scan)
        - Executes the found base motion in the correct order for that strategy
        - Retries arm goto

        Returns True if final arm goto succeeds, else False.
        """
        assert self.reachy is not None
        arm = self._get_arm(side)
        A_cur = T_base_ee.copy()

        print("\n[0] Try arm without moving base")
        if self._try_arm(arm, A_cur):
            print("[SUCCESS] Reached without base move.")
            return True

        print("\n[1] Strategy-1: translation-first search")
        best = self._find_base_assist_strategy1(arm, A_cur)

        
        # If strategy-1 fails to find ANY candidate, fall back to strategy-2
        if best is None:
            print("\n[1->2] Strategy-1 found no solution. Falling back to Strategy-2 (yaw+translation scan).")
            best = self._find_base_assist_strategy2(arm, A_cur)
            if best is None:
                print("[FAIL] No feasible base offset found in either strategy.")
                return 1

            dx_goal, dy_goal, yaw_deg_goal = best
            print(f"\n[BASE] (Strategy-2) Best motion: dx={dx_goal:+.3f}, dy={dy_goal:+.3f}, yaw={yaw_deg_goal:+.1f} deg")

            # Execute Strategy-2 order: rotate THEN translate (matches order="rt")
            yaw_abs = self._execute_yaw_in_steps(yaw_deg_goal)
            A_cur = self._apply_base_se2(A_cur, 0.0, 0.0, np.deg2rad(yaw_deg_goal), order="rt")

            trans_abs = self._execute_translation_in_steps(dx_goal, dy_goal)
            A_cur = self._apply_base_se2(A_cur, dx_goal, dy_goal, 0.0, order="rt")

            print(f"[BASE] Executed yaw_abs={yaw_abs:.1f}deg, trans_abs={trans_abs:.3f}m")

        else:
            dx_goal, dy_goal, yaw_deg_goal = best
            print(f"\n[BASE] (Strategy-1) Best motion: dx={dx_goal:+.3f}, dy={dy_goal:+.3f}, yaw={yaw_deg_goal:+.1f} deg")

            # Execute Strategy-1 order: translate THEN rotate (matches order="tr")
            trans_abs = self._execute_translation_in_steps(dx_goal, dy_goal)
            A_cur = self._apply_base_se2(A_cur, dx_goal, dy_goal, 0.0, order="tr")

            yaw_abs = self._execute_yaw_in_steps(yaw_deg_goal)
            A_cur = self._apply_base_se2(A_cur, 0.0, 0.0, np.deg2rad(yaw_deg_goal), order="tr")

            print(f"[BASE] Executed trans_abs={trans_abs:.3f}m, yaw_abs={yaw_abs:.1f}deg")

        print("\n[2] Retrying arm.goto after base assist")
        # ok = self._try_arm(arm, A_cur)
        if self._try_arm(arm, A_cur):
            print("[SUCCESS] Reached after base assist.")
            return True

        print("[FAIL] Feasible in search but failed after base motion (drift/slip/frames).")
        return False
        

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

    def get_depth_intrinsics(self) -> Dict[str, Any]:
        """Return intrinsics for depth camera (RGB or DEPTH view)."""
        # self.connect()
        assert self.reachy is not None

        h, w, distortion_model, D, K, R, P = self.reachy.cameras.depth.get_parameters()
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