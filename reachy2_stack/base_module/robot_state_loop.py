"""Unified robot state acquisition loop (camera + odometry)."""

from __future__ import annotations

import time
from queue import Queue, Full
from typing import TYPE_CHECKING, Optional, Union
from multiprocessing import Event as MPEvent
from threading import Event as ThreadEvent

import numpy as np

from .robot_state import RobotState

if TYPE_CHECKING:
    from reachy2_stack.core.client import ReachyClient


def robot_state_loop(
    client: "ReachyClient",
    state_queue: Queue,
    stop_event: Union[ThreadEvent, MPEvent],
    hz: int = 20,
    grab_depth: bool = True,
    grab_teleop: bool = False,
) -> None:
    """Unified sensor acquisition loop.

    Grabs camera + odometry together and queues RobotState.
    This merges the functionality of camera_loop and odometry_loop
    into a single coherent state producer.

    Args:
        client: ReachyClient instance
        state_queue: Queue to put RobotState objects
        stop_event: Event to signal loop termination
        hz: Acquisition rate in Hz
        grab_depth: Whether to grab depth camera frames
        grab_teleop: Whether to grab teleop camera frames
    """
    reachy = client.connect_reachy
    interval = 1.0 / hz

    # Cache intrinsics (only query once)
    depth_K: Optional[np.ndarray] = None
    teleop_left_K: Optional[np.ndarray] = None
    teleop_right_K: Optional[np.ndarray] = None

    # Cache extrinsics (only query once, they don't change)
    T_base_depth: Optional[np.ndarray] = None
    T_base_teleop_left: Optional[np.ndarray] = None
    T_base_teleop_right: Optional[np.ndarray] = None

    print(f"[ROBOT_STATE] Starting unified state loop at {hz} Hz")
    print(f"[ROBOT_STATE] Depth camera: {grab_depth}, Teleop cameras: {grab_teleop}")

    while not stop_event.is_set():
        t0 = time.time()

        # Build state
        state = RobotState(timestamp=time.time())

        # === Odometry (I/O - releases GIL) ===
        try:
            odom = client.get_mobile_odometry()
            state.odom_x = odom.get("x", 0.0)
            state.odom_y = odom.get("y", 0.0)
            state.odom_theta = odom.get("theta", 0.0)
        except Exception as e:
            print(f"[ROBOT_STATE] Odometry error: {e}")

        # === Depth camera (I/O - releases GIL) ===
        if grab_depth:
            try:
                rgb_frame, _ = reachy.cameras.depth.get_frame()
                depth_frame, _ = reachy.cameras.depth.get_depth_frame()

                state.rgb = np.asarray(rgb_frame)
                state.depth = np.asarray(depth_frame)

                # Cache depth intrinsics on first successful frame
                if depth_K is None:
                    try:
                        intrinsics = client.get_depth_intrinsics()
                        if intrinsics and "K" in intrinsics:
                            depth_K = np.array(intrinsics["K"])
                            print("[ROBOT_STATE] Depth intrinsics cached")
                    except Exception as e:
                        print(f"[ROBOT_STATE] Could not get depth intrinsics: {e}")

                # Cache depth extrinsics on first successful frame
                if T_base_depth is None:
                    try:
                        T_base_depth = client.get_depth_extrinsics()
                        print("[ROBOT_STATE] Depth extrinsics cached")
                    except Exception as e:
                        print(f"[ROBOT_STATE] Could not get depth extrinsics: {e}")

                state.depth_intrinsics = depth_K
                state.depth_extrinsics = T_base_depth

            except Exception as e:
                print(f"[ROBOT_STATE] Depth camera error: {e}")

        # === Teleop cameras (optional, I/O - releases GIL) ===
        if grab_teleop:
            try:
                state.teleop_left = np.asarray(client.get_teleop_rgb_left())
                state.teleop_right = np.asarray(client.get_teleop_rgb_right())

                # Cache teleop left intrinsics
                if teleop_left_K is None:
                    try:
                        intrinsics = client.get_teleop_intrinsics_left()
                        if intrinsics and "K" in intrinsics:
                            teleop_left_K = np.array(intrinsics["K"])
                            print("[ROBOT_STATE] Teleop left intrinsics cached")
                    except Exception as e:
                        print(f"[ROBOT_STATE] Could not get teleop left intrinsics: {e}")

                # Cache teleop right intrinsics
                if teleop_right_K is None:
                    try:
                        intrinsics = client.get_teleop_intrinsics_right()
                        if intrinsics and "K" in intrinsics:
                            teleop_right_K = np.array(intrinsics["K"])
                            print("[ROBOT_STATE] Teleop right intrinsics cached")
                    except Exception as e:
                        print(f"[ROBOT_STATE] Could not get teleop right intrinsics: {e}")

                # Cache teleop left extrinsics
                if T_base_teleop_left is None:
                    try:
                        T_base_teleop_left = client.get_teleop_extrinsics_left()
                        print("[ROBOT_STATE] Teleop left extrinsics cached")
                    except Exception as e:
                        print(f"[ROBOT_STATE] Could not get teleop left extrinsics: {e}")

                # Cache teleop right extrinsics
                if T_base_teleop_right is None:
                    try:
                        T_base_teleop_right = client.get_teleop_extrinsics_right()
                        print("[ROBOT_STATE] Teleop right extrinsics cached")
                    except Exception as e:
                        print(f"[ROBOT_STATE] Could not get teleop right extrinsics: {e}")

                state.teleop_left_intrinsics = teleop_left_K
                state.teleop_left_extrinsics = T_base_teleop_left
                state.teleop_right_intrinsics = teleop_right_K
                state.teleop_right_extrinsics = T_base_teleop_right

            except Exception as e:
                print(f"[ROBOT_STATE] Teleop camera error: {e}")

        # === Queue state (drop if full to avoid blocking) ===
        try:
            state_queue.put_nowait(state)
        except Full:
            # Queue full, drop this state (consumer is slow)
            pass

        # Rate limit
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    print("[ROBOT_STATE] Unified state loop stopped")
