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

                # Cache intrinsics on first successful frame
                if depth_K is None:
                    try:
                        intrinsics = client.get_depth_intrinsics()
                        if intrinsics and "K" in intrinsics:
                            depth_K = np.array(intrinsics["K"])
                            print("[ROBOT_STATE] Depth intrinsics cached")
                    except Exception as e:
                        print(f"[ROBOT_STATE] Could not get intrinsics: {e}")

                state.intrinsics = depth_K

            except Exception as e:
                print(f"[ROBOT_STATE] Depth camera error: {e}")

        # === Teleop cameras (optional, I/O - releases GIL) ===
        if grab_teleop:
            try:
                state.teleop_left = np.asarray(client.get_teleop_rgb_left())
                state.teleop_right = np.asarray(client.get_teleop_rgb_right())
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
