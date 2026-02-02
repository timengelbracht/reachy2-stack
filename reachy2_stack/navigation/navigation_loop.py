"""Navigation loop for path following and execution."""

from __future__ import annotations

import time
from queue import Queue, Empty
from typing import TYPE_CHECKING, Optional, List, Tuple, Union
from multiprocessing import Event as MPEvent
from threading import Event as ThreadEvent

import numpy as np

if TYPE_CHECKING:
    from reachy2_stack.core.client import ReachyClient
    from reachy2_stack.infra.world_model import WorldModel
    from reachy2_stack.base_module.robot_state import RobotState


def navigation_loop(
    state_queue: Queue,
    path_queue: Queue,
    stop_event: Union[ThreadEvent, MPEvent],
    client: "ReachyClient",
    world_model: Optional["WorldModel"] = None,
    control_mode: str = "position",
    distance_tolerance: float = 0.05,
    angle_tolerance: float = 5.0,
    velocity_gain: float = 0.5,
    max_velocity: float = 0.3,
    status_callback: Optional[callable] = None,
) -> None:
    """Navigation loop that follows paths from a planner.

    This loop:
    1. Listens for paths from a planner server (via path_queue)
    2. Executes paths using position or velocity control
    3. Monitors progress and completion

    Args:
        state_queue: Queue of RobotState (for current pose)
        path_queue: Queue of paths from planner. Each path is a list of
                    (x, y, theta_degrees) tuples in odom or world frame.
        stop_event: Event to signal loop termination
        client: ReachyClient instance
        world_model: Optional WorldModel for world-frame navigation
        control_mode: "position" (goto) or "velocity" (continuous control)
        distance_tolerance: Position tolerance in meters
        angle_tolerance: Angle tolerance in degrees
        velocity_gain: Proportional gain for velocity control
        max_velocity: Maximum velocity in m/s for velocity control
        status_callback: Optional callback(status_dict) for progress updates
    """
    from reachy2_stack.control.base import BaseController

    base = BaseController(client, world_model)

    current_path: Optional[List[Tuple[float, float, float]]] = None
    current_waypoint_idx = 0
    executing = False

    def report_status(msg: str, **kwargs):
        """Report status via callback and print."""
        print(f"[NAV] {msg}")
        if status_callback:
            status_callback({"message": msg, **kwargs})

    report_status("Navigation loop started")

    while not stop_event.is_set():
        # === Check for new path (non-blocking) ===
        try:
            new_path = path_queue.get_nowait()
            if new_path is None:
                # None signals "stop current navigation"
                if executing:
                    report_status("Navigation cancelled")
                    client.goto_base_defined_speed(0, 0, 0)
                executing = False
                current_path = None
                continue

            current_path = new_path
            current_waypoint_idx = 0
            executing = True
            report_status(
                f"Received new path with {len(current_path)} waypoints",
                waypoints=len(current_path),
            )
        except Empty:
            pass

        # === If not executing, sleep and continue ===
        if not executing or current_path is None:
            time.sleep(0.05)
            continue

        # === Get current state (drain queue to get latest) ===
        latest_state: Optional["RobotState"] = None
        while True:
            try:
                latest_state = state_queue.get_nowait()
            except Empty:
                break

        if latest_state is None:
            time.sleep(0.05)
            continue

        # === Execute current waypoint ===
        if current_waypoint_idx < len(current_path):
            target_x, target_y, target_theta = current_path[current_waypoint_idx]

            if control_mode == "position":
                # Position control: use goto (blocking per waypoint)
                report_status(
                    f"Going to waypoint {current_waypoint_idx + 1}/{len(current_path)}: "
                    f"({target_x:.2f}, {target_y:.2f}, {target_theta:.1f}°)",
                    waypoint=current_waypoint_idx + 1,
                    total=len(current_path),
                )

                if world_model is not None:
                    base.goto_world(
                        target_x,
                        target_y,
                        target_theta,
                        wait=True,
                        distance_tolerance=distance_tolerance,
                        angle_tolerance=angle_tolerance,
                        degrees=True,
                    )
                else:
                    base.goto_odom(
                        target_x,
                        target_y,
                        target_theta,
                        wait=True,
                        distance_tolerance=distance_tolerance,
                        angle_tolerance=angle_tolerance,
                        degrees=True,
                    )
                current_waypoint_idx += 1

            elif control_mode == "velocity":
                # Velocity control: compute and send velocities
                dx = target_x - latest_state.odom_x
                dy = target_y - latest_state.odom_y
                dist = np.sqrt(dx * dx + dy * dy)

                # Check if we've reached the waypoint
                if dist < distance_tolerance:
                    report_status(
                        f"Reached waypoint {current_waypoint_idx + 1}/{len(current_path)}",
                        waypoint=current_waypoint_idx + 1,
                        total=len(current_path),
                    )
                    current_waypoint_idx += 1
                else:
                    # Proportional control towards target
                    vx = np.clip(dx * velocity_gain, -max_velocity, max_velocity)
                    vy = np.clip(dy * velocity_gain, -max_velocity, max_velocity)

                    # Angular velocity towards target heading
                    dtheta = target_theta - latest_state.odom_theta
                    # Normalize to [-180, 180]
                    while dtheta > 180:
                        dtheta -= 360
                    while dtheta < -180:
                        dtheta += 360
                    vtheta = np.clip(dtheta * velocity_gain, -90, 90)

                    client.goto_base_defined_speed(vx, vy, vtheta)
                    time.sleep(0.1)  # Control loop rate

        else:
            # === Path complete ===
            report_status(
                "Path complete!",
                waypoint=len(current_path),
                total=len(current_path),
                complete=True,
            )
            executing = False
            current_path = None
            client.goto_base_defined_speed(0, 0, 0)  # Stop

    # Cleanup
    client.goto_base_defined_speed(0, 0, 0)
    report_status("Navigation loop stopped")
