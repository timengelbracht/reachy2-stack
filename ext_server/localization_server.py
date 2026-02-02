#!/usr/bin/env python3
"""Localization server with passthrough and noisy modes.

This server receives RobotState from clients and returns an updated
RobotState with T_world_cam set.

Modes:
- passthrough: T_world_cam = T_odom_cam (no noise)
- noisy: T_world_cam = T_odom_cam + noise (Gaussian perturbation)

The client applies offset correction using:
    T_world_odom = T_world_cam @ inv(T_odom_cam)
    T_world_base = T_world_odom @ T_odom_base

Usage:
    # Passthrough mode (default)
    python ext_server/localization_server.py --port 5556

    # Noisy mode with default noise
    python ext_server/localization_server.py --port 5556 --mode noisy

    # Noisy mode with custom noise levels
    python ext_server/localization_server.py --port 5556 --mode noisy --pos-noise 0.05 --rot-noise 2.0
"""

import argparse
import time
import zmq
import sys
import os

import numpy as np
from scipy.spatial.transform import Rotation

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reachy2_stack.base_module.robot_state import RobotState
from reachy2_stack.base_module.localization_loop import LocalizationMessage


def add_noise_to_transform(
    T: np.ndarray,
    position_noise_std: float,
    rotation_noise_std_deg: float,
) -> np.ndarray:
    """Add Gaussian noise to an SE(3) transform.

    Args:
        T: 4x4 transformation matrix
        position_noise_std: Standard deviation of position noise (meters)
        rotation_noise_std_deg: Standard deviation of rotation noise (degrees)

    Returns:
        Noisy 4x4 transformation matrix
    """
    T_noisy = T.copy()

    # Add position noise (Gaussian in x, y, z)
    if position_noise_std > 0:
        pos_noise = np.random.normal(0, position_noise_std, size=3)
        T_noisy[:3, 3] += pos_noise

    # Add rotation noise (small random rotation)
    if rotation_noise_std_deg > 0:
        # Generate random axis
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        # Generate random angle (degrees -> radians)
        angle_deg = np.random.normal(0, rotation_noise_std_deg)
        angle_rad = np.deg2rad(angle_deg)

        # Create rotation from axis-angle using scipy
        rotvec = axis * angle_rad
        R_noise = Rotation.from_rotvec(rotvec).as_matrix()

        # Apply noise rotation: R_noisy = R_noise @ R_original
        R_original = T_noisy[:3, :3]
        T_noisy[:3, :3] = R_noise @ R_original

    return T_noisy


class LocalizationServer:
    """Localization server with passthrough and noisy modes.

    Receives RobotState and returns T_world_cam:
    - passthrough: T_world_cam = T_odom_cam
    - noisy: T_world_cam = T_odom_cam + Gaussian noise

    The server returns T_world_cam (camera pose in world frame) because
    visual localization estimates the camera pose, not the robot base pose.
    """

    def __init__(
        self,
        port: int = 5556,
        mode: str = "passthrough",
        position_noise_std: float = 0.02,
        rotation_noise_std_deg: float = 1.0,
    ):
        """Initialize localization server.

        Args:
            port: ZeroMQ port to listen on
            mode: "passthrough" or "noisy"
            position_noise_std: Standard deviation of position noise in meters
                (only used in noisy mode)
            rotation_noise_std_deg: Standard deviation of rotation noise in degrees
                (only used in noisy mode)
        """
        self.port = port
        self.mode = mode
        self.position_noise_std = position_noise_std
        self.rotation_noise_std_deg = rotation_noise_std_deg

        # Initialize ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # Reply socket
        self.socket.bind(f"tcp://*:{port}")
        print(f"[LOCALIZATION SERVER] Listening on port {port}")

        # Stats
        self.frame_count = 0
        self.last_print_time = time.time()

    def localize_passthrough(self, robot_state: RobotState) -> RobotState:
        """Passthrough localization: T_world_cam = T_odom_cam.

        Args:
            robot_state: Input RobotState with odometry and camera extrinsics

        Returns:
            Updated RobotState with T_world_cam = T_odom_cam
        """
        # Compute T_odom_cam from odometry and camera extrinsics
        T_odom_cam = robot_state.get_T_odom_depth_cam()

        # In passthrough mode, world frame = odom frame
        robot_state.T_world_cam = T_odom_cam

        # Also compute T_world_base for reference
        T_odom_base = robot_state.get_T_odom_base()
        robot_state.T_world_base = T_odom_base

        return robot_state

    def localize_noisy(self, robot_state: RobotState) -> RobotState:
        """Noisy passthrough: T_world_cam = T_odom_cam + noise.

        Adds Gaussian noise to both position and orientation to simulate
        real visual localization uncertainty.

        Args:
            robot_state: Input RobotState with odometry and camera extrinsics

        Returns:
            Updated RobotState with noisy T_world_cam
        """
        # Compute T_odom_cam from odometry and camera extrinsics
        T_odom_cam = robot_state.get_T_odom_depth_cam()

        if T_odom_cam is not None:
            # Add noise to the transform
            T_world_cam_noisy = add_noise_to_transform(
                T_odom_cam,
                self.position_noise_std,
                self.rotation_noise_std_deg,
            )
            robot_state.T_world_cam = T_world_cam_noisy
        else:
            robot_state.T_world_cam = None

        # Also compute T_world_base with noise for reference
        T_odom_base = robot_state.get_T_odom_base()
        if T_odom_base is not None:
            T_world_base_noisy = add_noise_to_transform(
                T_odom_base,
                self.position_noise_std,
                self.rotation_noise_std_deg,
            )
            robot_state.T_world_base = T_world_base_noisy
        else:
            robot_state.T_world_base = None

        return robot_state

    def localize(self, robot_state: RobotState) -> RobotState:
        """Perform localization based on current mode.

        Args:
            robot_state: Input RobotState

        Returns:
            Updated RobotState with T_world_cam set
        """
        if self.mode == "noisy":
            return self.localize_noisy(robot_state)
        else:
            return self.localize_passthrough(robot_state)

    def run(self):
        """Run the server loop."""
        print("[LOCALIZATION SERVER] Ready to receive frames")
        if self.mode == "noisy":
            print(
                f"[LOCALIZATION SERVER] Mode: noisy "
                f"(pos_std={self.position_noise_std:.3f}m, "
                f"rot_std={self.rotation_noise_std_deg:.1f}deg)"
            )
        else:
            print("[LOCALIZATION SERVER] Mode: passthrough (T_world_cam = T_odom_cam)")

        try:
            while True:
                # Receive request
                try:
                    message = self.socket.recv()
                except zmq.ZMQError as e:
                    print(f"[LOCALIZATION SERVER] ZMQ error: {e}")
                    continue

                try:
                    # Unpack request
                    robot_state = LocalizationMessage.unpack_request(message, compressed=True)

                    # Perform localization
                    localized_state = self.localize(robot_state)

                    # Pack response
                    response = LocalizationMessage.pack_response(
                        robot_state=localized_state,
                        success=True,
                        compress=True,
                    )

                    self.frame_count += 1

                    # Print stats every 20 frames
                    if self.frame_count % 20 == 0:
                        elapsed = time.time() - self.last_print_time
                        fps = 20.0 / elapsed
                        T = localized_state.T_world_cam
                        if T is not None:
                            pos = T[:3, 3]
                            print(
                                f"[LOCALIZATION SERVER] Frame {self.frame_count}: "
                                f"T_world_cam pos = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                                f"{fps:.1f} FPS"
                            )
                        self.last_print_time = time.time()

                except Exception as e:
                    print(f"[LOCALIZATION SERVER] Error processing frame: {e}")
                    import traceback
                    traceback.print_exc()

                    # Create a dummy RobotState for error response
                    error_state = RobotState(timestamp=time.time())

                    # Send error response
                    response = LocalizationMessage.pack_response(
                        robot_state=error_state,
                        success=False,
                        error_msg=str(e),
                        compress=True,
                    )

                # Send response
                self.socket.send(response)

        except KeyboardInterrupt:
            print("\n[LOCALIZATION SERVER] Shutting down...")
        finally:
            self.socket.close()
            self.context.term()


def main():
    parser = argparse.ArgumentParser(
        description="Localization server with passthrough and noisy modes"
    )
    parser.add_argument("--port", type=int, default=5556, help="ZeroMQ port")
    parser.add_argument(
        "--mode",
        choices=["passthrough", "noisy"],
        default="passthrough",
        help="Localization mode: 'passthrough' (exact) or 'noisy' (with Gaussian noise)",
    )
    parser.add_argument(
        "--pos-noise",
        type=float,
        default=0.02,
        help="Position noise standard deviation in meters (noisy mode only)",
    )
    parser.add_argument(
        "--rot-noise",
        type=float,
        default=1.0,
        help="Rotation noise standard deviation in degrees (noisy mode only)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LOCALIZATION SERVER")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    if args.mode == "noisy":
        print(f"  Position noise std: {args.pos_noise:.3f} m")
        print(f"  Rotation noise std: {args.rot_noise:.1f} deg")
    else:
        print("  T_world_cam = T_odom_cam (camera pose)")
        print("  World frame = Odometry frame")
    print("=" * 60 + "\n")

    server = LocalizationServer(
        port=args.port,
        mode=args.mode,
        position_noise_std=args.pos_noise,
        rotation_noise_std_deg=args.rot_noise,
    )
    server.run()


if __name__ == "__main__":
    main()
