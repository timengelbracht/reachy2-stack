#!/usr/bin/env python3
"""Localization server (passthrough mode).

This server receives RobotState from clients and returns an updated
RobotState with T_world_base set. Currently implements passthrough
behavior where T_world_base = T_odom_base.

Usage:
    python ext_server/localization_server.py --port 5556
"""

import argparse
import time
import zmq
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reachy2_stack.base_module.robot_state import RobotState
from reachy2_stack.base_module.localization_loop import LocalizationMessage


class LocalizationServer:
    """Localization server (passthrough mode).

    Receives RobotState, sets T_world_base = T_odom_base, and returns it.
    This is useful for testing the pipeline. The world frame is the same
    as the odometry frame.
    """

    def __init__(self, port: int = 5556):
        """Initialize localization server.

        Args:
            port: ZeroMQ port to listen on
        """
        self.port = port

        # Initialize ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # Reply socket
        self.socket.bind(f"tcp://*:{port}")
        print(f"[LOCALIZATION SERVER] Listening on port {port}")

        # Stats
        self.frame_count = 0
        self.last_print_time = time.time()

    def localize(self, robot_state: RobotState) -> RobotState:
        """Passthrough localization: T_world_base = T_odom_base.

        Args:
            robot_state: Input RobotState with odometry

        Returns:
            Updated RobotState with T_world_base = T_odom_base
        """
        # Compute T_odom_base from odometry
        T_odom_base = robot_state.get_T_odom_base()

        # In passthrough mode, world frame = odom frame
        robot_state.T_world_base = T_odom_base

        return robot_state

    def run(self):
        """Run the server loop."""
        print("[LOCALIZATION SERVER] Ready to receive frames")
        print("[LOCALIZATION SERVER] Mode: passthrough (T_world_base = T_odom_base)")

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
                        T = localized_state.T_world_base
                        if T is not None:
                            pos = T[:3, 3]
                            print(
                                f"[LOCALIZATION SERVER] Frame {self.frame_count}: "
                                f"pos = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
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
    parser = argparse.ArgumentParser(description="Localization server (passthrough)")
    parser.add_argument("--port", type=int, default=5556, help="ZeroMQ port")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LOCALIZATION SERVER (Passthrough)")
    print("=" * 60)
    print("  T_world_base = T_odom_base")
    print("  World frame = Odometry frame")
    print("=" * 60 + "\n")

    server = LocalizationServer(port=args.port)
    server.run()


if __name__ == "__main__":
    main()
