"""Localization loop that communicates with localization server."""

from __future__ import annotations

import time
import zlib
from queue import Queue, Empty, Full
from typing import Optional, Dict, Any, Union, Tuple, TYPE_CHECKING
from threading import Event as ThreadEvent
from multiprocessing import Event as MPEvent

import numpy as np
import msgpack
import msgpack_numpy as m

from .robot_state import RobotState

if TYPE_CHECKING:
    from .camera_pose_buffer import CameraPoseBuffer
    from .localization_fusion import LocalizationFusion

# Enable msgpack-numpy for efficient numpy array serialization
m.patch()


class LocalizationMessage:
    """Message format for localization client-server communication.

    Serializes RobotState for network transmission and deserializes responses.
    """

    @staticmethod
    def pack_request(
        robot_state: RobotState,
        include_images: bool = True,
        compress: bool = True,
    ) -> bytes:
        """Pack RobotState into a message for the localization server.

        Args:
            robot_state: RobotState object to serialize
            include_images: Whether to include RGB/depth images (needed for visual localization)
            compress: Use zlib compression (recommended)

        Returns:
            Serialized message bytes
        """
        msg = {
            "timestamp": robot_state.timestamp,
            "odom_x": robot_state.odom_x,
            "odom_y": robot_state.odom_y,
            "odom_theta": robot_state.odom_theta,
            # Calibration data
            "depth_intrinsics": robot_state.depth_intrinsics,
            "depth_extrinsics": robot_state.depth_extrinsics,
            "teleop_left_intrinsics": robot_state.teleop_left_intrinsics,
            "teleop_left_extrinsics": robot_state.teleop_left_extrinsics,
            "teleop_right_intrinsics": robot_state.teleop_right_intrinsics,
            "teleop_right_extrinsics": robot_state.teleop_right_extrinsics,
            # Existing world pose (if any)
            "T_world_base": robot_state.T_world_base,
        }

        if include_images:
            msg["rgb"] = robot_state.rgb
            msg["depth"] = robot_state.depth
            msg["teleop_left"] = robot_state.teleop_left
            msg["teleop_right"] = robot_state.teleop_right

        packed = msgpack.packb(msg, use_bin_type=True)

        if compress:
            packed = zlib.compress(packed, level=1)  # Fast compression

        return packed

    @staticmethod
    def unpack_request(data: bytes, compressed: bool = True) -> RobotState:
        """Unpack localization request message (used by server).

        Args:
            data: Serialized message bytes
            compressed: Whether data is compressed

        Returns:
            RobotState object reconstructed from message
        """
        if compressed:
            data = zlib.decompress(data)

        msg = msgpack.unpackb(data, raw=False)

        # Reconstruct RobotState
        state = RobotState(
            timestamp=msg["timestamp"],
            odom_x=msg["odom_x"],
            odom_y=msg["odom_y"],
            odom_theta=msg["odom_theta"],
            # Images (may be None if not included)
            rgb=msg.get("rgb"),
            depth=msg.get("depth"),
            teleop_left=msg.get("teleop_left"),
            teleop_right=msg.get("teleop_right"),
            # Calibration
            depth_intrinsics=msg.get("depth_intrinsics"),
            depth_extrinsics=msg.get("depth_extrinsics"),
            teleop_left_intrinsics=msg.get("teleop_left_intrinsics"),
            teleop_left_extrinsics=msg.get("teleop_left_extrinsics"),
            teleop_right_intrinsics=msg.get("teleop_right_intrinsics"),
            teleop_right_extrinsics=msg.get("teleop_right_extrinsics"),
            # Existing world pose
            T_world_base=msg.get("T_world_base"),
        )

        return state

    @staticmethod
    def pack_response(
        robot_state: RobotState,
        success: bool = True,
        error_msg: str = "",
        compress: bool = True,
    ) -> bytes:
        """Pack localization response (used by server).

        Args:
            robot_state: Updated RobotState with T_world_base set
            success: Whether localization succeeded
            error_msg: Error message if failed
            compress: Use zlib compression

        Returns:
            Serialized message bytes
        """
        msg = {
            "success": success,
            "error_msg": error_msg,
            "timestamp": robot_state.timestamp,
            "odom_x": robot_state.odom_x,
            "odom_y": robot_state.odom_y,
            "odom_theta": robot_state.odom_theta,
            # The key outputs: world frame poses
            "T_world_base": robot_state.T_world_base,
            "T_world_cam": robot_state.T_world_cam,  # Camera pose from localization
            # Forward calibration data
            "depth_intrinsics": robot_state.depth_intrinsics,
            "depth_extrinsics": robot_state.depth_extrinsics,
            "teleop_left_intrinsics": robot_state.teleop_left_intrinsics,
            "teleop_left_extrinsics": robot_state.teleop_left_extrinsics,
            "teleop_right_intrinsics": robot_state.teleop_right_intrinsics,
            "teleop_right_extrinsics": robot_state.teleop_right_extrinsics,
            # Images (optional, may not need to send back)
            "rgb": robot_state.rgb,
            "depth": robot_state.depth,
            "teleop_left": robot_state.teleop_left,
            "teleop_right": robot_state.teleop_right,
        }

        packed = msgpack.packb(msg, use_bin_type=True)

        if compress:
            packed = zlib.compress(packed, level=1)

        return packed

    @staticmethod
    def unpack_response(data: bytes, compressed: bool = True) -> Tuple[Optional[RobotState], bool, str]:
        """Unpack localization response message (used by client).

        Args:
            data: Serialized message bytes
            compressed: Whether data is compressed

        Returns:
            Tuple of (RobotState or None, success, error_msg)
        """
        if compressed:
            data = zlib.decompress(data)

        msg = msgpack.unpackb(data, raw=False)

        success = msg.get("success", False)
        error_msg = msg.get("error_msg", "")

        if not success:
            return None, False, error_msg

        # Reconstruct RobotState
        state = RobotState(
            timestamp=msg["timestamp"],
            odom_x=msg["odom_x"],
            odom_y=msg["odom_y"],
            odom_theta=msg["odom_theta"],
            # Images
            rgb=msg.get("rgb"),
            depth=msg.get("depth"),
            teleop_left=msg.get("teleop_left"),
            teleop_right=msg.get("teleop_right"),
            # Calibration
            depth_intrinsics=msg.get("depth_intrinsics"),
            depth_extrinsics=msg.get("depth_extrinsics"),
            teleop_left_intrinsics=msg.get("teleop_left_intrinsics"),
            teleop_left_extrinsics=msg.get("teleop_left_extrinsics"),
            teleop_right_intrinsics=msg.get("teleop_right_intrinsics"),
            teleop_right_extrinsics=msg.get("teleop_right_extrinsics"),
            # The key outputs: world frame poses
            T_world_base=msg.get("T_world_base"),
            T_world_cam=msg.get("T_world_cam"),  # Camera pose from localization
        )

        return state, True, ""


def localization_loop(
    state_queue: Queue,
    localized_queue: Queue,
    stop_event: Union[ThreadEvent, MPEvent],
    server_host: str = "localhost",
    server_port: int = 5556,
    hz: float = 10.0,
    timeout_ms: int = 5000,
    include_images: bool = True,
    camera_buffer: Optional["CameraPoseBuffer"] = None,
    fusion: Optional["LocalizationFusion"] = None,
    use_offset_correction: bool = True,
) -> None:
    """Localization loop that communicates with localization server.

    This loop:
    1. Reads RobotState from state_queue
    2. Sends to localization server via ZeroMQ
    3. Receives localized RobotState with T_world_base
    4. Applies offset correction (if enabled) for delayed measurements
    5. Outputs to localized_queue

    Offset Correction:
    Visual localization returns poses for images captured in the past. Without
    correction, this causes pose jumps. When use_offset_correction=True:
    - Look up camera pose at keyframe time from camera_buffer
    - Compute world-odom offset: T_world_odom = T_world_cam @ inv(T_odom_cam)
    - Apply offset to current odometry for smooth pose output

    GIL behavior:
    - socket.send() and socket.recv() RELEASE the GIL (ZeroMQ C extension I/O)
    - Other threads can continue while waiting for server response
    - LocalizationMessage.pack_request() holds GIL (CPU-bound msgpack + zlib)

    Args:
        state_queue: Input queue of RobotState objects
        localized_queue: Output queue for localized RobotState objects
        stop_event: Event to signal loop termination
        server_host: Localization server hostname or IP
        server_port: Localization server port
        hz: Processing rate in Hz
        timeout_ms: ZeroMQ receive/send timeout in milliseconds
        include_images: Whether to send images to server (needed for visual localization)
        camera_buffer: Optional buffer for looking up camera poses at past timestamps.
            Required when use_offset_correction=True.
        fusion: Optional LocalizationFusion instance for offset correction.
            Required when use_offset_correction=True.
        use_offset_correction: Whether to apply offset correction for delayed
            localization results. If True, requires camera_buffer and fusion.
    """
    import zmq

    interval = 1.0 / hz

    print(f"[LOCALIZATION] Starting localization loop at {hz} Hz")
    print(f"[LOCALIZATION] Server: {server_host}:{server_port}")
    print(f"[LOCALIZATION] Include images: {include_images}")

    # Initialize ZeroMQ client
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # Request socket
    socket.connect(f"tcp://{server_host}:{server_port}")
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

    def reset_socket():
        """Reset socket after timeout or error."""
        nonlocal socket
        try:
            socket.close()
        except Exception:
            pass
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{server_host}:{server_port}")
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

    # Wait for server connection
    server_connected = False
    retry_interval = 2.0

    print(f"[LOCALIZATION] Waiting for server connection...")

    while not stop_event.is_set() and not server_connected:
        # Drain queue to get latest state
        latest_state: Optional[RobotState] = None
        while True:
            try:
                latest_state = state_queue.get_nowait()
            except Empty:
                break

        if latest_state is None:
            time.sleep(0.5)
            continue

        state = latest_state

        try:
            # Pack and send test request
            request = LocalizationMessage.pack_request(
                robot_state=state,
                include_images=include_images,
                compress=True,
            )

            # === I/O - releases GIL ===
            socket.send(request)
            response_data = socket.recv()
            # === End I/O ===

            localized_state, success, error_msg = LocalizationMessage.unpack_response(
                response_data, compressed=True
            )

            if success:
                server_connected = True
                print(f"[LOCALIZATION] Connected to server successfully!")
            else:
                print(f"[LOCALIZATION] Server error: {error_msg}, retrying...")
                time.sleep(retry_interval)

        except zmq.Again:
            print(f"[LOCALIZATION] Server timeout, retrying in {retry_interval}s...")
            reset_socket()
            time.sleep(retry_interval)

        except Exception as e:
            print(f"[LOCALIZATION] Connection error: {e}, retrying...")
            reset_socket()
            time.sleep(retry_interval)

    if not server_connected:
        print("[LOCALIZATION] Stopped before server connection established")
        socket.close()
        context.term()
        return

    print(f"[LOCALIZATION] Sending frames at {hz} Hz")

    frame_count = 0
    error_count = 0
    max_consecutive_errors = 5

    while not stop_event.is_set():
        t0 = time.time()

        # Drain queue to get latest state
        latest_state = None
        while True:
            try:
                latest_state = state_queue.get_nowait()
            except Empty:
                break

        if latest_state is None:
            time.sleep(0.1)
            continue

        state = latest_state
        frame_count += 1

        try:
            # Pack request (CPU-bound, holds GIL)
            request = LocalizationMessage.pack_request(
                robot_state=state,
                include_images=include_images,
                compress=True,
            )

            # === I/O - releases GIL ===
            socket.send(request)
            response_data = socket.recv()
            # === End I/O ===

            localized_state, success, error_msg = LocalizationMessage.unpack_response(
                response_data, compressed=True
            )

            if success and localized_state is not None:
                output_state = localized_state

                # Apply offset correction if enabled
                # Server returns T_world_cam directly (camera pose in world frame)
                if (
                    use_offset_correction
                    and camera_buffer is not None
                    and fusion is not None
                    and localized_state.T_world_cam is not None
                ):
                    from .localization_fusion import LocalizationResult

                    # Create localization result using T_world_cam directly
                    result = LocalizationResult(
                        timestamp_keyframe=localized_state.timestamp,
                        timestamp_received=time.time(),
                        T_world_cam_keyframe=localized_state.T_world_cam,
                    )

                    fusion_updated = fusion.update_from_localization(result)

                    # Get current camera pose from buffer and apply offset
                    current_pose = camera_buffer.get_latest()
                    if current_pose is not None and fusion.is_initialized:
                        T_world_base_now = fusion.get_T_world_base(
                            current_pose.T_odom_base
                        )

                        # Create output state with CURRENT timestamp and corrected pose
                        output_state = RobotState(
                            timestamp=current_pose.timestamp,
                            odom_x=localized_state.odom_x,
                            odom_y=localized_state.odom_y,
                            odom_theta=localized_state.odom_theta,
                            T_world_base=T_world_base_now,
                            # Forward calibration data
                            depth_intrinsics=localized_state.depth_intrinsics,
                            depth_extrinsics=localized_state.depth_extrinsics,
                            teleop_left_intrinsics=localized_state.teleop_left_intrinsics,
                            teleop_left_extrinsics=localized_state.teleop_left_extrinsics,
                            teleop_right_intrinsics=localized_state.teleop_right_intrinsics,
                            teleop_right_extrinsics=localized_state.teleop_right_extrinsics,
                        )

                        if frame_count % 20 == 0 and fusion_updated:
                            latency = result.latency
                            stats = fusion.get_stats()
                            print(
                                f"[LOCALIZATION] Offset correction applied: "
                                f"latency={latency:.2f}s, updates={stats.update_count}"
                            )

                # Output to queue
                try:
                    localized_queue.put_nowait(output_state)
                except Full:
                    pass  # Drop if queue full

                # Reset error count on success
                error_count = 0

                if frame_count % 20 == 0:
                    T = output_state.T_world_base
                    if T is not None:
                        pos = T[:3, 3]
                        print(
                            f"[LOCALIZATION] Frame {frame_count}: "
                            f"T_world_base pos = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                        )
                    else:
                        print(f"[LOCALIZATION] Frame {frame_count}: T_world_base = None")
            else:
                print(f"[LOCALIZATION] Server error: {error_msg}")
                error_count += 1

        except zmq.Again:
            print(f"[LOCALIZATION] Timeout waiting for server response")
            error_count += 1
            reset_socket()

        except Exception as e:
            print(f"[LOCALIZATION] Error: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            reset_socket()

        # Check for too many consecutive errors
        if error_count >= max_consecutive_errors:
            print(f"[LOCALIZATION] Too many consecutive errors ({error_count}), stopping")
            break

        # Rate limit
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    # Cleanup
    socket.close()
    context.term()
    print("[LOCALIZATION] Localization loop stopped")
