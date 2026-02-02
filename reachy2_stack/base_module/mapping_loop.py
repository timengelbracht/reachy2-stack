"""Mapping loop that communicates with wavemap server."""

from __future__ import annotations

import time
import zlib
from queue import Queue, Empty, Full
from typing import Optional, Dict, Any, Union
from threading import Event as ThreadEvent
from multiprocessing import Event as MPEvent

import numpy as np
import msgpack
import msgpack_numpy as m

from .robot_state import RobotState
from .map_state import MapState

# Enable msgpack-numpy for efficient numpy array serialization
m.patch()


class MappingMessage:
    """Message format for mapping client-server communication."""

    @staticmethod
    def pack_request(
        rgb: np.ndarray,
        depth: np.ndarray,
        T_world_cam: np.ndarray,
        timestamp: float,
        intrinsics: Optional[Dict[str, Any]] = None,
        compress: bool = True,
    ) -> bytes:
        """Pack RGB, depth, camera pose, and intrinsics into a message.

        Args:
            rgb: RGB image (H, W, 3) as uint8
            depth: Depth image (H, W) as float32 (in meters)
            T_world_cam: 4x4 camera pose in world/odom frame
            timestamp: Frame timestamp
            intrinsics: Camera intrinsics dict with 'K', 'width', 'height'
            compress: Use zlib compression (recommended)

        Returns:
            Serialized message bytes
        """
        msg = {
            "rgb": rgb,
            "depth": depth,
            "T_world_cam": T_world_cam,
            "timestamp": timestamp,
            "rgb_shape": rgb.shape,
            "depth_shape": depth.shape,
            "intrinsics": intrinsics,
        }

        packed = msgpack.packb(msg, use_bin_type=True)

        if compress:
            packed = zlib.compress(packed, level=1)  # Fast compression

        return packed

    @staticmethod
    def unpack_request(data: bytes, compressed: bool = True) -> Dict[str, Any]:
        """Unpack mapping request message (used by server).

        Args:
            data: Serialized message bytes
            compressed: Whether data is compressed

        Returns:
            Dictionary with 'rgb', 'depth', 'T_world_cam', 'timestamp', 'intrinsics'
        """
        if compressed:
            data = zlib.decompress(data)

        msg = msgpack.unpackb(data, raw=False)
        return msg

    @staticmethod
    def pack_response(
        occupied_points: Optional[np.ndarray] = None,
        free_points: Optional[np.ndarray] = None,
        occupied_colors: Optional[np.ndarray] = None,
        success: bool = True,
        error_msg: str = "",
    ) -> bytes:
        """Pack mapping result (used by server).

        Args:
            occupied_points: Occupied voxel points (N, 3) as float32
            free_points: Free voxel points (M, 3) as float32
            occupied_colors: Colors for occupied points (N, 3) as uint8
            success: Whether mapping succeeded
            error_msg: Error message if failed

        Returns:
            Serialized message bytes
        """
        msg = {
            "success": success,
            "error_msg": error_msg,
            "occupied_points": occupied_points if occupied_points is not None else np.zeros((0, 3), dtype=np.float32),
            "free_points": free_points if free_points is not None else np.zeros((0, 3), dtype=np.float32),
            "occupied_colors": occupied_colors if occupied_colors is not None else np.zeros((0, 3), dtype=np.uint8),
        }

        packed = msgpack.packb(msg, use_bin_type=True)
        return packed

    @staticmethod
    def unpack_response(data: bytes) -> Dict[str, Any]:
        """Unpack mapping response message (used by client).

        Args:
            data: Serialized message bytes

        Returns:
            Dictionary with 'success', 'occupied_points', 'free_points', 'occupied_colors'
        """
        msg = msgpack.unpackb(data, raw=False)
        return msg


def mapping_loop(
    state_queue: Queue,
    map_queue: Queue,
    stop_event: Union[ThreadEvent, MPEvent],
    server_host: str = "localhost",
    server_port: int = 5555,
    hz: float = 2.0,
    timeout_ms: int = 5000,
    depth_scale: float = 0.001,
) -> None:
    """Mapping loop that communicates with wavemap server.

    This loop:
    1. Reads RobotState from state_queue
    2. Sends RGBD + camera pose to wavemap server via ZeroMQ
    3. Outputs MapState to map_queue

    GIL behavior:
    - socket.send() and socket.recv() RELEASE the GIL (ZeroMQ C extension I/O)
    - Other threads can continue while waiting for server response
    - MappingMessage.pack_request() holds GIL (CPU-bound msgpack + zlib)

    Args:
        state_queue: Input queue of RobotState objects
        map_queue: Output queue for MapState objects
        stop_event: Event to signal loop termination
        server_host: Wavemap server hostname or IP
        server_port: Wavemap server port
        hz: Processing rate in Hz
        timeout_ms: ZeroMQ receive/send timeout in milliseconds
        depth_scale: Scale factor to convert depth to meters (e.g., 0.001 for mm)
    """
    import zmq

    interval = 1.0 / hz

    print(f"[MAPPING] Starting mapping loop at {hz} Hz")
    print(f"[MAPPING] Server: {server_host}:{server_port}")
    print(f"[MAPPING] Depth scale: {depth_scale}")

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

    print(f"[MAPPING] Waiting for server connection...")

    while not stop_event.is_set() and not server_connected:
        # Drain queue to get latest state
        latest_state: Optional[RobotState] = None
        while True:
            try:
                latest_state = state_queue.get_nowait()
            except Empty:
                break

        if latest_state is None or not latest_state.has_depth():
            time.sleep(0.5)
            continue

        state = latest_state

        try:
            # Get camera pose in odom frame
            T_odom_cam = state.get_T_odom_depth_cam()
            if T_odom_cam is None:
                print("[MAPPING] Waiting for camera extrinsics...")
                time.sleep(0.5)
                continue

            # Prepare intrinsics
            intrinsics = None
            if state.depth_intrinsics is not None:
                intrinsics = {
                    "K": state.depth_intrinsics,
                    "width": state.rgb.shape[1],
                    "height": state.rgb.shape[0],
                }

            # Scale depth to meters
            depth_meters = state.depth.astype(np.float32) * depth_scale

            # Pack and send test request
            request = MappingMessage.pack_request(
                rgb=state.rgb,
                depth=depth_meters,
                T_world_cam=T_odom_cam,
                timestamp=state.timestamp,
                intrinsics=intrinsics,
                compress=True,
            )

            # === I/O - releases GIL ===
            socket.send(request)
            response_data = socket.recv()
            # === End I/O ===

            response = MappingMessage.unpack_response(response_data)

            if response.get("success", False):
                server_connected = True
                print(f"[MAPPING] Connected to server successfully!")
            else:
                error_msg = response.get("error_msg", "Unknown error")
                print(f"[MAPPING] Server error: {error_msg}, retrying...")
                time.sleep(retry_interval)

        except zmq.Again:
            print(f"[MAPPING] Server timeout, retrying in {retry_interval}s...")
            reset_socket()
            time.sleep(retry_interval)

        except Exception as e:
            print(f"[MAPPING] Connection error: {e}, retrying...")
            reset_socket()
            time.sleep(retry_interval)

    if not server_connected:
        print("[MAPPING] Stopped before server connection established")
        socket.close()
        context.term()
        return

    print(f"[MAPPING] Sending frames at {hz} Hz")

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

        if latest_state is None or not latest_state.has_depth():
            time.sleep(0.1)
            continue

        state = latest_state
        frame_count += 1

        try:
            # Get camera pose in odom frame
            T_odom_cam = state.get_T_odom_depth_cam()
            if T_odom_cam is None:
                continue

            # Prepare intrinsics
            intrinsics = None
            if state.depth_intrinsics is not None:
                intrinsics = {
                    "K": state.depth_intrinsics,
                    "width": state.rgb.shape[1],
                    "height": state.rgb.shape[0],
                }

            # Scale depth to meters
            depth_meters = state.depth.astype(np.float32) * depth_scale

            # Pack request (CPU-bound, holds GIL)
            request = MappingMessage.pack_request(
                rgb=state.rgb,
                depth=depth_meters,
                T_world_cam=T_odom_cam,
                timestamp=state.timestamp,
                intrinsics=intrinsics,
                compress=True,
            )

            # === I/O - releases GIL ===
            socket.send(request)
            response_data = socket.recv()
            # === End I/O ===

            response = MappingMessage.unpack_response(response_data)

            if response.get("success", False):
                # Extract point clouds (make writable copies)
                occupied_pts = np.array(response["occupied_points"], dtype=np.float32)
                free_pts = np.array(response["free_points"], dtype=np.float32)
                occupied_colors = np.array(response["occupied_colors"], dtype=np.uint8)

                # Create MapState
                map_state = MapState(
                    timestamp=time.time(),
                    occupied_points=occupied_pts,
                    occupied_colors=occupied_colors,
                    free_points=free_pts,
                    robot_state=state,
                )

                # Output to queue
                try:
                    map_queue.put_nowait(map_state)
                except Full:
                    pass  # Drop if queue full

                # Reset error count on success
                error_count = 0

                if frame_count % 10 == 0:
                    print(
                        f"[MAPPING] Frame {frame_count}: "
                        f"{map_state.num_occupied()} occupied, "
                        f"{map_state.num_free()} free voxels"
                    )
            else:
                error_msg = response.get("error_msg", "Unknown error")
                print(f"[MAPPING] Server error: {error_msg}")
                error_count += 1

        except zmq.Again:
            print(f"[MAPPING] Timeout waiting for server response")
            error_count += 1
            reset_socket()

        except Exception as e:
            print(f"[MAPPING] Error: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            reset_socket()

        # Check for too many consecutive errors
        if error_count >= max_consecutive_errors:
            print(f"[MAPPING] Too many consecutive errors ({error_count}), stopping")
            break

        # Rate limit
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    # Cleanup
    socket.close()
    context.term()
    print("[MAPPING] Mapping loop stopped")
