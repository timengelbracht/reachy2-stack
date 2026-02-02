"""3D mapping from RGBD camera data."""

import time
import threading
import zlib
from queue import Queue
from typing import Optional, Dict, Any, Union

import numpy as np
import open3d as o3d
import msgpack
import msgpack_numpy as m

from .camera import CameraState
from .odometry import OdometryState
from .utils import rgbd_to_pointcloud

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
            depth: Depth image (H, W) as float32 or uint16
            T_world_cam: 4x4 camera pose in world frame
            timestamp: Frame timestamp
            intrinsics: Camera intrinsics dict with 'K', 'width', 'height', 'fx', 'fy', 'cx', 'cy'
            compress: Use zlib compression for images (recommended)

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
        """Unpack mapping request message.

        Args:
            data: Serialized message bytes
            compressed: Whether data is compressed

        Returns:
            Dictionary with 'rgb', 'depth', 'T_world_cam', 'timestamp'
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
        """Pack mapping result (occupied/free point clouds).

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
        """Unpack mapping response message.

        Args:
            data: Serialized message bytes

        Returns:
            Dictionary with 'success', 'occupied_points', 'free_points', 'occupied_colors'
        """
        msg = msgpack.unpackb(data, raw=False)
        return msg


def mapping_loop(
    camera_state: CameraState,
    odom_state: OdometryState,
    stop_evt: threading.Event,
    client,
    mapping_hz: float = 2.0,
    depth_scale: float = 1.0,
    depth_trunc: float = 3.0,
) -> None:
    """Generate point clouds from RGBD frames and transform to world frame.

    This loop:
    1. Gets latest RGB/depth frames from camera_state
    2. Converts RGBD to point cloud in camera frame
    3. Transforms point cloud to world frame using odometry
    4. Updates point cloud in odom_state for visualization

    Args:
        camera_state: CameraState instance for getting RGB/depth frames
        odom_state: OdometryState instance for getting pose and storing point clouds
        stop_evt: Threading event to signal loop termination
        client: ReachyClient instance for getting camera intrinsics
        mapping_hz: Point cloud generation rate in Hz
        depth_scale: Scale factor for depth (1.0 if depth is in meters, 0.001 if in mm)
        depth_trunc: Maximum depth value to include in point cloud (in meters)
    """
    dt = 1.0 / max(1e-6, mapping_hz)

    # Get camera intrinsics
    intrinsics = None
    try:
        intrinsics = client.get_depth_intrinsics()
        print(f"[MAPPING] Camera intrinsics loaded")
        print(f"[MAPPING] Point cloud generation at {mapping_hz} Hz")
        print(f"[MAPPING] Depth scale: {depth_scale}, truncation: {depth_trunc}m")
    except Exception as e:
        print(f"[MAPPING] ERROR: Could not get camera intrinsics: {e}")
        print(f"[MAPPING] Mapping disabled")
        return

    print("[MAPPING] Mapping loop started")

    frame_count = 0

    while not stop_evt.is_set():
        t0 = time.time()

        try:
            # Get latest frames from camera state
            frames = camera_state.get_frames()

            if frames is None:
                # No frames available yet, wait
                time.sleep(dt)
                continue

            rgb_frame, depth_frame, frame_timestamp = frames
            frame_count += 1

            # Convert RGBD to point cloud in camera frame
            pcd_camera = rgbd_to_pointcloud(
                rgb_frame,
                depth_frame,
                intrinsics,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
            )

            num_points = len(pcd_camera.points)

            if num_points > 0:
                # Update point cloud in odometry state (transforms to world frame internally)
                odom_state.update_pointcloud(pcd_camera)

                if frame_count % 10 == 0:  # Print status every 10 frames
                    print(f"[MAPPING] Frame {frame_count}: {num_points} points generated")
            else:
                if frame_count % 10 == 0:
                    print(f"[MAPPING] Frame {frame_count}: 0 points (check depth values)")

        except Exception as e:
            print(f"[MAPPING] Error: {e}")
            import traceback
            traceback.print_exc()

        # Pace the loop
        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    print("[MAPPING] Mapping loop stopped")


def mapping_loop_client(
    camera_state: CameraState,
    odom_state: OdometryState,
    stop_evt: threading.Event,
    server_host: str = "localhost",
    server_port: int = 5555,
    mapping_hz: float = 2.0,
    timeout_ms: int = 5000,
    depth_scale: float = 0.001,
) -> None:
    """Client-side mapping loop that communicates with wavemap server.

    This loop sends RGBD frames + camera pose to a remote wavemap server
    and receives back occupied/free point clouds for visualization.

    Args:
        camera_state: CameraState instance for getting RGB/depth frames and intrinsics
        odom_state: OdometryState instance for getting camera pose and storing point clouds
        stop_evt: Threading event to signal loop termination
        server_host: Wavemap server hostname or IP
        server_port: Wavemap server port
        mapping_hz: Request rate in Hz
        timeout_ms: ZeroMQ receive timeout in milliseconds
        depth_scale: Scale factor to convert depth to meters (e.g., 0.001 for mm to m)
    """
    import zmq

    dt = 1.0 / max(1e-6, mapping_hz)

    # Get camera intrinsics from camera state (set by camera_loop on startup)
    intrinsics = None
    # Wait a bit for camera_loop to initialize intrinsics
    for _ in range(10):
        intrinsics = camera_state.get_depth_intrinsics()
        if intrinsics is not None:
            print(f"[MAPPING CLIENT] Retrieved intrinsics {intrinsics}")
            break
        time.sleep(0.1)
        

    if intrinsics is not None:
        print(f"[MAPPING CLIENT] Camera intrinsics loaded from CameraState")
    else:
        print(f"[MAPPING CLIENT] Warning: No intrinsics in CameraState")
        print(f"[MAPPING CLIENT] Server will estimate intrinsics")

    # Initialize ZeroMQ client
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # Request socket
    socket.connect(f"tcp://{server_host}:{server_port}")
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

    print(f"[MAPPING CLIENT] Connecting to wavemap server at {server_host}:{server_port}...")

    # Wait for server to be available before entering main loop
    server_connected = False
    wait_retry_interval = 2.0  # Seconds between connection attempts

    while not stop_evt.is_set() and not server_connected:
        # Wait for frames first
        frames = camera_state.get_frames()
        if frames is None:
            time.sleep(0.5)
            continue

        rgb_frame, depth_frame, frame_timestamp = frames
        depth_meters = depth_frame.astype(np.float32) * depth_scale
        T_world_cam = odom_state.get_camera_pose()

        try:
            # Try to send a frame
            request = MappingMessage.pack_request(
                rgb=rgb_frame,
                depth=depth_meters,
                T_world_cam=T_world_cam,
                timestamp=frame_timestamp,
                intrinsics=intrinsics,
                compress=True,
            )
            socket.send(request)
            response_data = socket.recv()
            response = MappingMessage.unpack_response(response_data)

            if response["success"]:
                server_connected = True
                print(f"[MAPPING CLIENT] Connected to server successfully!")
            else:
                print(f"[MAPPING CLIENT] Server returned error, retrying...")
                time.sleep(wait_retry_interval)

        except zmq.Again:
            print(f"[MAPPING CLIENT] Waiting for server... (will retry in {wait_retry_interval}s)")
            # Reset socket after timeout
            socket.close()
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{server_host}:{server_port}")
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            time.sleep(wait_retry_interval)

        except Exception as e:
            print(f"[MAPPING CLIENT] Connection error: {e}, retrying...")
            try:
                socket.close()
            except:
                pass
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{server_host}:{server_port}")
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            time.sleep(wait_retry_interval)

    if not server_connected:
        print("[MAPPING CLIENT] Stopped before server connection established")
        socket.close()
        context.term()
        return

    print(f"[MAPPING CLIENT] Sending frames at {mapping_hz} Hz")

    frame_count = 0
    error_count = 0
    max_consecutive_errors = 5

    while not stop_evt.is_set():
        t0 = time.time()

        try:
            # Get latest frames from camera state
            frames = camera_state.get_frames()

            if frames is None:
                time.sleep(dt)
                continue

            rgb_frame, depth_frame, frame_timestamp = frames
            frame_count += 1

            # Scale depth to meters
            depth_meters = depth_frame.astype(np.float32) * depth_scale

            # Get camera pose in world frame from odometry
            T_world_cam = odom_state.get_camera_pose()

            # Pack and send request
            request = MappingMessage.pack_request(
                rgb=rgb_frame,
                depth=depth_meters,
                T_world_cam=T_world_cam,
                timestamp=frame_timestamp,
                intrinsics=intrinsics,
                compress=True,
            )

            socket.send(request)

            # Receive response
            response_data = socket.recv()
            response = MappingMessage.unpack_response(response_data)

            if response["success"]:
                # Extract point clouds (make writable copies from msgpack)
                occupied_pts = np.array(response["occupied_points"], dtype=np.float32)
                free_pts = np.array(response["free_points"], dtype=np.float32)
                occupied_colors = np.array(response["occupied_colors"], dtype=np.uint8)

                # Create Open3D point cloud for occupied voxels
                pcd_occupied = o3d.geometry.PointCloud()
                pcd_occupied.points = o3d.utility.Vector3dVector(occupied_pts)
                pcd_occupied.colors = o3d.utility.Vector3dVector(occupied_colors.astype(np.float32) / 255.0)

                pcd_free = o3d.geometry.PointCloud()
                pcd_free.points = o3d.utility.Vector3dVector(free_pts)
                pcd_free.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[0.8, 0.8, 0.8]]), (len(free_pts), 1))
                )  # Light gray for free space  
                

                # Update odometry state with occupied point cloud
                # (Already in world frame, no need to transform)
                with odom_state.lock:
                    odom_state.pointcloud_world = pcd_occupied
                    odom_state.pointcloud_timestamp = time.time()

                # Reset error count on success
                error_count = 0

                if frame_count % 10 == 0:
                    print(
                        f"[MAPPING CLIENT] Frame {frame_count}: "
                        f"{len(occupied_pts)} occupied, {len(free_pts)} free voxels"
                    )
            else:
                error_msg = response.get("error_msg", "Unknown error")
                print(f"[MAPPING CLIENT] Server error: {error_msg}")
                error_count += 1

        except zmq.Again:
            print(f"[MAPPING CLIENT] Timeout waiting for server response")
            error_count += 1
            # Need to reset socket after timeout
            socket.close()
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{server_host}:{server_port}")
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        except Exception as e:
            print(f"[MAPPING CLIENT] Error: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            # Reset socket on error
            try:
                socket.close()
            except:
                pass
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{server_host}:{server_port}")
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        # Check for too many consecutive errors
        if error_count >= max_consecutive_errors:
            print(f"[MAPPING CLIENT] Too many consecutive errors ({error_count}), stopping")
            break

        # Pace the loop
        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    # Cleanup
    socket.close()
    context.term()
    print("[MAPPING CLIENT] Mapping client loop stopped")


def mapping_loop_unified(
    state_queue: "Queue",
    result_queue: "Queue",
    stop_event: "Union[threading.Event, Any]",
    mode: str = "server",
    server_host: str = "localhost",
    server_port: int = 5555,
    hz: float = 2.0,
    timeout_ms: int = 5000,
    depth_scale: float = 0.001,
    depth_trunc: float = 3.5,
) -> None:
    """Unified mapping loop that processes RobotState from a queue.

    This is the new-style mapping loop that:
    1. Reads RobotState objects from state_queue
    2. Processes using either local or server mode
    3. Outputs results to result_queue for visualization

    Modes:
        - "local": CPU-bound pointcloud generation (blocks GIL)
        - "server": I/O-bound ZMQ to wavemap server (parallel-friendly)

    Args:
        state_queue: Queue of RobotState objects
        result_queue: Queue for output results (for visualization)
        stop_event: Event to signal loop termination
        mode: "local" or "server"
        server_host: Wavemap server hostname (for server mode)
        server_port: Wavemap server port (for server mode)
        hz: Processing rate in Hz
        timeout_ms: ZMQ timeout in milliseconds (for server mode)
        depth_scale: Scale factor to convert depth to meters
        depth_trunc: Maximum depth in meters (for local mode)
    """
    from queue import Empty, Full
    from .robot_state import RobotState
    from .utils import rgbd_to_pointcloud

    interval = 1.0 / hz
    print(f"[MAPPING UNIFIED] Starting in {mode} mode at {hz} Hz")

    # Initialize based on mode
    socket = None
    context = None

    if mode == "server":
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{server_host}:{server_port}")
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        print(f"[MAPPING UNIFIED] Connected to server at {server_host}:{server_port}")

    latest_state: Optional[RobotState] = None
    frame_count = 0

    while not stop_event.is_set():
        t0 = time.time()

        # Drain state queue to get latest
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
            if mode == "server":
                # === SERVER MODE (I/O - releases GIL) ===
                T_world_cam = state.get_pose()

                # Prepare intrinsics dict
                intrinsics = None
                if state.intrinsics is not None:
                    intrinsics = {"K": state.intrinsics}

                # Scale depth to meters
                depth_meters = state.depth.astype(np.float32) * depth_scale

                # Pack and send request
                request = MappingMessage.pack_request(
                    rgb=state.rgb,
                    depth=depth_meters,
                    T_world_cam=T_world_cam,
                    timestamp=state.timestamp,
                    intrinsics=intrinsics,
                    compress=True,
                )
                socket.send(request)

                # Wait for response (I/O - releases GIL)
                response_data = socket.recv()
                result = MappingMessage.unpack_response(response_data)

                if result["success"]:
                    # Queue result for visualization
                    try:
                        result_queue.put_nowait({
                            "occupied_points": np.array(result["occupied_points"], dtype=np.float32),
                            "free_points": np.array(result["free_points"], dtype=np.float32),
                            "occupied_colors": np.array(result["occupied_colors"], dtype=np.uint8),
                            "state": state,
                        })
                    except Full:
                        pass

                    if frame_count % 10 == 0:
                        print(
                            f"[MAPPING UNIFIED] Frame {frame_count}: "
                            f"{len(result['occupied_points'])} occupied voxels"
                        )

            else:
                # === LOCAL MODE (CPU - holds GIL) ===
                # Build intrinsics dict for rgbd_to_pointcloud
                if state.intrinsics is None:
                    print("[MAPPING UNIFIED] Warning: No intrinsics available")
                    time.sleep(interval)
                    continue

                K = state.intrinsics
                intrinsics = {
                    "K": K,
                    "width": state.rgb.shape[1],
                    "height": state.rgb.shape[0],
                }

                # Generate point cloud
                pcd = rgbd_to_pointcloud(
                    state.rgb,
                    state.depth,
                    intrinsics,
                    depth_scale=depth_scale,
                    depth_trunc=depth_trunc,
                )

                if len(pcd.points) > 0:
                    # Transform to world/odom frame
                    pcd.transform(state.get_pose())

                    # Queue result
                    try:
                        result_queue.put_nowait({
                            "points": np.asarray(pcd.points),
                            "colors": np.asarray(pcd.colors),
                            "state": state,
                        })
                    except Full:
                        pass

                    if frame_count % 10 == 0:
                        print(f"[MAPPING UNIFIED] Frame {frame_count}: {len(pcd.points)} points")

        except Exception as e:
            print(f"[MAPPING UNIFIED] Error: {e}")
            import traceback
            traceback.print_exc()

            # Reset socket on error (server mode)
            if mode == "server" and socket is not None:
                import zmq
                try:
                    socket.close()
                except:
                    pass
                socket = context.socket(zmq.REQ)
                socket.connect(f"tcp://{server_host}:{server_port}")
                socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
                socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        # Rate limit
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    # Cleanup
    if socket is not None:
        socket.close()
    if context is not None:
        context.term()
    print("[MAPPING UNIFIED] Mapping loop stopped")