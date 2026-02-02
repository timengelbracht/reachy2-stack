#!/usr/bin/env python3
"""Wavemap volumetric mapping server.

This server receives RGBD frames + camera poses from clients and maintains
a volumetric occupancy map using wavemap. It returns occupied and free
point clouds for visualization.

Usage:
    python ext_server/wavemap_server.py --port 5555
"""

import argparse
import time
import zmq
import sys
import os
import numpy as np
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reachy2_stack.base_module.mapping import MappingMessage

try:
    from advanced_mapping.wavemap import WaveMapper
    WAVEMAP_AVAILABLE = True
except ImportError:
    print("[WAVEMAP SERVER] Warning: WaveMapper not found. Running in dummy mode.")
    WAVEMAP_AVAILABLE = False


class WavemapServer:
    """Volumetric mapping server using wavemap."""

    def __init__(
        self,
        port: int = 5555,
        voxel_size: float = 0.05,
        min_range: float = 0.1,
        max_range: float = 5.0,
        query_resolution: float = 0.02,
        integrate_every_n: int = 5,
    ):
        """Initialize wavemap server.

        Args:
            port: ZeroMQ port to listen on
            voxel_size: Voxel size in meters for volumetric map (min_cell_width)
            min_range: Minimum depth range in meters
            max_range: Maximum depth range in meters
            query_resolution: Resolution for querying occupancy grid
            integrate_every_n: Integrate map every N frames (batching for performance)
        """
        self.port = port
        self.voxel_size = voxel_size
        self.min_range = min_range
        self.max_range = max_range
        self.query_resolution = query_resolution
        self.integrate_every_n = integrate_every_n

        # Initialize ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # Reply socket
        self.socket.bind(f"tcp://*:{port}")
        print(f"[WAVEMAP SERVER] Listening on port {port}")

        # Wavemap will be initialized after receiving first frame with intrinsics
        self.mapper = None
        self.intrinsics_initialized = False

        # Frame buffer for batched integration
        self.frame_buffer_count = 0

        # Stats
        self.frame_count = 0
        self.last_print_time = time.time()
        self.last_integration_time = time.time()

    def initialize_mapper(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: Optional[dict] = None,
    ) -> bool:
        """Initialize WaveMapper with camera intrinsics from first frame.

        Args:
            rgb: RGB image to get dimensions
            depth: Depth image
            intrinsics: Camera intrinsics dict (optional)

        Returns:
            True if initialization successful
        """
        if not WAVEMAP_AVAILABLE:
            print("[WAVEMAP SERVER] Cannot initialize: WaveMapper not available")
            return False

        try:
            height, width = depth.shape[:2]
            print(f"[WAVEMAP SERVER] Initializing mapper with image size {width}x{height}")

            # Use provided intrinsics or estimate
            if intrinsics is not None and "K" in intrinsics:
                K = intrinsics["K"]
                print("K:", K)
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
                print("[WAVEMAP SERVER] Using provided camera intrinsics")
            else:
                print("[WAVEMAP SERVER] No intrinsics provided, stopping initialization")
                return False

            params = {
                "min_cell_width": self.voxel_size,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "min_range": self.min_range,
                "max_range": self.max_range,
                "resolution": self.query_resolution,
            }

            self.mapper = WaveMapper(params)
            self.intrinsics_initialized = True

            print(f"[WAVEMAP SERVER] Initialized mapper:")
            print(f"  Image size: {width}x{height}")
            print(f"  Voxel size: {self.voxel_size}m")
            print(f"  Range: {self.min_range}m - {self.max_range}m")
            print(f"  Query resolution: {self.query_resolution}m")

            return True

        except Exception as e:
            print(f"[WAVEMAP SERVER] Failed to initialize mapper: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        T_world_cam: np.ndarray,
        timestamp: float,
        intrinsics: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process RGBD frame and update volumetric map.

        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth image (H, W)
            T_world_cam: Camera pose in world frame (4x4)
            timestamp: Frame timestamp
            intrinsics: Camera intrinsics dict (optional)

        Returns:
            Tuple of (occupied_points, free_points, occupied_colors)
        """
        # Initialize mapper on first frame
        if not self.intrinsics_initialized:
            if not self.initialize_mapper(rgb, depth, intrinsics):
                # Return empty point clouds on initialization failure
                return (
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.uint8),
                )

        if not WAVEMAP_AVAILABLE or self.mapper is None:
            # Dummy mode - return random point clouds
            num_occupied = np.random.randint(100, 500)
            num_free = np.random.randint(50, 200)
            occupied_points = np.random.randn(num_occupied, 3).astype(np.float32) * 2.0
            free_points = np.random.randn(num_free, 3).astype(np.float32) * 3.0
            occupied_colors = np.random.randint(0, 255, (num_occupied, 3), dtype=np.uint8)
            return occupied_points, free_points, occupied_colors

        # Insert depth frame into buffer (depth should already be in meters from client)
        self.mapper.insert_depth_to_buffer(depth, T_world_cam)
        # self.mapper.insert_depth_to_buffer(depth, np.linalg.inv(T_world_cam))
        self.frame_buffer_count += 1

        # Update voxel colors
        # self.update_voxel_colors(rgb, depth, T_world_cam)

        # Integrate and query map every N frames (batching for performance)
        if self.frame_buffer_count >= self.integrate_every_n:
            t0 = time.time()

            # Integrate buffered frames
            self.mapper.integrate_from_buffer()

            # Query occupancy grid
            self.mapper.interpolate_occupancy_grid()
            grid = self.mapper.get_occupancy_grid()

            occupied_points = grid.get("occupied", np.zeros((0, 3), dtype=np.float32))
            free_points = grid.get("free", np.zeros((0, 3), dtype=np.float32))

            # Get colors for occupied voxels
            # occupied_colors = self.get_voxel_colors(occupied_points)
            # set all occ color to green
            occupied_colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (len(occupied_points), 1))

            integration_time = time.time() - t0
            self.last_integration_time = time.time()
            self.frame_buffer_count = 0

            if len(occupied_points) > 0:
                print(
                    f"[WAVEMAP SERVER] Integration: {len(occupied_points)} occupied, "
                    f"{len(free_points)} free | {integration_time:.3f}s"
                )

            return occupied_points, free_points, occupied_colors
        else:
            # Return previous results while buffering
            # Query current map state
            self.mapper.interpolate_occupancy_grid()
            grid = self.mapper.get_occupancy_grid()

            occupied_points = grid.get("occupied", np.zeros((0, 3), dtype=np.float32))
            free_points = grid.get("free", np.zeros((0, 3), dtype=np.float32))
            # occupied_colors = self.get_voxel_colors(occupied_points)
            # set all occ color to green
            occupied_colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (len(occupied_points), 1))

            return occupied_points, free_points, occupied_colors

    def run(self):
        """Run the server loop."""
        print("[WAVEMAP SERVER] Ready to receive frames")

        try:
            while True:
                # Receive request
                try:
                    message = self.socket.recv()
                except zmq.ZMQError as e:
                    print(f"[WAVEMAP SERVER] ZMQ error: {e}")
                    continue

                try:
                    # Unpack request
                    request = MappingMessage.unpack_request(message, compressed=True)
                    rgb = request["rgb"]
                    depth = request["depth"]
                    T_world_cam = request["T_world_cam"]
                    timestamp = request["timestamp"]
                    intrinsics = request.get("intrinsics", None)

                    # Process frame
                    occupied_pts, free_pts, occupied_colors = self.process_frame(
                        rgb, depth, T_world_cam, timestamp, intrinsics
                    )

                    # Pack response
                    response = MappingMessage.pack_response(
                        occupied_points=occupied_pts,
                        free_points=free_pts,
                        occupied_colors=occupied_colors,
                        success=True,
                    )

                    self.frame_count += 1

                    # Print stats every 10 frames
                    if self.frame_count % 10 == 0:
                        elapsed = time.time() - self.last_print_time
                        fps = 10.0 / elapsed
                        print(
                            f"[WAVEMAP SERVER] Frame {self.frame_count}: "
                            f"{len(occupied_pts)} occupied, {len(free_pts)} free | "
                            f"{fps:.1f} FPS"
                        )
                        self.last_print_time = time.time()

                except Exception as e:
                    print(f"[WAVEMAP SERVER] Error processing frame: {e}")
                    import traceback
                    traceback.print_exc()

                    # Send error response
                    response = MappingMessage.pack_response(
                        success=False,
                        error_msg=str(e),
                    )

                # Send response
                self.socket.send(response)

        except KeyboardInterrupt:
            print("\n[WAVEMAP SERVER] Shutting down...")
        finally:
            self.socket.close()
            self.context.term()


def main():
    parser = argparse.ArgumentParser(description="Wavemap volumetric mapping server")
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ port")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size in meters")
    parser.add_argument("--min-range", type=float, default=0.1, help="Minimum depth range in meters")
    parser.add_argument("--max-range", type=float, default=5.0, help="Maximum depth range in meters")
    parser.add_argument("--query-resolution", type=float, default=0.1, help="Query grid resolution")
    parser.add_argument("--integrate-every", type=int, default=5, help="Integrate every N frames")
    args = parser.parse_args()

    if not WAVEMAP_AVAILABLE:
        print("\n" + "="*60)
        print("WARNING: WaveMapper not available!")
        print("Server will run in DUMMY MODE (random point clouds)")
        print("To enable wavemap:")
        print("  1. Install pywavemap: pip install pywavemap")
        print("  2. Ensure advanced_mapping/wavemap.py is accessible")
        print("="*60 + "\n")

    server = WavemapServer(
        port=args.port,
        voxel_size=args.voxel_size,
        min_range=args.min_range,
        max_range=args.max_range,
        query_resolution=args.query_resolution,
        integrate_every_n=args.integrate_every,
    )
    server.run()


if __name__ == "__main__":
    main()
