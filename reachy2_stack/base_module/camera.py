"""Camera display using OpenCV."""

import time
import threading
from typing import Optional

import numpy as np
import cv2

from reachy2_stack.core.client import ReachyClient

class CameraState:
    """Thread-safe container for camera state and latest frames."""
    def __init__(self):
        self.lock = threading.Lock()
        # Depth camera (RGBD)
        self.rgb_frame: Optional[np.ndarray] = None
        self.depth_frame: Optional[np.ndarray] = None
        self.timestamp: float = 0.0
        # Teleop cameras (stereo RGB)
        self.teleop_left_frame: Optional[np.ndarray] = None
        self.teleop_right_frame: Optional[np.ndarray] = None
        self.teleop_timestamp: float = 0.0

    def update_frames(self, rgb_frame: np.ndarray, depth_frame: np.ndarray):
        """Update the latest RGB and depth frames from depth camera.

        Args:
            rgb_frame: Latest RGB frame as a numpy array
            depth_frame: Latest depth frame as a numpy array
        """
        with self.lock:
            self.rgb_frame = rgb_frame
            self.depth_frame = depth_frame
            self.timestamp = time.time()

    def update_teleop_frames(self, left_frame: np.ndarray, right_frame: np.ndarray):
        """Update the latest left and right teleop camera frames.

        Args:
            left_frame: Latest left teleop RGB frame
            right_frame: Latest right teleop RGB frame
        """
        with self.lock:
            self.teleop_left_frame = left_frame
            self.teleop_right_frame = right_frame
            self.teleop_timestamp = time.time()

    def get_frames(self) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        """Get the latest RGB and depth frames from depth camera.

        Returns:
            Tuple of (rgb_frame, depth_frame, timestamp) or None if frames are not available
        """
        with self.lock:
            if self.rgb_frame is not None and self.depth_frame is not None:
                return self.rgb_frame.copy(), self.depth_frame.copy(), self.timestamp
            else:
                print("[CAM STATE] No frames available")
                return None

    def get_teleop_frames(self) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        """Get the latest left and right teleop camera frames.

        Returns:
            Tuple of (left_frame, right_frame, timestamp) or None if frames are not available
        """
        with self.lock:
            if self.teleop_left_frame is not None and self.teleop_right_frame is not None:
                return self.teleop_left_frame.copy(), self.teleop_right_frame.copy(), self.teleop_timestamp
            else:
                return None

def camera_loop(
        reachy: 'ReachyClient',
        camera_state: CameraState,
        stop_evt: threading.Event,
        show_rgb: bool = True,
        show_depth: bool = True,
        show_teleop: bool = False,
        render_fps: float = 20.0,
        grab_every_n: int = 1,
        grab_teleop_every_n: int = 1,
        depth_colormap=cv2.COLORMAP_JET,
        depth_minmax=None,
        depth_normalize_percentile: bool = True,
        depth_percentile_range=(1, 99),
        client: Optional['ReachyClient'] = None,
    ) -> None:

    """Camera State Update Loop with OpenCV Display.

    This loop only handles frame acquisition and optional visualization.
    Point cloud generation is handled separately in mapping.py.

    Args:
        reachy: Reachy SDK instance
        camera_state: CameraState instance for sharing frames
        stop_evt: Threading event to signal loop termination
        show_rgb: Show RGB camera feed from depth camera
        show_depth: Show depth camera feed
        show_teleop: Show teleop stereo cameras (left + right)
        render_fps: Frame rate for display
        grab_every_n: Grab depth camera frames every N iterations (reduces load)
        grab_teleop_every_n: Grab teleop frames every N iterations (0 = disabled, default 1)
        depth_colormap: OpenCV colormap for depth visualization
        depth_minmax: Fixed depth range (lo, hi) or None for auto-scale
        depth_normalize_percentile: Use percentile normalization for better visibility
        depth_percentile_range: Percentile range for normalization
        client: ReachyClient instance for teleop camera access
    """
    if not (show_rgb or show_depth or show_teleop):
        return

    dt = 1.0 / max(1e-6, render_fps)

    # Create OpenCV windows
    if show_rgb:
        cv2.namedWindow("Reachy RGB Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reachy RGB Camera", 640, 480)
    if show_depth:
        cv2.namedWindow("Reachy Depth Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reachy Depth Camera", 640, 480)
    if show_teleop:
        cv2.namedWindow("Reachy Teleop Cameras (L+R)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reachy Teleop Cameras (L+R)", 1280, 480)

    grab_count = 0
    teleop_grab_count = 0

    # Check if teleop cameras are enabled
    enable_teleop = show_teleop and grab_teleop_every_n > 0 and client is not None
    if show_teleop and (grab_teleop_every_n <= 0 or client is None):
        print("[CAM] Warning: Teleop cameras disabled (grab_teleop_every_n must be > 0 and client required)")
        enable_teleop = False

    print("[CAM] Camera display started")
    if enable_teleop:
        print(f"[CAM] Teleop cameras enabled (grabbing every {grab_teleop_every_n} frames)")

    while not stop_evt.is_set():
        t0 = time.time()
        grab_count += 1
        teleop_grab_count += 1

        # Grab depth camera frames every N-th iteration
        if grab_count % grab_every_n == 0:
            try:
                rgb, _ = reachy.cameras.depth.get_frame()
                depth, _ = reachy.cameras.depth.get_depth_frame()
                camera_state.update_frames(np.asarray(rgb), np.asarray(depth))

                if show_rgb:
                    frame = np.asarray(rgb)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        # OpenCV expects BGR, so no conversion needed if it's already BGR
                        cv2.imshow("Reachy RGB Camera", frame)
                    else:
                        cv2.imshow("Reachy RGB Camera", frame)

                if show_depth:
            
                    d = np.asarray(depth)

                    if d.ndim == 3 and d.shape[2] == 3:
                        # Already colorized
                        cv2.imshow("Reachy Depth Camera", d)
                    else:
                        # Single-channel depth, normalize for display
                        dd = np.squeeze(d).astype(np.float32)
                        if dd.ndim == 2:
                            # Normalize to 0-255 range for visualization
                            if depth_minmax is not None:
                                # Use fixed depth range
                                lo, hi = depth_minmax
                                dd = np.clip(dd, lo, hi)
                                dd_norm = ((dd - lo) / (hi - lo) * 255).astype(np.uint8)
                            elif depth_normalize_percentile:
                                # Percentile-based normalization (better visibility, clips outliers)
                                lo_percentile, hi_percentile = depth_percentile_range
                                dd_valid = dd[dd > 0]  # Ignore zero/invalid depth values

                                if len(dd_valid) > 0:
                                    lo = np.percentile(dd_valid, lo_percentile)
                                    hi = np.percentile(dd_valid, hi_percentile)

                                    # Clip and normalize
                                    dd_clipped = np.clip(dd, lo, hi)
                                    if hi > lo:
                                        dd_norm = ((dd_clipped - lo) / (hi - lo) * 255).astype(np.uint8)
                                    else:
                                        dd_norm = np.zeros_like(dd, dtype=np.uint8)
                                else:
                                    dd_norm = np.zeros_like(dd, dtype=np.uint8)
                            else:
                                # Simple min/max auto-normalize
                                dd_min, dd_max = dd.min(), dd.max()
                                if dd_max > dd_min:
                                    dd_norm = ((dd - dd_min) / (dd_max - dd_min) * 255).astype(np.uint8)
                                else:
                                    dd_norm = np.zeros_like(dd, dtype=np.uint8)

                            # Apply colormap
                            dd_color = cv2.applyColorMap(dd_norm, depth_colormap)
                            cv2.imshow("Reachy Depth Camera", dd_color)

            except Exception as e:
                print(f"[CAM] Depth camera grab error: {e}")

        # Grab teleop camera frames every N-th iteration
        if enable_teleop and teleop_grab_count % grab_teleop_every_n == 0:
            try:
                left_frame = client.get_teleop_rgb_left()
                right_frame = client.get_teleop_rgb_right()

                left_img = np.asarray(left_frame)
                right_img = np.asarray(right_frame)

                # Update camera state
                camera_state.update_teleop_frames(left_img, right_img)

                # Display combined left+right image
                if show_teleop:
                    # Concatenate left and right horizontally
                    combined = np.hstack([left_img, right_img])
                    cv2.imshow("Reachy Teleop Cameras (L+R)", combined)

            except Exception as e:
                print(f"[CAM] Teleop camera grab error: {e}")

        # OpenCV needs waitKey to process window events (even if we don't use the key)
        cv2.waitKey(1)

        # Pace rendering
        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    # Cleanup
    if show_rgb:
        cv2.destroyWindow("Reachy RGB Camera")
    if show_depth:
        cv2.destroyWindow("Reachy Depth Camera")
    if show_teleop:
        cv2.destroyWindow("Reachy Teleop Cameras (L+R)")
    print("[CAM] Camera display stopped")
