"""Camera display using matplotlib."""

import time
import threading
from typing import Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

from reachy2_stack.core.client import ReachyClient

class CameraState:
    """Thread-safe container for camera state, frames, and intrinsics."""
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
        # Camera intrinsics
        self.depth_intrinsics: Optional[dict] = None
        self.teleop_left_intrinsics: Optional[dict] = None
        self.teleop_right_intrinsics: Optional[dict] = None
        # Camera extrinsics (optional)
        self.depth_extrinsics: Optional[np.ndarray] = None
        self.teleop_left_extrinsics: Optional[np.ndarray] = None
        self.teleop_right_extrinsics: Optional[np.ndarray] = None

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

    def set_depth_intrinsics(self, intrinsics: dict):
        """Set depth camera intrinsics.

        Args:
            intrinsics: Dict with 'K', 'width', 'height' (K is 3x3 matrix)
        """
        with self.lock:
            self.depth_intrinsics = intrinsics

    def set_teleop_intrinsics(self, left_intrinsics: dict, right_intrinsics: dict):
        """Set teleop camera intrinsics.

        Args:
            left_intrinsics: Dict with 'K', 'width', 'height' for left camera
            right_intrinsics: Dict with 'K', 'width', 'height' for right camera
        """
        with self.lock:
            self.teleop_left_intrinsics = left_intrinsics
            self.teleop_right_intrinsics = right_intrinsics

    def set_depth_extrinsics(self, extrinsics: np.ndarray):
        """Set depth camera extrinsics (T_base_cam).

        Args:
            extrinsics: 4x4 transformation matrix (base <- camera)
        """
        with self.lock:
            self.depth_extrinsics = extrinsics

    def set_teleop_extrinsics(self, left_extrinsics: np.ndarray, right_extrinsics: np.ndarray):
        """Set teleop camera extrinsics.

        Args:
            left_extrinsics: 4x4 transformation matrix for left camera
            right_extrinsics: 4x4 transformation matrix for right camera
        """
        with self.lock:
            self.teleop_left_extrinsics = left_extrinsics
            self.teleop_right_extrinsics = right_extrinsics

    def get_depth_intrinsics(self) -> Optional[dict]:
        """Get depth camera intrinsics (thread-safe).

        Returns:
            Dict with 'K', 'width', 'height' or None if not set
        """
        with self.lock:
            if self.depth_intrinsics is not None:
                return self.depth_intrinsics.copy()
            return None

    def get_teleop_intrinsics(self) -> Optional[tuple[dict, dict]]:
        """Get teleop camera intrinsics (thread-safe).

        Returns:
            Tuple of (left_intrinsics, right_intrinsics) or None if not set
        """
        with self.lock:
            if self.teleop_left_intrinsics is not None and self.teleop_right_intrinsics is not None:
                return self.teleop_left_intrinsics.copy(), self.teleop_right_intrinsics.copy()
            return None

    def get_depth_extrinsics(self) -> Optional[np.ndarray]:
        """Get depth camera extrinsics (thread-safe).

        Returns:
            4x4 transformation matrix or None if not set
        """
        with self.lock:
            if self.depth_extrinsics is not None:
                return self.depth_extrinsics.copy()
            return None

    def get_teleop_extrinsics(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Get teleop camera extrinsics (thread-safe).

        Returns:
            Tuple of (left_extrinsics, right_extrinsics) or None if not set
        """
        with self.lock:
            if self.teleop_left_extrinsics is not None and self.teleop_right_extrinsics is not None:
                return self.teleop_left_extrinsics.copy(), self.teleop_right_extrinsics.copy()
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
    # Fetch and store camera intrinsics/extrinsics on startup
    if client is not None:
        try:
            # Depth camera intrinsics
            depth_intrinsics = client.get_depth_intrinsics()
            camera_state.set_depth_intrinsics(depth_intrinsics)
            print("[CAM] Depth camera intrinsics loaded")
        except Exception as e:
            print(f"[CAM] Warning: Could not get depth intrinsics: {e}")

        try:
            # Depth camera extrinsics
            depth_extrinsics = client.get_depth_extrinsics()
            camera_state.set_depth_extrinsics(np.array(depth_extrinsics))
            print("[CAM] Depth camera extrinsics loaded")
        except Exception as e:
            print(f"[CAM] Warning: Could not get depth extrinsics: {e}")

        try:
            # Teleop camera intrinsics
            teleop_left_intrinsics = client.get_teleop_intrinsics_left()
            teleop_right_intrinsics = client.get_teleop_intrinsics_right()
            camera_state.set_teleop_intrinsics(teleop_left_intrinsics, teleop_right_intrinsics)
            print("[CAM] Teleop camera intrinsics loaded")
        except Exception as e:
            print(f"[CAM] Warning: Could not get teleop intrinsics: {e}")

        try:
            # Teleop camera extrinsics
            teleop_left_extrinsics = client.get_teleop_extrinsics_left()
            teleop_right_extrinsics = client.get_teleop_extrinsics_right()
            camera_state.set_teleop_extrinsics(
                np.array(teleop_left_extrinsics),
                np.array(teleop_right_extrinsics)
            )
            print("[CAM] Teleop camera extrinsics loaded")
        except Exception as e:
            print(f"[CAM] Warning: Could not get teleop extrinsics: {e}")

    dt = 1.0 / max(1e-6, render_fps)

    grab_count = 0
    teleop_grab_count = 0

    # Check if teleop cameras are enabled
    enable_teleop = show_teleop and grab_teleop_every_n > 0 and client is not None
    if show_teleop and (grab_teleop_every_n <= 0 or client is None):
        print("[CAM] Warning: Teleop cameras disabled (grab_teleop_every_n must be > 0 and client required)")
        enable_teleop = False

    # Set up matplotlib for interactive mode
    matplotlib.use('TkAgg')
    plt.ion()

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Reachy Camera Feeds", fontsize=14)

    ax_rgb = axes[0, 0]
    ax_depth = axes[0, 1]
    ax_teleop_left = axes[1, 0]
    ax_teleop_right = axes[1, 1]

    # Set titles
    ax_rgb.set_title("RGB Camera")
    ax_depth.set_title("Depth Camera")
    ax_teleop_left.set_title("Teleop Left")
    ax_teleop_right.set_title("Teleop Right")

    # Remove axes ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Initialize image objects (will be updated in loop)
    im_rgb = None
    im_depth = None
    im_teleop_left = None
    im_teleop_right = None

    # Show "Not enabled" for disabled cameras
    if not show_rgb:
        ax_rgb.text(0.5, 0.5, "Not enabled", ha='center', va='center',
                    fontsize=16, transform=ax_rgb.transAxes, color='gray')
    if not show_depth:
        ax_depth.text(0.5, 0.5, "Not enabled", ha='center', va='center',
                      fontsize=16, transform=ax_depth.transAxes, color='gray')
    if not enable_teleop:
        ax_teleop_left.text(0.5, 0.5, "Not enabled", ha='center', va='center',
                            fontsize=16, transform=ax_teleop_left.transAxes, color='gray')
        ax_teleop_right.text(0.5, 0.5, "Not enabled", ha='center', va='center',
                             fontsize=16, transform=ax_teleop_right.transAxes, color='gray')

    plt.tight_layout()
    plt.show(block=False)

    print("[CAM] Camera display started (matplotlib)")
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
                    # Convert BGR to RGB for matplotlib
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame

                    if im_rgb is None:
                        im_rgb = ax_rgb.imshow(frame_rgb)
                    else:
                        im_rgb.set_data(frame_rgb)

                if show_depth:
                    d = np.asarray(depth)

                    if d.ndim == 3 and d.shape[2] == 3:
                        # Already colorized, convert BGR to RGB
                        depth_vis = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
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
                                # Percentile-based normalization
                                lo_percentile, hi_percentile = depth_percentile_range
                                dd_valid = dd[dd > 0]

                                if len(dd_valid) > 0:
                                    lo = np.percentile(dd_valid, lo_percentile)
                                    hi = np.percentile(dd_valid, hi_percentile)
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

                            # Apply colormap and convert BGR to RGB
                            dd_color = cv2.applyColorMap(dd_norm, depth_colormap)
                            depth_vis = cv2.cvtColor(dd_color, cv2.COLOR_BGR2RGB)
                        else:
                            depth_vis = dd

                    if im_depth is None:
                        im_depth = ax_depth.imshow(depth_vis)
                    else:
                        im_depth.set_data(depth_vis)

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

                # Convert BGR to RGB for matplotlib
                left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

                if im_teleop_left is None:
                    im_teleop_left = ax_teleop_left.imshow(left_rgb)
                else:
                    im_teleop_left.set_data(left_rgb)

                if im_teleop_right is None:
                    im_teleop_right = ax_teleop_right.imshow(right_rgb)
                else:
                    im_teleop_right.set_data(right_rgb)

            except Exception as e:
                print(f"[CAM] Teleop camera grab error: {e}")

        # Update matplotlib display
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            # Window was closed
            break

        # Pace rendering
        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    # Cleanup
    plt.close(fig)
    print("[CAM] Camera display stopped")
