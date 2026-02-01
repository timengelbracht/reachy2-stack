"""Camera display using OpenCV."""

import time
import threading
from typing import Optional

import numpy as np
import cv2
import open3d as o3d


def camera_loop(
    reachy,
    stop_evt: threading.Event,
    show_rgb: bool = True,
    show_depth: bool = True,
    render_fps: float = 20.0,
    grab_every_n: int = 1,
    depth_colormap=cv2.COLORMAP_JET,
    depth_minmax=None,
    depth_normalize_percentile: bool = True,
    depth_percentile_range=(1, 99),
    odom_state=None,
    generate_pointcloud: bool = False,
    pcd_every_n: int = 30,
    depth_scale: float = 1.0,
    depth_trunc: float = 3.0,
    client=None,
) -> None:
    """Display camera feeds using OpenCV (thread-safe).

    Args:
        reachy: Reachy SDK instance
        stop_evt: Threading event to signal loop termination
        show_rgb: Show RGB camera feed
        show_depth: Show depth camera feed
        render_fps: Frame rate for display
        grab_every_n: Grab frames every N iterations (reduces load)
        depth_colormap: OpenCV colormap for depth visualization
        depth_minmax: Fixed depth range (lo, hi) or None for auto-scale
        depth_normalize_percentile: Use percentile normalization for better visibility
        depth_percentile_range: Percentile range for normalization
        odom_state: Optional OdometryState instance for point cloud updates
        generate_pointcloud: Generate point clouds from RGBD frames
        pcd_every_n: Generate point cloud every N frames
        depth_scale: Scale factor for depth (1.0 if depth is in meters)
        depth_trunc: Maximum depth value to include in point cloud (in meters)
        client: Optional ReachyClient instance for getting camera intrinsics
    """
    if not (show_rgb or show_depth):
        return

    dt = 1.0 / max(1e-6, render_fps)

    # Create OpenCV windows
    if show_rgb:
        cv2.namedWindow("Reachy RGB Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reachy RGB Camera", 640, 480)
    if show_depth:
        cv2.namedWindow("Reachy Depth Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reachy Depth Camera", 640, 480)

    grab_count = 0
    pcd_count = 0

    # Get camera intrinsics for point cloud generation
    intrinsics = None
    if generate_pointcloud and odom_state is not None:
        if client is None:
            print(f"[CAM] Warning: client is required for point cloud generation")
            generate_pointcloud = False
        else:
            try:
                intrinsics = client.get_depth_intrinsics()
                print(f"[CAM] Point cloud generation enabled (every {pcd_every_n} frames)")
            except Exception as e:
                print(f"[CAM] Could not get camera intrinsics: {e}")
                generate_pointcloud = False

    print("[CAM] Camera display started")

    while not stop_evt.is_set():
        t0 = time.time()
        grab_count += 1

        # Grab frames only every N-th iteration
        if grab_count % grab_every_n == 0:
            try:
                if show_rgb:
                    frame, _ = reachy.cameras.depth.get_frame()
                    frame = np.asarray(frame)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        # OpenCV expects BGR, so no conversion needed if it's already BGR
                        cv2.imshow("Reachy RGB Camera", frame)
                    else:
                        cv2.imshow("Reachy RGB Camera", frame)

                if show_depth:
                    d, _ = reachy.cameras.depth.get_depth_frame()
                    d = np.asarray(d)

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

                # Generate point cloud from RGBD
                if generate_pointcloud and odom_state is not None and intrinsics is not None:
                    if pcd_count % pcd_every_n == 0:
                        try:
                            # Get RGB and depth frames
                            rgb_frame, _ = reachy.cameras.depth.get_frame()
                            depth_frame, _ = reachy.cameras.depth.get_depth_frame()

                            rgb = np.asarray(rgb_frame)
                            depth = np.asarray(depth_frame).squeeze()

                            # Create point cloud in camera frame
                            from .utils import rgbd_to_pointcloud
                            pcd_camera = rgbd_to_pointcloud(
                                rgb,
                                depth,
                                intrinsics,
                                depth_scale=depth_scale,
                                depth_trunc=depth_trunc,
                            )
                            # Update odometry state with new point cloud
                            odom_state.update_pointcloud(pcd_camera)
                        except Exception as e:
                            print(f"[CAM] point cloud generation error: {e}")
                            import traceback
                            traceback.print_exc()

                    # Increment counter after processing
                    pcd_count += 1

            except Exception as e:
                print(f"[CAM] grab error: {e}")

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
    print("[CAM] Camera display stopped")
