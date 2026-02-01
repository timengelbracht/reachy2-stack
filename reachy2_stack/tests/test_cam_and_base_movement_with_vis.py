#!/usr/bin/env python3
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import cv2
from pynput import keyboard
import open3d as o3d

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient

# ---------------- CONFIG ----------------
HOST = "192.168.1.71"

# Teleop
CMD_HZ = 30
VX = 0.6
VY = 0.6
WZ = 110.0

# Camera rendering (using OpenCV - compatible with Open3D)
RENDER_FPS = 20
GRAB_EVERY_N = 1
SHOW_RGB = True  # Show RGB camera feed
SHOW_DEPTH = True  # Show depth camera feed

# Depth display
DEPTH_CMAP = "viridis"
DEPTH_MINMAX = None

# Open3D visualization
VIS_UPDATE_HZ = 10  # Update rate for 3D visualization
SHOW_TRAJECTORY = True  # Show odometry trail
MAX_TRAIL_POINTS = 500  # Maximum trajectory points to keep
COORD_FRAME_SIZE = 0.3  # Size of coordinate frame axes
# --------------------------------------


@dataclass
class OdometryState:
    """Thread-safe container for odometry data."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # in degrees
    timestamp: float = 0.0
    trajectory: List[List[float]] = field(default_factory=list)

    def update(self, x: float, y: float, theta: float):
        with self.lock:
            self.x = x
            self.y = y
            self.theta = theta
            self.timestamp = time.time()
            # Add to trajectory (xy plane, z=0)
            self.trajectory.append([x, y, 0.0])
            # Limit trajectory length
            if len(self.trajectory) > MAX_TRAIL_POINTS:
                self.trajectory.pop(0)

    def get_pose(self):
        with self.lock:
            return self.x, self.y, self.theta

    def get_trajectory(self):
        with self.lock:
            return np.array(self.trajectory) if self.trajectory else np.zeros((0, 3))


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1]


def create_coordinate_frame(size: float = 1.0) -> o3d.geometry.TriangleMesh:
    """Create a coordinate frame mesh (RGB = XYZ)."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def pose_to_transform(x: float, y: float, theta_deg: float) -> np.ndarray:
    """Convert 2D pose (x, y, theta) to 4x4 transform matrix."""
    theta = np.deg2rad(theta_deg)
    T = np.eye(4)
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta)
    T[0, 3] = x
    T[1, 3] = y
    return T


def camera_loop(reachy, stop_evt: threading.Event) -> None:
    if not (SHOW_RGB or SHOW_DEPTH):
        return

    dt = 1.0 / max(1e-6, RENDER_FPS)

    plt.ion()
    ncols = int(SHOW_RGB) + int(SHOW_DEPTH)
    fig, axes = plt.subplots(1, ncols, num="Reachy Live Camera", figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    ax_i = 0
    ax_rgb = axes[ax_i] if SHOW_RGB else None
    if SHOW_RGB:
        ax_rgb.set_title("Frame")
        ax_rgb.axis("off")
        ax_i += 1

    ax_depth = axes[ax_i] if SHOW_DEPTH else None
    if SHOW_DEPTH:
        ax_depth.set_title("Depth")
        ax_depth.axis("off")

    rgb_artist = None
    depth_artist = None

    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    grab_count = 0
    last_rgb = None
    last_depth = None

    while not stop_evt.is_set():
        t0 = time.time()
        grab_count += 1

        if grab_count % GRAB_EVERY_N == 0:
            try:
                if SHOW_RGB:
                    frame, _ = reachy.cameras.depth.get_frame()
                    frame = np.asarray(frame)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        last_rgb = _bgr_to_rgb(frame)
                    else:
                        last_rgb = frame
                if SHOW_DEPTH:
                    d, _ = reachy.cameras.depth.get_depth_frame()
                    d = np.asarray(d)
                    if d.ndim == 3 and d.shape[2] == 3:
                        last_depth = _bgr_to_rgb(d)
                    else:
                        dd = np.squeeze(d)
                        last_depth = dd.astype(np.float32) if dd.ndim == 2 else None
            except Exception as e:
                print("[CAM] grab error:", e)

        if rgb_artist is None and SHOW_RGB:
            if last_rgb is None:
                last_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb_artist = ax_rgb.imshow(last_rgb)

        if depth_artist is None and SHOW_DEPTH:
            if last_depth is None:
                last_depth = np.zeros((480, 640), dtype=np.float32)
            if isinstance(last_depth, np.ndarray) and last_depth.ndim == 2:
                depth_artist = ax_depth.imshow(last_depth, cmap=DEPTH_CMAP)
            else:
                depth_artist = ax_depth.imshow(last_depth)

        if SHOW_RGB and last_rgb is not None:
            rgb_artist.set_data(last_rgb)

        if SHOW_DEPTH and last_depth is not None:
            if isinstance(last_depth, np.ndarray) and last_depth.ndim == 2:
                dshow = last_depth
                if DEPTH_MINMAX is not None:
                    lo, hi = DEPTH_MINMAX
                    dshow = np.clip(dshow, lo, hi)
                if getattr(depth_artist, "cmap", None) is None:
                    ax_depth.clear()
                    ax_depth.set_title("Depth")
                    ax_depth.axis("off")
                    depth_artist = ax_depth.imshow(dshow, cmap=DEPTH_CMAP)
                    fig.canvas.draw()
                    bg = fig.canvas.copy_from_bbox(fig.bbox)
                else:
                    depth_artist.set_data(dshow)
            else:
                if getattr(depth_artist, "cmap", None) is not None:
                    ax_depth.clear()
                    ax_depth.set_title("Depth")
                    ax_depth.axis("off")
                    depth_artist = ax_depth.imshow(last_depth)
                    fig.canvas.draw()
                    bg = fig.canvas.copy_from_bbox(fig.bbox)
                else:
                    depth_artist.set_data(last_depth)

        fig.canvas.restore_region(bg)
        if SHOW_RGB:
            ax_rgb.draw_artist(rgb_artist)
        if SHOW_DEPTH:
            ax_depth.draw_artist(depth_artist)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    plt.close(fig)


def teleop_loop(client: ReachyClient, stop_evt: threading.Event) -> None:
    pressed = set()

    def on_press(key):
        if key == keyboard.Key.esc:
            stop_evt.set()
            return False
        if key in (keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right):
            pressed.add(key)
            return
        if key == keyboard.Key.space:
            pressed.add("space")
            return
        try:
            ch = key.char
        except Exception:
            ch = None
        if ch == ",":
            pressed.add(",")
        elif ch == ".":
            pressed.add(".")

    def on_release(key):
        if key in (keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right):
            pressed.discard(key)
            return
        if key == keyboard.Key.space:
            pressed.discard("space")
            return
        try:
            ch = key.char
        except Exception:
            ch = None
        if ch == ",":
            pressed.discard(",")
        elif ch == ".":
            pressed.discard(".")

    keyboard.Listener(on_press=on_press, on_release=on_release).start()

    dt = 1.0 / CMD_HZ
    print(
        "\n[TELEOP]\n"
        "↑/↓ : forward/back\n"
        "←/→ : left/right\n"
        ",/. : rotate left/right\n"
        "SPACE: stop\n"
        "ESC: quit\n"
    )

    while not stop_evt.is_set():
        vx = vy = wz = 0.0

        if keyboard.Key.up in pressed:
            vx += VX
        if keyboard.Key.down in pressed:
            vx -= VX
        if keyboard.Key.left in pressed:
            vy += VY
        if keyboard.Key.right in pressed:
            vy -= VY

        if "," in pressed:
            wz += WZ
        if "." in pressed:
            wz -= WZ

        if "space" in pressed:
            vx = vy = wz = 0.0

        client.goto_base_defined_speed(vx, vy, wz)
        time.sleep(dt)

    client.goto_base_defined_speed(0.0, 0.0, 0.0)


def odometry_loop(client: ReachyClient, odom_state: OdometryState, stop_evt: threading.Event) -> None:
    """Poll odometry from robot and update shared state."""
    dt = 1.0 / 30.0  # Poll at 30 Hz

    while not stop_evt.is_set():
        try:
            odom = client.get_mobile_odometry()
            x = odom.get("x", 0.0)
            y = odom.get("y", 0.0)
            theta = odom.get("theta", 0.0)
            odom_state.update(x, y, theta)
        except Exception as e:
            print(f"[ODOM] error: {e}")

        time.sleep(dt)


def open3d_vis_loop(odom_state: OdometryState, stop_evt: threading.Event) -> None:
    """Run Open3D visualization in main thread (non-blocking mode)."""

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Reachy Odometry", width=800, height=600)

    # Create geometries
    origin_frame = create_coordinate_frame(COORD_FRAME_SIZE)
    base_frame = create_coordinate_frame(COORD_FRAME_SIZE)

    # Create trajectory line
    trajectory_line = o3d.geometry.LineSet()
    trajectory_line.paint_uniform_color([0.0, 0.5, 1.0])  # Blue trail

    # Add ground grid (make it larger for better spatial reference)
    grid_size = 50.0  # 50 meter grid
    grid_div = 50
    lines = []
    points = []
    for i in range(grid_div + 1):
        t = -grid_size / 2 + i * grid_size / grid_div
        # Lines parallel to X
        points.append([- grid_size / 2, t, 0])
        points.append([grid_size / 2, t, 0])
        lines.append([len(points) - 2, len(points) - 1])
        # Lines parallel to Y
        points.append([t, -grid_size / 2, 0])
        points.append([t, grid_size / 2, 0])
        lines.append([len(points) - 2, len(points) - 1])

    grid_lines = o3d.geometry.LineSet()
    grid_lines.points = o3d.utility.Vector3dVector(np.array(points))
    grid_lines.lines = o3d.utility.Vector2iVector(np.array(lines))
    grid_lines.paint_uniform_color([0.5, 0.5, 0.5])

    # Add invisible bounding points to expand the scene and allow more zoom out
    # This tricks Open3D into allowing a larger zoom range
    bounding_points = o3d.geometry.PointCloud()
    bound_pts = np.array([
        [-30, -30, -10],
        [30, 30, 10],
    ])
    bounding_points.points = o3d.utility.Vector3dVector(bound_pts)
    bounding_points.paint_uniform_color([0.1, 0.1, 0.1])  # Nearly invisible

    # Add all geometries
    vis.add_geometry(origin_frame)
    vis.add_geometry(base_frame)
    vis.add_geometry(grid_lines)
    vis.add_geometry(bounding_points)  # Invisible bounds to allow more zoom
    if SHOW_TRAJECTORY:
        vis.add_geometry(trajectory_line)

    # Set up view with better initial position
    ctr = vis.get_view_control()

    # Change field of view for wider viewing angle (default is 60)
    ctr.change_field_of_view(step=20.0)  # Increase FOV

    # Set a nice initial viewpoint (top-down angled view)
    ctr.set_zoom(0.05)  # Zoom out much more for wider view
    ctr.set_front([0.3, 0.3, -0.9])  # Looking down at an angle
    ctr.set_lookat([0, 0, 0])  # Look at origin
    ctr.set_up([0, 0, 1])  # Z is up

    # Adjust render options for better viewing
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    render_opt.point_size = 2.0
    render_opt.line_width = 2.0

    # Try to set viewing frustum - check available attributes
    try:
        # Different Open3D versions use different names
        if hasattr(render_opt, 'depth_far'):
            render_opt.depth_far = 1000.0
            print(f"[VIS] Set depth_far to 1000.0")
        if hasattr(render_opt, 'depth_near'):
            render_opt.depth_near = 0.01
            print(f"[VIS] Set depth_near to 0.01")

        # Debug: print available attributes
        depth_attrs = [attr for attr in dir(render_opt) if 'depth' in attr.lower() or 'clip' in attr.lower() or 'far' in attr.lower() or 'near' in attr.lower()]
        if depth_attrs:
            print(f"[VIS] Available depth/clip attributes: {depth_attrs}")
    except Exception as e:
        print(f"[VIS] Note: Could not set depth clipping: {e}")

    dt = 1.0 / VIS_UPDATE_HZ
    prev_T = np.eye(4)  # Track previous transform

    print("\n[VIS] Open3D window opened")
    print("[VIS] Red=X, Green=Y, Blue=Z")
    print("[VIS] Gray grid = ground plane")
    print("[VIS] Blue trail = odometry trajectory")
    print("\n[VIS] Mouse controls:")
    print("  - Left drag: Rotate view")
    print("  - Right drag / Middle drag: Pan")
    print("  - Scroll: Zoom in/out")
    print("  - Ctrl+Left drag: Roll view")
    print("  - Shift+Left drag: Pan (alternative)\n")

    while not stop_evt.is_set():
        t0 = time.time()

        # Get current pose
        x, y, theta = odom_state.get_pose()
        T_base = pose_to_transform(x, y, theta)

        # Update base frame: apply inverse of previous, then new transform
        T_delta = T_base @ np.linalg.inv(prev_T)
        base_frame.transform(T_delta)
        prev_T = T_base.copy()

        # Update trajectory
        if SHOW_TRAJECTORY:
            traj = odom_state.get_trajectory()
            if len(traj) > 1:
                trajectory_line.points = o3d.utility.Vector3dVector(traj)
                lines = [[i, i + 1] for i in range(len(traj) - 1)]
                trajectory_line.lines = o3d.utility.Vector2iVector(np.array(lines))

        # Update visualization
        vis.update_geometry(base_frame)
        if SHOW_TRAJECTORY:
            vis.update_geometry(trajectory_line)

        if not vis.poll_events():
            stop_evt.set()
            break
        vis.update_renderer()

        # Pace updates
        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    vis.destroy_window()
    print("\n[VIS] Open3D window closed")


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    if reachy.mobile_base is None:
        print("[BASE] No mobile base.")
        return

    client.turn_on_all()

    # Shared state
    stop_evt = threading.Event()
    odom_state = OdometryState(lock=threading.Lock())

    # Start threads
    cam_thread = threading.Thread(target=camera_loop, args=(reachy, stop_evt), daemon=True)
    teleop_thread = threading.Thread(target=teleop_loop, args=(client, stop_evt), daemon=True)
    odom_thread = threading.Thread(target=odometry_loop, args=(client, odom_state, stop_evt), daemon=True)

    try:
        cam_thread.start()
        teleop_thread.start()
        odom_thread.start()

        # Run Open3D in main thread (blocking until window closes or ESC pressed)
        open3d_vis_loop(odom_state, stop_evt)

    finally:
        stop_evt.set()
        cam_thread.join(timeout=2.0)
        teleop_thread.join(timeout=2.0)
        odom_thread.join(timeout=2.0)
        try:
            client.goto_base_defined_speed(0.0, 0.0, 0.0)
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
