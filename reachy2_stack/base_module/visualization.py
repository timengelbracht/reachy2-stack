"""Open3D 3D visualization for odometry and trajectory."""

import time
import threading

import numpy as np
import open3d as o3d

from .odometry import OdometryState
from .utils import create_coordinate_frame, pose_to_transform


def open3d_vis_loop(
    odom_state: OdometryState,
    stop_evt: threading.Event,
    vis_update_hz: float = 10.0,
    show_trajectory: bool = True,
    show_camera: bool = True,
    show_pointcloud: bool = False,
    coord_frame_size: float = 0.3,
    grid_size: float = 50.0,
    grid_div: int = 50,
) -> None:
    """Run Open3D visualization in main thread (non-blocking mode).

    Args:
        odom_state: Shared OdometryState object
        stop_evt: Threading event to signal loop termination
        vis_update_hz: Update rate for visualization in Hz
        show_trajectory: Show odometry trajectory trail
        show_camera: Show camera coordinate frame
        show_pointcloud: Show point cloud from RGBD
        coord_frame_size: Size of coordinate frame axes
        grid_size: Size of ground grid in meters
        grid_div: Number of grid divisions
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Reachy Odometry", width=800, height=600)

    # Create geometries
    origin_frame = create_coordinate_frame(coord_frame_size)
    base_frame = create_coordinate_frame(coord_frame_size)
    camera_frame = create_coordinate_frame(coord_frame_size * 0.7)  # Slightly smaller for distinction

    # Create trajectory line
    trajectory_line = o3d.geometry.LineSet()
    # paint in hightlight yellow
    trajectory_line.paint_uniform_color([0.0, 0.5, 0.5])  # Cyan color for trajectory

    # Create point cloud geometry (initially with a single point at origin)
    # This ensures Open3D can properly update it later
    pointcloud_geom = None
    if show_pointcloud:
        print("creating pointcloud geometry")
        pointcloud_geom = o3d.geometry.PointCloud()
        # Add a dummy point at origin so geometry is not empty
        pointcloud_geom.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
        pointcloud_geom.colors = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]]))

    # Add ground grid
    lines = []
    points = []
    height = -0.5  # Slightly below origin
    for i in range(grid_div + 1):
        t = -grid_size / 2 + i * grid_size / grid_div
        # Lines parallel to X
        points.append([-grid_size / 2, t, height])
        points.append([grid_size / 2, t, height])
        lines.append([len(points) - 2, len(points) - 1])
        # Lines parallel to Y
        points.append([t, -grid_size / 2, height])
        points.append([t, grid_size / 2, height])
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
    if show_camera:
        vis.add_geometry(camera_frame)
    vis.add_geometry(grid_lines)
    vis.add_geometry(bounding_points)  # Invisible bounds to allow more zoom
    if show_trajectory:
        vis.add_geometry(trajectory_line)
    if show_pointcloud and pointcloud_geom is not None:
        print("showing pointcloud")
        vis.add_geometry(pointcloud_geom)

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

    dt = 1.0 / vis_update_hz
    prev_T = np.eye(4)  # Track previous base transform
    prev_T_cam = np.eye(4)  # Track previous camera transform

    print("\n[VIS] Open3D window opened")
    print("[VIS] Red=X, Green=Y, Blue=Z")
    print("[VIS] Gray grid = ground plane")
    print("[VIS] Cyan trail = odometry trajectory")
    if show_camera:
        print("[VIS] Small frame = camera")
    if show_pointcloud:
        print("[VIS] Point cloud = RGBD camera data in world frame")
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
        T_delta_base = T_base @ np.linalg.inv(prev_T)
        base_frame.transform(T_delta_base)
        prev_T = T_base.copy()

        # Update camera frame
        if show_camera:
            T_world_cam = odom_state.get_camera_pose()
            T_delta_cam = T_world_cam @ np.linalg.inv(prev_T_cam)
            camera_frame.transform(T_delta_cam)
            prev_T_cam = T_world_cam.copy()

        # Update trajectory
        if show_trajectory:
            traj = odom_state.get_trajectory()
            if len(traj) > 1:
                trajectory_line.points = o3d.utility.Vector3dVector(traj)
                lines_traj = [[i, i + 1] for i in range(len(traj) - 1)]
                trajectory_line.lines = o3d.utility.Vector2iVector(np.array(lines_traj))
                # Reapply color after updating geometry (Open3D may reset it)
                colors = [[0.0, 0.5, 0.5] for _ in range(len(lines_traj))]  # Cyan for each line segment
                trajectory_line.colors = o3d.utility.Vector3dVector(colors)

        # Update point cloud
        if show_pointcloud and pointcloud_geom is not None:
            print("updating pointcloud")
            pcd_world = odom_state.get_pointcloud()
            if pcd_world is not None and len(pcd_world.points) > 0:
                print(f"  Point cloud has {len(pcd_world.points)} points")
                pointcloud_geom.points = pcd_world.points
                pointcloud_geom.colors = pcd_world.colors

        # Update visualization
        vis.update_geometry(base_frame)
        if show_camera:
            vis.update_geometry(camera_frame)
        if show_trajectory:
            vis.update_geometry(trajectory_line)
        if show_pointcloud and pointcloud_geom is not None:
            vis.update_geometry(pointcloud_geom)

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
