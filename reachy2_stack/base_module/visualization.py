"""Open3D 3D visualization for odometry and trajectory."""

import time
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent

from .utils import rgbd_to_pointcloud


def vis_process(
    vis_queue: "MPQueue",
    stop_event: "MPEvent",
    fps: int = 30,
    show_trajectory: bool = True,
    show_pointcloud: bool = True,
    show_camera_frames: bool = True,
    show_images: bool = True,
    coord_frame_size: float = 0.5,
    grid_size: float = 50.0,
    grid_div: int = 50,
    depth_scale: float = 0.001,
    depth_trunc: float = 3.5,
) -> None:
    """Visualization in a separate process.

    Reads from vis_queue and renders with Open3D (3D) and Matplotlib (images).
    This function is designed to run in a separate process
    (via multiprocessing.Process) so it has its own GIL and
    doesn't block I/O threads.

    Supports both:
    - RobotState objects (new unified format with camera calibration)
    - Dict format (backward compatible with existing code)

    Args:
        vis_queue: multiprocessing.Queue containing either:
            - RobotState objects (with full camera calibration)
            - dicts with:
                - 'odom_x', 'odom_y', 'odom_theta': odometry updates
                - 'occupied_points', 'occupied_colors': occupied voxels
                - 'free_points': free voxels
                - 'points', 'colors': generic point cloud
        stop_event: multiprocessing.Event to signal shutdown
        fps: Target rendering frame rate
        show_trajectory: Show odometry trajectory trail
        show_pointcloud: Show point clouds from RGBD (in world frame)
        show_camera_frames: Show coordinate frames for all cameras
        show_images: Show camera images (RGB, depth, teleop) with Matplotlib
        coord_frame_size: Size of coordinate frame axes
        grid_size: Size of ground grid in meters
        grid_div: Number of grid divisions
        depth_scale: Scale factor for depth (0.001 if depth in mm)
        depth_trunc: Maximum depth to include in meters
    """
    # Import inside process to avoid serialization issues
    import open3d as o3d
    import numpy as np
    import cv2
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
    import matplotlib.pyplot as plt
    from .robot_state import RobotState
    from .map_state import MapState

    print("[VIS PROCESS] Starting visualization process")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Reachy Visualization", width=1280, height=720)

    # Create geometries
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size)
    robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size)

    # Camera frames (smaller to distinguish from robot frame)
    depth_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size * 0.6)
    teleop_left_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size * 0.4)
    teleop_right_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size * 0.4)

    traj_pcd = o3d.geometry.PointCloud()
    occ_pcd = o3d.geometry.PointCloud()
    free_pcd = o3d.geometry.PointCloud()

    # Create ground grid
    lines = []
    points = []
    height = -0.1
    for i in range(grid_div + 1):
        t = -grid_size / 2 + i * grid_size / grid_div
        points.append([-grid_size / 2, t, height])
        points.append([grid_size / 2, t, height])
        lines.append([len(points) - 2, len(points) - 1])
        points.append([t, -grid_size / 2, height])
        points.append([t, grid_size / 2, height])
        lines.append([len(points) - 2, len(points) - 1])

    grid_lines = o3d.geometry.LineSet()
    grid_lines.points = o3d.utility.Vector3dVector(np.array(points))
    grid_lines.lines = o3d.utility.Vector2iVector(np.array(lines))
    grid_lines.paint_uniform_color([0.3, 0.3, 0.3])

    # Add geometries
    vis.add_geometry(origin_frame)
    vis.add_geometry(robot_frame)
    vis.add_geometry(grid_lines)
    if show_camera_frames:
        vis.add_geometry(depth_cam_frame)
        vis.add_geometry(teleop_left_frame)
        vis.add_geometry(teleop_right_frame)
    if show_trajectory:
        vis.add_geometry(traj_pcd)
    if show_pointcloud:
        vis.add_geometry(occ_pcd)
        vis.add_geometry(free_pcd)

    # Set up view
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=20.0)
    ctr.set_zoom(0.1)
    ctr.set_front([0.3, 0.3, 0.9])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])

    # Render options
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
    render_opt.point_size = 2.0

    trajectory = []
    interval = 1.0 / fps

    # Track previous transforms for incremental updates
    prev_robot_T = np.eye(4)
    prev_depth_T = np.eye(4)
    prev_teleop_left_T = np.eye(4)
    prev_teleop_right_T = np.eye(4)

    # Set up matplotlib figure for camera images (2x2 grid)
    fig = None
    axs = None
    im_rgb = None
    im_depth = None
    im_teleop_left = None
    im_teleop_right = None

    if show_images:
        plt.ion()  # Enable interactive mode
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.canvas.manager.set_window_title("Camera Views")

        # Initialize with placeholder images
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder_depth = np.zeros((480, 640), dtype=np.uint8)

        axs[0, 0].set_title("RGB (Depth Camera)")
        im_rgb = axs[0, 0].imshow(placeholder)
        axs[0, 0].axis('off')

        axs[0, 1].set_title("Depth")
        im_depth = axs[0, 1].imshow(placeholder_depth, cmap='jet', vmin=0, vmax=255)
        axs[0, 1].axis('off')

        axs[1, 0].set_title("Teleop Left")
        im_teleop_left = axs[1, 0].imshow(placeholder)
        axs[1, 0].axis('off')

        axs[1, 1].set_title("Teleop Right")
        im_teleop_right = axs[1, 1].imshow(placeholder)
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    print("[VIS PROCESS] Visualization ready")
    print("[VIS PROCESS] Robot frame = large, Depth cam = medium, Teleop cams = small")
    if show_images:
        print("[VIS PROCESS] Matplotlib image window enabled (close window to quit)")

    while not stop_event.is_set():
        t0 = time.time()

        # Drain queue (get all pending updates)
        latest_state = None
        latest_map_state = None
        while True:
            try:
                data = vis_queue.get_nowait()

                # Check if it's a RobotState object
                if isinstance(data, RobotState):
                    latest_state = data
                    # Also update trajectory from RobotState
                    trajectory.append([data.odom_x, data.odom_y, 0.0])
                    if len(trajectory) > 500:
                        trajectory.pop(0)

                # Check if it's a MapState object
                elif isinstance(data, MapState):
                    latest_map_state = data
                    # Update trajectory from MapState's robot_state if available
                    if data.robot_state is not None:
                        trajectory.append([data.robot_state.odom_x, data.robot_state.odom_y, 0.0])
                        if len(trajectory) > 500:
                            trajectory.pop(0)

                    # Update occupied point cloud from wavemap server
                    if data.has_map():
                        occ_pcd.points = o3d.utility.Vector3dVector(data.occupied_points)
                        if data.occupied_colors is not None and len(data.occupied_colors) > 0:
                            colors = data.occupied_colors.astype(np.float64) / 255.0
                            occ_pcd.colors = o3d.utility.Vector3dVector(colors)
                        else:
                            occ_pcd.paint_uniform_color([0.0, 1.0, 0.0])

                    # Update free point cloud from wavemap server
                    if data.has_free_space():
                        free_pcd.points = o3d.utility.Vector3dVector(data.free_points)
                        free_pcd.paint_uniform_color([0.5, 0.5, 0.5])

                else:
                    # Dict format (backward compatibility)
                    if "odom_x" in data and data["odom_x"] is not None:
                        trajectory.append([data["odom_x"], data["odom_y"], 0.0])
                        if len(trajectory) > 500:
                            trajectory.pop(0)

                        # Update robot frame position
                        theta_rad = np.deg2rad(data.get("odom_theta", 0))
                        new_T = np.eye(4)
                        new_T[0, 0] = np.cos(theta_rad)
                        new_T[0, 1] = -np.sin(theta_rad)
                        new_T[1, 0] = np.sin(theta_rad)
                        new_T[1, 1] = np.cos(theta_rad)
                        new_T[0, 3] = data["odom_x"]
                        new_T[1, 3] = data["odom_y"]

                        T_delta = new_T @ np.linalg.inv(prev_robot_T)
                        robot_frame.transform(T_delta)
                        prev_robot_T = new_T.copy()

                    # Update occupied point cloud (from external mapping)
                    if "occupied_points" in data and data["occupied_points"] is not None:
                        pts = data["occupied_points"]
                        if len(pts) > 0:
                            occ_pcd.points = o3d.utility.Vector3dVector(pts)
                            if "occupied_colors" in data and data["occupied_colors"] is not None:
                                colors = data["occupied_colors"]
                                if colors.dtype == np.uint8:
                                    colors = colors.astype(np.float64) / 255.0
                                occ_pcd.colors = o3d.utility.Vector3dVector(colors)
                            else:
                                occ_pcd.paint_uniform_color([0.0, 1.0, 0.0])

                    # Update free point cloud
                    if "free_points" in data and data["free_points"] is not None:
                        pts = data["free_points"]
                        if len(pts) > 0:
                            free_pcd.points = o3d.utility.Vector3dVector(pts)
                            free_pcd.paint_uniform_color([0.5, 0.5, 0.5])

                    # Generic point cloud (from local mapping mode)
                    if "points" in data and data["points"] is not None:
                        pts = data["points"]
                        if len(pts) > 0:
                            occ_pcd.points = o3d.utility.Vector3dVector(pts)
                            if "colors" in data and data["colors"] is not None:
                                occ_pcd.colors = o3d.utility.Vector3dVector(data["colors"])
                            else:
                                occ_pcd.paint_uniform_color([0.0, 1.0, 0.0])

            except Exception:
                # Queue empty or error
                break

        # If we got a MapState with robot_state, use that for frame updates
        if latest_map_state is not None and latest_map_state.robot_state is not None:
            if latest_state is None:
                latest_state = latest_map_state.robot_state

        # Process latest RobotState if available
        if latest_state is not None:
            state = latest_state

            # Update robot base frame
            T_odom_base = state.get_T_odom_base()
            T_delta = T_odom_base @ np.linalg.inv(prev_robot_T)
            robot_frame.transform(T_delta)
            prev_robot_T = T_odom_base.copy()

            # Update camera frames
            if show_camera_frames:
                # Depth camera frame
                T_odom_depth = state.get_T_odom_depth_cam()
                if T_odom_depth is not None:
                    T_delta = T_odom_depth @ np.linalg.inv(prev_depth_T)
                    depth_cam_frame.transform(T_delta)
                    prev_depth_T = T_odom_depth.copy()

                # Teleop left camera frame
                T_odom_teleop_left = state.get_T_odom_teleop_left()
                if T_odom_teleop_left is not None:
                    T_delta = T_odom_teleop_left @ np.linalg.inv(prev_teleop_left_T)
                    teleop_left_frame.transform(T_delta)
                    prev_teleop_left_T = T_odom_teleop_left.copy()

                # Teleop right camera frame
                T_odom_teleop_right = state.get_T_odom_teleop_right()
                if T_odom_teleop_right is not None:
                    T_delta = T_odom_teleop_right @ np.linalg.inv(prev_teleop_right_T)
                    teleop_right_frame.transform(T_delta)
                    prev_teleop_right_T = T_odom_teleop_right.copy()

            # Unproject RGBD to point cloud in world frame
            if show_pointcloud and state.has_depth() and state.has_depth_calibration():
                try:
                    # Build intrinsics dict for rgbd_to_pointcloud
                    K = state.depth_intrinsics
                    intrinsics_dict = {
                        "K": K,
                        "width": state.rgb.shape[1],
                        "height": state.rgb.shape[0],
                    }

                    # Create point cloud in camera frame
                    pcd_camera = rgbd_to_pointcloud(
                        state.rgb,
                        state.depth,
                        intrinsics_dict,
                        depth_scale=depth_scale,
                        depth_trunc=depth_trunc,
                    )

                    # Transform to odom/world frame
                    T_odom_depth = state.get_T_odom_depth_cam()
                    if T_odom_depth is not None and len(pcd_camera.points) > 0:
                        pcd_camera.transform(T_odom_depth)
                        occ_pcd.points = pcd_camera.points
                        occ_pcd.colors = pcd_camera.colors

                except Exception:
                    # Silently ignore point cloud errors (may happen on first frames)
                    pass

            # Display camera images with Matplotlib (single canvas)
            if show_images and fig is not None:
                try:
                    # RGB image from depth camera (images come in BGR, convert to RGB)
                    if state.rgb is not None:
                        rgb_display = state.rgb
                        if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
                            # Convert BGR to RGB for matplotlib
                            rgb_rgb = cv2.cvtColor(rgb_display, cv2.COLOR_BGR2RGB)
                        else:
                            rgb_rgb = rgb_display
                        im_rgb.set_data(rgb_rgb)

                    # Depth image (colorized)
                    if state.depth is not None:
                        depth_img = state.depth.copy()
                        # Apply depth scale to get meters
                        if depth_scale != 1.0:
                            depth_meters = depth_img * depth_scale
                        else:
                            depth_meters = depth_img
                        # Normalize for visualization (0 to depth_trunc meters)
                        depth_norm = np.clip(depth_meters / depth_trunc, 0, 1)
                        depth_uint8 = (depth_norm * 255).astype(np.uint8)
                        im_depth.set_data(depth_uint8)

                    # Teleop left camera (BGR to RGB)
                    if state.teleop_left is not None:
                        teleop_left_display = state.teleop_left
                        if len(teleop_left_display.shape) == 3 and teleop_left_display.shape[2] == 3:
                            teleop_left_rgb = cv2.cvtColor(teleop_left_display, cv2.COLOR_BGR2RGB)
                        else:
                            teleop_left_rgb = teleop_left_display
                        im_teleop_left.set_data(teleop_left_rgb)

                    # Teleop right camera (BGR to RGB)
                    if state.teleop_right is not None:
                        teleop_right_display = state.teleop_right
                        if len(teleop_right_display.shape) == 3 and teleop_right_display.shape[2] == 3:
                            teleop_right_rgb = cv2.cvtColor(teleop_right_display, cv2.COLOR_BGR2RGB)
                        else:
                            teleop_right_rgb = teleop_right_display
                        im_teleop_right.set_data(teleop_right_rgb)

                    # Update matplotlib figure
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                    # Check if figure was closed
                    if not plt.fignum_exists(fig.number):
                        stop_event.set()
                        break

                except Exception:
                    # Ignore matplotlib errors
                    pass

        # Update trajectory visualization
        if show_trajectory and len(trajectory) > 0:
            traj_pcd.points = o3d.utility.Vector3dVector(trajectory)
            traj_pcd.paint_uniform_color([0.0, 0.5, 0.5])  # Cyan

        # Update visualization
        vis.update_geometry(robot_frame)
        if show_camera_frames:
            vis.update_geometry(depth_cam_frame)
            vis.update_geometry(teleop_left_frame)
            vis.update_geometry(teleop_right_frame)
        if show_trajectory:
            vis.update_geometry(traj_pcd)
        if show_pointcloud:
            vis.update_geometry(occ_pcd)
            vis.update_geometry(free_pcd)

        if not vis.poll_events():
            stop_event.set()
            break
        vis.update_renderer()

        # Rate limit
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    vis.destroy_window()
    if show_images and fig is not None:
        plt.close(fig)
    print("[VIS PROCESS] Visualization process stopped")
