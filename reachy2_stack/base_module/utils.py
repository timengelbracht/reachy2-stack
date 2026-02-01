"""Utility functions for coordinate transformations and visualization."""

import numpy as np
import open3d as o3d


def create_coordinate_frame(size: float = 1.0) -> o3d.geometry.TriangleMesh:
    """Create a coordinate frame mesh (RGB = XYZ).

    Args:
        size: Size of the coordinate frame axes

    Returns:
        Open3D TriangleMesh representing coordinate frame
    """
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict,
    depth_scale: float = 1.0,
    depth_trunc: float = 3.0,
) -> o3d.geometry.PointCloud:
    """Convert RGBD image to Open3D point cloud.

    Args:
        rgb: RGB image (H, W, 3) in BGR format
        depth: Depth image (H, W) in meters or mm
        intrinsics: Camera intrinsics dict with 'K', 'width', 'height'
        depth_scale: Scale factor to convert depth to meters (1.0 if already in meters)
        depth_trunc: Maximum depth value to include (in meters)

    Returns:
        Open3D PointCloud in camera frame
    """
    # Convert BGR to RGB
    if rgb.shape[2] == 3:
        rgb_image = rgb[:, :, ::-1]  # BGR to RGB
    else:
        rgb_image = rgb

    # Create Open3D images
    o3d_rgb = o3d.geometry.Image(rgb_image.astype(np.uint8))
    o3d_depth = o3d.geometry.Image((depth * depth_scale).astype(np.float32))

    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb,
        o3d_depth,
        depth_scale=1.0,  # Already scaled above
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    # Get intrinsics
    K = intrinsics.get("K", np.eye(3))
    width = intrinsics.get("width", rgb.shape[1])
    height = intrinsics.get("height", rgb.shape[0])

    # Create Open3D camera intrinsic
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
    )

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsic)

    return pcd


def pose_to_transform(x: float, y: float, theta_deg: float) -> np.ndarray:
    """Convert 2D pose (x, y, theta) to 4x4 transform matrix.

    Args:
        x: X position in meters
        y: Y position in meters
        theta_deg: Rotation angle in degrees

    Returns:
        4x4 homogeneous transformation matrix
    """
    theta = np.deg2rad(theta_deg)
    T = np.eye(4)
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta)
    T[0, 3] = x
    T[1, 3] = y
    return T
