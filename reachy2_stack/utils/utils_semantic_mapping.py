#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import open3d as o3d
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from pathlib import Path

    
def _get_color_for_class(class_name: str) -> np.ndarray:
    """Deterministic color per class (extend as you add more)."""
    palette: Dict[str, np.ndarray] = {
        "cabinet": np.array([0.12156863, 0.46666667, 0.70588235]),  # blue-ish
        "fridge":  np.array([0.83921569, 0.15294118, 0.15686275]),  # red-ish
    }
    return palette.get(class_name, np.array([0.17254902, 0.62745098, 0.17254902]))  # green as default


def visualize_semantic_map_in_mesh(
    mesh_path: Path,
    semantic_db_path: Path,
    sphere_radius_front: float = 0.03,
    sphere_radius_handle: float = 0.02,
    show_pointclouds: bool = True,
    max_points_per_cloud: int | None = 5000,
) -> None:
    """
    GUI version using O3DVisualizer that shows:
      - mesh
      - front/handle spheres
      - 3D point clouds (optional)
      - a 3D text label with the index [i] next to each front
      - (NEW) handle articulation trajectories as red polylines
    """

    # ---------------- colors ----------------
    INSTANCE_COLORS = np.array([
        [0.121, 0.466, 0.705],  # blue
        [1.000, 0.498, 0.054],  # orange
        [0.172, 0.627, 0.172],  # green
        [0.839, 0.152, 0.156],  # red
        [0.580, 0.403, 0.741],  # purple
        [0.549, 0.337, 0.294],  # brown
        [0.890, 0.466, 0.760],  # pink
        [0.498, 0.498, 0.498],  # gray
        [0.737, 0.741, 0.133],  # olive
        [0.090, 0.745, 0.811],  # cyan
    ], dtype=float)

    TRAJ_COLOR = np.array([1.0, 0.0, 0.0], dtype=float)  # pure red for trajectories

    # ---------------- data load ----------------
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    data = np.load(semantic_db_path, allow_pickle=True)
    class_names = data["class_names"]
    front_positions = data["front_positions"]
    handle_positions = data["handle_positions"]
    front_points3d = data.get("front_points3d", None)
    handle_points3d = data.get("handle_points3d", None)

    # NEW: articulation trajectories (N, T, 3), optional
    articulation_traj = data.get("articulation_trajectory", None)

    N = len(front_positions)

    print("\n[visualize_semantic_map_in_mesh_with_labels] Articulated objects in DB:")
    for i in range(N):
        print(f"  [{i}] class={class_names[i]}  "
              f"front={np.round(front_positions[i], 3)}  "
              f"handle={np.round(handle_positions[i], 3)}")

    # ---------------- GUI init ----------------
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer(
        "Semantic Map in Mesh (with 3D labels + trajectories)",
        1800,
        1200,
    )

    vis.show_axes = True

    # add mesh
    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultLit"
    vis.add_geometry("scene_mesh", mesh, mesh_mat)

    # world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    frame_mat = rendering.MaterialRecord()
    frame_mat.shader = "defaultLit"
    vis.add_geometry("world_frame", world_frame, frame_mat)

    # ---------------- add instances ----------------
    for i in range(N):
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        p_front = np.asarray(front_positions[i])
        p_handle = np.asarray(handle_positions[i])

        # front sphere
        fs = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius_front)
        fs.translate(p_front)
        fs.paint_uniform_color(color)
        mat_fs = rendering.MaterialRecord()
        mat_fs.shader = "defaultLit"
        vis.add_geometry(f"front_{i}", fs, mat_fs)

        # handle sphere
        hs = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius_handle)
        hs.translate(p_handle)
        hs.paint_uniform_color(color)
        mat_hs = rendering.MaterialRecord()
        mat_hs.shader = "defaultLit"
        vis.add_geometry(f"handle_{i}", hs, mat_hs)

        # 3D label: index i, slightly offset from front center
        label_pos = p_front + np.array([0.03, 0.03, 0.03])
        vis.add_3d_label(label_pos.tolist(), f"{i}")

        # front cloud
        if show_pointclouds and front_points3d is not None:
            pts_f = np.asarray(front_points3d[i])
            if pts_f.size > 0:
                if max_points_per_cloud and pts_f.shape[0] > max_points_per_cloud:
                    idx = np.random.choice(pts_f.shape[0], max_points_per_cloud, replace=False)
                    pts_f = pts_f[idx]
                pc_f = o3d.geometry.PointCloud()
                pc_f.points = o3d.utility.Vector3dVector(pts_f)
                pc_f.paint_uniform_color(color)
                mat_pf = rendering.MaterialRecord()
                mat_pf.shader = "defaultUnlit"
                vis.add_geometry(f"front_cloud_{i}", pc_f, mat_pf)

        # handle cloud
        if show_pointclouds and handle_points3d is not None:
            pts_h = np.asarray(handle_points3d[i])
            if pts_h.size > 0:
                if max_points_per_cloud and pts_h.shape[0] > max_points_per_cloud:
                    idx = np.random.choice(pts_h.shape[0], max_points_per_cloud, replace=False)
                    pts_h = pts_h[idx]
                pc_h = o3d.geometry.PointCloud()
                pc_h.points = o3d.utility.Vector3dVector(pts_h)
                pc_h.paint_uniform_color(color * 0.85)
                mat_ph = rendering.MaterialRecord()
                mat_ph.shader = "defaultUnlit"
                vis.add_geometry(f"handle_cloud_{i}", pc_h, mat_ph)

        # -------------- NEW: handle trajectory polyline --------------
        if articulation_traj is not None:
            traj_i = np.asarray(articulation_traj[i])
            # Expect shape (T, 3) and at least 2 points, and finite
            if traj_i.ndim == 2 and traj_i.shape[0] >= 2 and traj_i.shape[1] == 3:
                finite_mask = np.isfinite(traj_i).all(axis=1)
                traj_i = traj_i[finite_mask]
                if traj_i.shape[0] >= 2:
                    # Build LineSet: connect successive points
                    num_pts = traj_i.shape[0]
                    lines = np.column_stack([
                        np.arange(num_pts - 1),
                        np.arange(1, num_pts)
                    ]).astype(np.int32)

                    line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(traj_i),
                        lines=o3d.utility.Vector2iVector(lines),
                    )
                    colors = np.tile(TRAJ_COLOR, (lines.shape[0], 1))
                    line_set.colors = o3d.utility.Vector3dVector(colors)

                    mat_traj = rendering.MaterialRecord()
                    mat_traj.shader = "unlitLine"
                    # thickness only works in some backends, but fine to set
                    mat_traj.line_width = 2.0

                    vis.add_geometry(f"traj_{i}", line_set, mat_traj)

    vis.reset_camera_to_default()

    # O3DVisualizer *is* the window; add it to the app directly
    app.add_window(vis)
    app.run()


def _mask_to_points_from_xyz(
    mask: np.ndarray,
    xyz: np.ndarray,
    max_points: int | None = 5000,
) -> np.ndarray:
    """
    Lift a 2D binary mask into a set of 3D points using the XYZcut map.

    Returns:
        points_world: (M, 3) world coords, optionally downsampled to max_points.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pts = xyz[ys, xs, :]  # (N, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if max_points is not None and pts.shape[0] > max_points:
        # random subsample for memory
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    return pts.astype(np.float32)


def _bbox_from_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned 3D bounding box from (N,3) points."""
    if points.shape[0] == 0:
        return np.zeros(3), np.zeros(3)
    return points.min(axis=0), points.max(axis=0)


def _iou3d(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    """IoU of two axis-aligned 3D boxes."""
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_sizes = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_sizes))

    vol_a = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    vol_b = float(np.prod(np.maximum(b_max - b_min, 0.0)))
    union = vol_a + vol_b - inter_vol
    if union <= 0.0:
        return 0.0
    return inter_vol / union


def estimate_plane_normal(points: np.ndarray) -> np.ndarray:
    """
    Estimate unit plane normal from 3D points via PCA/SVD.
    points: (N, 3)

    Returns:
        normal: (3,) unit vector. If not enough points, returns NaNs.
    """
    if points.shape[0] < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    pts = points.astype(float)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    # center
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid

    # covariance
    cov = pts_centered.T @ pts_centered / max(pts_centered.shape[0] - 1, 1)

    # eigen-decomposition: smallest eigenvalue → normal
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]

    # normalize
    n = np.linalg.norm(normal)
    if n < 1e-8:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    normal = normal / n
    return normal

def estimate_handle_longest_axis(points: np.ndarray) -> np.ndarray:
    """
    Estimate the main (longest) axis of a handle from its 3D points.

    Args:
        points: (N, 3) array of 3D points on/around the handle in world coordinates.

    Returns:
        axis: (3,) unit vector corresponding to the dominant direction of the handle.
              If estimation fails (too few / invalid points), returns NaNs.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got {points.shape}")

    if points.shape[0] < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    pts = points.astype(float)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    # center
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid

    # covariance
    cov = pts_centered.T @ pts_centered / max(pts_centered.shape[0] - 1, 1)

    # eigen-decomposition: largest eigenvalue → longest axis
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]

    # normalize
    n = np.linalg.norm(principal_axis)
    if n < 1e-8:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    principal_axis = principal_axis / n
    return principal_axis

def compute_articulation_geometry(
    front_points: np.ndarray,
    handle_centroid: np.ndarray,
    front_normal: np.ndarray,
) -> Dict[str, Any]:
    """
    Geometric descriptors for a front + handle pair.
    Computes articulation type (prismatic vs revolute) and generates 
    a 20-step opening trajectory in world coordinates.
    """

    # 1. Filter points: get all front points with same z as handle centroid (slice)
    z_handle = handle_centroid[2]
    # Tolerance can be adjusted based on point cloud density
    relevant_points = front_points[np.abs(front_points[:, 2] - z_handle) < 0.02] 

    # Fallback if slice is empty (use all points projected to Z plane)
    if len(relevant_points) < 10:
        relevant_points = front_points.copy()
        relevant_points[:, 2] = z_handle

    vert = np.array([0, 0, 1])  # World Z-axis
    normal = front_normal / (np.linalg.norm(front_normal) + 1e-8)

    # 2. Project handle centroid onto the plane of the front
    # p0 is the centroid of the slice
    p0 = relevant_points.mean(axis=0)
    distance_plane = np.dot(normal, handle_centroid - p0)
    handle_centroid_proj = handle_centroid - distance_plane * normal

    # 3. Define Lateral Directions (Left/Right on the face)
    # Vector orthogonal to both Up(Z) and Normal -> Horizontal tangent
    direction_to_bound_1 = np.cross(vert, normal)
    direction_to_bound_1 /= np.linalg.norm(direction_to_bound_1) + 1e-8

    # direction to bound 2 is opposite
    direction_to_bound_2 = -direction_to_bound_1

    # 4. Find Spatial Bounds of the front face
    # Project front points onto these tangent vectors
    projections_1 = (relevant_points - p0) @ direction_to_bound_1
    projections_2 = (relevant_points - p0) @ direction_to_bound_2

    idx_left = np.argmax(projections_1)
    idx_right = np.argmax(projections_2)

    # These are the 3D coordinates of the left/right edges at handle height
    # Note: "Left" here is relative to the cross product direction
    bound_1_pt = relevant_points[idx_left]
    bound_2_pt = relevant_points[idx_right]

    # 5. Calculate distances from Handle Projection to edges
    # We project bounds to the same plane to be safe, though they should be close
    dist_1 = np.linalg.norm(bound_1_pt - handle_centroid_proj)
    dist_2 = np.linalg.norm(bound_2_pt - handle_centroid_proj)

    # Calculate Eccentricity to classify Drawer vs Door
    # If handle is roughly centered, ratio is ~1.0. If offset, ratio >> 1.0
    if dist_1 > dist_2:
        eccentricity = dist_1 / (dist_2 + 1e-6)
    else:
        eccentricity = dist_2 / (dist_1 + 1e-6)

    print(f"Eccentricity: {eccentricity:.3f}")

    # Threshold: > 2.0 implies handle is clearly on one side -> Revolute (Door)
    # < 2.0 implies handle is somewhat centered -> Prismatic (Drawer)
    if eccentricity < 2.0:
        joint_type = "prismatic"
    else:
        joint_type = "revolute"

    # -------------------------------------------------------
    # Trajectory Generation
    # -------------------------------------------------------
    handle_trajectory = []
    
    # Number of steps for the animation/visualization path
    num_steps = 20
    
    if joint_type == "prismatic":
        axis = front_normal
        hinge_position = handle_centroid # Virtual "origin" for a drawer is the handle itself
        
        # Drawers open OUT along the normal
        max_extension = 0.4  # meters
        
        start_pos = handle_centroid
        end_pos = handle_centroid + (normal * max_extension)
        
        for t in np.linspace(0.0, 1.0, num=num_steps):
            pos = start_pos * (1 - t) + end_pos * t
            handle_trajectory.append(pos)

    else: # Revolute
        # 1. Identify Hinge Location
        # The hinge is on the side FARTHEST from the handle.
        if dist_1 > dist_2:
            # Handle is far from Bound 1 -> Hinge is at Bound 1
            hinge_position = bound_1_pt
            # We are closer to Bound 2 (Handle side).
            # To open OUT, we need to check rotation direction.
            # Standard convention: check cross product or just geometric intuition.
            # If hinge is "Left" (relative to face), we usually rotate CCW (+).
            # If hinge is "Right", we usually rotate CW (-).
            # Here we use the sign of the cross product with normal to determine sign.
            
            # Heuristic: Rotate such that the first step moves roughly along the normal
            rotation_sign = 1.0 
        else:
            # Handle is far from Bound 2 -> Hinge is at Bound 2
            hinge_position = bound_2_pt
            rotation_sign = -1.0

        # Refine Hinge Z to be exactly Handle Z (pure Z rotation)
        hinge_position[2] = z_handle
        axis = vert # [0, 0, 1]

        # Lever arm vector (Hinge -> Handle)
        radius_vec = handle_centroid - hinge_position
        radius_len = np.linalg.norm(radius_vec)

        # Check "Opening" direction
        # We want the tangent at t=0 to align with positive normal (opening out)
        # Tangent of rotation = Axis x Radius
        tangent = np.cross(axis * rotation_sign, radius_vec)
        if np.dot(tangent, normal) < 0:
            # If the computed rotation moves 'into' the object, flip sign
            rotation_sign *= -1.0
            
        max_angle_deg = 90.0
        max_angle_rad = np.radians(max_angle_deg) * rotation_sign

        # Generate Arc
        for t in np.linspace(0.0, 1.0, num=num_steps):
            theta = max_angle_rad * t
            
            # Rotation Matrix around Z axis
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])
            
            # Apply rotation to the radius vector, then add hinge offset
            rotated_radius = R @ radius_vec
            pos = hinge_position + rotated_radius
            handle_trajectory.append(pos)

    return {
        "type": joint_type,
        "axis": axis,  # Direction vector of axis
        "origin": hinge_position, # Point on axis
        "trajectory": np.array(handle_trajectory), # (20, 3)
        "eccentricity": eccentricity
    }

def compute_articulation_geometry_old(
    front_points: np.ndarray,
    handle_centroid: np.ndarray,
    front_normal: np.ndarray,
) -> Dict[str, Any]:
    """
    Geometric descriptors for a front + handle pair.

    """

    
    # get all front points with same z as handle centroid
    z_handle = handle_centroid[2]
    front_points = front_points[np.abs(front_points[:, 2] - z_handle) < 0.005]

    vert = np.array([0, 0, 1])  # z-axis
    normal = front_normal

    # project handle centroid onto the plane of the front
    p0 = front_points.mean(axis=0)
    distance = np.dot(normal, handle_centroid - p0)
    handle_centroid_proj = handle_centroid - distance * normal

    # vector orthogonal to both (direction from front center to bound 1)
    direction_to_bound_1 = np.cross(vert, normal)
    direction_to_bound_1 /= np.linalg.norm(direction_to_bound_1) + 1e-8

    # direction to bound 2 is opposite
    direction_to_bound_2 = -direction_to_bound_1

    # project front points onto the two directions to find bounds
    projections_1 = (front_points - p0) @ direction_to_bound_1
    projections_2 = (front_points - p0) @ direction_to_bound_2

    idx_left = np.argmax(projections_1)
    idx_right = np.argmax(projections_2)

    left_bound = front_points[idx_left]
    right_bound = front_points[idx_right]

    v_left = left_bound - handle_centroid_proj
    # v_left[2] = 0.0
    # v_left /= np.linalg.norm(v_left) + 1e-8

    v_right = right_bound - handle_centroid_proj
    # v_right[2] = 0.0
    # v_right /= np.linalg.norm(v_right) + 1e-8

    dist_right = np.linalg.norm(v_right)
    dist_left = np.linalg.norm(v_left)

    print("dist_right:", dist_right)
    print("dist_left:", dist_left)

    eccentricity = dist_right / dist_left if dist_left > 1e-4 else np.nan
    if eccentricity < 1.0:
        eccentricity = 1.0 / eccentricity
    print("eccentricity:", eccentricity)
    print("##")

    if eccentricity < 2:
        type = "prismatic"
    else:
        type = "revolute"

    # # estimate articulation axis
    # handle_trajectory = []
    # if type == "prismatic":
    #     axis = front_normal
    #     position = handle_centroid
    #     interaction_point = handle_centroid
    #     max_interaction_dist = 0.2
    #     max_interaction_angle = None
    #     lever = None
    #     handle_closed_position = handle_centroid 
    #     handle_open_position = handle_centroid + front_normal * max_interaction_dist
    #     # trajectory over 20 steps
    #     for t in np.linspace(0.0, 1.0, num=20):
    #         pos = handle_closed_position * (1 - t) + handle_open_position * t
    #         handle_trajectory.append(pos)
    # else:
    #     # far bound determins the position of the axis
    #     position = right_bound if dist_right > dist_left else left_bound
    #     axis = vert
    #     interaction_point = handle_centroid
    #     max_interaction_angle = 90.0
    #     max_interaction_dist = None
    #     # lever from axis to handle centroid projection
    #     lever = handle_centroid_proj - position
    #     handle_closed_position = handle_centroid
    #     # open position is rotated 90 degrees around axis



        a = 2

    






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True, help="Path to scene mesh (e.g. .ply)")
    parser.add_argument("--semantic-db", type=Path, required=True, help="Path to semantic_map.npz")
    args = parser.parse_args()

    visualize_semantic_map_in_mesh(args.mesh, args.semantic_db)
