import numpy as np
import open3d as o3d
from pathlib import Path
from reachy2_stack.infra.world_model import WorldModel
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from reachy2_stack.utils.utils_testing import create_camera_frustum

def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def pretty_mat(name: str, T: np.ndarray) -> None:
    """Nicely print a 4x4 transform matrix."""
    print(f"{name}:")
    with np.printoptions(precision=3, suppress=True):
        print(T)
    print()


def create_frame(T_world_X: np.ndarray, size: float = 0.2, name: str = ""):
    """
    Create a coordinate frame mesh transformed by T_world_X (world ← X).
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T_world_X)
    return frame

def visualize_world_model_in_mesh(
    mesh_path: Path,
    world_model: WorldModel,
) -> None:
    """
    Visualize:
      - world & base frames
      - cameras as frustums
      - end-effectors as frames
      - mesh (if available)
    """
    print_header("VISUALIZER: Setting up Open3D scene")

    # ------------------------------------------------------
    # Load mesh (optional)
    # ------------------------------------------------------
    mesh = None
    if mesh_path is not None and mesh_path.exists():
        print(f"[VISUALIZER] Loading mesh: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
    else:
        print(f"[VISUALIZER] Mesh path not found: {mesh_path} (continuing without mesh)")

    # ------------------------------------------------------
    # Get transforms from WorldModel
    # ------------------------------------------------------
    T_world_base = world_model.get_T_world_base()
    pretty_mat("T_world_base (world ← base)", T_world_base)

    # ------------------------------------------------------
    # Setup GUI
    # ------------------------------------------------------
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("WorldModel View", 1600, 1000)
    vis.show_axes = True

    # --- Add mesh ---
    if mesh is not None:
        mesh_mat = rendering.MaterialRecord()
        mesh_mat.shader = "defaultLit"
        vis.add_geometry("scene_mesh", mesh, mesh_mat)

    # --- Add world origin frame ---
    frame_world = create_frame(np.eye(4), size=0.25)
    mat_world = rendering.MaterialRecord()
    mat_world.shader = "defaultLit"
    vis.add_geometry("frame_world", frame_world, mat_world)

    # --- Add base frame ---
    frame_base = create_frame(T_world_base, size=0.25)
    mat_base = rendering.MaterialRecord()
    mat_base.shader = "defaultLit"
    vis.add_geometry("frame_base", frame_base, mat_base)
    vis.add_3d_label(T_world_base[:3, 3].tolist(), "base")

    # ------------------------------------------------------
    # Cameras: teleop_left, teleop_right, depth_rgb
    # ------------------------------------------------------
    print_header("VISUALIZER: Adding cameras")

    camera_ids = ["teleop_left", "teleop_right", "depth_rgb"]
    colors = {
        "teleop_left":  [1.0, 0.0, 0.0],  # red
        "teleop_right": [0.0, 0.0, 1.0],  # blue
        "depth_rgb":    [0.0, 1.0, 0.0],  # green
    }

    for cam_id in camera_ids:
        T_world_cam = world_model.get_T_world_cam(cam_id)
        if T_world_cam is None:
            print(f"[VISUALIZER] Camera '{cam_id}' not present in world model, skipping.")
            continue

        K = world_model.get_intrinsics(cam_id)
        if K is None:
            print(f"[VISUALIZER] Camera '{cam_id}' has no intrinsics, skipping.")
            continue

        image_size = world_model.get_image_size(cam_id)
        if image_size is None:
            # Fallback: estimate from principal point (cx, cy)
            cx = K[0, 2]
            cy = K[1, 2]
            height = int(cy * 2)
            width = int(cx * 2)
        else:
            # (H, W) as stored in world model
            height, width = image_size

        print(f"[VISUALIZER] Adding camera '{cam_id}' with size (H,W)=({height},{width})")
        pretty_mat(f"T_world_cam ({cam_id}) (world ← cam)", T_world_cam)

        color = np.array(colors.get(cam_id, [1.0, 1.0, 1.0]))

        frustum = create_camera_frustum(
            T_wc=T_world_cam,
            K=K,
            width=width,
            height=height,
            scale=0.15,
            color=color,
        )

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 3.0
        vis.add_geometry(f"frustum_{cam_id}", frustum, mat)

        label_pos = T_world_cam[:3, 3] + np.array([0.0, 0.0, 0.05])
        vis.add_3d_label(label_pos.tolist(), cam_id)

    # ------------------------------------------------------
    # End-effectors
    # ------------------------------------------------------
    print_header("VISUALIZER: Adding end-effector frames")

    ee_sides = ["left", "right"]
    for side in ee_sides:
        if side == "left":
            T_world_ee = world_model.get_T_world_ee_left()
        else:
            T_world_ee = world_model.get_T_world_ee_right()

        if T_world_ee is None:
            print(f"[VISUALIZER] EE '{side}' not set, skipping.")
            continue

        pretty_mat(f"T_world_ee_{side} (world ← ee)", T_world_ee)
        frame_ee = create_frame(T_world_ee, size=0.15)
        mat_ee = rendering.MaterialRecord()
        mat_ee.shader = "defaultLit"
        vis.add_geometry(f"frame_ee_{side}", frame_ee, mat_ee)

        label_pos = T_world_ee[:3, 3] + np.array([0.0, 0.0, 0.03])
        vis.add_3d_label(label_pos.tolist(), f"ee_{side}")

    vis.reset_camera_to_default()
    gui.Application.instance.add_window(vis)
    gui.Application.instance.run()
