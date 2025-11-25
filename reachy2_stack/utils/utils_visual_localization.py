import numpy as np
from reachy2_stack.utils.utils_dataclass import TeleopCameraData
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from pathlib import Path
from typing import Dict, Any

def colmap_camera_cfg_from_reachy_camera_intrinsics(cam_data: TeleopCameraData) -> dict:
    """Convert Reachy camera intrinsics to COLMAP camera configuration dictionary.
    Args:
        cam_data (TeleopCameraData): Reachy camera data containing intrinsics.
    Returns:
        dict: COLMAP camera configuration dictionary.
    """
    K = cam_data.intrinsics["K"]
    D = cam_data.intrinsics["D"]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    return {
        "model": "OPENCV_FISHEYE",
        "width": cam_data.intrinsics["width"],
        "height": cam_data.intrinsics["height"],
        "params": [
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            float(D[0]),
            float(D[1]),
            float(D[2]),
            float(D[3]),
        ],
    }


def generate_dummy_transformations_files(out_path: str) -> None:
    """Generate dummy transformation files for testing."""
    transform = np.eye(4, dtype=np.float32)  # 4x4 identity matrix
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for _ in range(7):
            f.write("dummy\n")
        for row in transform:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")



def create_camera_frustum(
    T_wc: np.ndarray, 
    K: np.ndarray, 
    width: int, 
    height: int, 
    scale: float = 0.15, 
    color: np.ndarray = np.array([1., 0., 0.])
) -> o3d.geometry.LineSet:
    """
    Creates a wireframe frustum. 
    Uses K_inv to ensure the visual frustum matches the actual calibrated geometry 
    (including principal point offsets).
    """
    # 1. Define image corners in pixel coordinates
    corners_pix = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T  # 3x4

    # 2. Back-project to Normalized Camera Coordinates (Z=1)
    K_inv = np.linalg.inv(K)
    corners_cam = K_inv @ corners_pix

    # 3. Scale to visual size
    corners_cam = corners_cam * scale

    # 4. Define Center and transform to World
    points_cam = np.hstack((np.zeros((3,1)), corners_cam)) # 0,0,0 + 4 corners
    
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3:4]
    
    points_world = (R_wc @ points_cam) + t_wc
    points_world = points_world.T

    # 5. Define Lines
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Rays
        [1, 2], [2, 3], [3, 4], [4, 1]   # Base
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    
    return line_set

def visualize_localization_in_mesh(
    mesh_path: Path,
    loc_results: Dict[str, Dict[str, Any]],
) -> None:
    """
    Visualizes HLoc results using ONLY the mesh and the loc_results dictionary.
    """
    
    print(f"\n[VISUALIZER] Loading Mesh: {mesh_path}")
    
    # 1. Load Mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # 2. Setup GUI
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Reachy Localization Results", 1600, 1000)
    vis.show_axes = True

    # Add Mesh
    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultLit"
    vis.add_geometry("scene_mesh", mesh, mesh_mat)

    # 3. Add Cameras from loc_results
    # Define colors for specific keys (fallback to white if unknown)
    colors = {
        "teleop_left":  [1.0, 0.0, 0.0], # Red
        "teleop_right": [0.0, 0.0, 1.0], # Blue
        "depth":        [0.0, 1.0, 0.0], # Green
    }

    count = 0
    for cam_name, res in loc_results.items():
        if "T_wc" not in res or "K" not in res:
            continue

        print(f"[VISUALIZER] Adding {cam_name}...")
        count += 1
        
        T_wc = res["T_wc"]
        K = res["K"]

        # --- LOGIC TO DETERMINE W/H WITHOUT DATACLASS ---
        if "w" in res and "h" in res:
            width = int(res["w"])
            height = int(res["h"])
        else:
            # Estimate from Principal Point (cx, cy)
            # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            cx = K[0, 2]
            cy = K[1, 2]
            width = int(cx * 2)
            height = int(cy * 2)
        # ------------------------------------------------

        color = np.array(colors.get(cam_name, [1.0, 1.0, 1.0]))

        # Create Frustum
        frustum = create_camera_frustum(T_wc, K, width, height, scale=0.15, color=color)
        
        # Material
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 3.0
        
        vis.add_geometry(f"frustum_{cam_name}", frustum, mat)
        
        # Label
        label_pos = T_wc[:3, 3] + np.array([0, 0, 0.05]) # Offset label slightly up
        vis.add_3d_label(label_pos.tolist(), cam_name)

    if count == 0:
        print("[VISUALIZER] Warning: No valid poses found in loc_results.")

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()