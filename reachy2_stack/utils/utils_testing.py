import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def show_image(img: np.ndarray | None, title: str) -> None:
    if img is None:
        print(f"{title}: None (not displaying)")
        return

    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

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