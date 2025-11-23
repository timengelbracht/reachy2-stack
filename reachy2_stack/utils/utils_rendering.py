import numpy as np
import open3d as o3d

class DepthRenderer:
    def __init__(
        self,
        mesh: o3d.t.geometry.TriangleMesh,
        K: np.ndarray,
        clip_max_dist: float = 10.0,
    ) -> None:
        """
        Simple depth renderer.

        Parameters
        ----------
        mesh : o3d.t.geometry.TriangleMesh
            Mesh in world coordinates.
        K : (3, 3) array-like
            Pinhole intrinsics matrix.
            Width/height are inferred as 2*cx, 2*cy.
        clip_max_dist : float
            Depth values larger than this are set to 0.
        """
        self.K = np.asarray(K, dtype=np.float64)
        self.clip_max_dist = float(clip_max_dist)

        # infer image size from intrinsics (assuming principal point at center)
        self.width = int(round(self.K[0, 2] * 2))
        self.height = int(round(self.K[1, 2] * 2))

        # --- set up renderer once -----------------------------------
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.width, self.height
        )
        self.renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"

        # store legacy mesh once
        self.mesh_legacy = mesh.to_legacy()
        self.renderer.scene.add_geometry("mesh", self.mesh_legacy, mat)

    def render(self, w_T_wc: np.ndarray) -> np.ndarray:
        """
        Render a depth image for a given camera pose.

        Parameters
        ----------
        w_T_wc : (4, 4) array-like
            Camera pose in world coordinates (same convention as your
            original function – whatever Open3D expects here).

        Returns
        -------
        depth : (H, W) float32
            Depth image in meters. Values outside [0.01, clip_max_dist]
            are set to 0.
        """
        w_T_wc = np.asarray(w_T_wc, dtype=np.float64)

        # Update camera each call, but reuse renderer + mesh
        self.renderer.setup_camera(self.K, w_T_wc, self.width, self.height)

        depth_img = self.renderer.render_to_depth_image(z_in_view_space=True)
        depth = np.asarray(depth_img, dtype=np.float32)

        # Clip depth values: mark invalid as 0
        depth[depth > self.clip_max_dist] = 0.0
        depth[depth < 0.01] = 0.0

        return depth
    
def depth_to_world_xyz(depth: np.ndarray, K: np.ndarray, w_T_wc: np.ndarray) -> np.ndarray:
    """
    Convert a depth image (pixel-wise depth) into 3D points in world coordinates.
    Parameters
    ----------
    depth : np.ndarray
        Depth image of shape (H, W), where each value is the depth along the camera z-axis.
    K : np.ndarray
        Camera intrinsic matrix of shape (3, 3) with fx, fy, cx, cy.
    w_T_wc : np.ndarray
        4x4 homogeneous transform from camera frame to world frame (world_T_camera).
        The top-left 3x3 block is rotation R_wc and the top-right 3x1 is translation t_wc.
    Returns
    -------
    np.ndarray
        Array of shape (H, W, 3) and dtype float32 containing 3D points in world coordinates.
    Notes
    -----
    - Uses a pinhole camera model: x = (u - cx) * z / fx, y = (v - cy) * z / fy.
    - Depth units must match the translation units in w_T_wc.
    """
    
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # pixel coordinates

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Reconstruct 3D camera coordinates
    x_cam = ((u - cx) * depth) / fx
    y_cam = ((v - cy) * depth) / fy
    z_cam = depth

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    pts_cam_flat = pts_cam.reshape(-1, 3)  # (H*W, 3)

    # Convert camera-to-world rotation
    R_wc = w_T_wc[:3, :3]  # 3×3 rotation matrix
    t_wc = w_T_wc[:3, 3]   # 3×1
    pts_w = (R_wc @ pts_cam_flat.T + t_wc[:, None]).T  # 3×N

    return pts_w.reshape(H, W, 3).astype(np.float32)
