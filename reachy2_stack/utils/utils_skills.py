import numpy as np

def pose_from_normal_and_handle(
    normal_world: np.ndarray,
    handle_axis_world: np.ndarray,
    handle_center_world: np.ndarray,
) -> np.ndarray:
    """
    Build T_world_ee from:
      - plane normal        → ee +Z
      - longest handle axis → ee +X (projected into the plane)
      - handle center       → translation
    All vectors are in WORLD frame.
    """
    n = np.asarray(normal_world, dtype=float)
    a = np.asarray(handle_axis_world, dtype=float)
    p = np.asarray(handle_center_world, dtype=float)

    # --- 1) ee +Z: align with plane normal ---
    z_w = n / np.linalg.norm(n)

    # --- 2) ee +X: longest handle axis, projected onto plane (orthogonal to z_w) ---
    a_u = a / np.linalg.norm(a)
    x_tmp = a_u - np.dot(a_u, z_w) * z_w   # remove component along normal

    x_norm = np.linalg.norm(x_tmp)
    if x_norm < 1e-6:
        raise ValueError("Handle axis is (almost) parallel to the plane normal.")

    x_w = x_tmp / x_norm

    # --- 3) ee +Y: complete right-handed frame ---
    # This ensures x_w × y_w = z_w
    y_w = np.cross(z_w, x_w)

    # --- 4) Assemble 4x4 pose matrix (world ← ee) ---
    T_world_ee = np.eye(4)
    T_world_ee[:3, :3] = np.column_stack([x_w, y_w, z_w])
    T_world_ee[:3, 3] = p

    return T_world_ee