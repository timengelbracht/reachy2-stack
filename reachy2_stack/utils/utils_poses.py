import numpy as np

def T_to_xytheta(T: np.ndarray) -> tuple[float, float, float]:
    """Extract (x, y, theta_deg) from a 4x4 world←base transform."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4")

    x = float(T[0, 3])
    y = float(T[1, 3])
    theta = float(np.arctan2(T[1, 0], T[0, 0]))  # radians
    theta_deg = float(np.rad2deg(theta))
    return x, y, theta_deg

def xytheta_to_T(x: float, y: float, theta_deg: float) -> np.ndarray:
    """Build a 4x4 world←base transform from (x, y, theta_deg)."""
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    T = np.array(
        [
            [cos_theta, -sin_theta, 0.0, x],
            [sin_theta, cos_theta, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return T