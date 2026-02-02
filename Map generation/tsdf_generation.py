import os
import glob
import numpy as np
import pandas as pd
import open3d as o3d

# ----------------------------
# User config
# ----------------------------
DATA_DIR = "/exchange/out/run_20260129_173750"
RGB_DIR = os.path.join(DATA_DIR, "rgb")
DEPTH_DIR = "/exchange/out/pngs"
ODOM_CSV = os.path.join(DATA_DIR, "odom.csv")
EXTRINSIC_TXT = os.path.join(DATA_DIR, "camera_extrinsics.txt")
INTRINSIC_TXT = os.path.join(DATA_DIR, "camera_intrinsics.txt")

DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 3.5

VOXEL_LENGTH = 0.02
SDF_TRUNC = 0.06

THETA_IS_DEGREES = True
MAX_DT_NS = 30e6

MAX_FRAMES = 2000 


# ----------------------------
# Helpers
# ----------------------------
def load_extrinsic_T_base_cam(path: str) -> np.ndarray:
    """
    Expect a 4x4 matrix in text file (space-separated or similar).

    IMPORTANT (aligned with your OdometryState module):
      This must be T_base_cam = (base <- cam), i.e. camera-to-base transform.

    Then camera pose in world is:
      T_world_cam = T_world_base @ T_base_cam
    """
    T = np.loadtxt(path)
    if T.shape != (4, 4):
        raise ValueError(f"Extrinsic must be 4x4, got {T.shape}")
    return T


def load_intrinsics_from_txt(path: str):
    """
    Returns:
      width, height, fx, fy, cx, cy, K (3x3)
    """
    width = height = None
    K_rows = []

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    i = 0
    while i < len(lines):
        parts = lines[i].split()

        if len(parts) >= 2 and parts[0] == "image_width":
            width = int(float(parts[1]))
        elif len(parts) >= 2 and parts[0] == "image_height":
            height = int(float(parts[1]))
        elif parts[0] == "camera_matrix_K":
            for j in range(1, 4):
                row = [float(x) for x in lines[i + j].split()]
                if len(row) != 3:
                    raise ValueError(f"Expected 3 values in K row, got {row}")
                K_rows.append(row)
            i += 3

        i += 1

    if width is None or height is None:
        raise ValueError("Could not parse image_width/image_height from intrinsics file.")
    if len(K_rows) != 3:
        raise ValueError("Could not parse camera_matrix_K (3 rows) from intrinsics file.")

    K = np.array(K_rows, dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return width, height, fx, fy, cx, cy, K


def pose2d_to_T(x, y, theta) -> np.ndarray:
    """T_world_base (world <- base) from planar odom pose."""
    if THETA_IS_DEGREES:
        theta = np.deg2rad(theta)

    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = x
    T[1, 3] = y
    return T


def list_timestamped_images(folder: str, exts=(".png", ".jpg", ".jpeg")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))

    items = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            t = float(stem)
        except ValueError:
            continue
        items.append((t, p))

    items.sort(key=lambda x: x[0])
    return items


def match_by_nearest_timestamp(rgb_items, depth_items, max_dt_ns=None):
    if not rgb_items or not depth_items:
        return []

    depth_ts = np.array([t for t, _ in depth_items], dtype=np.float64)
    pairs = []

    for t_rgb, rgb_path in rgb_items:
        j = int(np.searchsorted(depth_ts, t_rgb))

        candidates = []
        if 0 <= j - 1 < len(depth_items):
            candidates.append(depth_items[j - 1])
        if 0 <= j < len(depth_items):
            candidates.append(depth_items[j])

        if not candidates:
            continue

        t_d, depth_path = min(candidates, key=lambda td: abs(td[0] - t_rgb))
        dt = abs(t_d - t_rgb)

        if (max_dt_ns is not None) and (dt > max_dt_ns):
            continue

        pairs.append((t_rgb, rgb_path, t_d, depth_path, dt))

    # return pairs[:MAX_FRAMES]
    return pairs


def interp_odom_pose(odom_df: pd.DataFrame, t: float) -> tuple[float, float, float]:
    ts = odom_df["timestamp"].values
    if t <= ts[0]:
        row = odom_df.iloc[0]
        return float(row.x), float(row.y), float(row.theta)
    if t >= ts[-1]:
        row = odom_df.iloc[-1]
        return float(row.x), float(row.y), float(row.theta)

    idx = int(np.searchsorted(ts, t))
    t0, t1 = ts[idx - 1], ts[idx]
    r0, r1 = odom_df.iloc[idx - 1], odom_df.iloc[idx]

    alpha = (t - t0) / (t1 - t0 + 1e-12)

    x = (1 - alpha) * float(r0.x) + alpha * float(r1.x)
    y = (1 - alpha) * float(r0.y) + alpha * float(r1.y)
    theta = (1 - alpha) * float(r0.theta) + alpha * float(r1.theta)
    return x, y, theta


def check_image_sizes(color_img, depth_img, width, height, rgb_path, depth_path):
    c = np.asarray(color_img)
    d = np.asarray(depth_img)
    if c.shape[1] != width or c.shape[0] != height:
        raise ValueError(
            f"RGB image size {c.shape[1]}x{c.shape[0]} does not match intrinsics "
            f"{width}x{height}. Example file: {rgb_path}"
        )
    if d.shape[1] != width or d.shape[0] != height:
        raise ValueError(
            f"Depth image size {d.shape[1]}x{d.shape[0]} does not match intrinsics "
            f"{width}x{height}. Example file: {depth_path}"
        )


def make_trajectory_line(points_xyz: np.ndarray) -> o3d.geometry.LineSet:
    if len(points_xyz) < 2:
        return o3d.geometry.LineSet()

    lines = [[i, i + 1] for i in range(len(points_xyz) - 1)]
    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_xyz),
        lines=o3d.utility.Vector2iVector(lines),
    )


# ----------------------------
# Main
# ----------------------------
def run_tsdf_fusion():
    odom_df = pd.read_csv(ODOM_CSV)
    odom_df = odom_df.sort_values("timestamp").reset_index(drop=True)

    T_base_cam = load_extrinsic_T_base_cam(EXTRINSIC_TXT)
    print("Loaded extrinsics T_base_cam (base <- cam):")
    print(T_base_cam)

    WIDTH, HEIGHT, FX, FY, CX, CY, K = load_intrinsics_from_txt(INTRINSIC_TXT)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)

    print("Loaded intrinsics:")
    print(f"  size = {WIDTH} x {HEIGHT}")
    print(f"  fx, fy = {FX:.6f}, {FY:.6f}")
    print(f"  cx, cy = {CX:.6f}, {CY:.6f}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LENGTH,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    rgb_items = list_timestamped_images(RGB_DIR, exts=(".png", ".jpg", ".jpeg"))
    depth_items = list_timestamped_images(DEPTH_DIR, exts=(".png",))

    if not rgb_items:
        raise RuntimeError(f"No RGB images found in {RGB_DIR}")
    if not depth_items:
        raise RuntimeError(f"No depth images found in {DEPTH_DIR}")

    pairs = match_by_nearest_timestamp(rgb_items, depth_items, max_dt_ns=MAX_DT_NS)

    print(f"RGB frames:   {len(rgb_items)}")
    print(f"Depth frames: {len(depth_items)}")
    print(f"Matched:      {len(pairs)}  (max dt = {MAX_DT_NS/1e6:.1f} ms)")

    if not pairs:
        raise RuntimeError("No RGB-Depth pairs matched. Increase MAX_DT_NS or verify timestamps.")

    dts = np.array([dt for *_, dt in pairs], dtype=np.float64)
    print(f"dt (ms): mean={dts.mean()/1e6:.2f}, median={np.median(dts)/1e6:.2f}, max={dts.max()/1e6:.2f}")

    # ---- camera frustum visualization buffers ----
    cam_positions = []
    cam_frustums = []
    FRUSTUM_EVERY = 25   # draw one frustum every N frames
    FRUSTUM_SCALE = 0.25 # size of the camera rectangle/frustum in world units (meters)

    integrated = 0
    size_checked = False

    for t_rgb, rgb_path, t_d, depth_path, dt in pairs:
        x, y, theta = interp_odom_pose(odom_df, t_rgb)
        T_world_base = pose2d_to_T(x, y, theta)

        # T_world_cam (world <- cam)
        T_world_cam = T_world_base @ T_base_cam

        # Open3D wants world->camera (cam <- world)
        T_cam_world = np.linalg.inv(T_world_cam)

        color = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(depth_path)

        if not size_checked:
            check_image_sizes(color, depth, WIDTH, HEIGHT, rgb_path, depth_path)
            size_checked = True

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=DEPTH_SCALE,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )

        volume.integrate(rgbd, intrinsic, T_cam_world)

        # ---- collect camera pose + frustum ----
        cam_positions.append(T_world_cam[:3, 3].copy())

        if integrated % FRUSTUM_EVERY == 0:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=WIDTH,
                view_height_px=HEIGHT,
                intrinsic=K,              # 3x3
                extrinsic=T_cam_world,    # world->camera
                scale=FRUSTUM_SCALE
            )
            cam_frustums.append(frustum)

        integrated += 1

        if integrated % 50 == 0:
            print(f"Integrated {integrated} frames... (last dt={dt/1e6:.2f} ms)")

    print(f"Done. Integrated {integrated} frames.")

    pcd = volume.extract_point_cloud()
    pcd = pcd.voxel_down_sample(VOXEL_LENGTH)
    pcd_path = os.path.join(DATA_DIR, "tsdf_fused.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud: {pcd_path}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = os.path.join(DATA_DIR, "tsdf_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Saved mesh: {mesh_path}")

    # ---- visualize: cloud + frustums ----
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    cam_positions_np = np.array(cam_positions) if cam_positions else np.zeros((0, 3))
    traj = make_trajectory_line(cam_positions_np)

    # geoms = [pcd, world_frame, traj] + cam_frustums
    geoms = [pcd]
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    run_tsdf_fusion()
