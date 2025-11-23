import math
import cv2
import numpy as np
import re
from pathlib import Path
import scipy.io as sio
from scipy.spatial.transform import Rotation as R  
import math
import cv2
import json

from reachy2_stack.utils.utils_images import load_image


from reachy2_stack.utils.utils_images import write_depth_vis
from reachy2_stack.utils.utils_pointclouds import load_mesh
from reachy2_stack.utils.utils_rendering import DepthRenderer, depth_to_world_xyz
from dataclasses import dataclass

@dataclass
class MappingInputLeica:
    """Minimal representation for map building from leica."""
    pano_path: Path        # RGB panorama
    pano_pose_path: Path   # pose .txt of the panorama
    mesh_path: Path        # mesh of the scan used for depth rendering


@dataclass
class HLocMapConfig:
    maps_root: Path         # root folder for all maps
    location_name: str      # e.g. "kitchen_1"
    overwrite: bool = False

def make_equirect_to_pinhole(equi_img: np.ndarray,
                        rot_mat: np.ndarray,
                        hfov_deg: float,
                        vfov_deg: float,
                        out_w: int,
                        out_h: int) -> np.ndarray:
    """
    Convert an equirectangular panorama to a pinhole camera view.

    Parameters
    - equi_img: H_e x W_e x C numpy array containing the equirectangular panorama
      (pixel ordering is preserved, e.g. BGR for OpenCV images).
    - rot_mat: 3x3 rotation matrix that rotates ray directions from the pinhole
      camera coordinate frame into the panorama/world coordinate frame. Should
      be a proper orthonormal rotation matrix.
    - hfov_deg, vfov_deg: horizontal and vertical field-of-view in degrees for
      the output pinhole view.
    - out_w, out_h: width and height of the output pinhole image in pixels.

    Returns
    - out_h x out_w x C numpy array sampled from the panorama using linear
      interpolation. Horizontal sampling wraps around the panorama borders.
    """
    H_e, W_e = equi_img.shape[:2]

    # compute the tangent extents for each axis
    tan_h = math.tan(math.radians(hfov_deg / 2))
    tan_v = math.tan(math.radians(vfov_deg / 2))

    # screen coords in camera space
    xs = np.linspace(-tan_h, +tan_h, out_w)
    ys = np.linspace(-tan_v, +tan_v, out_h)
    xv, yv = np.meshgrid(xs, -ys)       # note the -ys to flip vertically
    zv = np.ones_like(xv)

    dirs = (rot_mat @ np.stack([xv, yv, zv], -1).reshape(-1,3).T).T

    # convert to spherical coords
    lon = np.arctan2(dirs[:,0], dirs[:,2])   # range [-π, π]
    lat = np.arcsin(dirs[:,1] / np.linalg.norm(dirs, axis=1))  # [-π/2, π/2]

    # map to equirectangular pixel coords
    uf = (lon / (2 * math.pi) + 0.5) * W_e
    vf = (0.5 - lat / math.pi) * H_e

    map_x = uf.reshape(out_h, out_w).astype(np.float32)
    map_y = vf.reshape(out_h, out_w).astype(np.float32)

    # sample with wrap‑around horizontally
    return cv2.remap(
        equi_img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )


def load_leica_pano(pano_path: Path,
                    pano_pose_path: Path) -> dict:
    """Load leica panorama and its pose from files.
    Args:
        pano_path (Path): Path to the panorama image file.
        pano_pose_path (Path): Path to the panorama pose text file.
    Returns:
        dict: Dictionary containing 'rgb' (the panorama image as a NumPy array)
              and 'pose' (the pose data as a dictionary).
    """
    
    data = {"LDR": {}, "HDR": {}}
    section = None

    # regex to capture numbers inside the brackets
    array_re = re.compile(r"\[([^\]]+)\]")
    

    for line in pano_pose_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("Ldr Image"):
            section = "LDR"
        elif line.startswith("Hdr Image"):
            section = "HDR"
        elif section and line.startswith("position"):
            m = array_re.search(line)
            if m:
                nums = [float(x) for x in m.group(1).split(",")]
                data[section]["position"] = nums
        elif section and line.startswith("orientation"):
            m = array_re.search(line)
            if m:
                nums = [float(x) for x in m.group(1).split(",")]
                data[section]["orientation"] = nums
    
    pano = {}
    pano['rgb'] = load_image(pano_path)
    pano['pose'] = data

    return pano

def save_mat(mat_path: str | Path, rgb_image: np.ndarray, xyz_array) -> None:
    """
    Save RGB and XYZ arrays to a MATLAB .mat file.

    This function writes the provided RGB image and XYZ array to a .mat file using
    scipy.io.savemat. The data are stored under the keys "RGBcut" and "XYZcut"
    respectively.

    Parameters
    ----------
    mat_path : str | pathlib.Path
        Path to the output .mat file. The path will be converted to a string before
        writing. If the file already exists it will be overwritten by savemat.
    rgb_image : numpy.ndarray
        RGB image array to save. Common shapes are (H, W, 3) for color images, but
        other shapes are accepted and will be saved as-is.
    xyz_array : numpy.ndarray
        Array of XYZ coordinates (for example shape (H, W, 3) or (N, 3)). This array
        will be saved under the key "XYZcut".

    Returns
    -------
    None

    Raises
    ------
    OSError, IOError
        If the target path cannot be written to (propagated from underlying I/O
        operations).
    ValueError, TypeError
        If scipy.io.savemat cannot serialize the provided objects (propagated from
        scipy).

    Notes
    -----
    - The function uses scipy.io.savemat with do_compression=False.
    - The saved .mat file will contain two variables: "RGBcut" and "XYZcut".

    Example
    -------
    >>> save_mat("output.mat", rgb_image, xyz_array)
    """

    sio.savemat(str(mat_path),
                {"RGBcut": rgb_image, "XYZcut": xyz_array},
                do_compression=False)
    
def process_leica_for_hloc(
    pano_path: Path,
    pano_pose_path: Path,
    mesh_path: Path,
    out_base: Path,
    overwrite: bool = False,
) -> Path:

    # create rectified crops of the panorama and depth renderings
    hfov = 90.0
    vfov = 120.0
    W, H = 1024, 1364
    step = hfov * 0.1

    # compute intrinsics
    fx = (W/2) / math.tan(math.radians(hfov/2))
    fy = (H/2) / math.tan(math.radians(vfov/2))
    K  = np.array([[fx,0,W/2],
        [0, fy,H/2],
        [0,  0,  1]])
    
    if out_base.exists() and not overwrite:
        print(f"Map directory {out_base} already exists. Use overwrite=True to replace.")
        return

    rgb_out  = out_base / "rgb"
    depth_vis_out = out_base / "depth_vis"
    pose_out = out_base / "poses"
    xyz_out = out_base / "xyz"
    for d in (rgb_out, pose_out, xyz_out, depth_vis_out):
        d.mkdir(parents=True, exist_ok=True)

    pano_data = load_leica_pano(
        pano_path,
        pano_pose_path,
    )
    pano_img = pano_data["rgb"]
    pano_pose = pano_data["pose"]["HDR"]

       # get initial camera pose and rots around world axes
    euler0 = R.from_quat(pano_pose["orientation"], scalar_first=True).as_euler("xyz", degrees=True)
    rot_initial_around_world_z = euler0[2] 

    # define default camera pose for tiel cropping and rendering
    # no rotation, translation from metadata
    t0 = np.array(pano_pose["position"])

    # create a grid of yaw angles to cover the full 360 degrees
    yaws = np.arange(0.0, 360.0, step) 

    T_o3d_leica = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
    
    # render depth from mesh
    mesh = load_mesh(mesh_path)
    depth_renderer = DepthRenderer(
        mesh,
        K=K,
        clip_max_dist=10.0,
    )
    idx = 0
    for yaw_deg in yaws:
        yaw   = math.radians(yaw_deg)

        # compute target camera pose
        # yaw the camera around its own y-axis for horizontal rotation in camera frame
        R_yaw = R.from_euler("y", yaw, degrees=False).as_matrix()
        R_wc_target = R_yaw # can prolly remove R_wc_initial, since we use identny quat for initial pose
        t_wc_target = t0 

        # compute new camera pose in world coordinates after rotation
        ext = np.eye(4, dtype=float)
        ext[:3, :3] = R_wc_target  # convert from Leica to Open3D coordinate system
        ext[:3, 3] = t_wc_target
        ext_inv_o3d = np.linalg.inv(ext) @ T_o3d_leica  # convert from Leica to Open3D coordinate system  # convert from Open3D to Leica coordinate system
        ext_save = np.linalg.inv(T_o3d_leica) @ ext  # save in Open3D coordinate system

        #RGB pinhole tile in the equirectangular image
        # adjust yaw to match the initial rotation for crops
        # this is needed to align the crops with the original panorama
        yaw_adjusted = yaw_deg + rot_initial_around_world_z  
        R_yaw = R.from_euler("y", yaw_adjusted, degrees=True).as_matrix()
        tile_rgb = make_equirect_to_pinhole(
            equi_img=pano_img, 
            rot_mat=R_yaw, 
            hfov_deg=hfov, 
            vfov_deg=vfov, 
            out_w=W, 
            out_h=H
        )

        # write RGB tile
        fn = f"{idx:03d}.jpg"
        cv2.imwrite(str(rgb_out/fn), tile_rgb)

        # 2) render depth tile
        tile_depth = depth_renderer.render(ext_inv_o3d)

        # 3) write depth visualization
        depth_vis_fn = fn.replace(".jpg", "_vis.png")
        write_depth_vis(str(depth_vis_out/depth_vis_fn), tile_depth)

        # 4) write xyz tile to mat
        xyz_tile = depth_to_world_xyz(tile_depth, K,  ext_save) #ext_save
        xyz_fn = fn + ".mat"
        save_mat(str(xyz_out/xyz_fn), rgb_image=tile_rgb, xyz_array=xyz_tile)

        q = R.from_matrix(ext_save[:3, :3]).as_quat(scalar_first=False) 
            # convert to quaternion

        # 5) Dump JSON
        pose = {
            "w_T_wc": ext_save.tolist(),
            "K": K.tolist(),
            "h": H,
            "w": W}
        
        with open(pose_out/fn.replace(".jpg",".json"), "w") as f:
            json.dump(pose, f, indent=2)

        idx += 1

    print(f"[LEICA] Exported {idx} tiles → {out_base}")

