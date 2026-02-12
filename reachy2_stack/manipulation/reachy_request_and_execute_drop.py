#!/usr/bin/env python3
"""
Request trajectory from VIDbot server and execute drop action.

Usage:
    python reachy_request_and_execute_drop.py \
        --vidbot-url http://GPU_IP:9000/infer_and_return_trajectories \
        -o "apple" -i "place"
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/exchange")

import io
import os
import time
import binascii
import argparse
import tempfile

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.arm import ArmController
from reachy2_stack.control.gripper import GripperController


# ===================== CONFIGURATION =====================

# Arm scan position - positions arm so camera sees both object and background
Q_SCAN = np.array([
    -15.97136943,
     11.45268959,
     -4.05706748,
   -115.52284166,
    -12.24387651,
     40.98972057,
      5.18980185
])

# Timing
SCAN_POSITION_SETTLE_TIME = 2.0  # seconds to wait after moving to scan position
GRIPPER_OPEN_SETTLE_TIME = 2.0  # seconds to wait after opening gripper

# Trajectory subsampling
PRE_T_SUBSAMPLE_FACTOR = 3  # take every Nth point from pre trajectory
POST_T_SUBSAMPLE_FACTOR = 6  # take every Nth point from post trajectory (1 = no subsampling)
POST_T_SKIP_FIRST = 5  # skip first N poses of post trajectory (to avoid arm hitting base)

# Gripper release timing (fraction of trajectory, 0.0 to 1.0)
# 0.8 = open gripper when 80% through the trajectory
GRIPPER_OPEN_FRACTION = 0.7

# ===========================================================


def is_valid_T(T: np.ndarray, tol: float = 1e-3) -> bool:
    """Validate a 4x4 transformation matrix."""
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], np.array([0, 0, 0, 1], dtype=float), atol=tol):
        return False
    R = T[:3, :3]
    if not np.isfinite(R).all() or not np.isfinite(T[:3, 3]).all():
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=5e-2):
        return False
    det = np.linalg.det(R)
    return 0.5 < det < 1.5


def validate_trajectory(traj_T: np.ndarray, label: str) -> None:
    """Validate trajectory shape and key poses."""
    if traj_T.ndim != 3 or traj_T.shape[1:] != (4, 4):
        raise ValueError(f"{label}: expected (N,4,4), got {traj_T.shape}")

    # Check first, second, and last poses
    check_indices = [0, min(1, len(traj_T) - 1), len(traj_T) - 1]
    for idx in check_indices:
        if not is_valid_T(traj_T[idx]):
            raise ValueError(f"{label}: invalid transform at idx {idx}\n{traj_T[idx]}")


def extract_position(T: np.ndarray) -> np.ndarray:
    """Extract XYZ position from 4x4 transform matrix."""
    return T[:3, 3]


def extract_positions(traj_T: np.ndarray) -> np.ndarray:
    """Extract all XYZ positions from trajectory. Returns (N, 3) array."""
    return traj_T[:, :3, 3]


def print_trajectory_info(traj_T: np.ndarray, label: str) -> None:
    """Print detailed trajectory information."""
    positions = extract_positions(traj_T)
    n_steps = len(traj_T)

    start_pos = positions[0]
    end_pos = positions[-1]

    # Calculate total path length
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths)

    # Straight-line distance
    direct_distance = np.linalg.norm(end_pos - start_pos)

    print(f"\n{'='*60}")
    print(f"TRAJECTORY: {label}")
    print(f"{'='*60}")
    print(f"  Steps: {n_steps}")
    print(f"  Start position (XYZ): [{start_pos[0]:7.3f}, {start_pos[1]:7.3f}, {start_pos[2]:7.3f}] m")
    print(f"  End position (XYZ):   [{end_pos[0]:7.3f}, {end_pos[1]:7.3f}, {end_pos[2]:7.3f}] m")
    print(f"  Direct distance:      {direct_distance:.3f} m")
    print(f"  Path length:          {total_length:.3f} m")
    print(f"  Path/Direct ratio:    {total_length/max(direct_distance, 1e-6):.2f}")

    # Bounding box
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    print(f"  Bounding box:")
    print(f"    X: [{mins[0]:7.3f}, {maxs[0]:7.3f}] m  (range: {maxs[0]-mins[0]:.3f})")
    print(f"    Y: [{mins[1]:7.3f}, {maxs[1]:7.3f}] m  (range: {maxs[1]-mins[1]:.3f})")
    print(f"    Z: [{mins[2]:7.3f}, {maxs[2]:7.3f}] m  (range: {maxs[2]-mins[2]:.3f})")
    print(f"{'='*60}\n")


def visualize_trajectory(pre_T: np.ndarray, post_T: np.ndarray, title: str = "VIDbot Trajectory") -> None:
    """
    Visualize trajectories in 3D using matplotlib.

    - pre_T shown in blue (approach)
    - post_T shown in orange (retract)
    - Start points marked with 'o'
    - End points marked with 'x'
    """
    fig = plt.figure(figsize=(12, 5))

    pre_pos = extract_positions(pre_T)
    post_pos = extract_positions(post_T)

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot pre trajectory (approach)
    ax1.plot(pre_pos[:, 0], pre_pos[:, 1], pre_pos[:, 2], 'b-', linewidth=2, label='pre (approach)')
    ax1.scatter(*pre_pos[0], c='blue', s=100, marker='o', label='pre start')
    ax1.scatter(*pre_pos[-1], c='blue', s=100, marker='x', label='pre end')

    # Plot post trajectory (retract)
    ax1.plot(post_pos[:, 0], post_pos[:, 1], post_pos[:, 2], 'orange', linewidth=2, label='post (retract)')
    ax1.scatter(*post_pos[0], c='orange', s=100, marker='o', label='post start')
    ax1.scatter(*post_pos[-1], c='orange', s=100, marker='x', label='post end')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{title} - 3D View')
    ax1.legend(loc='upper left', fontsize=8)

    # Top-down view (XY plane)
    ax2 = fig.add_subplot(122)

    ax2.plot(pre_pos[:, 0], pre_pos[:, 1], 'b-', linewidth=2, label='pre (approach)')
    ax2.scatter(pre_pos[0, 0], pre_pos[0, 1], c='blue', s=100, marker='o')
    ax2.scatter(pre_pos[-1, 0], pre_pos[-1, 1], c='blue', s=100, marker='x')

    ax2.plot(post_pos[:, 0], post_pos[:, 1], 'orange', linewidth=2, label='post (retract)')
    ax2.scatter(post_pos[0, 0], post_pos[0, 1], c='orange', s=100, marker='o')
    ax2.scatter(post_pos[-1, 0], post_pos[-1, 1], c='orange', s=100, marker='x')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'{title} - Top-Down (XY)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def execute_trajectory(
    arm: ArmController,
    traj_T: np.ndarray,
    step_sleep: float,
    label: str,
    gripper: GripperController | None = None,
    open_gripper_at_fraction: float | None = None,
    stop_after_gripper_open: bool = False,
) -> bool:
    """
    Execute a trajectory of end-effector poses.

    Args:
        arm: Arm controller
        traj_T: Trajectory as (N, 4, 4) array of transforms
        step_sleep: Sleep time between steps
        label: Label for logging
        gripper: Optional gripper controller
        open_gripper_at_fraction: If set, open gripper after reaching this
            fraction of the trajectory (0.0 to 1.0). The gripper opens
            AFTER the arm reaches the target position.
        stop_after_gripper_open: If True, stop trajectory execution after opening gripper

    Returns:
        True if successful, False otherwise
    """
    validate_trajectory(traj_T, label)

    n_steps = len(traj_T)
    open_at_idx = None
    if open_gripper_at_fraction is not None and gripper is not None:
        open_at_idx = int(n_steps * open_gripper_at_fraction)
        open_at_idx = max(0, min(open_at_idx, n_steps - 1))

    print(f"\n--- Executing {label} --- steps={n_steps}")
    if open_at_idx is not None:
        print(f"    Gripper will open after step {open_at_idx}/{n_steps - 1}")
        if stop_after_gripper_open:
            print(f"    Will stop after gripper opens")

    for i, T_base_ee in enumerate(traj_T):
        # Move arm to pose
        ok = arm.goto_pose_base_with_base_assist(T_base_ee=T_base_ee)
        if not ok:
            print(f"{label}: FAIL at step {i}/{n_steps - 1}")
            return False

        time.sleep(step_sleep)

        # Open gripper AFTER reaching the position
        if open_at_idx is not None and i == open_at_idx:
            print(f"    Opening gripper at step {i}")
            gripper.open()
            time.sleep(GRIPPER_OPEN_SETTLE_TIME)
            if stop_after_gripper_open:
                print(f"    Stopping trajectory after gripper open")
                return True

    return True


def capture_rgb_depth_png_bytes(reachy) -> tuple[bytes, bytes]:
    """
    Capture RGB + depth from Reachy and return as PNG bytes.

    RGB: uint8 PNG
    Depth: uint16 PNG (raw, no scaling)
    """
    # Capture from Reachy
    rgb_bgr, _ = reachy.cameras.depth.get_frame()
    rgb = np.asarray(rgb_bgr, dtype=np.uint8)

    depth_raw, _ = reachy.cameras.depth.get_depth_frame()
    depth = np.asarray(depth_raw, dtype=np.uint16)

    # Save to temp files and read back as bytes
    with tempfile.TemporaryDirectory() as tmp:
        rgb_path = os.path.join(tmp, "rgb.png")
        depth_path = os.path.join(tmp, "depth.png")

        if not cv2.imwrite(rgb_path, rgb):
            raise RuntimeError("Failed to write RGB PNG")
        if not cv2.imwrite(depth_path, depth):
            raise RuntimeError("Failed to write depth PNG")

        with open(rgb_path, "rb") as f:
            rgb_bytes = f.read()
        with open(depth_path, "rb") as f:
            depth_bytes = f.read()

    return rgb_bytes, depth_bytes


def request_trajectory_from_vidbot(
    url: str,
    rgb_bytes: bytes,
    depth_bytes: bytes,
    object_name: str,
    action: str,
    visualize: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Send RGBD to VIDbot server and get trajectory.

    Returns:
        (pre_T, post_T, metadata) - trajectories and optional metadata
    """
    files = {
        "rgb_png": ("000000.png", rgb_bytes, "image/png"),
        "depth_png": ("000000.png", depth_bytes, "image/png"),
    }
    data = {
        "object_name": object_name,
        "action": action,
        "visualize": "true" if visualize else "false",
    }

    print(f"Requesting inference from: {url}")
    response = requests.post(url, files=files, data=data, timeout=600)
    response.raise_for_status()
    resp = response.json()

    # Decode trajectory payload
    payload_hex = resp["payload_npz_bytes"]
    payload = binascii.unhexlify(payload_hex)

    z = np.load(io.BytesIO(payload), allow_pickle=True)
    pre_T = np.asarray(z["pre_T"], dtype=float)
    post_T = np.asarray(z["post_T"], dtype=float)

    metadata = {}
    if "best_idx" in z.files:
        metadata["best_idx"] = int(z["best_idx"])
    if "best_loss" in z.files:
        metadata["best_loss"] = float(z["best_loss"])

    return pre_T, post_T, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute drop action using VIDbot trajectory")
    parser.add_argument("--reachy-host", type=str, default="192.168.1.71")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("-o", "--object", dest="object_name", type=str, required=True,
                        help="Object name (e.g., 'apple')")
    parser.add_argument("-i", "--action", dest="action", type=str, required=True,
                        help="Action verb (e.g., 'place', 'drop')")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Enable visualization on VIDbot server")
    parser.add_argument("--vidbot-url", type=str, required=True,
                        help="VIDbot server URL")
    parser.add_argument("--step-sleep", type=float, default=0.5,
                        help="Sleep between trajectory steps")
    parser.add_argument("--gripper-open-fraction", type=float, default=GRIPPER_OPEN_FRACTION,
                        help="Fraction of trajectory at which to open gripper (0.0-1.0)")
    parser.add_argument("--show-trajectory", action="store_true",
                        help="Print and visualize the trajectory before executing")
    parser.add_argument("--trajectory-only", action="store_true",
                        help="Only show trajectory (don't execute on robot)")
    args = parser.parse_args()

    print(f"Object: {args.object_name}, Action: {args.action}")

    # Connect to Reachy
    cfg = ReachyConfig(host=args.reachy_host)
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()

    arm = ArmController(client=client, side=args.side, world=None)
    gripper = GripperController(client=client, side=args.side)
    reachy = client.connect_reachy

    try:
        # Move to scan position (camera can see object + background)
        print(f"\nMoving to scan position...")
        arm.goto_joints(Q_SCAN, duration=3.0, wait=True)
        time.sleep(SCAN_POSITION_SETTLE_TIME)

        # Capture RGBD
        print("Capturing RGBD...")
        rgb_bytes, depth_bytes = capture_rgb_depth_png_bytes(reachy)

        # Get trajectory from VIDbot
        pre_T, post_T, metadata = request_trajectory_from_vidbot(
            url=args.vidbot_url,
            rgb_bytes=rgb_bytes,
            depth_bytes=depth_bytes,
            object_name=args.object_name,
            action=args.action,
            visualize=args.visualize,
        )

        # Subsample pre_T
        pre_T = pre_T[::PRE_T_SUBSAMPLE_FACTOR]

        # Skip first N poses of post_T, but extend pre_T to reach the new start
        # Append skipped portion of post_T to pre_T (excluding first pose to avoid duplicate)
        if POST_T_SKIP_FIRST > 0:
            skipped_poses = post_T[1:POST_T_SKIP_FIRST + 1]  # poses 1 to N (0 is already at end of pre_T)
            pre_T = np.concatenate([pre_T, skipped_poses], axis=0)

        # Now trim post_T and subsample
        post_T = post_T[POST_T_SKIP_FIRST::POST_T_SUBSAMPLE_FACTOR]

        print(f"\nTrajectory shapes: pre={pre_T.shape}, post={post_T.shape}")
        if metadata:
            print(f"Metadata: {metadata}")

        # Print and visualize trajectory if requested
        if args.show_trajectory or args.trajectory_only:
            print_trajectory_info(pre_T, "pre_T (approach)")
            print_trajectory_info(post_T, "post_T (retract)")
            visualize_trajectory(pre_T, post_T, title=f"{args.object_name} - {args.action}")

        # Exit early if only showing trajectory
        if args.trajectory_only:
            print("--trajectory-only flag set, skipping execution")
            return 0

        # Execute pre trajectory (approach to start of post_T, no gripper action)
        ok = execute_trajectory(
            arm=arm,
            traj_T=pre_T,
            step_sleep=args.step_sleep,
            label="pre (approach)",
        )
        if not ok:
            return 1

        # Execute post trajectory (move to drop position, open gripper, then stop)
        time.sleep(0.3)
        ok = execute_trajectory(
            arm=arm,
            traj_T=post_T,
            step_sleep=args.step_sleep,
            label="post (drop)",
            gripper=gripper,
            open_gripper_at_fraction=args.gripper_open_fraction,
            stop_after_gripper_open=True,
        )
        if not ok:
            return 1

        # Return to default (scan) position
        print("\n--- Returning to default position ---")
        time.sleep(0.3)
        arm.goto_joints(Q_SCAN, duration=3.0, wait=True)

        return 0

    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


# python reachy2_stack/manipulation/reachy_request_and_execute_drop.py   -o toy -i "drop in box"   --vidbot-url http://192.168.1.223:9
# 000/infer_and_return_trajectories   -v