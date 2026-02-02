#!/usr/bin/env python3
"""Unified test script for robot_state_loop, mapping_loop, localization, and vis_process.

This script tests:
- robot_state_thread: Unified camera + odometry -> RobotState queue
- mapping_thread: Sends RGBD to wavemap server, outputs MapState
- localization_thread: Sends images to localization server with offset correction
- teleop_thread: Keyboard control
- vis_process: Visualization with Open3D (3D) and Matplotlib (images)

Architecture:
    robot_state_loop ──► RobotState ──┬──► vis_process (images, trajectory)
                                      │
                                      ├──► mapping_loop ──► MapState ──► vis_process (point clouds)
                                      │
                                      └──► localization_loop ──► Localized RobotState (with offset correction)

Usage:
    # Without mapping or localization (just camera + vis)
    python reachy2_stack/tests/module_test_unified.py

    # With mapping (requires wavemap server running)
    python ext_server/wavemap_server.py --port 5555  # Terminal 1
    python reachy2_stack/tests/module_test_unified.py --mapping  # Terminal 2

    # With localization (requires localization server running)
    python ext_server/localization_server.py --port 5556  # Terminal 1
    python reachy2_stack/tests/module_test_unified.py --localization  # Terminal 2

    # Enable teleop cameras
    python reachy2_stack/tests/module_test_unified.py --mapping --teleop
"""

import argparse
import time
import threading
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
from queue import Queue
from typing import Optional

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.base_module import teleop_loop
from reachy2_stack.base_module.robot_state_loop import robot_state_loop
from reachy2_stack.base_module.mapping_loop import mapping_loop
from reachy2_stack.base_module.localization_loop import localization_loop
from reachy2_stack.base_module.camera_pose_buffer import CameraPoseBuffer
from reachy2_stack.base_module.localization_fusion import LocalizationFusion
from reachy2_stack.base_module.visualization import vis_process

# ---------------- CONFIG ----------------
HOST = "192.168.1.71"

# Robot state acquisition
STATE_HZ = 15  # Hz for camera + odometry

# Teleop
CMD_HZ = 30
VX = 0.6
VY = 0.6
WZ = 100.0

# Mapping
MAPPING_HZ = 2.0  # Hz for mapping requests
MAPPING_SERVER_HOST = "localhost"
MAPPING_SERVER_PORT = 5555

# Localization
LOCALIZATION_HZ = 10.0  # Hz for localization requests
LOCALIZATION_SERVER_HOST = "localhost"
LOCALIZATION_SERVER_PORT = 5556
FUSION_SMOOTH_ALPHA = 1.0  # Offset blend factor (0=ignore vision, 1=trust fully)

# Visualization
VIS_FPS = 15
DEPTH_SCALE = 0.001  # mm to meters
DEPTH_TRUNC = 3.5  # meters
# --------------------------------------


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified module test")
    parser.add_argument(
        "--host",
        default=HOST,
        help="Robot host IP",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Enable teleop cameras",
    )
    parser.add_argument(
        "--mapping",
        action="store_true",
        help="Enable mapping (requires wavemap server)",
    )
    parser.add_argument(
        "--server-host",
        default=MAPPING_SERVER_HOST,
        help="Wavemap server host",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=MAPPING_SERVER_PORT,
        help="Wavemap server port",
    )
    parser.add_argument(
        "--mapping-hz",
        type=float,
        default=MAPPING_HZ,
        help="Mapping request rate (Hz)",
    )
    parser.add_argument(
        "--state-hz",
        type=int,
        default=STATE_HZ,
        help="Robot state acquisition rate (Hz)",
    )
    parser.add_argument(
        "--vis-fps",
        type=int,
        default=VIS_FPS,
        help="Visualization frame rate (FPS)",
    )
    parser.add_argument(
        "--localization",
        action="store_true",
        help="Enable localization (requires localization server)",
    )
    parser.add_argument(
        "--loc-server-host",
        default=LOCALIZATION_SERVER_HOST,
        help="Localization server host",
    )
    parser.add_argument(
        "--loc-server-port",
        type=int,
        default=LOCALIZATION_SERVER_PORT,
        help="Localization server port",
    )
    parser.add_argument(
        "--loc-hz",
        type=float,
        default=LOCALIZATION_HZ,
        help="Localization request rate (Hz)",
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=FUSION_SMOOTH_ALPHA,
        help="Fusion offset blend factor (0=ignore vision, 1=trust fully)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("UNIFIED MODULE TEST")
    print("=" * 60)
    print(f"Robot host: {args.host}")
    print(f"State acquisition: {args.state_hz} Hz")
    print(f"Teleop cameras: {args.teleop}")
    print(f"Mapping: {args.mapping}")
    if args.mapping:
        print(f"  Server: {args.server_host}:{args.server_port}")
        print(f"  Rate: {args.mapping_hz} Hz")
    print(f"Localization: {args.localization}")
    if args.localization:
        print(f"  Server: {args.loc_server_host}:{args.loc_server_port}")
        print(f"  Rate: {args.loc_hz} Hz")
        print(f"  Fusion alpha: {args.fusion_alpha}")
    print(f"Visualization FPS: {args.vis_fps}")
    print("=" * 60 + "\n")

    # Setup
    cfg = ReachyConfig(host=args.host)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    if reachy.mobile_base is None:
        print("[ERROR] No mobile base found.")
        return

    client.turn_on_all()

    # === Events ===
    stop_event = MPEvent()  # Works across processes

    # === Queues ===
    state_queue = Queue(maxsize=5)  # RobotState from robot_state_thread
    mapping_state_queue = Queue(maxsize=5)  # RobotState for mapping thread
    map_queue = Queue(maxsize=5)  # MapState from mapping thread
    loc_state_queue = Queue(maxsize=5)  # RobotState for localization thread
    localized_queue = Queue(maxsize=5)  # Localized RobotState from localization thread
    vis_queue = MPQueue(maxsize=20)  # For vis process (RobotState + MapState)

    # === Localization Fusion (shared state) ===
    camera_buffer: Optional[CameraPoseBuffer] = None
    fusion: Optional[LocalizationFusion] = None

    if args.localization:
        # Create camera pose buffer for delayed localization fusion
        camera_buffer = CameraPoseBuffer(
            max_age_seconds=5.0,
            expected_hz=args.state_hz,
        )

        # Get camera extrinsics for fusion (need to query once)
        T_base_cam = client.get_depth_extrinsics()
        if T_base_cam is not None:
            fusion = LocalizationFusion(
                camera_buffer=camera_buffer,
                T_base_cam=T_base_cam,
                smooth_alpha=args.fusion_alpha,
                max_translation_jump=5.0,
                max_rotation_jump=180.0,
            )
            print(f"[MAIN] Localization fusion initialized")
        else:
            print("[WARN] Could not get depth extrinsics, fusion disabled")

    # === THREADS (main process, I/O-bound) ===
    threads = [
        threading.Thread(
            target=robot_state_loop,
            args=(client, state_queue, stop_event),
            kwargs={
                "hz": args.state_hz,
                "grab_depth": True,
                "grab_teleop": args.teleop,
                "camera_buffer": camera_buffer,  # For localization fusion
            },
            daemon=True,
            name="robot_state",
        ),
        threading.Thread(
            target=teleop_loop,
            args=(client, stop_event),
            kwargs={
                "cmd_hz": CMD_HZ,
                "vx": VX,
                "vy": VY,
                "wz": WZ,
            },
            daemon=True,
            name="teleop",
        ),
    ]

    # Add mapping thread if enabled
    if args.mapping:
        threads.append(
            threading.Thread(
                target=mapping_loop,
                args=(mapping_state_queue, map_queue, stop_event),
                kwargs={
                    "server_host": args.server_host,
                    "server_port": args.server_port,
                    "hz": args.mapping_hz,
                    "depth_scale": DEPTH_SCALE,
                },
                daemon=True,
                name="mapping",
            )
        )

    # Add localization thread if enabled
    if args.localization and fusion is not None:
        threads.append(
            threading.Thread(
                target=localization_loop,
                args=(loc_state_queue, localized_queue, stop_event),
                kwargs={
                    "server_host": args.loc_server_host,
                    "server_port": args.loc_server_port,
                    "hz": args.loc_hz,
                    "include_images": True,
                    "camera_buffer": camera_buffer,
                    "fusion": fusion,
                    "use_offset_correction": True,
                },
                daemon=True,
                name="localization",
            )
        )

    # === PROCESS (separate GIL for visualization) ===
    vis_proc = Process(
        target=vis_process,
        args=(vis_queue, stop_event),
        kwargs={
            "fps": args.vis_fps,
            "show_trajectory": True,
            "show_pointcloud": False,
            "show_camera_frames": True,
            "show_images": True,  # Show RGB, depth, teleop images
            "depth_scale": DEPTH_SCALE,
            "depth_trunc": DEPTH_TRUNC,
        },
        daemon=True,
        name="visualization",
    )

    try:
        # Start all threads
        for t in threads:
            print(f"[MAIN] Starting thread: {t.name}")
            t.start()

        # Start visualization process
        print("[MAIN] Starting visualization process")
        vis_proc.start()

        print("\n[MAIN] All components started")
        print("[MAIN] Press Ctrl+C to stop\n")

        # Main loop: forward RobotState and MapState to vis process
        while not stop_event.is_set():
            try:
                # Get RobotState and forward to vis + mapping + localization
                try:
                    state = state_queue.get_nowait()
                    # Send RobotState to vis process (for images, trajectory)
                    try:
                        vis_queue.put_nowait(state)
                    except:
                        pass  # Queue full, drop

                    # Also send to mapping thread if enabled
                    if args.mapping:
                        try:
                            mapping_state_queue.put_nowait(state)
                        except:
                            pass  # Queue full, drop

                    # Also send to localization thread if enabled
                    if args.localization:
                        try:
                            loc_state_queue.put_nowait(state)
                        except:
                            pass  # Queue full, drop
                except:
                    pass  # No state ready

                # Get MapState from mapping and forward to vis
                if args.mapping:
                    try:
                        map_state = map_queue.get_nowait()
                        # Send MapState to vis process (for point clouds)
                        try:
                            vis_queue.put_nowait(map_state)
                        except:
                            pass  # Queue full, drop
                    except:
                        pass  # No map state ready

                # Drain localized queue (with offset-corrected T_world_base)
                # The localized states could be forwarded to navigation or vis
                if args.localization:
                    try:
                        _ = localized_queue.get_nowait()
                    except:
                        pass  # No localized state ready

                time.sleep(0.01)  # 100 Hz main loop

            except KeyboardInterrupt:
                break

    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")

    finally:
        stop_event.set()

        # Wait for threads
        for t in threads:
            t.join(timeout=2.0)
            if t.is_alive():
                print(f"[MAIN] Warning: thread {t.name} did not stop cleanly")

        # Wait for vis process
        vis_proc.join(timeout=2.0)
        if vis_proc.is_alive():
            print("[MAIN] Terminating visualization process")
            vis_proc.terminate()

        # Cleanup robot
        try:
            client.goto_base_defined_speed(0.0, 0.0, 0.0)
            reachy.mobile_base.turn_off()
        except Exception:
            pass

        client.close()
        print("[MAIN] Shutdown complete")


if __name__ == "__main__":
    main()
