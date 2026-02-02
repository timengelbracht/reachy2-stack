#!/usr/bin/env python3
"""Unified test script with hybrid threading + multiprocessing architecture.

This script demonstrates the new architecture:
- robot_state_thread: Unified camera + odometry -> RobotState queue
- teleop_thread: Keyboard control (unchanged)
- mapping_thread: Processes RobotState (local or server mode)
- navigation_thread: Path following (optional)
- vis_process: Separate process for visualization (no GIL contention)

Usage:
    # Local mapping mode (CPU-bound, holds GIL)
    python reachy2_stack/tests/module_test_unified.py

    # Server mapping mode (I/O-bound, parallel-friendly)
    # First start the wavemap server:
    #   python ext_server/wavemap_server.py --port 5555
    # Then run:
    python reachy2_stack/tests/module_test_unified.py --mode server
"""

import argparse
import time
import threading
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
from queue import Queue

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.base_module import teleop_loop
from reachy2_stack.base_module.robot_state_loop import robot_state_loop
from reachy2_stack.base_module.mapping import mapping_loop_unified
from reachy2_stack.base_module.visualization import vis_process

# ---------------- CONFIG ----------------
HOST = "192.168.1.71"

# Robot state acquisition
STATE_HZ = 20  # Hz for camera + odometry

# Teleop
CMD_HZ = 30
VX = 0.6
VY = 0.6
WZ = 110.0

# Mapping
MAPPING_MODE = "local"  # "local" or "server"
MAPPING_HZ = 2.0
MAPPING_SERVER_HOST = "localhost"
MAPPING_SERVER_PORT = 5555
DEPTH_SCALE = 0.001  # mm to meters
DEPTH_TRUNC = 3.5  # meters

# Visualization
VIS_FPS = 30
# --------------------------------------


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified module test")
    parser.add_argument(
        "--mode",
        choices=["local", "server"],
        default=MAPPING_MODE,
        help="Mapping mode: local (CPU) or server (ZMQ)",
    )
    parser.add_argument(
        "--host",
        default=HOST,
        help="Robot host IP",
    )
    parser.add_argument(
        "--server-host",
        default=MAPPING_SERVER_HOST,
        help="Wavemap server host (for server mode)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=MAPPING_SERVER_PORT,
        help="Wavemap server port (for server mode)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("UNIFIED MODULE TEST")
    print("=" * 60)
    print(f"Robot host: {args.host}")
    print(f"Mapping mode: {args.mode}")
    if args.mode == "server":
        print(f"Wavemap server: {args.server_host}:{args.server_port}")
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
    mapping_result_queue = Queue(maxsize=5)  # Results from mapping_thread
    vis_queue = MPQueue(maxsize=20)  # For vis process

    # === THREADS (main process, I/O-bound) ===
    threads = [
        threading.Thread(
            target=robot_state_loop,
            args=(client, state_queue, stop_event),
            kwargs={
                "hz": STATE_HZ,
                "grab_depth": True,
                "grab_teleop": False,
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
        threading.Thread(
            target=mapping_loop_unified,
            args=(state_queue, mapping_result_queue, stop_event),
            kwargs={
                "mode": args.mode,
                "server_host": args.server_host,
                "server_port": args.server_port,
                "hz": MAPPING_HZ,
                "depth_scale": DEPTH_SCALE,
                "depth_trunc": DEPTH_TRUNC,
            },
            daemon=True,
            name="mapping",
        ),
    ]

    # === PROCESS (separate GIL for visualization) ===
    vis_proc = Process(
        target=vis_process,
        args=(vis_queue, stop_event),
        kwargs={
            "fps": VIS_FPS,
            "show_trajectory": True,
            "show_pointcloud": True,
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

        # Main loop: forward data from mapping results to vis process
        last_state = None
        while not stop_event.is_set():
            try:
                # Get mapping results and forward to vis
                try:
                    result = mapping_result_queue.get_nowait()

                    # Build vis data packet
                    vis_data = {}

                    # Add odometry from state
                    if "state" in result and result["state"] is not None:
                        state = result["state"]
                        vis_data["odom_x"] = state.odom_x
                        vis_data["odom_y"] = state.odom_y
                        vis_data["odom_theta"] = state.odom_theta

                    # Add point cloud data
                    if "occupied_points" in result:
                        vis_data["occupied_points"] = result["occupied_points"]
                    if "occupied_colors" in result:
                        vis_data["occupied_colors"] = result["occupied_colors"]
                    if "free_points" in result:
                        vis_data["free_points"] = result["free_points"]
                    if "points" in result:
                        vis_data["points"] = result["points"]
                    if "colors" in result:
                        vis_data["colors"] = result["colors"]

                    # Send to vis process
                    try:
                        vis_queue.put_nowait(vis_data)
                    except:
                        pass  # Queue full, drop

                except:
                    pass  # No results ready

                time.sleep(0.02)  # 50 Hz main loop

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
