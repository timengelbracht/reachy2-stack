#!/usr/bin/env python3
from __future__ import annotations

import threading
import time

from reachy2_stack.core.client import ReachyClient, ReachyConfig
from reachy2_stack.control.arm import ArmController
from reachy2_stack.control.gripper import GripperController
from reachy2_stack.skills.wave import WaveSkill
from reachy2_stack.skills.expression import ExpressionSkill
from reachy2_stack.skills.audio import AudioSkill
import numpy as np

HOST = "192.168.1.71"              # Reachy IP
SIDE = "right"                      # "left" or "right"
LOCAL_AUDIO = "/exchange/data/audio/carey.wav"   # local path on your machine
REMOTE_AUDIO = "carey.wav"         # name on Reachy after upload
SONG_DURATION = 10.0               # seconds (approx length of the song)


def main() -> None:
    # --- connect + power on --------------------------------------------------
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)

    # Trigger SDK connection (property-style)
    client.connect()
    client.turn_on_all()

    q_apex = np.array([
        -20.0,   # shoulder pitch
        0.0,   # shoulder roll
        0.0,   # arm yaw
        -130.0,  # elbow pitch -120,
        0.0,   # wrist yaw
        0.0,   # wrist pitch
        -30.0,   # wrist roll
    ], dtype=float)


    # --- build skills --------------------------------------------------------
    arm = ArmController(client=client, side="right", world=None)
    arm_left = ArmController(client=client, side="left", world=None)
    wave = WaveSkill(arm=arm)
    expr = ExpressionSkill(client=client)
    audio = AudioSkill(client=client)
    gripper_left = GripperController(client=client, side="left")
    gripper_left.set_opening(opening_percent=35.0)

    # move to apex first
    arm_left.goto_joints(q_apex, duration=2.0, wait=True)



    # Upload audio (safe to call even if already on Reachy)
    print(f"Uploading audio file: {LOCAL_AUDIO}")
    audio.upload(LOCAL_AUDIO)
    print("Files on Reachy:", audio.list_files())

    # Shared stop flag: set when song is done (timer) or user stops
    stop_event = threading.Event()

    # --- worker: motions (wave + happy in a loop) ---------------------------
    def motion_worker() -> None:
        while not stop_event.is_set():
            # Shorter wave chunk so we can check stop_event between cycles
            wave.run(
                n_waves=2,
                move_duration=0.6,
                pause=0.1,
            )
            if stop_event.is_set():
                break

            expr.happy(
                head_duration=1.0,
                antenna_duration=0.4,
                pause=0.1,
            )

    # --- worker: audio playback ---------------------------------------------
    def audio_worker() -> None:
        # Just play the file once; motion will run while audio plays
        audio.play_once(REMOTE_AUDIO)

    # --- worker: timer to auto-stop after SONG_DURATION ---------------------
    def timer_worker() -> None:
        if SONG_DURATION is None:
            return
        time.sleep(SONG_DURATION)
        if not stop_event.is_set():
            print("\n[Timer] Song duration reached. Stopping motions.")
            stop_event.set()

    # --- start threads -------------------------------------------------------
    motion_thread = threading.Thread(target=motion_worker, daemon=True)
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    timer_thread = threading.Thread(target=timer_worker, daemon=True)

    motion_thread.start()
    audio_thread.start()
    timer_thread.start()

    # --- wait for user input to stop early ----------------------------------
    input("\nPress ENTER at any time to stop wave + happy + audio...\n")
    stop_event.set()

    # Make sure everything winds down
    print("Stopping audio and waiting for motion thread to finish...")
    audio.stop()
    motion_thread.join(timeout=5.0)

    # --- back to default posture --------------------------------------------
    print("Returning to default posture...")
    client.goto_posture("default")

    print("Demo finished.")


if __name__ == "__main__":
    main()
