#!/usr/bin/env python3
from __future__ import annotations

from reachy2_stack.core.client import ReachyClient, ReachyConfig
from reachy2_stack.skills.audio import AudioSkill

HOST = "192.168.1.71"             # Your Reachy IP
LOCAL_AUDIO = "/exchange/data/audio/santa.wav"   # Path to local file
REMOTE_NAME = "santa.wav"          # Filename on Reachy


def main() -> None:
    # --- Connect to Reachy ---------------------------------------------------
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect_reachy  # initialize SDK

    audio = AudioSkill(client)

    print("=== AUDIO STOP TEST ===")

    # --- Upload --------------------------------------------------------------
    print(f"Uploading: {LOCAL_AUDIO}")
    audio.upload(LOCAL_AUDIO)

    print("Files on Reachy:")
    print(audio.list_files())

    # --- Play ----------------------------------------------------------------
    print("\nPlaying audio...")
    audio.play_once(REMOTE_NAME)

    # --- Wait for terminal input to stop ------------------------------------
    input("\nPress ENTER to STOP audio playback...")

    print("Stopping audio!")
    audio.stop()

    print("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    main()