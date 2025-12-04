#!/usr/bin/env python3
from __future__ import annotations

from reachy2_stack.core.client import ReachyClient, ReachyConfig
from reachy2_stack.control.arm import ArmController
from reachy2_stack.skills.wave import WaveSkill  # adjust import if needed

HOST = "192.168.1.71"   # your Reachy IP
SIDE = "left"          # or "left"


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect_reachy
    client.turn_on_all()

    arm = ArmController(client=client, side=SIDE, world=None)
    wave = WaveSkill(arm=arm)

    wave.run(
        n_waves=5,
        move_duration=0.6,
        pause=0.1,
    )

    client.goto_posture("default")


if __name__ == "__main__":
    main()
