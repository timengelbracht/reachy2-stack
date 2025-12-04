#!/usr/bin/env python
from __future__ import annotations

import time

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.gripper import GripperController  # adjust import if needed


# --- CONFIG -------------------------------------------------------------------
HOST = "192.168.1.72"   # <- change to your Reachy IP
SIDE = "right"          # "left" or "right"
# ------------------------------------------------------------------------------


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    gripper = GripperController(client=client, side=SIDE)

    try:
        print("[GRIPPER] Turning arm on...")
        if SIDE == "right":
            reachy.r_arm.turn_on()
        else:
            reachy.l_arm.turn_on()

        time.sleep(0.5)

        # 1) Fully open / close ----------------------------------------------
        print("[GRIPPER] Opening fully...")
        gripper.open()
        time.sleep(2.0)

        print("[GRIPPER] Closing fully...")
        gripper.close()
        time.sleep(2.0)

        # 2) set_opening (non-goto) ------------------------------------------
        print("[GRIPPER] set_opening to 50%...")
        gripper.set_opening(50.0)
        time.sleep(2.0)

        opening_norm = gripper.get_opening_normalized()
        print(f"[GRIPPER] Normalized opening after set_opening: {opening_norm:.3f}")

        # 3) goto() with percentage=True, blocking ---------------------------
        print("[GRIPPER] goto() to 80% opening (percentage=True, wait=True)...")
        g1 = gripper.goto(
            target=80.0,
            duration=2.0,
            wait=True,
            interpolation_mode="minimum_jerk",
            percentage=True,
        )
        print(f"[GRIPPER] Goto finished, handle={getattr(g1, 'id', None)}")
        opening_norm = gripper.get_opening_normalized()
        print(f"[GRIPPER] Normalized opening after goto 80%: {opening_norm:.3f}")

        time.sleep(1.0)

        # 4) goto() non-blocking + poll via client.is_goto_finished ----------
        print("[GRIPPER] goto() to 10% opening (non-blocking)...")
        g2 = gripper.goto(
            target=10.0,
            duration=3.0,
            wait=False,
            interpolation_mode="linear",
            percentage=True,
        )
        print(f"[GRIPPER] Non-blocking goto handle={getattr(g2, 'id', None)}")

        print("[GRIPPER] Waiting for g2 to finish...")
        while not client.is_goto_finished(g2):
            time.sleep(0.1)

        opening_norm = gripper.get_opening_normalized()
        print(f"[GRIPPER] Normalized opening after goto 10%: {opening_norm:.3f}")

        print("[GRIPPER] Gripper test DONE ✅")

    finally:
        print("[GRIPPER] Turning arm off smoothly and closing client.")
        client.turn_off_all()


if __name__ == "__main__":
    main()
