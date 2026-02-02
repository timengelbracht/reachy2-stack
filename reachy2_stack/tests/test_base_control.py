#!/usr/bin/env python
from __future__ import annotations

import time

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.control.base import BaseController  # adjust import if needed


# --- CONFIG -------------------------------------------------------------------
HOST = "192.168.1.71"   # <- change to your Reachy IP
# ------------------------------------------------------------------------------


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    if not hasattr(reachy, "mobile_base") or reachy.mobile_base is None:
        print("[BASE] No mobile_base available on this Reachy. Exiting.")
        client.close()
        return

    base = BaseController(client=client, world=None)
    client.turn_on_all()

    try:
        print("[BASE] Resetting odometry...")
        base.reset_odometry()
        time.sleep(0.5)

        # 1) Simple goto forward 0.3 m --------------------------------------
        print("[BASE] Goto x=0.30, y=0.00, theta=0° (wait=True)...")
        g1 = base.goto_odom(
            x=0.3,
            y=0.0,
            theta=0.0,
            wait=True,
            distance_tolerance=0.05,
            angle_tolerance=5.0,
            timeout=20.0,
        )
        print(f"[BASE] Goto finished, handle={getattr(g1, 'id', None)}")
        print("[BASE] Odometry after g1:", reachy.mobile_base.get_current_odometry())

        time.sleep(1.0)

        # 2) Rotate by 90° in robot frame -----------------------------------
        print("[BASE] rotate_by +90° (wait=True)...")
        g2 = base.rotate_by(theta=90.0, wait=True, degrees=True)
        print(f"[BASE] rotate_by finished, handle={getattr(g2, 'id', None)}")
        print("[BASE] Odometry after rotation:", reachy.mobile_base.get_current_odometry())

        time.sleep(1.0)

        # 3) translate_by forward 0.2 m in robot frame ----------------------
        print("[BASE] translate_by x=0.2 (wait=True)...")
        g3 = base.translate_by(x=0.2, y=0.0, wait=True)
        print(f"[BASE] translate_by finished, handle={getattr(g3, 'id', None)}")
        print("[BASE] Odometry after translate:", reachy.mobile_base.get_current_odometry())

        # 4) Chained gotos (non-blocking) -----------------------------------
        print("[BASE] Chaining two odom gotos (non-blocking)...")
        g4 = base.goto_odom(x=0.0, y=0.0, theta=0.0, wait=False)
        
        g5 = base.goto_odom(x=0.4, y=0.0, theta=0.0, wait=False)

        print(f"[BASE] g4 id={getattr(g4, 'id', None)}, g5 id={getattr(g5, 'id', None)}")
        print("[BASE] Polling g5 via client.is_goto_finished...")
        while not client.is_goto_finished(g5):
            time.sleep(0.1)

        print("[BASE] Final odometry after chained gotos:",
              reachy.mobile_base.get_current_odometry())

        print("[BASE] Base test DONE ✅")

    finally:
        print("[BASE] Turning mobile base off and closing client.")
        try:
            reachy.mobile_base.turn_off()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()