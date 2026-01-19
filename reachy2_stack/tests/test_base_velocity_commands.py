#!/usr/bin/env python
from __future__ import annotations

import time
import numpy as np
from matplotlib import pyplot as plt
import json

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig
from reachy2_stack.control.arm import ArmController  # adjust import if needed
from reachy2_stack.control.base import BaseController 


# --- CONFIG -------------------------------------------------------------------
HOST = "192.168.1.71"
SAVE_PATH = "/exchange/arm_traj.npz"
# ------------------------------------------------------------------------------


def main() -> None:
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    reachy = client.connect_reachy

    #initialize the base controller
    base = BaseController(client=client, world=None)

    reachy.mobile_base.turn_off()
    base.reset_odometry()
    #turn on the drivers. Warning, do not be close to the robot or try to move it/ its arms
    reachy.mobile_base.turn_on()

    for _ in range (25):
        client.goto_base_defined_speed(0.6,0.0,0.0)
        time.sleep(0.1)
    
    for _ in range (25):
        client.goto_base_defined_speed(-0.6,0.0,0.0)
        time.sleep(0.1)

    for _ in range (25):
        client.goto_base_defined_speed(0.0,0.6,0.0)
        time.sleep(0.1)

    for _ in range (25):
        client.goto_base_defined_speed(0.0,-0.6,0.0)
        time.sleep(0.1)
    
    for _ in range (25):
        client.goto_base_defined_speed(0.0,0.0,110.0)
        time.sleep(0.1)

    for _ in range (25):
        client.goto_base_defined_speed(0.0,0.0,-110.0)
        time.sleep(0.1)

    # turn everything off 

    reachy.mobile_base.turn_off()
    client.close()

if __name__ == "__main__":
    main()
