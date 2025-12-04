from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel  

import time



@dataclass
class GripperController:
    """Small helper for gripper control."""

    client: ReachyClient
    side: str  # "left" or "right"
    poll_dt: float = 0.02

    def goto(
        self,
        target: float,
        duration: float = 2.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
        degrees: bool = True,
        percentage: bool = False,
    ):
        return self.client.gripper_goto(
            side=self.side,
            target=target,
            duration=duration,
            wait=wait,
            interpolation_mode=interpolation_mode,
            degrees=degrees,
            percentage=percentage,
        )

    def open(
        self,
        duration: float = 1.0,
        wait: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> None:
        """Fully open gripper using goto; optionally block until finished."""
        handle = self.goto(
            target=100.0,          # 100% open
            duration=duration,
            wait=False,            # we handle waiting manually
            interpolation_mode=interpolation_mode,
            degrees=False,
            percentage=True,
        )
        if wait:
            while not self.client.is_goto_finished(handle):
                time.sleep(self.poll_dt)

    def close(
        self,
        duration: float = 1.0,
        wait: bool = True,
        interpolation_mode: str = "minimum_jerk",
    ) -> None:
        """Fully close gripper using goto; optionally block until finished."""
        handle = self.goto(
            target=0.0,            # 0% open = fully closed
            duration=duration,
            wait=False,
            interpolation_mode=interpolation_mode,
            degrees=False,
            percentage=True,
        )
        if wait:
            while not self.client.is_goto_finished(handle):
                time.sleep(self.poll_dt)

    def set_opening(
        self,
        opening_percent: float,
        duration: float = 1.0,
        wait: bool = False,
        interpolation_mode: str = "minimum_jerk",
    ) -> None:
        """Set gripper opening in [0, 100] percent."""
        handle = self.goto(
            target=float(opening_percent),
            duration=duration,
            wait=False,
            interpolation_mode=interpolation_mode,
            degrees=False,
            percentage=True,
        )
        if wait:
            while not self.client.is_goto_finished(handle):
                time.sleep(self.poll_dt)

    def get_opening_normalized(self) -> float:
        if self.side == "right":
            return self.client.get_gripper_opening_right()
        elif self.side == "left":
            return self.client.get_gripper_opening_left()
        else:
            raise ValueError(f"Unknown side: {self.side!r}")