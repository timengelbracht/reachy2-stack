from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from reachy2_stack.control.arm import ArmController


@dataclass
class WaveSkill:
    """Minimal waving skill in joint space.

    - Uses a fixed apex pose in joint space (q_apex) depending on arm side.
    - Waves by moving only the elbow pitch joint (index 3)
      between -80° and -120°.
    """

    arm: ArmController

    def _get_q_apex(self) -> np.ndarray:
        """Return side-specific apex pose."""
        side = self.arm.side
        if side == "right":
            return np.array([
                0.0,    # shoulder pitch
                -50.0,  # shoulder roll
                -90.0,  # arm yaw
                -120.0, # elbow pitch
                0.0,    # wrist yaw
                0.0,    # wrist pitch
                -90.0,  # wrist roll
            ], dtype=float)
        elif side == "left":
            return np.array([
                0.0,    # shoulder pitch
                50.0,   # shoulder roll
                90.0,   # arm yaw
                -120.0, # elbow pitch
                0.0,    # wrist yaw
                0.0,    # wrist pitch
                -90.0,  # wrist roll
            ], dtype=float)
        else:
            raise ValueError(f"Unsupported arm side: {side!r} (expected 'left' or 'right').")

    def run(
        self,
        n_waves: int = 3,
        move_duration: float = 0.6,
        pause: float = 0.1,
    ) -> None:
        # --- base pose (apex) -------------------------------------------------
        q_apex = self._get_q_apex()

        ELBOW_IDX = 3

        # poses for elbow wave
        q_up = q_apex.copy()
        q_down = q_apex.copy()

        q_up[ELBOW_IDX] = -80.0    # a bit more open
        q_down[ELBOW_IDX] = -120.0 # a bit more closed

        # 1) go to apex first
        self.arm.goto_joints(q_apex, duration=1.5, wait=True)

        # 2) wave: -80 -> -120 -> -80 (per cycle)
        for _ in range(n_waves):
            self.arm.goto_joints(q_up, duration=move_duration, wait=True)
            time.sleep(pause)

            self.arm.goto_joints(q_down, duration=move_duration, wait=True)
            time.sleep(pause)

            self.arm.goto_joints(q_up, duration=move_duration, wait=True)
            time.sleep(pause)

        # 3) back to clean apex at the end
        self.arm.goto_joints(q_apex, duration=move_duration, wait=True)
