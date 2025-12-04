from __future__ import annotations

from dataclasses import dataclass
import time

from reachy2_stack.core.client import ReachyClient


@dataclass
class ExpressionSkill:
    """A set of expressive head + antenna animations.

    Currently supports:
        - happy()
    More expressions can be added later.
    """

    client: ReachyClient

    # ----------------------------------------------------------
    # Internal utilities
    # ----------------------------------------------------------

    def _head_on(self):
        """Ensure head + antennas are stiff before moving."""
        reachy = self.client.connect_reachy
        reachy.head.turn_on()

    def _neutral_head(self, duration=0.8):
        """Move head back to neutral roll/pitch/yaw."""
        reachy = self.client.connect_reachy
        reachy.head.goto([0.0, 0.0, 0.0], duration=duration, wait=True)

    def _neutral_antennas(self, duration=0.4):
        reachy = self.client.connect_reachy
        reachy.head.l_antenna.goto(0.0, duration=duration, wait=False)
        reachy.head.r_antenna.goto(0.0, duration=duration, wait=True)

    # ----------------------------------------------------------
    # PUBLIC EXPRESSIONS
    # ----------------------------------------------------------

    def happy(
        self,
        head_duration: float = 1.0,
        antenna_duration: float = 0.4,
        pause: float = 0.1,
    ) -> None:
        """A cute 'happy' gesture:
        - slight head tilt
        - look around
        - perk up antennas
        - little antenna wiggle
        """

        reachy = self.client.connect_reachy
        self._head_on()

        # 1) Neutral first
        self._neutral_head(duration=0.4)
        time.sleep(pause)

        # 2) Friendly slight head tilt
        reachy.head.goto(
            [5.0, -10.0, 5.0],   # roll, pitch, yaw
            duration=head_duration,
            wait=True,
        )
        time.sleep(pause)

        # 3) Look slightly right
        reachy.head.look_at(x=0.5, y=-0.3, z=0.1, duration=head_duration, wait=True)
        time.sleep(pause)

        # 4) Look slightly left
        reachy.head.look_at(x=0.5, y=0.3, z=0.1, duration=head_duration, wait=True)
        time.sleep(pause)

        # 5) Return front
        reachy.head.look_at(x=0.5, y=0.0, z=0.2, duration=head_duration, wait=True)
        time.sleep(pause)

        # 6) Antennas: perk up
        reachy.head.l_antenna.goto(60.0, duration=antenna_duration, wait=False)
        reachy.head.r_antenna.goto(-60.0, duration=antenna_duration, wait=True)
        time.sleep(pause)

        # 7) Antenna wiggle
        for _ in range(2):
            reachy.head.l_antenna.goto(30.0, duration=antenna_duration, wait=False)
            reachy.head.r_antenna.goto(-30.0, duration=antenna_duration, wait=True)
            time.sleep(pause)

            reachy.head.l_antenna.goto(60.0, duration=antenna_duration, wait=False)
            reachy.head.r_antenna.goto(-60.0, duration=antenna_duration, wait=True)
            time.sleep(pause)

        # 8) Reset everything
        self._neutral_antennas()
        self._neutral_head()
