#!/usr/bin/env python3
from __future__ import annotations

from reachy2_stack.core.client import ReachyClient, ReachyConfig
from reachy2_stack.skills.expression import ExpressionSkill

HOST = "192.168.1.71"  # Reachy IP


def main():
    cfg = ReachyConfig(host=HOST)
    client = ReachyClient(cfg)
    client.connect()
    client.turn_on_all()

    expr = ExpressionSkill(client)
    expr.happy()


if __name__ == "__main__":
    main()
