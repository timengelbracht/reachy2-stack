#!/usr/bin/env python3
"""
Interactive debugging script for verifying:
- Can we connect to Reachy?
- Do we get valid joint states?
- Are config + client wired correctly?

Run with:
    python tests/test_robot_connection.py
"""
import time
import pprint

from reachy2_stack.core.client import ReachyClient, ReachyConfig


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # ------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------
    print_header("Loading config")
    cfg = ReachyConfig.from_yaml("config/config.yaml")
    pprint.pprint(cfg)

    # ------------------------------------------------------
    # 2. Connect to robot or simulator
    # ------------------------------------------------------
    print_header("Connecting to Reachy")
    client = ReachyClient(cfg)
    client.connect()
    print("Connected.")


if __name__ == "__main__":
    main()
