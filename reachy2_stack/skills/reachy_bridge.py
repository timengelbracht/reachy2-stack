#!/usr/bin/env python3
"""
reachy_bridge.py

Runs in the *Reachy SDK environment* (your /root/.pyenv/versions/3.11.9/bin/python)
and talks to a separate WBC solver running in another environment (Pinocchio / numpy mismatch)
via ZeroMQ.

Why this exists:
- You said Reachy SDK and Pinocchio environments have incompatible numpy versions.
- So: keep robot control here, keep WBC math in another process/env.
- This bridge:
    1) reads robot state (arm joints + base state)
    2) sends state + desired end-effector pose to the WBC solver server
    3) receives {base_cmd, q_cmd}
    4) executes the command on the robot using the SDK (joint position commands + base cmd)

Expected solver side:
- a ZMQ REP server (e.g., wbc_server.py) listening on tcp://127.0.0.1:5557
- request/response payloads are JSON

If you want, I can also give you the matching `wbc_server.py` that wraps your existing
`wbc_solver.py` and runs under the Pinocchio env.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zmq

from reachy2_stack.utils.utils_dataclass import ReachyConfig
from reachy2_stack.core.client import ReachyClient
from reachy2_stack.infra.world_model import WorldModel, WorldModelConfig
from reachy2_stack.control.arm import ArmController
# If you have a BaseController in your stack, import it.
# from reachy2_stack.control.base import BaseController


# ---------------- CONFIG ----------------
HOST = "192.168.1.71"
SIDE = "right"  # "right" or "left"

# ZMQ endpoint where the solver (running in Pinocchio env) is listening:
SOLVER_ENDPOINT = "tcp://127.0.0.1:5557"

# Control loop
RATE_HZ = 20.0
DT = 1.0 / RATE_HZ

# Safety / limits (tune for your robot)
MAX_BASE_V = 0.25        # m/s
MAX_BASE_W = 0.8         # rad/s
MAX_JOINT_STEP = 0.25    # rad per cycle (simple safety clamp)
# ----------------------------------------


@dataclass
class BaseState:
    x: float
    y: float
    theta: float


@dataclass
class SolverDiag:
    iters: int = 0
    pos_err: float = 0.0
    ori_err: float = 0.0
    converged: float = 0.0


class WbcSolverClient:
    """ZMQ REQ client: send state+target, receive base_cmd+q_cmd."""

    def __init__(self, endpoint: str, timeout_ms: int = 2000) -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(endpoint)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)

    def solve(
        self,
        side: str,
        base_state: BaseState,
        q: np.ndarray,
        T_base_ee_des: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BaseState, np.ndarray, SolverDiag]:
        req: Dict[str, Any] = {
            "side": side,
            "base_state": {"x": base_state.x, "y": base_state.y, "theta": base_state.theta},
            "q": q.astype(float).tolist(),
            "T_base_ee_des": T_base_ee_des.astype(float).tolist(),
            "weights": weights or {},
            "extra": extra or {},
        }

        self._sock.send_string(json.dumps(req))
        raw = self._sock.recv_string()
        resp = json.loads(raw)
        if "error" in resp:
            raise RuntimeError(f"WBC_SERVER error: {resp['error']}")

        base_cmd = BaseState(
            x=float(resp["base_cmd"]["x"]),
            y=float(resp["base_cmd"]["y"]),
            theta=float(resp["base_cmd"]["theta"]),
        )
        q_cmd = np.array(resp["q_cmd"], dtype=float)

        d = resp.get("diag", {})
        diag = SolverDiag(
            iters=int(d.get("iters", 0)),
            pos_err=float(d.get("pos_err", 0.0)),
            ori_err=float(d.get("ori_err", 0.0)),
            converged=float(d.get("converged", 0.0)),
        )
        return base_cmd, q_cmd, diag


class ReachyBridge:
    """
    - Reads robot state via ReachyClient/ArmController
    - Calls WBC solver over ZMQ
    - Applies commands via SDK (arm joint positions + base cmd)
    """

    def __init__(self, host: str, side: str, solver_endpoint: str) -> None:
        self.side = side

        cfg = ReachyConfig(host=host)
        self.client = ReachyClient(cfg)
        self.client.connect()

        # World model can help access odom/world frames (even without ROS)
        self.world = WorldModel(WorldModelConfig(location_name="lab", localization_mode="odom"))

        self.arm = ArmController(client=self.client, side=self.side, world=self.world)
        self.reachy = self.client.connect_reachy

        # If you have a BaseController, you can use it; otherwise use SDK base object.
        # self.base = BaseController(client=self.client, world=self.world)
        self.solver = WbcSolverClient(solver_endpoint)

        # Keep a local base estimate if you cannot query odom cleanly.
        self._base_est = BaseState(x=0.0, y=0.0, theta=0.0)

    # -------------------- State IO --------------------

    def get_arm_q(self) -> np.ndarray:
        if self.side == "right":
            q_raw, _ = self.client.get_joint_state_right()
        else:
            q_raw, _ = self.client.get_joint_state_left()

        # q_raw is already in radians (your print proves it)
        return np.array(q_raw, dtype=float)

    def get_base_state(self) -> BaseState:
        """
        Best effort.
        If you have a proper odom getter in your stack, use it here.
        Otherwise we keep an internal estimate updated from commanded velocities.
        """
        # If there is an SDK getter (example names — adjust to your API):
        #   bs = self.client.get_base_state() or self.world.get_base_pose() ...
        # return BaseState(bs.x, bs.y, bs.theta)

        return self._base_est

    # -------------------- Command Application --------------------

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    def _apply_arm_q(self, q_cmd: np.ndarray, duration: float = 0.5) -> None:
        q_cur = self.get_arm_q()
        dq = q_cmd - q_cur
        step = np.clip(dq, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        q_next = q_cur + step

        # Send radians directly (no rad2deg!)
        self.arm.goto_joints(q_next, duration=max(duration, 0.5), wait=False)


    def _apply_base_cmd(self, base_cmd: BaseState, dt: float) -> None:
        """
        You said base accepts (x,y,theta) or velocity commands.
        Here we interpret solver output as a *desired base pose increment* relative to current estimate.
        If your solver returns absolute base pose, set self._base_est directly.
        If your SDK accepts velocity commands, compute v,w and send those.
        """

        # Interpret solver output as *absolute* base pose in odom:
        # (This is what your terminal printed: BaseState(x=?, y=?, theta=?))
        # We'll compute a velocity command that moves toward it.
        cur = self.get_base_state()
        dx = base_cmd.x - cur.x
        dy = base_cmd.y - cur.y
        dth = base_cmd.theta - cur.theta

        # naive proportional conversion -> velocities
        # (replace with something better if you want)
        v_x = self._clamp(dx / max(dt, 1e-6), -MAX_BASE_V, MAX_BASE_V)
        v_y = self._clamp(dy / max(dt, 1e-6), -MAX_BASE_V, MAX_BASE_V)
        w_z = self._clamp(dth / max(dt, 1e-6), -MAX_BASE_W, MAX_BASE_W)

        # Apply to robot base
        # Adjust these calls to your actual SDK:
        # Examples (you MUST adapt to your Reachy SDK API):
        #
        # 1) If you have a velocity API:
        #     self.reachy.mobile_base.set_speed(vx=v_x, vy=v_y, vtheta=w_z)
        #
        # 2) If you have a goto pose API:
        #     self.reachy.mobile_base.goto(x=base_cmd.x, y=base_cmd.y, theta=base_cmd.theta)
        #
        # I’ll provide a safe "try a few common names" approach.

        mb = getattr(self.reachy, "mobile_base", None)
        if mb is not None:
            if hasattr(mb, "set_speed"):
                mb.set_speed(vx=v_x, vy=v_y, vtheta=w_z)
            elif hasattr(mb, "set_velocity"):
                mb.set_velocity(vx=v_x, vy=v_y, omega=w_z)
            elif hasattr(mb, "goto"):
                mb.goto(x=base_cmd.x, y=base_cmd.y, theta=base_cmd.theta)
            else:
                # As a last resort, do nothing but keep estimate updated.
                pass

        # Update internal estimate (dead-reckoning)
        self._base_est = BaseState(
            x=cur.x + v_x * dt,
            y=cur.y + v_y * dt,
            theta=cur.theta + w_z * dt,
        )

    # -------------------- Main Loop --------------------

    def run_cartesian(
        self,
        T_base_ee_des: np.ndarray,
        run_seconds: float = 5.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Hold a desired EE pose (in *base frame*, consistent with your ArmController FK),
        and let WBC move base+arm to reduce error.
        """
        t0 = time.time()
        next_t = t0

        while True:
            now = time.time()
            if now - t0 > run_seconds:
                break
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                continue
            next_t += DT

            # Read state
            q = self.get_arm_q()
            print(q)
            base_state = self.get_base_state()

            # Ask solver
            try:
                base_cmd, q_cmd, diag = self.solver.solve(
                    side=self.side,
                    base_state=base_state,
                    q=q,
                    T_base_ee_des=T_base_ee_des,
                    weights=weights,
                )
            except zmq.error.Again:
                print("[BRIDGE] Solver timeout")
                continue
            except Exception as e:
                print(f"[BRIDGE] Solver error: {e}")
                continue

            # Apply
            self._apply_arm_q(q_cmd, duration=0.0)
            self._apply_base_cmd(base_cmd, dt=DT)

            print(
                f"[BRIDGE] pos_err={diag.pos_err:.3f}  ori_err={diag.ori_err:.3f}  "
                f"iters={diag.iters:3d}  conv={diag.converged:.0f}  "
                f"base=({base_cmd.x:+.3f},{base_cmd.y:+.3f},{base_cmd.theta:+.3f})"
            )

    def close(self) -> None:
        try:
            if self.side == "right":
                self.reachy.r_arm.turn_off_smoothly()
            else:
                self.reachy.l_arm.turn_off_smoothly()
        except Exception:
            pass
        self.client.close()


def main() -> None:
    bridge = ReachyBridge(host=HOST, side=SIDE, solver_endpoint=SOLVER_ENDPOINT)

    try:
        bridge.client.turn_on_all()
        bridge.client.goto_posture("default", wait=True)

        # Example: take current FK pose and nudge 5 cm “forward” in base frame
        # NOTE: ArmController.forward_kinematics() in your snippet returns a 4x4 T_base_ee.
        T_fk = bridge.arm.forward_kinematics()
        T_des = np.array(T_fk, dtype=float).copy()

        # IMPORTANT: in your earlier example you used T_target[2,3] but commented "+X".
        # In homogeneous transforms: [0,3]=x, [1,3]=y, [2,3]=z.
        # So choose what you actually want.
        T_des[0, 3] += 0.05  # +5 cm in +X of base frame

        print("[BRIDGE] Running WBC to reach desired EE pose for 5s...")
        bridge.run_cartesian(T_base_ee_des=T_des, run_seconds=5.0)

        print("[BRIDGE] Done ✅")

    finally:
        bridge.close()


if __name__ == "__main__":
    main()
