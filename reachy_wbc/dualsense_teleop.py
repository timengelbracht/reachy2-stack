#!/usr/bin/env python3
"""
DualSense PS5 teleop for Reachy2 whole-body IK controller.

Reads the DualSense controller via evdev and publishes incremental
EE pose commands to /target_ee_pose_world (PoseStamped, frame: odom).

Controls:
  Left stick X/Y     → move EE in torso frame (forward-back, left-right)
  Right stick Y       → move EE in Z (up-down)
  Right stick X       → rotate EE yaw
  D-pad up/down       → rotate EE pitch
  D-pad left/right    → rotate EE roll
  L1 / R1             → decrease / increase speed
  Cross (X)           → re-center on current EE pose (sync)
  Circle (O)          → toggle orientation lock (position-only mode)
  Triangle            → print current target pose
  L2 analog           → fine-control mode (proportional slow-down)

Usage:
  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=0
  /usr/bin/python3 dualsense_teleop.py
"""

import threading
import numpy as np
from scipy.spatial.transform import Rotation

import evdev
from evdev import ecodes

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

# Import FK from the IK controller
from wholebody_ik_controller import wholebody_fk, R_ARM_JOINTS

# ── DualSense constants ──────────────────────────────────────────────────

DS_AXIS_CENTER = 128  # midpoint for 0-255 range
DS_AXIS_MAX = 127.0
DS_DEADZONE = 15      # raw units (out of 255)

# Axis codes
AX_LX  = ecodes.ABS_X      # 0: Left stick horizontal
AX_LY  = ecodes.ABS_Y      # 1: Left stick vertical
AX_L2  = ecodes.ABS_Z      # 2: L2 trigger (0=released, 255=full)
AX_RX  = ecodes.ABS_RX     # 3: Right stick horizontal
AX_RY  = ecodes.ABS_RY     # 4: Right stick vertical
AX_R2  = ecodes.ABS_RZ     # 5: R2 trigger
AX_DX  = ecodes.ABS_HAT0X  # 16: D-pad X (-1/0/1)
AX_DY  = ecodes.ABS_HAT0Y  # 17: D-pad Y (-1/0/1)

# Button codes
BTN_CROSS    = ecodes.BTN_SOUTH   # 304
BTN_CIRCLE   = ecodes.BTN_EAST    # 305
BTN_TRIANGLE = ecodes.BTN_NORTH   # 307
BTN_SQUARE   = ecodes.BTN_WEST    # 308
BTN_L1       = ecodes.BTN_TL      # 310
BTN_R1       = ecodes.BTN_TR      # 311


def _normalize_stick(raw, center=DS_AXIS_CENTER, deadzone=DS_DEADZONE):
    """Normalize 0-255 raw value to -1..+1 with deadzone."""
    val = raw - center
    if abs(val) < deadzone:
        return 0.0
    sign = 1.0 if val > 0 else -1.0
    return np.clip(sign * (abs(val) - deadzone) / (DS_AXIS_MAX - deadzone), -1.0, 1.0)


# ── ROS2 Node ─────────────────────────────────────────────────────────────

class DualSenseTeleop(Node):
    def __init__(self):
        super().__init__("dualsense_teleop")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("pos_speed", 0.015)       # m/s at full stick
        self.declare_parameter("rot_speed", 0.06)        # rad/s at full stick
        self.declare_parameter("dpad_rot_speed", 0.05)   # rad/s for d-pad
        self.declare_parameter("publish_rate", 10.0)     # Hz for target pub
        self.declare_parameter("z_min", 0.3)
        self.declare_parameter("z_max", 1.8)

        self.pos_speed = self.get_parameter("pos_speed").value
        self.rot_speed = self.get_parameter("rot_speed").value
        self.dpad_rot_speed = self.get_parameter("dpad_rot_speed").value
        self.pub_rate = self.get_parameter("publish_rate").value
        self.z_min = self.get_parameter("z_min").value
        self.z_max = self.get_parameter("z_max").value

        # ── Speed multiplier (L1/R1 adjustable) ──────────────────────────
        self.speed_scale = 1.0

        # ── EE target state ───────────────────────────────────────────────
        self.target_pos = None   # np.array([x, y, z]) in odom frame
        self.target_rot = None   # Rotation object (quaternion internally, avoids gimbal lock)
        self.ori_locked = False
        self.synced = False      # have we synced with actual EE pose?

        # ── Robot state ───────────────────────────────────────────────────
        self.base_x = None
        self.base_y = None
        self.base_yaw = None
        self.current_joints = np.zeros(7)
        self.joints_received = False

        # ── Joystick state ────────────────────────────────────────────────
        self.axes = {
            AX_LX: DS_AXIS_CENTER, AX_LY: DS_AXIS_CENTER,
            AX_RX: DS_AXIS_CENTER, AX_RY: DS_AXIS_CENTER,
            AX_L2: 0, AX_R2: 0,
            AX_DX: 0, AX_DY: 0,
        }
        self.buttons = {}

        # ── Publisher ─────────────────────────────────────────────────────
        self.target_pub = self.create_publisher(
            PoseStamped, "/target_ee_pose_world", 10
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        # ── Timers ────────────────────────────────────────────────────────
        self.create_timer(1.0 / self.pub_rate, self._control_tick)
        self.create_timer(1.0, self._status_tick)

        # ── Start evdev reader thread ─────────────────────────────────────
        self._find_and_start_reader()

        self.get_logger().info(
            "DualSense teleop started.\n"
            "  Left stick  → XY position\n"
            "  Right stick → Z (vert) / Yaw (horiz)\n"
            "  D-pad       → Roll / Pitch\n"
            "  L1/R1       → Speed -/+\n"
            "  Cross (X)   → Re-sync to current EE\n"
            "  Circle (O)  → Toggle orientation lock\n"
            "  Triangle    → Print pose\n"
            "  L2 trigger  → Fine-control (hold)\n"
            "\nWaiting for /odom and /joint_states..."
        )

    # ── evdev reader ──────────────────────────────────────────────────────

    def _find_and_start_reader(self):
        dev = None
        for path in evdev.list_devices():
            d = evdev.InputDevice(path)
            if "DualSense" in d.name:
                dev = d
                break
        if dev is None:
            self.get_logger().error("DualSense not found! Connect and restart.")
            return
        self.get_logger().info(f"Found DualSense at {dev.path}")
        t = threading.Thread(target=self._evdev_loop, args=(dev,), daemon=True)
        t.start()

    def _evdev_loop(self, dev):
        for event in dev.read_loop():
            if event.type == ecodes.EV_ABS:
                self.axes[event.code] = event.value
            elif event.type == ecodes.EV_KEY:
                if event.value == 1:  # press
                    self._on_button(event.code)

    def _on_button(self, code):
        if code == BTN_CROSS:
            self._sync_to_current_ee()
        elif code == BTN_CIRCLE:
            self.ori_locked = not self.ori_locked
            state = "ON (position-only)" if self.ori_locked else "OFF"
            self.get_logger().info(f"Orientation lock: {state}")
        elif code == BTN_TRIANGLE:
            if self.target_pos is not None:
                rpy = self.target_rot.as_euler('xyz')
                self.get_logger().info(
                    f"Target: pos=({self.target_pos[0]:.3f}, {self.target_pos[1]:.3f}, "
                    f"{self.target_pos[2]:.3f}), rpy=({np.degrees(rpy[0]):.1f}, "
                    f"{np.degrees(rpy[1]):.1f}, {np.degrees(rpy[2]):.1f})deg, "
                    f"speed={self.speed_scale:.1f}x"
                )
        elif code == BTN_L1:
            self.speed_scale = max(0.1, self.speed_scale - 0.25)
            self.get_logger().info(f"Speed: {self.speed_scale:.2f}x")
        elif code == BTN_R1:
            self.speed_scale = min(3.0, self.speed_scale + 0.25)
            self.get_logger().info(f"Speed: {self.speed_scale:.2f}x")

    # ── Robot state callbacks ─────────────────────────────────────────────

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.base_x = p.x
        self.base_y = p.y
        self.base_yaw = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz')[2]

    def _joint_cb(self, msg):
        name_list = list(msg.name)
        for i, jname in enumerate(R_ARM_JOINTS):
            if jname in name_list:
                self.current_joints[i] = msg.position[name_list.index(jname)]
        self.joints_received = True

        # Auto-sync on first receipt
        if not self.synced and self.base_x is not None:
            self._sync_to_current_ee()

    # ── Sync target to current actual EE pose ─────────────────────────────

    def _sync_to_current_ee(self):
        if self.base_x is None or not self.joints_received:
            self.get_logger().warn("Cannot sync: missing odom or joint_states")
            return
        T_ee = wholebody_fk(self.base_x, self.base_y, self.base_yaw,
                            self.current_joints)
        self.target_pos = T_ee[:3, 3].copy()
        self.target_rot = Rotation.from_matrix(T_ee[:3, :3])
        self.synced = True
        self.get_logger().info(
            f"Synced to EE: ({self.target_pos[0]:.3f}, {self.target_pos[1]:.3f}, "
            f"{self.target_pos[2]:.3f})"
        )

    # ── Control loop ──────────────────────────────────────────────────────

    def _control_tick(self):
        if self.target_pos is None:
            return

        dt = 1.0 / self.pub_rate

        # Read sticks
        lx = _normalize_stick(self.axes[AX_LX])
        ly = _normalize_stick(self.axes[AX_LY])
        rx = _normalize_stick(self.axes[AX_RX])
        ry = _normalize_stick(self.axes[AX_RY])
        dpad_x = self.axes[AX_DX]
        dpad_y = self.axes[AX_DY]

        # L2 fine control: 0=normal, 255=maximum slow-down (0.2x)
        l2_raw = self.axes[AX_L2]
        fine_factor = 1.0 - 0.8 * (l2_raw / 255.0)

        scale = self.speed_scale * fine_factor

        # Any input at all?
        has_input = (abs(lx) > 0 or abs(ly) > 0 or abs(rx) > 0 or abs(ry) > 0
                     or dpad_x != 0 or dpad_y != 0)
        if not has_input:
            return

        # Position deltas in torso/base frame, then rotated to odom
        # LY up (negative raw) → forward (torso +X), LX right → right (torso -Y)
        dx_torso = -ly * self.pos_speed * scale * dt
        dy_torso = -lx * self.pos_speed * scale * dt
        dz = -ry * self.pos_speed * scale * dt  # RY up → +Z (same in both frames)

        if self.base_yaw is not None:
            c = np.cos(self.base_yaw)
            s = np.sin(self.base_yaw)
            dx_odom = c * dx_torso - s * dy_torso
            dy_odom = s * dx_torso + c * dy_torso
        else:
            dx_odom = dx_torso
            dy_odom = dy_torso

        self.target_pos[0] += dx_odom
        self.target_pos[1] += dy_odom
        self.target_pos[2] = np.clip(self.target_pos[2] + dz, self.z_min, self.z_max)

        # Orientation deltas (only if not locked) — quaternion accumulation avoids gimbal lock
        if not self.ori_locked:
            dyaw   =  -rx * self.rot_speed * scale * dt        # RX right → negative yaw
            droll  =  -dpad_x * self.dpad_rot_speed * scale * dt
            dpitch =  -dpad_y * self.dpad_rot_speed * scale * dt

            # Apply deltas as small rotations in the current EE frame
            delta_rot = Rotation.from_euler('xyz', [droll, dpitch, dyaw])
            self.target_rot = self.target_rot * delta_rot

        # Publish
        self._publish_target()

    def _publish_target(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x = float(self.target_pos[0])
        msg.pose.position.y = float(self.target_pos[1])
        msg.pose.position.z = float(self.target_pos[2])
        quat = self.target_rot.as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.target_pub.publish(msg)

    def _status_tick(self):
        if self.base_x is None:
            self.get_logger().info("Waiting for /odom...", throttle_duration_sec=5.0)
        elif not self.joints_received:
            self.get_logger().info("Waiting for /joint_states...", throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = DualSenseTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
