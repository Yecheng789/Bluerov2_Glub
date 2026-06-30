#!/usr/bin/env python3
import sys
import time
import select
import termios
import tty
from typing import Dict

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class KeyboardCmdVel(Node):
    """
    Keyboard teleop for BlueROV2 via geometry_msgs/Twist on cmd_vel.

    Recommended key mapping:
      w / s : forward / backward       -> linear.x
      a / d : left / right             -> linear.y
      r / f : up / down                -> linear.z
      u / o : roll  + / -              -> angular.x
      i / k : pitch + / -              -> angular.y
      j / l : yaw   + / -              -> angular.z

      1 / 2 : decrease / increase linear scale
      3 / 4 : decrease / increase angular scale
      space/x: emergency stop
      h      : print help
      Ctrl-C : quit
    """

    def __init__(self) -> None:
        super().__init__("keyboard_cmd_vel")

        # Parameters
        self.declare_parameter("cmd_vel_topic", "/itrl_rov_1/cmd_vel")
        self.declare_parameter("loop_hz", 30.0)
        self.declare_parameter("hold_timeout", 0.25)

        self.declare_parameter("linear_scale", 0.35)
        # Use a safer default angular scale to avoid aggressive yaw steps
        self.declare_parameter("angular_scale", 0.50)

        self.declare_parameter("linear_step", 0.05)
        self.declare_parameter("angular_step", 0.10)

        self.declare_parameter("max_linear_scale", 0.80)
        self.declare_parameter("max_angular_scale", 1.50)
        
        self.declare_parameter("debug_keys", False)  # Enable verbose key debug logging
        # Smoothing / slew rate (units: per second). Prevents abrupt steps.
        self.declare_parameter("slew_rate", 5.0)

        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.loop_hz = float(self.get_parameter("loop_hz").value)
        self.hold_timeout = float(self.get_parameter("hold_timeout").value)

        self.linear_scale = float(self.get_parameter("linear_scale").value)
        self.angular_scale = float(self.get_parameter("angular_scale").value)
        self.linear_step = float(self.get_parameter("linear_step").value)
        self.angular_step = float(self.get_parameter("angular_step").value)
        self.max_linear_scale = float(self.get_parameter("max_linear_scale").value)
        self.max_angular_scale = float(self.get_parameter("max_angular_scale").value)
        self.debug_keys = bool(self.get_parameter("debug_keys").value)
        self.slew_rate = float(self.get_parameter("slew_rate").value)

        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Per-axis command state with independent timeout
        self.axis_cmd: Dict[str, float] = {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.0,
        }
        self.axis_expiry: Dict[str, float] = {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.0,
        }

        # Previous published axis values (used for smoothing)
        self.prev_axis_cmd: Dict[str, float] = {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.0,
        }

        self.fd = None
        self.old_term_settings = None
        self.tty_ok = False
        self._setup_terminal()

        self.timer = self.create_timer(1.0 / self.loop_hz, self._on_timer)

        self.get_logger().info("=" * 60)
        self.get_logger().info("Keyboard Teleop for BlueROV2 (6DoF)")
        self.get_logger().info(f"Publishing Twist to: {self.cmd_vel_topic}")
        self.get_logger().info(f"Loop: {self.loop_hz} Hz | Hold timeout: {self.hold_timeout:.3f}s")
        self.get_logger().info(f"Linear scale: {self.linear_scale:.2f} | Angular scale: {self.angular_scale:.2f}")
        self.get_logger().info("=" * 60)
        self._print_help()
        self._publish_current()

    def _setup_terminal(self) -> None:
        if not sys.stdin.isatty():
            self.get_logger().warn(
                "stdin is not a TTY. Keyboard input will not work in this shell."
            )
            return

        try:
            self.fd = sys.stdin.fileno()
            self.old_term_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.tty_ok = True
            self.get_logger().info("✓ Terminal configured for keyboard input (cbreak mode)")
        except Exception as exc:
            self.get_logger().error(f"✗ Failed to configure terminal: {exc}")
            self.tty_ok = False

    def _restore_terminal(self) -> None:
        if self.tty_ok and self.fd is not None and self.old_term_settings is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_term_settings)
            except Exception:
                pass

    def destroy_node(self):
        self._restore_terminal()
        super().destroy_node()

    def _print_help(self) -> None:
        msg = f"""
================ BlueROV2 keyboard cmd_vel teleop ================
Publishing to: {self.cmd_vel_topic}

Motion:
  w / s : forward / backward       (linear.x)
  a / d : left / right             (linear.y)
  r / f : up / down                (linear.z)

Rotation:
  u / o : roll  + / -
  i / k : pitch + / -
  j / l : yaw   + / -

Scales:
  1 / 2 : decrease / increase linear scale   (current: {self.linear_scale:.2f})
  3 / 4 : decrease / increase angular scale  (current: {self.angular_scale:.2f})

Safety:
  space or x : STOP all axes
  h          : print this help again
  Ctrl-C     : quit
==================================================================
"""
        print(msg, flush=True)

    def _read_key_nonblocking(self):
        if not self.tty_ok:
            return None

        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None

        ch = sys.stdin.read(1)
        if ch == "\x03":  # Ctrl-C
            raise KeyboardInterrupt

        # Swallow ANSI escape sequences (e.g. arrow keys) so they do not pollute input
        # ESC character starts a sequence, read up to 3 more bytes safely
        if ch == "\x1b":
            # Read escape sequence characters (usually [, then A-Z or other)
            for _ in range(3):
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    _ = sys.stdin.read(1)
                else:
                    break
            return None

        return ch.lower()

    def _set_axis_for_hold(self, axis_name: str, value: float) -> None:
        now = time.monotonic()
        self.axis_cmd[axis_name] = value
        self.axis_expiry[axis_name] = now + self.hold_timeout

    def _stop_all(self) -> None:
        for k in self.axis_cmd:
            self.axis_cmd[k] = 0.0
            self.axis_expiry[k] = 0.0
        self._publish_current()
        self.get_logger().info("STOP")

    def _handle_key(self, key: str) -> None:
        if key is None:
            return

        if key in (" ", "x"):
            self._stop_all()
            return

        if key == "h":
            self._print_help()
            return

        if key == "1":
            self.linear_scale = max(0.05, self.linear_scale - self.linear_step)
            self.get_logger().info(f"linear_scale = {self.linear_scale:.2f}")
            return

        if key == "2":
            self.linear_scale = min(self.max_linear_scale, self.linear_scale + self.linear_step)
            self.get_logger().info(f"linear_scale = {self.linear_scale:.2f}")
            return

        if key == "3":
            self.angular_scale = max(0.05, self.angular_scale - self.angular_step)
            self.get_logger().info(f"angular_scale = {self.angular_scale:.2f}")
            return

        if key == "4":
            self.angular_scale = min(self.max_angular_scale, self.angular_scale + self.angular_step)
            self.get_logger().info(f"angular_scale = {self.angular_scale:.2f}")
            return

        # Translational axes
        if key == "w":
            self._set_axis_for_hold("linear_x", +self.linear_scale)
            if self.debug_keys:
                self.get_logger().info(f"[KEY] w -> linear.x = +{self.linear_scale:.2f}")
            return
        if key == "s":
            self._set_axis_for_hold("linear_x", -self.linear_scale)
            if self.debug_keys:
                self.get_logger().info(f"[KEY] s -> linear.x = -{self.linear_scale:.2f}")
            return
        if key == "a":
            self._set_axis_for_hold("linear_y", +self.linear_scale)
            if self.debug_keys:
                self.get_logger().info(f"[KEY] a -> linear.y = +{self.linear_scale:.2f}")
            return
        if key == "d":
            self._set_axis_for_hold("linear_y", -self.linear_scale)
            if self.debug_keys:
                self.get_logger().info(f"[KEY] d -> linear.y = -{self.linear_scale:.2f}")
            return
        if key == "r":
            self._set_axis_for_hold("linear_z", +self.linear_scale)
            if self.debug_keys:
                self.get_logger().info(f"[KEY] r -> linear.z = +{self.linear_scale:.2f}")
            return
        if key == "f":
            self._set_axis_for_hold("linear_z", -self.linear_scale)
            if self.debug_keys:
                self.get_logger().info(f"[KEY] f -> linear.z = -{self.linear_scale:.2f}")
            return

        # Rotational axes
        if key == "u":
            self._set_axis_for_hold("angular_x", +self.angular_scale)
            self.get_logger().info(f"[KEY] u -> angular.x (ROLL) = +{self.angular_scale:.2f}")
            return
        if key == "o":
            self._set_axis_for_hold("angular_x", -self.angular_scale)
            self.get_logger().info(f"[KEY] o -> angular.x (ROLL) = -{self.angular_scale:.2f}")
            return
        if key == "i":
            self._set_axis_for_hold("angular_y", +self.angular_scale)
            self.get_logger().info(f"[KEY] i -> angular.y (PITCH) = +{self.angular_scale:.2f}")
            return
        if key == "k":
            self._set_axis_for_hold("angular_y", -self.angular_scale)
            self.get_logger().info(f"[KEY] k -> angular.y (PITCH) = -{self.angular_scale:.2f}")
            return
        if key == "j":
            self._set_axis_for_hold("angular_z", +self.angular_scale)
            self.get_logger().info(f"[KEY] j -> angular.z (YAW) = +{self.angular_scale:.2f}")
            return
        if key == "l":
            self._set_axis_for_hold("angular_z", -self.angular_scale)
            self.get_logger().info(f"[KEY] l -> angular.z (YAW) = -{self.angular_scale:.2f}")
            return

    def _publish_current(self) -> None:
        # Apply simple first-order smoothing (slew) to avoid abrupt steps
        # alpha = slew_rate * dt; here dt ~= 1/loop_hz
        if self.loop_hz > 0:
            alpha = min(1.0, self.slew_rate * (1.0 / self.loop_hz))
        else:
            alpha = 1.0

        # Clamp targets to configured maximums
        max_ang = float(self.max_angular_scale)

        targets = {
            "linear_x": float(self.axis_cmd["linear_x"]),
            "linear_y": float(self.axis_cmd["linear_y"]),
            "linear_z": float(self.axis_cmd["linear_z"]),
            "angular_x": float(self.axis_cmd["angular_x"]),
            "angular_y": float(self.axis_cmd["angular_y"]),
            "angular_z": max(-max_ang, min(max_ang, float(self.axis_cmd["angular_z"]))),
        }

        # Smooth and publish
        msg = Twist()
        for axis in ["linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"]:
            prev = self.prev_axis_cmd.get(axis, 0.0)
            tgt = targets[axis]
            applied = prev + alpha * (tgt - prev)
            self.prev_axis_cmd[axis] = applied

            if axis.startswith("linear_"):
                if axis == "linear_x":
                    msg.linear.x = float(applied)
                elif axis == "linear_y":
                    msg.linear.y = float(applied)
                elif axis == "linear_z":
                    msg.linear.z = float(applied)
            else:
                if axis == "angular_x":
                    msg.angular.x = float(applied)
                elif axis == "angular_y":
                    msg.angular.y = float(applied)
                elif axis == "angular_z":
                    msg.angular.z = float(applied)

        # Log published command (only if non-zero or debug enabled)
        if any([msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z]):
            self.get_logger().debug(
                f"[PUB] lin=({msg.linear.x:+.2f}, {msg.linear.y:+.2f}, {msg.linear.z:+.2f}) "
                f"ang=({msg.angular.x:+.2f}, {msg.angular.y:+.2f}, {msg.angular.z:+.2f})"
            )

        self.pub.publish(msg)

    def _expire_axes(self) -> None:
        now = time.monotonic()
        for axis_name, expiry in self.axis_expiry.items():
            if self.axis_cmd[axis_name] != 0.0 and now > expiry:
                self.get_logger().debug(f"[EXPIRE] {axis_name} timeout after {self.hold_timeout:.3f}s")
                self.axis_cmd[axis_name] = 0.0

    def _on_timer(self) -> None:
        try:
            while True:
                key = self._read_key_nonblocking()
                if key is None:
                    break
                self._handle_key(key)

            self._expire_axes()
            self._publish_current()

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.get_logger().error(f"Timer loop error: {exc}")

def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeyboardCmdVel()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop_all()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()