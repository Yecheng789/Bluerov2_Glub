#!/usr/bin/env python3
"""
Simple WASD/QE/RF keyboard teleop -> geometry_msgs/Twist.

Keys (hold = OS repeat):
  w/s : linear.x  +/-
  a/d : linear.y  +/-
  r/f : linear.z  +/-
  q/e : angular.z +/-
  x   : stop (zero twist)
  1/2 : decrease/increase linear speed
  3/4 : decrease/increase yaw speed
  h   : help
  Ctrl-C to exit

Run this in a REAL terminal (not via ros2 launch).
"""

import sys
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class WasdTeleop(Node):
    def __init__(self):
        super().__init__("wasd_teleop")

        self.declare_parameter("cmd_vel_topic", "/itrl_rov_1/cmd_vel")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("lin_step", 0.2)   # normalized step per key press
        self.declare_parameter("yaw_step", 0.4)   # normalized step per key press

        self.declare_parameter("lin_scale", 1.0)  # scales linear.*
        self.declare_parameter("yaw_scale", 1.0)  # scales angular.z

        self.declare_parameter("key_timeout", 0.15)  # seconds
        self.last_key_time = self.get_clock().now()

        topic = self.get_parameter("cmd_vel_topic").value
        rate_hz = float(self.get_parameter("rate_hz").value)

        self.pub = self.create_publisher(Twist, topic, 10)
        self.timer = self.create_timer(1.0 / rate_hz, self.tick)

        self.lin_scale = float(self.get_parameter("lin_scale").value)
        self.yaw_scale = float(self.get_parameter("yaw_scale").value)

        self.cmd = Twist()

        # terminal setup
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        self.get_logger().info(f"WASD teleop publishing Twist on: {topic}")
        self.print_help()

    def destroy_node(self):
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        except Exception:
            pass
        super().destroy_node()

    def print_help(self):
        self.get_logger().info(
            "Keys: w/s x, a/d y, r/f z, q/e yaw, x stop, 1/2 lin scale, 3/4 yaw scale, h help"
        )

    def read_key(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            return sys.stdin.read(1)
        return None

    def tick(self):
        ch = self.read_key()

        now = self.get_clock().now()
        # If no new key recently, decay to zero (prevents channel "sticking"/mixing)
        key_timeout = float(self.get_parameter("key_timeout").value)
        if ch is None:
            if (now - self.last_key_time).nanoseconds * 1e-9 > key_timeout:
                self.cmd = Twist()
                self.pub.publish(self.cmd)
            return

        # got a key
        self.last_key_time = now

        ch = ch.lower()

        lin_step = float(self.get_parameter("lin_step").value)
        yaw_step = float(self.get_parameter("yaw_step").value)

        if ch == "h":
            self.print_help()
            return

        if ch == "x":
            self.cmd = Twist()
            self.pub.publish(self.cmd)
            return

        # Speed scaling
        if ch == "1":
            self.lin_scale = clamp(self.lin_scale - 0.1, 0.1, 2.0)
            self.get_logger().info(f"lin_scale={self.lin_scale:.2f}")
            return
        if ch == "2":
            self.lin_scale = clamp(self.lin_scale + 0.1, 0.1, 2.0)
            self.get_logger().info(f"lin_scale={self.lin_scale:.2f}")
            return
        if ch == "3":
            self.yaw_scale = clamp(self.yaw_scale - 0.1, 0.1, 2.0)
            self.get_logger().info(f"yaw_scale={self.yaw_scale:.2f}")
            return
        if ch == "4":
            self.yaw_scale = clamp(self.yaw_scale + 0.1, 0.1, 2.0)
            self.get_logger().info(f"yaw_scale={self.yaw_scale:.2f}")
            return

        # Command mapping (normalized “sticks” in [-1,1])
        if ch == "w":
            self.cmd.linear.x = clamp(self.cmd.linear.x + lin_step, -1.0, 1.0)
        elif ch == "s":
            self.cmd.linear.x = clamp(self.cmd.linear.x - lin_step, -1.0, 1.0)
        elif ch == "a":
            self.cmd.linear.y = clamp(self.cmd.linear.y + lin_step, -1.0, 1.0)
        elif ch == "d":
            self.cmd.linear.y = clamp(self.cmd.linear.y - lin_step, -1.0, 1.0)
        elif ch == "r":
            self.cmd.linear.z = clamp(self.cmd.linear.z + lin_step, -1.0, 1.0)
        elif ch == "f":
            self.cmd.linear.z = clamp(self.cmd.linear.z - lin_step, -1.0, 1.0)
        elif ch == "q":
            self.cmd.angular.z = clamp(self.cmd.angular.z + yaw_step, -1.0, 1.0)
        elif ch == "e":
            self.cmd.angular.z = clamp(self.cmd.angular.z - yaw_step, -1.0, 1.0)
        else:
            return

        # Apply scales and publish
        msg = Twist()
        msg.linear.x = clamp(self.cmd.linear.x * self.lin_scale, -1.0, 1.0)
        msg.linear.y = clamp(self.cmd.linear.y * self.lin_scale, -1.0, 1.0)
        msg.linear.z = clamp(self.cmd.linear.z * self.lin_scale, -1.0, 1.0)
        msg.angular.z = clamp(self.cmd.angular.z * self.yaw_scale, -1.0, 1.0)

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = WasdTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()