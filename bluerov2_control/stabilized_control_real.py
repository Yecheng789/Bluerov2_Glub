#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Twist
from px4_msgs.msg import VehicleControlMode, VehicleThrustSetpoint, VehicleTorqueSetpoint
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class StabilizedControlReal(Node):
    def __init__(self):
        super().__init__("stabilized_control_real")

        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.declare_parameter("control_mode_topic", "/glub/fmu/out/vehicle_control_mode")
        self.declare_parameter("thrust_sp_topic", "/glub/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_sp_topic", "/glub/fmu/in/vehicle_torque_setpoint")
        self.declare_parameter("cmd_vel_topic", "/itrl_rov_1/cmd_vel")

        self.declare_parameter("thrust_scale", 0.45)
        self.declare_parameter("yaw_torque_scale", 0.45)
        self.declare_parameter("thrust_min", 0.18)
        self.declare_parameter("torque_min", 0.18)
        self.declare_parameter("thrust_sat", 0.60)
        self.declare_parameter("torque_sat", 0.60)

        self.declare_parameter("yaw_sign", -1.0)
        self.declare_parameter("z_sign", -1.0)
        self.declare_parameter("torque_z_sign", 1.0)
        self.declare_parameter("require_control_mode_gate", False)
        self.declare_parameter("cmd_timeout", 0.25)
        self.declare_parameter("cmd_deadband", 0.05)
        self.declare_parameter("loop_dt", 0.02)

        control_mode_topic = self.get_parameter("control_mode_topic").value
        thrust_topic = self.get_parameter("thrust_sp_topic").value
        torque_topic = self.get_parameter("torque_sp_topic").value
        cmd_topic = self.get_parameter("cmd_vel_topic").value

        self.sub_cm = self.create_subscription(
            VehicleControlMode,
            control_mode_topic,
            self.on_control_mode,
            px4_qos,
        )
        self.sub_cmd = self.create_subscription(Twist, cmd_topic, self.on_cmd_vel, 10)
        self.pub_thrust = self.create_publisher(VehicleThrustSetpoint, thrust_topic, px4_qos)
        self.pub_torque = self.create_publisher(VehicleTorqueSetpoint, torque_topic, px4_qos)

        self.enabled = False
        self.cmd_xyz = (0.0, 0.0, 0.0)
        self.cmd_yaw = 0.0
        self.cmd_last_us = 0
        self.last_cmd_log_us = 0
        self.last_output_log_us = 0

        self.timer = self.create_timer(float(self.get_parameter("loop_dt").value), self.tick)
        self.get_logger().info(
            f"real teleop wrench: cmd_vel={cmd_topic}, thrust={thrust_topic}, torque={torque_topic}"
        )

    def on_control_mode(self, msg: VehicleControlMode):
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)
        if gate and not self.enabled:
            self.enabled = True
            self.get_logger().info("Enabled real wrench output (armed + offboard).")
        elif (not gate) and self.enabled:
            self.enabled = False
            self.get_logger().info("Disabled real wrench output.")
            self.publish_wrench((0.0, 0.0, 0.0), 0.0)

    def on_cmd_vel(self, msg: Twist):
        yaw_sign = float(self.get_parameter("yaw_sign").value)
        z_sign = float(self.get_parameter("z_sign").value)

        surge = clamp(float(msg.linear.x), -1.0, 1.0)
        sway = clamp(-float(msg.linear.y), -1.0, 1.0)
        heave = clamp(z_sign * float(msg.linear.z), -1.0, 1.0)
        yaw = clamp(yaw_sign * float(msg.angular.z), -1.0, 1.0)

        self.cmd_xyz = (surge, sway, heave)
        self.cmd_yaw = yaw
        self.cmd_last_us = int(self.get_clock().now().nanoseconds / 1000)
        self.maybe_log_cmd()

    def get_cmd(self, now_us: int):
        timeout_us = int(float(self.get_parameter("cmd_timeout").value) * 1e6)
        if self.cmd_last_us == 0 or (now_us - self.cmd_last_us) > timeout_us:
            return (0.0, 0.0, 0.0, 0.0)
        sx, sy, sz = self.cmd_xyz
        return (sx, sy, sz, self.cmd_yaw)

    def publish_wrench(self, thrust_xyz, torque_z: float):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = 0
        thr.xyz = [float(thrust_xyz[0]), float(thrust_xyz[1]), float(thrust_xyz[2])]
        self.pub_thrust.publish(thr)

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = 0
        tor.xyz = [0.0, 0.0, float(torque_z)]
        self.pub_torque.publish(tor)

    def maybe_log_cmd(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        if now_us - self.last_cmd_log_us < 250000:
            return
        self.last_cmd_log_us = now_us
        sx, sy, sz = self.cmd_xyz
        self.get_logger().info(
            f"cmd_vel received: thrust_cmd=[{sx:+.2f},{sy:+.2f},{sz:+.2f}] yaw_cmd={self.cmd_yaw:+.2f}"
        )

    def maybe_log_output(self, thrust, torque_z):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        if now_us - self.last_output_log_us < 500000:
            return
        if max(abs(thrust[0]), abs(thrust[1]), abs(thrust[2]), abs(torque_z)) <= 1e-6:
            return
        self.last_output_log_us = now_us
        self.get_logger().info(
            f"wrench output: thrust=[{thrust[0]:+.3f},{thrust[1]:+.3f},{thrust[2]:+.3f}] torque_z={torque_z:+.3f}"
        )

    def tick(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        require_gate = bool(self.get_parameter("require_control_mode_gate").value)
        if require_gate and not self.enabled:
            self.publish_wrench((0.0, 0.0, 0.0), 0.0)
            return

        deadband = float(self.get_parameter("cmd_deadband").value)
        surge_u, sway_u, heave_u, yaw_u = self.get_cmd(now_us)

        thrust_scale = float(self.get_parameter("thrust_scale").value)
        yaw_torque_scale = float(self.get_parameter("yaw_torque_scale").value)
        thrust_min = float(self.get_parameter("thrust_min").value)
        torque_min = float(self.get_parameter("torque_min").value)
        thrust_sat = float(self.get_parameter("thrust_sat").value)
        torque_sat = float(self.get_parameter("torque_sat").value)
        torque_z_sign = float(self.get_parameter("torque_z_sign").value)

        surge = surge_u if abs(surge_u) > deadband else 0.0
        sway = sway_u if abs(sway_u) > deadband else 0.0
        heave = heave_u if abs(heave_u) > deadband else 0.0
        yaw = yaw_u if abs(yaw_u) > deadband else 0.0

        thrust = (
            self.apply_deadzone_boost(surge, thrust_scale, thrust_min, thrust_sat),
            self.apply_deadzone_boost(sway, thrust_scale, thrust_min, thrust_sat),
            self.apply_deadzone_boost(heave, thrust_scale, thrust_min, thrust_sat),
        )
        torque_z = self.apply_deadzone_boost(
            torque_z_sign * yaw,
            yaw_torque_scale,
            torque_min,
            torque_sat,
        )

        self.publish_wrench(thrust, torque_z)
        self.maybe_log_output(thrust, torque_z)

    @staticmethod
    def apply_deadzone_boost(cmd: float, scale: float, min_output: float, sat: float) -> float:
        if cmd == 0.0:
            return 0.0

        mag = min(abs(cmd), 1.0)
        max_output = min(abs(scale), abs(sat))
        min_output = min(abs(min_output), max_output)
        output = min_output + mag * (max_output - min_output)
        return math.copysign(output, cmd)


def main():
    rclpy.init()
    node = StabilizedControlReal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.publish_wrench((0.0, 0.0, 0.0), 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
