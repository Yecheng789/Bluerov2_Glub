#!/usr/bin/env python3

import rclpy
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class RealWrenchWatchdog(Node):
    def __init__(self):
        super().__init__("real_wrench_watchdog")

        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.declare_parameter("thrust_sp_topic", "/glub/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_sp_topic", "/glub/fmu/in/vehicle_torque_setpoint")
        self.declare_parameter("warn_threshold", 1e-4)
        self.declare_parameter("print_dt", 0.5)

        self.thrust = None
        self.torque = None
        self.last_print_us = 0

        self.create_subscription(
            VehicleThrustSetpoint,
            self.get_parameter("thrust_sp_topic").value,
            self.on_thrust,
            px4_qos,
        )
        self.create_subscription(
            VehicleTorqueSetpoint,
            self.get_parameter("torque_sp_topic").value,
            self.on_torque,
            px4_qos,
        )

    def on_thrust(self, msg: VehicleThrustSetpoint):
        self.thrust = tuple(float(v) for v in msg.xyz)
        self.maybe_print()

    def on_torque(self, msg: VehicleTorqueSetpoint):
        self.torque = tuple(float(v) for v in msg.xyz)
        self.maybe_print()

    def maybe_print(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        print_dt_us = int(float(self.get_parameter("print_dt").value) * 1e6)
        if now_us - self.last_print_us < print_dt_us:
            return
        self.last_print_us = now_us

        if self.thrust is None or self.torque is None:
            self.get_logger().info("waiting for thrust/torque setpoints...")
            return

        threshold = float(self.get_parameter("warn_threshold").value)
        values = self.thrust + self.torque
        max_abs = max(abs(v) for v in values)
        msg = (
            f"thrust=[{self.thrust[0]:+.4f},{self.thrust[1]:+.4f},{self.thrust[2]:+.4f}] "
            f"torque=[{self.torque[0]:+.4f},{self.torque[1]:+.4f},{self.torque[2]:+.4f}]"
        )
        if max_abs > threshold:
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)


def main():
    rclpy.init()
    node = RealWrenchWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
