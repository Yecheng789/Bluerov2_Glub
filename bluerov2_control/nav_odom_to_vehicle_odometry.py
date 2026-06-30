#!/usr/bin/env python3
"""Convert nav_msgs/Odometry into px4_msgs/VehicleOdometry."""

import math

import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def _cov_diag(covariance, indices):
    out = []
    for index in indices:
        try:
            out.append(_finite_or_nan(covariance[index]))
        except (IndexError, TypeError):
            out.append(float("nan"))
    return out


class NavOdomToVehicleOdometry(Node):
    """Bridge filtered MoCap odometry to the message type used by MPC nodes."""

    def __init__(self):
        super().__init__("nav_odom_to_vehicle_odometry")

        self.declare_parameter("input_odom_topic", "/mocap/glub/odom_ekf")
        self.declare_parameter(
            "output_vehicle_odometry_topic",
            "/mocap/glub/vehicle_odometry_ekf",
        )
        self.declare_parameter("pose_frame", "frd")
        self.declare_parameter("velocity_frame", "body_frd")
        self.declare_parameter("quality", 100)

        input_topic = str(self.get_parameter("input_odom_topic").value)
        output_topic = str(
            self.get_parameter("output_vehicle_odometry_topic").value
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(VehicleOdometry, output_topic, qos)
        self.sub = self.create_subscription(
            Odometry,
            input_topic,
            self.on_odom,
            qos,
        )
        self.get_logger().info(
            f"bridging nav odom '{input_topic}' -> vehicle odometry "
            f"'{output_topic}'"
        )

    def on_odom(self, msg: Odometry):
        out = VehicleOdometry()
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        out.timestamp = now_us

        stamp = msg.header.stamp
        stamp_us = int(stamp.sec * 1000000 + stamp.nanosec / 1000)
        out.timestamp_sample = stamp_us if stamp_us > 0 else now_us

        out.pose_frame = self._pose_frame_value()
        out.position = [
            _finite_or_nan(msg.pose.pose.position.x),
            _finite_or_nan(msg.pose.pose.position.y),
            _finite_or_nan(msg.pose.pose.position.z),
        ]
        out.q = [
            _finite_or_nan(msg.pose.pose.orientation.w),
            _finite_or_nan(msg.pose.pose.orientation.x),
            _finite_or_nan(msg.pose.pose.orientation.y),
            _finite_or_nan(msg.pose.pose.orientation.z),
        ]

        out.velocity_frame = self._velocity_frame_value()
        out.velocity = [
            _finite_or_nan(msg.twist.twist.linear.x),
            _finite_or_nan(msg.twist.twist.linear.y),
            _finite_or_nan(msg.twist.twist.linear.z),
        ]
        out.angular_velocity = [
            _finite_or_nan(msg.twist.twist.angular.x),
            _finite_or_nan(msg.twist.twist.angular.y),
            _finite_or_nan(msg.twist.twist.angular.z),
        ]

        out.position_variance = _cov_diag(msg.pose.covariance, [0, 7, 14])
        out.orientation_variance = _cov_diag(msg.pose.covariance, [21, 28, 35])
        out.velocity_variance = _cov_diag(msg.twist.covariance, [0, 7, 14])
        out.reset_counter = 0
        out.quality = int(self.get_parameter("quality").value)
        self.pub.publish(out)

    def _pose_frame_value(self):
        frame = str(self.get_parameter("pose_frame").value).strip().lower()
        if frame == "ned":
            return VehicleOdometry.POSE_FRAME_NED
        if frame == "frd":
            return VehicleOdometry.POSE_FRAME_FRD
        return VehicleOdometry.POSE_FRAME_UNKNOWN

    def _velocity_frame_value(self):
        frame = str(self.get_parameter("velocity_frame").value).strip().lower()
        if frame == "ned":
            return VehicleOdometry.VELOCITY_FRAME_NED
        if frame == "frd":
            return VehicleOdometry.VELOCITY_FRAME_FRD
        if frame in ("body_frd", "body-frd", "body"):
            return VehicleOdometry.VELOCITY_FRAME_BODY_FRD
        return VehicleOdometry.VELOCITY_FRAME_UNKNOWN


def main(args=None):
    rclpy.init(args=args)
    node = NavOdomToVehicleOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
