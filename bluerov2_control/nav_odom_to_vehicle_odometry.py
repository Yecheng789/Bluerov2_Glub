#!/usr/bin/env python3
"""Convert nav_msgs/Odometry into px4_msgs/VehicleOdometry."""

import math

import numpy as np
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


def _validated_body_frd_angular_velocity(values, max_norm_rad_s):
    """Validate an external BODY_FRD angular-rate measurement."""
    angular_velocity = np.asarray(values, dtype=float)
    if angular_velocity.shape != (3,) or not np.all(
        np.isfinite(angular_velocity)
    ):
        return None
    max_norm_rad_s = float(max_norm_rad_s)
    if not math.isfinite(max_norm_rad_s) or max_norm_rad_s <= 0.0:
        raise ValueError(
            "angular_velocity_override_max_norm_rad_s must be finite and > 0"
        )
    if float(np.linalg.norm(angular_velocity)) > max_norm_rad_s:
        return None
    return angular_velocity


def _axis_transform(frame, *, body=False):
    """Return a rotation that maps the named input frame to FRD/NED axes."""
    frame = str(frame).strip().lower()
    if frame in ("", "passthrough"):
        return np.eye(3, dtype=float)
    if body:
        if frame == "frd":
            return np.eye(3, dtype=float)
        if frame == "flu":
            return np.diag([1.0, -1.0, -1.0])
        raise ValueError("input_body_frame must be one of: passthrough, frd, flu")

    if frame == "ned":
        return np.eye(3, dtype=float)
    if frame == "nwu":
        return np.diag([1.0, -1.0, -1.0])
    if frame == "enu":
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )
    raise ValueError(
        "input_world_frame must be one of: passthrough, ned, nwu, enu"
    )


def _quat_xyzw_to_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return None
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quat_wxyz(rotation):
    """Convert a proper rotation matrix to a normalized wxyz quaternion."""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        return [float("nan")] * 4

    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        diagonal = np.diag(r)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0)) * 2.0
            if s <= 1e-12:
                return [float("nan")] * 4
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif index == 1:
            s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0)) * 2.0
            if s <= 1e-12:
                return [float("nan")] * 4
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0)) * 2.0
            if s <= 1e-12:
                return [float("nan")] * 4
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12 or not math.isfinite(norm):
        return [float("nan")] * 4
    q /= norm
    if q[0] < 0.0:
        q = -q
    return q.tolist()


def _transform_covariance_diagonal(covariance, transform):
    try:
        covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
        block = np.zeros((6, 6), dtype=float)
        block[0:3, 0:3] = transform
        block[3:6, 3:6] = transform
        transformed = block @ covariance @ block.T
        return np.diag(transformed)
    except (TypeError, ValueError):
        return np.full(6, np.nan, dtype=float)


class NavOdomToVehicleOdometry(Node):
    """Bridge filtered MoCap odometry to the message type used by MPC nodes."""

    def __init__(self):
        super().__init__("nav_odom_to_vehicle_odometry")

        self.declare_parameter("input_odom_topic", "/mocap/glub_fb/odom_ekf")
        self.declare_parameter(
            "output_vehicle_odometry_topic",
            "/mocap/glub_fb/vehicle_odometry_ekf",
        )
        self.declare_parameter("pose_frame", "frd")
        self.declare_parameter("velocity_frame", "body_frd")
        self.declare_parameter("input_world_frame", "passthrough")
        self.declare_parameter("input_body_frame", "passthrough")
        self.declare_parameter("require_explicit_frame_transform", False)
        self.declare_parameter("quality", 100)
        self.declare_parameter("angular_velocity_override_topic", "")
        self.declare_parameter(
            "angular_velocity_override_timeout_sec",
            0.10,
        )
        self.declare_parameter(
            "angular_velocity_override_max_norm_rad_s",
            5.0,
        )

        input_topic = str(self.get_parameter("input_odom_topic").value)
        output_topic = str(
            self.get_parameter("output_vehicle_odometry_topic").value
        )
        self._input_world_frame = str(
            self.get_parameter("input_world_frame").value
        ).strip().lower()
        self._input_body_frame = str(
            self.get_parameter("input_body_frame").value
        ).strip().lower()
        require_explicit = bool(
            self.get_parameter("require_explicit_frame_transform").value
        )
        if require_explicit and (
            self._input_world_frame in ("", "passthrough")
            or self._input_body_frame in ("", "passthrough")
        ):
            raise ValueError(
                "Explicit MoCap frames are required: set input_world_frame to "
                "ned/nwu/enu and input_body_frame to frd/flu"
            )
        self._world_to_ned = _axis_transform(self._input_world_frame)
        self._body_to_frd = _axis_transform(
            self._input_body_frame,
            body=True,
        )
        self._angular_velocity_override_topic = str(
            self.get_parameter("angular_velocity_override_topic").value
        ).strip()
        self._angular_velocity_override_timeout_sec = float(
            self.get_parameter(
                "angular_velocity_override_timeout_sec"
            ).value
        )
        if (
            not math.isfinite(self._angular_velocity_override_timeout_sec)
            or self._angular_velocity_override_timeout_sec <= 0.0
        ):
            raise ValueError(
                "angular_velocity_override_timeout_sec must be finite and > 0"
            )
        self._angular_velocity_override_max_norm_rad_s = float(
            self.get_parameter(
                "angular_velocity_override_max_norm_rad_s"
            ).value
        )
        _validated_body_frd_angular_velocity(
            np.zeros(3),
            self._angular_velocity_override_max_norm_rad_s,
        )
        self._angular_velocity_override = None
        self._angular_velocity_override_rx_sec = None

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
        if self._angular_velocity_override_topic:
            self.angular_velocity_sub = self.create_subscription(
                VehicleOdometry,
                self._angular_velocity_override_topic,
                self.on_angular_velocity_override,
                qos,
            )
        else:
            self.angular_velocity_sub = None
        self.get_logger().info(
            f"bridging nav odom '{input_topic}' -> vehicle odometry "
            f"'{output_topic}', world={self._input_world_frame}->NED, "
            f"body={self._input_body_frame}->FRD, "
            "angular_velocity="
            f"'{self._angular_velocity_override_topic or 'nav_odom'}'"
        )

    def _now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def on_angular_velocity_override(self, msg: VehicleOdometry):
        """Cache PX4's low-latency BODY_FRD angular velocity."""
        angular_velocity = _validated_body_frd_angular_velocity(
            msg.angular_velocity,
            self._angular_velocity_override_max_norm_rad_s,
        )
        if angular_velocity is None:
            self.get_logger().warn(
                "Rejected invalid external BODY_FRD angular velocity.",
                throttle_duration_sec=1.0,
            )
            return
        self._angular_velocity_override = angular_velocity
        self._angular_velocity_override_rx_sec = self._now_sec()

    def _fresh_angular_velocity_override(self):
        if not self._angular_velocity_override_topic:
            return None
        if (
            self._angular_velocity_override is None
            or self._angular_velocity_override_rx_sec is None
        ):
            return None
        age_sec = self._now_sec() - self._angular_velocity_override_rx_sec
        if not 0.0 <= age_sec <= self._angular_velocity_override_timeout_sec:
            return None
        return self._angular_velocity_override.copy()

    def on_odom(self, msg: Odometry):
        out = VehicleOdometry()
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        out.timestamp = now_us

        stamp = msg.header.stamp
        stamp_us = int(stamp.sec * 1000000 + stamp.nanosec / 1000)
        out.timestamp_sample = stamp_us if stamp_us > 0 else now_us

        out.pose_frame = self._pose_frame_value()
        position_input = np.array(
            [
                _finite_or_nan(msg.pose.pose.position.x),
                _finite_or_nan(msg.pose.pose.position.y),
                _finite_or_nan(msg.pose.pose.position.z),
            ],
            dtype=float,
        )
        out.position = (self._world_to_ned @ position_input).tolist()

        rotation_input = _quat_xyzw_to_matrix(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )
        if rotation_input is None:
            out.q = [float("nan")] * 4
            self.get_logger().warn(
                "Received invalid odometry quaternion; publishing NaN orientation.",
                throttle_duration_sec=1.0,
            )
        else:
            rotation_output = (
                self._world_to_ned
                @ rotation_input
                @ self._body_to_frd.T
            )
            out.q = _matrix_to_quat_wxyz(rotation_output)

        out.velocity_frame = self._velocity_frame_value()
        velocity_input = np.array(
            [
                _finite_or_nan(msg.twist.twist.linear.x),
                _finite_or_nan(msg.twist.twist.linear.y),
                _finite_or_nan(msg.twist.twist.linear.z),
            ],
            dtype=float,
        )
        angular_input = np.array(
            [
                _finite_or_nan(msg.twist.twist.angular.x),
                _finite_or_nan(msg.twist.twist.angular.y),
                _finite_or_nan(msg.twist.twist.angular.z),
            ],
            dtype=float,
        )
        out.velocity = (self._body_to_frd @ velocity_input).tolist()
        if self._angular_velocity_override_topic:
            angular_velocity = self._fresh_angular_velocity_override()
            if angular_velocity is None:
                self.get_logger().warn(
                    "Required external BODY_FRD angular velocity is missing "
                    "or stale; withholding VehicleOdometry.",
                    throttle_duration_sec=1.0,
                )
                return
            out.angular_velocity = angular_velocity.tolist()
        else:
            out.angular_velocity = (
                self._body_to_frd @ angular_input
            ).tolist()

        pose_diag = _transform_covariance_diagonal(
            msg.pose.covariance,
            self._world_to_ned,
        )
        twist_diag = _transform_covariance_diagonal(
            msg.twist.covariance,
            self._body_to_frd,
        )
        out.position_variance = pose_diag[0:3].tolist()
        out.orientation_variance = pose_diag[3:6].tolist()
        out.velocity_variance = twist_diag[0:3].tolist()
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
