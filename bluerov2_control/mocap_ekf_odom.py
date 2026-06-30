#!/usr/bin/env python3
"""EKF odometry helper for short MoCap dropouts."""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12 or not np.isfinite(norm):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q = q / norm
    if q[3] < 0.0:
        q = -q
    return q


def quat_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = np.asarray(q1, dtype=float)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def parse_quat_xyzw(text):
    text = str(text).strip()
    if not text:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    values = [float(value) for value in text.replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError(
            "orientation_correction_quat_xyzw must contain four values: "
            "x y z w"
        )
    return quat_normalize(np.array(values, dtype=float))


def quat_exp(rotvec):
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return quat_normalize(
            np.array(
                [
                    0.5 * rotvec[0],
                    0.5 * rotvec[1],
                    0.5 * rotvec[2],
                    1.0,
                ],
                dtype=float,
            )
        )
    axis = rotvec / angle
    half = 0.5 * angle
    return quat_normalize(
        np.concatenate([axis * math.sin(half), [math.cos(half)]])
    )


def quat_log(q):
    q = quat_normalize(q)
    vector = q[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-12:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, q[3])
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return vector * (angle / vector_norm)


def skew(v):
    v = np.asarray(v, dtype=float)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def quat_to_rotation_matrix(q):
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def orientation_residual(q_meas, q_pred):
    q_err = quat_multiply(q_meas, quat_conjugate(q_pred))
    return quat_log(q_err)


def base_link_z_axis_angle_rad(q):
    rotation = quat_to_rotation_matrix(q)
    base_link_z_in_mocap = rotation[:, 2]
    dot = float(np.clip(base_link_z_in_mocap[2], -1.0, 1.0))
    return float(math.acos(dot))


def all_finite(*arrays):
    return all(np.all(np.isfinite(np.asarray(array))) for array in arrays)


class QuaternionCvEkf:
    """Constant-velocity EKF for position and quaternion attitude."""

    def __init__(
        self,
        position_std=0.01,
        orientation_std=0.12,
        linear_accel_std=0.5,
        angular_accel_std=1.0,
        gyro_std=0.05,
        max_position_innovation_m=0.25,
        max_orientation_innovation_rad=0.75,
        max_base_link_z_axis_angle_rad=0.4,
        max_gyro_innovation_rad_s=2.0,
    ):
        self.position_std = position_std
        self.orientation_std = orientation_std
        self.linear_accel_std = linear_accel_std
        self.angular_accel_std = angular_accel_std
        self.gyro_std = gyro_std
        self.max_position_innovation_m = max_position_innovation_m
        self.max_orientation_innovation_rad = max_orientation_innovation_rad
        self.max_base_link_z_axis_angle_rad = max_base_link_z_axis_angle_rad
        self.max_gyro_innovation_rad_s = max_gyro_innovation_rad_s
        self.last_pose_rejection_reason = ""

        self.position = np.zeros(3, dtype=float)
        self.linear_velocity = np.zeros(3, dtype=float)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.angular_velocity = np.zeros(3, dtype=float)
        self.covariance = np.eye(12, dtype=float)
        self.initialized = False

    def initialize(self, position, orientation):
        self.position = np.asarray(position, dtype=float).copy()
        self.linear_velocity = np.zeros(3, dtype=float)
        self.orientation = quat_normalize(orientation)
        self.angular_velocity = np.zeros(3, dtype=float)
        self.covariance = np.diag(
            [
                self.position_std**2,
                self.position_std**2,
                self.position_std**2,
                1.0,
                1.0,
                1.0,
                self.orientation_std**2,
                self.orientation_std**2,
                self.orientation_std**2,
                1.0,
                1.0,
                1.0,
            ]
        )
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return
        dt = max(float(dt), 0.0)
        if dt <= 1e-9:
            return

        snapshot = self._snapshot()

        self.position = self.position + self.linear_velocity * dt
        delta_q = quat_exp(self.angular_velocity * dt)
        self.orientation = quat_normalize(
            quat_multiply(delta_q, self.orientation)
        )

        f = np.eye(12, dtype=float)
        f[0:3, 3:6] = np.eye(3) * dt
        f[6:9, 9:12] = np.eye(3) * dt

        q = np.zeros((12, 12), dtype=float)
        linear_accel_var = self.linear_accel_std**2
        angular_accel_var = self.angular_accel_std**2
        q[0:3, 0:3] = np.eye(3) * linear_accel_var * (dt**4) / 4.0
        q[0:3, 3:6] = np.eye(3) * linear_accel_var * (dt**3) / 2.0
        q[3:6, 0:3] = q[0:3, 3:6]
        q[3:6, 3:6] = np.eye(3) * linear_accel_var * (dt**2)
        q[6:9, 6:9] = np.eye(3) * angular_accel_var * (dt**4) / 4.0
        q[6:9, 9:12] = np.eye(3) * angular_accel_var * (dt**3) / 2.0
        q[9:12, 6:9] = q[6:9, 9:12]
        q[9:12, 9:12] = np.eye(3) * angular_accel_var * (dt**2)

        self.covariance = f @ self.covariance @ f.T + q
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        if not self._orientation_z_safe():
            self._restore(snapshot)

    def update_pose(self, position_meas, orientation_meas):
        if not all_finite(position_meas, orientation_meas):
            self.last_pose_rejection_reason = "non-finite position or orientation"
            return False

        self.last_pose_rejection_reason = ""
        orientation_meas = quat_normalize(orientation_meas)
        accepted, position_residual, attitude_residual = (
            self._check_pose_measurement(position_meas, orientation_meas)
        )

        if not self.initialized:
            if accepted:
                self.initialize(position_meas, orientation_meas)
            return accepted

        if not accepted:
            return False

        snapshot = self._snapshot()
        residual = np.concatenate([position_residual, attitude_residual])

        h = np.zeros((6, 12), dtype=float)
        h[0:3, 0:3] = np.eye(3)
        h[3:6, 6:9] = np.eye(3)

        r = np.diag(
            [
                self.position_std**2,
                self.position_std**2,
                self.position_std**2,
                self.orientation_std**2,
                self.orientation_std**2,
                self.orientation_std**2,
            ]
        )

        if not self._correct(residual, h, r):
            self._restore(snapshot)
            return False
        if not self._orientation_z_safe():
            self._restore(snapshot)
            return False
        return True

    def update_body_angular_velocity(self, angular_velocity_body_meas):
        if not self.initialized or not all_finite(angular_velocity_body_meas):
            return False

        angular_velocity_body_meas = np.asarray(
            angular_velocity_body_meas,
            dtype=float,
        )
        rotation = quat_to_rotation_matrix(self.orientation)
        angular_velocity_world_meas = rotation @ angular_velocity_body_meas
        residual = angular_velocity_world_meas - self.angular_velocity
        innovation_rad_s = float(np.linalg.norm(residual))
        accepted = self._within_gate(
            innovation_rad_s,
            self.max_gyro_innovation_rad_s,
        )
        if not accepted:
            return False

        snapshot = self._snapshot()

        h = np.zeros((3, 12), dtype=float)
        h[:, 6:9] = skew(self.angular_velocity)
        h[:, 9:12] = np.eye(3)
        r = np.eye(3, dtype=float) * self.gyro_std**2

        if not self._correct(residual, h, r):
            self._restore(snapshot)
            return False
        if not self._orientation_z_safe():
            self._restore(snapshot)
            return False
        return True

    def reset_copy(self):
        return QuaternionCvEkf(
            position_std=self.position_std,
            orientation_std=self.orientation_std,
            linear_accel_std=self.linear_accel_std,
            angular_accel_std=self.angular_accel_std,
            gyro_std=self.gyro_std,
            max_position_innovation_m=self.max_position_innovation_m,
            max_orientation_innovation_rad=self.max_orientation_innovation_rad,
            max_base_link_z_axis_angle_rad=self.max_base_link_z_axis_angle_rad,
            max_gyro_innovation_rad_s=self.max_gyro_innovation_rad_s,
        )

    def body_linear_velocity(self):
        rotation = quat_to_rotation_matrix(self.orientation)
        return rotation.T @ self.linear_velocity

    def body_angular_velocity(self):
        rotation = quat_to_rotation_matrix(self.orientation)
        return rotation.T @ self.angular_velocity

    def _check_pose_measurement(self, position_meas, orientation_meas):
        z_axis_angle_rad = base_link_z_axis_angle_rad(orientation_meas)
        z_axis_accepted = self._within_gate(
            z_axis_angle_rad,
            self.max_base_link_z_axis_angle_rad,
        )

        if not self.initialized:
            if not z_axis_accepted:
                self.last_pose_rejection_reason = (
                    "base_link_z_axis_angle="
                    f"{z_axis_angle_rad:.3f}rad > "
                    f"{self.max_base_link_z_axis_angle_rad:.3f}rad"
                )
            return z_axis_accepted, np.zeros(3), np.zeros(3)

        position_meas = np.asarray(position_meas, dtype=float)
        position_residual = position_meas - self.position
        attitude_residual = orientation_residual(
            orientation_meas,
            self.orientation,
        )
        position_innovation_m = float(np.linalg.norm(position_residual))
        orientation_innovation_rad = float(np.linalg.norm(attitude_residual))

        position_accepted = self._within_gate(
            position_innovation_m,
            self.max_position_innovation_m,
        )
        orientation_accepted = self._within_gate(
            orientation_innovation_rad,
            self.max_orientation_innovation_rad,
        )
        accepted = position_accepted and orientation_accepted and z_axis_accepted
        if not accepted:
            reasons = []
            if not position_accepted:
                reasons.append(
                    "position_innovation="
                    f"{position_innovation_m:.3f}m > "
                    f"{self.max_position_innovation_m:.3f}m"
                )
            if not orientation_accepted:
                reasons.append(
                    "orientation_innovation="
                    f"{orientation_innovation_rad:.3f}rad > "
                    f"{self.max_orientation_innovation_rad:.3f}rad"
                )
            if not z_axis_accepted:
                reasons.append(
                    "base_link_z_axis_angle="
                    f"{z_axis_angle_rad:.3f}rad > "
                    f"{self.max_base_link_z_axis_angle_rad:.3f}rad"
                )
            self.last_pose_rejection_reason = ", ".join(reasons)
        return accepted, position_residual, attitude_residual

    def _correct(self, residual, h, r):
        pht = self.covariance @ h.T
        s = h @ pht + r
        try:
            kalman_gain = np.linalg.solve(s.T, pht.T).T
        except np.linalg.LinAlgError:
            return False

        correction = kalman_gain @ residual
        self.position = self.position + correction[0:3]
        self.linear_velocity = self.linear_velocity + correction[3:6]
        self.orientation = quat_normalize(
            quat_multiply(quat_exp(correction[6:9]), self.orientation)
        )
        self.angular_velocity = self.angular_velocity + correction[9:12]

        identity = np.eye(12, dtype=float)
        ikh = identity - kalman_gain @ h
        self.covariance = (
            ikh @ self.covariance @ ikh.T
            + kalman_gain @ r @ kalman_gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return all_finite(
            self.position,
            self.linear_velocity,
            self.orientation,
            self.angular_velocity,
            self.covariance,
        )

    def _snapshot(self):
        return (
            self.position.copy(),
            self.linear_velocity.copy(),
            self.orientation.copy(),
            self.angular_velocity.copy(),
            self.covariance.copy(),
        )

    def _restore(self, snapshot):
        self.position = snapshot[0]
        self.linear_velocity = snapshot[1]
        self.orientation = snapshot[2]
        self.angular_velocity = snapshot[3]
        self.covariance = snapshot[4]

    def _orientation_z_safe(self):
        limit = self.max_base_link_z_axis_angle_rad
        if limit is None or limit <= 0.0:
            return True
        return base_link_z_axis_angle_rad(self.orientation) <= limit

    @staticmethod
    def _within_gate(value, limit):
        return limit is None or limit <= 0.0 or value <= limit


class MocapEkfOdom(Node):
    """ROS node that publishes filtered MoCap odometry."""

    def __init__(self):
        super().__init__("mocap_ekf_odom")

        self.declare_parameter("rigid_body_name", "glub")
        rigid_body_name = str(self.get_parameter("rigid_body_name").value)

        self.declare_parameter("pose_topic", "")
        self.declare_parameter("odom_topic", "")
        self.declare_parameter("imu_topic", "/mavros/imu/data")
        self.declare_parameter("parent_frame", "mocap")
        self.declare_parameter("child_frame", "")
        self.declare_parameter("child_frame_flu", "")
        self.declare_parameter("publish_rate_hz", 80.0)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_static_flu_tf", True)
        self.declare_parameter("use_imu_gyro", False)
        self.declare_parameter("imu_gyro_lpf_alpha", 0.8)
        self.declare_parameter("orientation_correction_quat_xyzw", "")
        self.declare_parameter("max_coast_sec", 1.0)
        self.declare_parameter("max_rejected_samples", 200)

        self.declare_parameter("position_std", 0.01)
        self.declare_parameter("orientation_std", 0.12)
        self.declare_parameter("linear_accel_std", 0.5)
        self.declare_parameter("angular_accel_std", 1.0)
        self.declare_parameter("gyro_std", 0.05)
        self.declare_parameter("max_position_innovation_m", 0.25)
        self.declare_parameter("max_orientation_innovation_rad", 0.75)
        self.declare_parameter("max_base_link_z_axis_angle_rad", 0.4)
        self.declare_parameter("max_gyro_innovation_rad_s", 2.0)

        default_pose_topic = f"/mocap/{rigid_body_name}/pose"
        default_odom_topic = f"/mocap/{rigid_body_name}/odom_ekf"
        default_child_frame = f"{rigid_body_name}/base_link_ekf"
        default_child_frame_flu = f"{rigid_body_name}/base_link_ekf_flu"

        self._pose_topic = self._param_or_default(
            "pose_topic",
            default_pose_topic,
        )
        self._odom_topic = self._param_or_default(
            "odom_topic",
            default_odom_topic,
        )
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._parent_frame = str(self.get_parameter("parent_frame").value)
        self._child_frame = self._param_or_default(
            "child_frame",
            default_child_frame,
        )
        self._child_frame_flu = self._param_or_default(
            "child_frame_flu",
            default_child_frame_flu,
        )
        self._publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value
        )
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._publish_static_flu_tf = bool(
            self.get_parameter("publish_static_flu_tf").value
        )
        self._use_imu_gyro = bool(self.get_parameter("use_imu_gyro").value)
        self._imu_gyro_lpf_alpha = float(
            self.get_parameter("imu_gyro_lpf_alpha").value
        )
        self._orientation_correction = parse_quat_xyzw(
            self.get_parameter("orientation_correction_quat_xyzw").value
        )
        self._max_coast_sec = float(self.get_parameter("max_coast_sec").value)
        self._max_rejected_samples = int(
            self.get_parameter("max_rejected_samples").value
        )

        self._filter = QuaternionCvEkf(
            position_std=float(self.get_parameter("position_std").value),
            orientation_std=float(self.get_parameter("orientation_std").value),
            linear_accel_std=float(
                self.get_parameter("linear_accel_std").value
            ),
            angular_accel_std=float(
                self.get_parameter("angular_accel_std").value
            ),
            gyro_std=float(self.get_parameter("gyro_std").value),
            max_position_innovation_m=float(
                self.get_parameter("max_position_innovation_m").value
            ),
            max_orientation_innovation_rad=float(
                self.get_parameter("max_orientation_innovation_rad").value
            ),
            max_base_link_z_axis_angle_rad=float(
                self.get_parameter("max_base_link_z_axis_angle_rad").value
            ),
            max_gyro_innovation_rad_s=float(
                self.get_parameter("max_gyro_innovation_rad_s").value
            ),
        )
        self._last_filter_sec = None
        self._last_pose_rx_sec = None
        self._consecutive_full_rejections = 0
        self._imu_gyro_lpf = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, qos)
        self.create_subscription(
            PoseStamped,
            self._pose_topic,
            self._pose_cb,
            qos,
        )
        if self._use_imu_gyro:
            self.create_subscription(Imu, self._imu_topic, self._imu_cb, qos)

        if self._publish_tf:
            self._tf_broadcaster = TransformBroadcaster(self)
        else:
            self._tf_broadcaster = None
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        if self._publish_static_flu_tf:
            self._static_tf_timer = self.create_timer(
                1.0,
                self._publish_static_body_frames_once,
            )
        else:
            self._static_tf_timer = None

        self.get_logger().info(
            "mocap_ekf_odom: "
            f"rigid_body='{rigid_body_name}', "
            f"pose='{self._pose_topic}', odom='{self._odom_topic}', "
            f"imu='{self._imu_topic}', use_imu_gyro={self._use_imu_gyro}, "
            f"parent_frame='{self._parent_frame}', "
            f"child_frame='{self._child_frame}', "
            f"rate={self._publish_rate_hz:.1f}Hz, "
            f"publish_tf={self._publish_tf}, "
            f"max_coast={self._max_coast_sec:.2f}s, "
            "orientation_correction_xyzw="
            f"[{self._orientation_correction[0]:.6f}, "
            f"{self._orientation_correction[1]:.6f}, "
            f"{self._orientation_correction[2]:.6f}, "
            f"{self._orientation_correction[3]:.6f}]"
        )

    def _param_or_default(self, name, default):
        value = str(self.get_parameter(name).value)
        return value if value.strip() else default

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose_cb(self, msg):
        now = self._now_sec()
        self._predict_to(now)

        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        orientation = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        orientation = quat_multiply(orientation, self._orientation_correction)

        accepted = self._filter.update_pose(position, orientation)
        self._handle_update_result(accepted, position, orientation)
        self._last_pose_rx_sec = now

        frame_id = msg.header.frame_id
        if frame_id and frame_id != self._parent_frame:
            self.get_logger().warn(
                "Received pose in frame "
                f"'{frame_id}', publishing odom in '{self._parent_frame}'.",
                throttle_duration_sec=2.0,
            )

    def _imu_cb(self, msg):
        if not self._use_imu_gyro or not self._filter.initialized:
            return

        frame_id = msg.header.frame_id
        if frame_id and frame_id != "base_link":
            self.get_logger().warn(
                "Received IMU gyro in frame "
                f"'{frame_id}', expected 'base_link' (FLU).",
                throttle_duration_sec=2.0,
            )

        raw_angular_velocity = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=float,
        )
        alpha = min(max(self._imu_gyro_lpf_alpha, 0.0), 1.0)
        if self._imu_gyro_lpf is None:
            filtered_angular_velocity = raw_angular_velocity
        else:
            filtered_angular_velocity = (
                alpha * raw_angular_velocity
                + (1.0 - alpha) * self._imu_gyro_lpf
            )
        self._imu_gyro_lpf = filtered_angular_velocity

        # MAVROS base_link is FLU; this EKF body frame follows FRD.
        angular_velocity_body = np.array(
            [
                filtered_angular_velocity[0],
                -filtered_angular_velocity[1],
                -filtered_angular_velocity[2],
            ],
            dtype=float,
        )
        accepted = self._filter.update_body_angular_velocity(
            angular_velocity_body
        )
        if not accepted:
            self.get_logger().warn(
                "Rejected IMU gyro outlier: "
                "max_gyro_innovation="
                f"{self._filter.max_gyro_innovation_rad_s:.3f}rad/s",
                throttle_duration_sec=1.0,
            )

    def _handle_update_result(self, accepted, position, orientation):
        if accepted:
            self._consecutive_full_rejections = 0
            return

        self._consecutive_full_rejections += 1
        self.get_logger().warn(
            "Rejected MoCap pose outlier or invalid sample: "
            f"consecutive_rejections={self._consecutive_full_rejections}, "
            f"reason={self._filter.last_pose_rejection_reason or 'unknown'}",
            throttle_duration_sec=1.0,
        )

        if (
            self._max_rejected_samples <= 0
            or self._consecutive_full_rejections < self._max_rejected_samples
        ):
            return

        z_angle = base_link_z_axis_angle_rad(quat_normalize(orientation))
        z_safe = self._filter._within_gate(
            z_angle,
            self._filter.max_base_link_z_axis_angle_rad,
        )
        if all_finite(position, orientation) and z_safe:
            self._filter.initialize(position, orientation)
            self._consecutive_full_rejections = 0
            self.get_logger().warn(
                "Reinitialized EKF after repeated MoCap pose rejections.",
                throttle_duration_sec=1.0,
            )
        else:
            self._reset_filter()
            self.get_logger().warn(
                "Reset EKF after repeated rejected or unsafe MoCap samples.",
                throttle_duration_sec=1.0,
            )

    def _predict_to(self, now):
        if self._last_filter_sec is None:
            self._last_filter_sec = now
            return
        if self._filter.initialized:
            self._filter.predict(now - self._last_filter_sec)
        self._last_filter_sec = now

    def _reset_filter(self):
        self._filter = self._filter.reset_copy()
        self._last_filter_sec = None
        self._last_pose_rx_sec = None
        self._consecutive_full_rejections = 0
        self._imu_gyro_lpf = None

    def _tick(self):
        if not self._filter.initialized or self._last_filter_sec is None:
            return

        now = self._now_sec()
        if (
            self._last_pose_rx_sec is not None
            and (now - self._last_pose_rx_sec) > self._max_coast_sec
        ):
            self._reset_filter()
            self.get_logger().warn(
                "MoCap coasting for too long; waiting for next valid pose.",
                throttle_duration_sec=1.0,
            )
            return

        self._predict_to(now)

        stamp = self.get_clock().now().to_msg()
        odom_msg = self._odom_message(stamp)
        self._odom_pub.publish(odom_msg)
        if self._tf_broadcaster is not None:
            self._tf_broadcaster.sendTransform(self._transform_message(stamp))

    def _odom_message(self, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self._parent_frame
        msg.child_frame_id = self._child_frame

        msg.pose.pose.position.x = float(self._filter.position[0])
        msg.pose.pose.position.y = float(self._filter.position[1])
        msg.pose.pose.position.z = float(self._filter.position[2])
        msg.pose.pose.orientation.x = float(self._filter.orientation[0])
        msg.pose.pose.orientation.y = float(self._filter.orientation[1])
        msg.pose.pose.orientation.z = float(self._filter.orientation[2])
        msg.pose.pose.orientation.w = float(self._filter.orientation[3])

        linear_body = self._filter.body_linear_velocity()
        angular_body = self._filter.body_angular_velocity()
        msg.twist.twist.linear.x = float(linear_body[0])
        msg.twist.twist.linear.y = float(linear_body[1])
        msg.twist.twist.linear.z = float(linear_body[2])
        msg.twist.twist.angular.x = float(angular_body[0])
        msg.twist.twist.angular.y = float(angular_body[1])
        msg.twist.twist.angular.z = float(angular_body[2])

        pose_cov = np.zeros((6, 6), dtype=float)
        pose_cov[0:3, 0:3] = self._filter.covariance[0:3, 0:3]
        pose_cov[3:6, 3:6] = self._filter.covariance[6:9, 6:9]
        msg.pose.covariance = pose_cov.reshape(-1).tolist()

        twist_cov = np.zeros((6, 6), dtype=float)
        twist_cov[0:3, 0:3] = self._filter.covariance[3:6, 3:6]
        twist_cov[3:6, 3:6] = self._filter.covariance[9:12, 9:12]
        msg.twist.covariance = twist_cov.reshape(-1).tolist()
        return msg

    def _transform_message(self, stamp):
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._parent_frame
        msg.child_frame_id = self._child_frame
        msg.transform.translation.x = float(self._filter.position[0])
        msg.transform.translation.y = float(self._filter.position[1])
        msg.transform.translation.z = float(self._filter.position[2])
        msg.transform.rotation.x = float(self._filter.orientation[0])
        msg.transform.rotation.y = float(self._filter.orientation[1])
        msg.transform.rotation.z = float(self._filter.orientation[2])
        msg.transform.rotation.w = float(self._filter.orientation[3])
        return msg

    def _publish_static_body_frames_once(self):
        stamp = self.get_clock().now().to_msg()

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self._child_frame
        transform.child_frame_id = self._child_frame_flu
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 1.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 0.0

        self._static_tf_broadcaster.sendTransform([transform])
        if self._static_tf_timer is not None:
            self._static_tf_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = MocapEkfOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
