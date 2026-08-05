#!/usr/bin/env python3
"""
PX4-attitude-backed position controller for the fixed-hook Gazebo mission.

The ROS node implements the position outer loop used by PX4's UUV position
controller and deliberately leaves attitude stabilization to PX4
``uuv_att_control``.  It publishes exactly one OffboardControlMode stream with
``attitude=True`` and one VehicleAttitudeSetpoint stream.  No wrench setpoint is
published by this node.

All navigation quantities use PX4 NED / body FRD conventions.  The hook model
is never moved; ``hook_alignment_raise_m`` changes the vehicle target instead.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    VehicleAttitudeSetpoint,
    VehicleControlMode,
    VehicleOdometry,
    VehicleRatesSetpoint,
)
from std_msgs.msg import Bool


DEFAULT_CONTROL_MODE_TIMEOUT_SEC = 1.25


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quat_norm_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("quaternion must contain four finite nonzero values")
    return q / norm


def quat_to_yaw_wxyz(quaternion):
    qw, qx, qy, qz = quat_norm_wxyz(quaternion)
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)


def level_yaw_quat_wxyz(yaw):
    half_yaw = 0.5 * float(yaw)
    return np.array(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        dtype=float,
    )


def quat_to_rotation_wxyz(quaternion):
    """Return the body-FRD to navigation-frame rotation matrix."""
    qw, qx, qy, qz = quat_norm_wxyz(quaternion)
    return np.array([
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qw * qz),
            2.0 * (qx * qz + qw * qy),
        ],
        [
            2.0 * (qx * qy + qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qw * qx),
        ],
        [
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ], dtype=float)


def move_towards_vector(current, target, max_speed, dt):
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - current
    distance = float(np.linalg.norm(delta))
    max_step = max(float(max_speed), 0.0) * max(float(dt), 0.0)
    if distance <= max(max_step, 1e-12):
        return target.copy()
    return current + (max_step / distance) * delta


def move_towards_yaw(current, target, max_rate, dt):
    error = wrap_pi(float(target) - float(current))
    max_step = max(float(max_rate), 0.0) * max(float(dt), 0.0)
    return wrap_pi(float(current) + clamp(error, -max_step, max_step))


def velocity_in_navigation_frame(msg, quaternion):
    """Convert VehicleOdometry velocity to the position/navigation frame."""
    velocity = np.asarray(msg.velocity, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("odometry velocity must contain three finite values")

    frame = int(msg.velocity_frame)
    if frame == int(VehicleOdometry.VELOCITY_FRAME_NED):
        if int(msg.pose_frame) != int(VehicleOdometry.POSE_FRAME_NED):
            raise ValueError("NED velocity requires an NED pose for this mission")
        return velocity
    if frame == int(VehicleOdometry.VELOCITY_FRAME_BODY_FRD):
        return quat_to_rotation_wxyz(quaternion) @ velocity
    if (
        frame == int(VehicleOdometry.VELOCITY_FRAME_FRD)
        and int(msg.pose_frame) == int(VehicleOdometry.POSE_FRAME_FRD)
    ):
        return velocity
    raise ValueError(
        f"unsupported odometry pose/velocity frames: "
        f"{int(msg.pose_frame)}/{frame}"
    )


def position_pd_thrust_body(
    position,
    velocity_nav,
    quaternion,
    reference_position,
    kp,
    kd,
    axis_limits,
    reference_velocity=None,
):
    """PX4-UUV-style navigation-frame P-D output rotated into body FRD."""
    position = np.asarray(position, dtype=float)
    velocity_nav = np.asarray(velocity_nav, dtype=float)
    reference_position = np.asarray(reference_position, dtype=float)
    kp = np.asarray(kp, dtype=float)
    kd = np.asarray(kd, dtype=float)
    limits = np.asarray(axis_limits, dtype=float)
    if reference_velocity is None:
        reference_velocity = np.zeros(3, dtype=float)
    reference_velocity = np.asarray(reference_velocity, dtype=float)

    values = np.concatenate([
        position,
        velocity_nav,
        reference_position,
        kp,
        kd,
        limits,
        reference_velocity,
    ])
    if not np.all(np.isfinite(values)) or np.any(limits <= 0.0):
        raise ValueError("position controller inputs and positive limits must be finite")

    thrust_nav = (
        kp * (reference_position - position)
        + kd * (reference_velocity - velocity_nav)
    )
    thrust_body = quat_to_rotation_wxyz(quaternion).T @ thrust_nav
    return np.clip(thrust_body, -limits, limits)


def retrieval_targets(
    *,
    box_xyz_sdf,
    hook_mount_xyz_sdf,
    hook_tip_extra_x,
    handle_offset_ned,
    approach_yaw,
    approach_clearance,
    pass_overshoot,
    backward_extra,
    alignment_raise,
):
    """Return ALIGN/PASS/BACK vehicle targets without moving hook geometry."""
    box_x, box_y, box_z_sdf = np.asarray(box_xyz_sdf, dtype=float)
    mount_x, mount_y, mount_z_sdf = np.asarray(
        hook_mount_xyz_sdf,
        dtype=float,
    )
    handle = np.array([box_x, box_y, -box_z_sdf], dtype=float)
    handle += np.asarray(handle_offset_ned, dtype=float)

    yaw = float(approach_yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation_yaw = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    forward = np.array([cos_yaw, sin_yaw, 0.0], dtype=float)
    hook_tip_body = np.array([
        mount_x + float(hook_tip_extra_x),
        mount_y,
        -mount_z_sdf,
    ], dtype=float)
    hook_tip_nav = rotation_yaw @ hook_tip_body

    align = handle - float(approach_clearance) * forward - hook_tip_nav
    forward_pass = handle + float(pass_overshoot) * forward - hook_tip_nav
    backward_pass = align - float(backward_extra) * forward

    raise_m = float(alignment_raise)
    if not math.isfinite(raise_m) or raise_m < 0.0:
        raise ValueError("alignment_raise must be finite and nonnegative")
    # NED +z points down.  Reduce the vehicle z target to raise the unchanged
    # hook, and keep the same correction throughout the pass.
    for target in (align, forward_pass, backward_pass):
        target[2] -= raise_m
    return align, forward_pass, backward_pass


class PositionPayloadRetrieval(Node):
    """Autonomous staged position mission using the PX4 UUV attitude loop."""

    def __init__(self):
        super().__init__("position_payload_retrieval")

        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.declare_parameter("odom_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter(
            "control_mode_topic",
            "/fmu/out/vehicle_control_mode",
        )
        self.declare_parameter(
            "attitude_setpoint_topic",
            "/fmu/in/vehicle_attitude_setpoint_v1",
        )
        self.declare_parameter(
            "offboard_control_mode_topic",
            "/fmu/in/offboard_control_mode",
        )
        self.declare_parameter(
            "rates_setpoint_topic",
            "/fmu/in/vehicle_rates_setpoint",
        )
        self.declare_parameter(
            "mission_enable_topic",
            "/bluerov2/mission_enable",
        )
        self.declare_parameter("require_mission_enable", False)

        self.declare_parameter("loop_rate_hz", 50.0)
        self.declare_parameter("odom_timeout_s", 0.30)
        self.declare_parameter(
            "control_mode_timeout_s",
            DEFAULT_CONTROL_MODE_TIMEOUT_SEC,
        )

        # The defaults match PX4 uuv_pos_control.  The final normalized output
        # is kept below the PX4 UUV_THRUST_SAT=0.1 default.
        self.declare_parameter("gain_x_p", 0.6)
        self.declare_parameter("gain_y_p", 0.6)
        self.declare_parameter("gain_z_p", 0.8)
        self.declare_parameter("gain_x_d", 0.25)
        self.declare_parameter("gain_y_d", 0.25)
        self.declare_parameter("gain_z_d", 0.30)
        self.declare_parameter("thrust_limit_xy", 0.08)
        self.declare_parameter("thrust_limit_z", 0.08)

        self.declare_parameter("align_speed_mps", 0.06)
        self.declare_parameter("descent_speed_mps", 0.06)
        self.declare_parameter("forward_pass_speed_mps", 0.05)
        self.declare_parameter("backward_pass_speed_mps", 0.06)
        self.declare_parameter("return_speed_mps", 0.06)
        self.declare_parameter("yaw_reference_rate_rad_s", 0.10)
        self.declare_parameter("mission_pos_tol_m", 0.08)
        self.declare_parameter("mission_yaw_tol_rad", math.radians(5.0))
        self.declare_parameter("settled_linear_speed_mps", 0.05)
        self.declare_parameter("settled_angular_speed_rad_s", 0.12)
        self.declare_parameter("align_hold_s", 2.0)

        self.declare_parameter("box_x", -1.5)
        self.declare_parameter("box_y", -1.5)
        self.declare_parameter("box_z_sdf", -96.5)
        self.declare_parameter("hook_mount_x", 0.42)
        self.declare_parameter("hook_mount_y", 0.04)
        self.declare_parameter("hook_mount_z_sdf", -0.08)
        self.declare_parameter("hook_tip_extra_x", 0.10)
        self.declare_parameter("handle_offset_world_x", 0.1)
        self.declare_parameter("handle_offset_world_y", -0.10)
        self.declare_parameter("handle_offset_world_z_down", -0.025)
        self.declare_parameter("approach_yaw", -1.57)
        self.declare_parameter("approach_clearance", 0.1)
        self.declare_parameter("pass_overshoot", 0.1)
        self.declare_parameter("backward_extra_m", 0.08)
        self.declare_parameter("hook_alignment_raise_m", 0.04)
        # Once vertically aligned, lower the vehicle slightly so the hook
        # opening, rather than its lower edge, passes through the handle.
        self.declare_parameter("align_hold_lower_m", 0.01)
        self.declare_parameter("align_hold_depth_tol_m", 0.01)
        self.declare_parameter("return_to_start", True)
        self.declare_parameter("shore_x", 0.0)
        self.declare_parameter("shore_y", 0.0)
        self.declare_parameter("shore_z", 95.0)
        self.declare_parameter("shore_yaw", 0.0)

        odom_topic = str(self.get_parameter("odom_topic").value)
        control_mode_topic = str(
            self.get_parameter("control_mode_topic").value
        )
        attitude_topic = str(
            self.get_parameter("attitude_setpoint_topic").value
        )
        heartbeat_topic = str(
            self.get_parameter("offboard_control_mode_topic").value
        )
        rates_topic = str(self.get_parameter("rates_setpoint_topic").value)
        mission_enable_topic = str(
            self.get_parameter("mission_enable_topic").value
        )

        self.sub_odom = self.create_subscription(
            VehicleOdometry,
            odom_topic,
            self.on_odometry,
            px4_qos,
        )
        self.sub_control_mode = self.create_subscription(
            VehicleControlMode,
            control_mode_topic,
            self.on_control_mode,
            px4_qos,
        )
        self.sub_mission_enable = self.create_subscription(
            Bool,
            mission_enable_topic,
            self.on_mission_enable,
            10,
        )
        self.pub_attitude = self.create_publisher(
            VehicleAttitudeSetpoint,
            attitude_topic,
            px4_qos,
        )
        self.pub_offboard_mode = self.create_publisher(
            OffboardControlMode,
            heartbeat_topic,
            10,
        )
        self.pub_rates = self.create_publisher(
            VehicleRatesSetpoint,
            rates_topic,
            px4_qos,
        )

        self.have_odom = False
        self.odom_valid = False
        self.position_ned = np.zeros(3, dtype=float)
        self.velocity_ned = np.zeros(3, dtype=float)
        self.angular_velocity_body = np.zeros(3, dtype=float)
        self.quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_odom_monotonic = None
        self.last_control_mode_monotonic = None
        self.armed_offboard = False
        self.attitude_control_enabled = False
        self.mission_enable = not bool(
            self.get_parameter("require_mission_enable").value
        )

        self.mission_state = "WAITING"
        self.home_position = None
        self.home_yaw = 0.0
        self.align_target = None
        self.forward_target = None
        self.backward_target = None
        self.reference_position = None
        self.reference_yaw = None
        self.hold_started_monotonic = None
        self.last_tick_monotonic = None

        loop_rate = float(self.get_parameter("loop_rate_hz").value)
        if not math.isfinite(loop_rate) or loop_rate <= 0.0:
            raise ValueError("loop_rate_hz must be finite and positive")
        self.timer = self.create_timer(1.0 / loop_rate, self.tick)

        self.get_logger().info(
            "Position payload retrieval ready: ROS position P-D outer loop + "
            "PX4 UUV attitude inner loop; roll/pitch setpoint fixed at zero."
        )

    def on_odometry(self, msg):
        receive_time = time.monotonic()
        try:
            position = np.asarray(msg.position, dtype=float)
            quaternion = quat_norm_wxyz(msg.q)
            angular_velocity = np.asarray(msg.angular_velocity, dtype=float)
            if (
                int(msg.pose_frame) != int(VehicleOdometry.POSE_FRAME_NED)
                or position.shape != (3,)
                or angular_velocity.shape != (3,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(angular_velocity))
            ):
                raise ValueError("mission requires finite NED odometry")
            velocity = velocity_in_navigation_frame(msg, quaternion)
        except ValueError as exc:
            self.odom_valid = False
            self.get_logger().error(
                f"Rejected VehicleOdometry: {exc}",
                throttle_duration_sec=1.0,
            )
            return

        self.position_ned = position
        self.quaternion_wxyz = quaternion
        self.velocity_ned = velocity
        self.angular_velocity_body = angular_velocity
        self.have_odom = True
        self.odom_valid = True
        self.last_odom_monotonic = receive_time

    def on_control_mode(self, msg):
        self.last_control_mode_monotonic = time.monotonic()
        self.armed_offboard = bool(msg.flag_armed) and bool(
            msg.flag_control_offboard_enabled
        )
        self.attitude_control_enabled = bool(
            msg.flag_control_attitude_enabled
        )

    def on_mission_enable(self, msg):
        self.mission_enable = bool(msg.data)
        if not self.mission_enable:
            self._reset_mission()
            self.get_logger().info("Position mission disabled; zero thrust hold.")
        else:
            self.get_logger().info(
                "Position mission enabled; it will start after Armed + Offboard."
            )

    def _now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def _odom_fresh(self, now_monotonic):
        if (
            not self.have_odom
            or not self.odom_valid
            or self.last_odom_monotonic is None
        ):
            return False
        timeout = float(self.get_parameter("odom_timeout_s").value)
        return timeout <= 0.0 or (
            0.0 <= now_monotonic - self.last_odom_monotonic <= timeout
        )

    def _control_mode_fresh(self, now_monotonic):
        if self.last_control_mode_monotonic is None:
            return False
        timeout = float(
            self.get_parameter("control_mode_timeout_s").value
        )
        return timeout <= 0.0 or (
            0.0
            <= now_monotonic - self.last_control_mode_monotonic
            <= timeout
        )

    def _control_active(self, now_monotonic):
        return (
            self.armed_offboard
            and self._control_mode_fresh(now_monotonic)
            and self.attitude_control_enabled
            and self.mission_enable
        )

    def _publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = self._now_us()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.pub_offboard_mode.publish(msg)

    def _publish_attitude_setpoint(
        self,
        thrust_body,
        yaw_reference,
        yaw_rate_reference=0.0,
    ):
        timestamp = self._now_us()
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = timestamp
        msg.yaw_sp_move_rate = float(yaw_rate_reference)
        msg.q_d = level_yaw_quat_wxyz(yaw_reference).tolist()
        msg.thrust_body = np.asarray(thrust_body, dtype=float).tolist()
        self.pub_attitude.publish(msg)

        # PX4's UUV attitude controller also consumes the rates setpoint while
        # attitude control is enabled.  Publish it from the same process so an
        # old manual/rate command cannot keep driving yaw after q_d settles.
        rates = VehicleRatesSetpoint()
        rates.timestamp = timestamp
        rates.roll = 0.0
        rates.pitch = 0.0
        rates.yaw = float(yaw_rate_reference)
        rates.thrust_body = [0.0, 0.0, 0.0]
        rates.reset_integral = False
        self.pub_rates.publish(rates)

    def _reset_mission(self):
        self.mission_state = "WAITING"
        self.home_position = None
        self.align_target = None
        self.forward_target = None
        self.backward_target = None
        self.reference_position = None
        self.reference_yaw = None
        self.hold_started_monotonic = None

    def _compute_targets(self):
        return retrieval_targets(
            box_xyz_sdf=(
                self.get_parameter("box_x").value,
                self.get_parameter("box_y").value,
                self.get_parameter("box_z_sdf").value,
            ),
            hook_mount_xyz_sdf=(
                self.get_parameter("hook_mount_x").value,
                self.get_parameter("hook_mount_y").value,
                self.get_parameter("hook_mount_z_sdf").value,
            ),
            hook_tip_extra_x=self.get_parameter("hook_tip_extra_x").value,
            handle_offset_ned=(
                self.get_parameter("handle_offset_world_x").value,
                self.get_parameter("handle_offset_world_y").value,
                self.get_parameter("handle_offset_world_z_down").value,
            ),
            approach_yaw=self.get_parameter("approach_yaw").value,
            approach_clearance=self.get_parameter("approach_clearance").value,
            pass_overshoot=self.get_parameter("pass_overshoot").value,
            backward_extra=self.get_parameter("backward_extra_m").value,
            alignment_raise=self.get_parameter("hook_alignment_raise_m").value,
        )

    def _start_mission(self, now_monotonic):
        self.home_position = self.position_ned.copy()
        self.home_yaw = quat_to_yaw_wxyz(self.quaternion_wxyz)
        (
            self.align_target,
            self.forward_target,
            self.backward_target,
        ) = self._compute_targets()
        self.reference_position = self.position_ned.copy()
        self.reference_yaw = self.home_yaw
        self.hold_started_monotonic = None
        self._enter_stage("ALIGN_XY", now_monotonic, reset_reference=False)
        self.get_logger().info(
            "Mission start: horizontal alignment first, then vertical descent; "
            f"home={self.home_position.tolist()}, "
            f"align={self.align_target.tolist()}."
        )

    def _enter_stage(self, state, now_monotonic, reset_reference=True):
        self.mission_state = state
        if reset_reference:
            self.reference_position = self.position_ned.copy()
            self.reference_yaw = quat_to_yaw_wxyz(self.quaternion_wxyz)
        self.hold_started_monotonic = (
            now_monotonic if state == "ALIGN_HOLD" else None
        )
        self.get_logger().info(f"Position mission -> {state}")

    def _stage_target(self):
        approach_yaw = float(self.get_parameter("approach_yaw").value)
        if self.mission_state == "ALIGN_XY":
            target = self.align_target.copy()
            target[2] = self.home_position[2]
            return target, approach_yaw
        if self.mission_state == "DESCEND":
            return self.align_target.copy(), approach_yaw
        engagement_lower = float(
            self.get_parameter("align_hold_lower_m").value
        )
        if self.mission_state == "ALIGN_HOLD":
            target = self.align_target.copy()
            target[2] += engagement_lower
            return target, approach_yaw
        if self.mission_state == "FORWARD_PASS":
            target = self.forward_target.copy()
            target[2] += engagement_lower
            return target, approach_yaw
        if self.mission_state == "BACKWARD_PASS":
            target = self.backward_target.copy()
            target[2] += engagement_lower
            return target, approach_yaw
        if bool(self.get_parameter("return_to_start").value):
            return self.home_position.copy(), self.home_yaw
        return np.array([
            float(self.get_parameter("shore_x").value),
            float(self.get_parameter("shore_y").value),
            float(self.get_parameter("shore_z").value),
        ]), float(self.get_parameter("shore_yaw").value)

    def _stage_speed(self):
        parameter = {
            "ALIGN_XY": "align_speed_mps",
            "DESCEND": "descent_speed_mps",
            "ALIGN_HOLD": "descent_speed_mps",
            "FORWARD_PASS": "forward_pass_speed_mps",
            "BACKWARD_PASS": "backward_pass_speed_mps",
            "RETURN_HOME": "return_speed_mps",
            "DONE": "return_speed_mps",
        }.get(self.mission_state, "align_speed_mps")
        return float(self.get_parameter(parameter).value)

    def _at_target(self, target, target_yaw):
        position_error = float(np.linalg.norm(self.position_ned - target))
        yaw = quat_to_yaw_wxyz(self.quaternion_wxyz)
        yaw_error = abs(wrap_pi(target_yaw - yaw))
        return (
            position_error
            <= float(self.get_parameter("mission_pos_tol_m").value)
            and yaw_error
            <= float(self.get_parameter("mission_yaw_tol_rad").value)
            and float(np.linalg.norm(self.velocity_ned))
            <= float(self.get_parameter("settled_linear_speed_mps").value)
            and float(np.linalg.norm(self.angular_velocity_body))
            <= float(
                self.get_parameter("settled_angular_speed_rad_s").value
            )
        )

    def _update_stage(self, now_monotonic):
        target, target_yaw = self._stage_target()
        at_target = self._at_target(target, target_yaw)

        if self.mission_state == "ALIGN_XY" and at_target:
            self._enter_stage("DESCEND", now_monotonic)
        elif self.mission_state == "DESCEND" and at_target:
            self._enter_stage("ALIGN_HOLD", now_monotonic)
        elif self.mission_state == "ALIGN_HOLD":
            depth_ready = abs(
                self.position_ned[2] - target[2]
            ) <= float(
                self.get_parameter("align_hold_depth_tol_m").value
            )
            if not at_target or not depth_ready:
                self.hold_started_monotonic = now_monotonic
            elif (
                now_monotonic - self.hold_started_monotonic
                >= float(self.get_parameter("align_hold_s").value)
            ):
                self._enter_stage("FORWARD_PASS", now_monotonic)
        elif self.mission_state == "FORWARD_PASS" and at_target:
            self._enter_stage("BACKWARD_PASS", now_monotonic)
        elif self.mission_state == "BACKWARD_PASS" and at_target:
            self._enter_stage("RETURN_HOME", now_monotonic)
        elif self.mission_state == "RETURN_HOME" and at_target:
            self._enter_stage("DONE", now_monotonic)

    def _controller_gains_and_limits(self):
        kp = np.array([
            float(self.get_parameter("gain_x_p").value),
            float(self.get_parameter("gain_y_p").value),
            float(self.get_parameter("gain_z_p").value),
        ])
        kd = np.array([
            float(self.get_parameter("gain_x_d").value),
            float(self.get_parameter("gain_y_d").value),
            float(self.get_parameter("gain_z_d").value),
        ])
        xy_limit = float(self.get_parameter("thrust_limit_xy").value)
        z_limit = float(self.get_parameter("thrust_limit_z").value)
        return kp, kd, np.array([xy_limit, xy_limit, z_limit])

    def tick(self):
        now_monotonic = time.monotonic()

        if self.last_tick_monotonic is None:
            dt = 1.0 / float(self.get_parameter("loop_rate_hz").value)
        else:
            dt = clamp(now_monotonic - self.last_tick_monotonic, 0.001, 0.05)
        self.last_tick_monotonic = now_monotonic

        if not self._odom_fresh(now_monotonic):
            self.get_logger().warn(
                "VehicleOdometry absent, invalid, or stale; no attitude "
                "setpoint is published.",
                throttle_duration_sec=1.0,
            )
            if self.have_odom:
                self._publish_attitude_setpoint(
                    np.zeros(3),
                    quat_to_yaw_wxyz(self.quaternion_wxyz),
                )
            self._reset_mission()
            return

        current_yaw = quat_to_yaw_wxyz(self.quaternion_wxyz)
        if not self._control_mode_fresh(now_monotonic):
            self.get_logger().warn(
                "VehicleControlMode absent or stale; zero thrust published "
                "and Offboard heartbeat stopped.",
                throttle_duration_sec=1.0,
            )
            self._publish_attitude_setpoint(np.zeros(3), current_yaw)
            self._reset_mission()
            return

        # Heartbeat and control output live in the same process.  It is only
        # streamed while the complete feedback chain is healthy.
        self._publish_offboard_heartbeat()
        if not self._control_active(now_monotonic):
            if self.mission_state != "WAITING":
                self._reset_mission()
            self._publish_attitude_setpoint(np.zeros(3), current_yaw)
            return

        if self.mission_state == "WAITING":
            self._start_mission(now_monotonic)

        self._update_stage(now_monotonic)
        target, target_yaw = self._stage_target()
        previous_reference_position = self.reference_position.copy()
        self.reference_position = move_towards_vector(
            self.reference_position,
            target,
            self._stage_speed(),
            dt,
        )
        reference_velocity = (
            self.reference_position - previous_reference_position
        ) / max(dt, 1e-6)
        previous_reference_yaw = self.reference_yaw
        self.reference_yaw = move_towards_yaw(
            self.reference_yaw,
            target_yaw,
            float(
                self.get_parameter("yaw_reference_rate_rad_s").value
            ),
            dt,
        )
        reference_yaw_rate = wrap_pi(
            self.reference_yaw - previous_reference_yaw
        ) / max(dt, 1e-6)

        kp, kd, limits = self._controller_gains_and_limits()
        try:
            thrust_body = position_pd_thrust_body(
                self.position_ned,
                self.velocity_ned,
                self.quaternion_wxyz,
                self.reference_position,
                kp,
                kd,
                limits,
                reference_velocity,
            )
        except ValueError as exc:
            self.get_logger().error(
                f"Position controller rejected state: {exc}",
                throttle_duration_sec=1.0,
            )
            thrust_body = np.zeros(3)

        self._publish_attitude_setpoint(
            thrust_body,
            self.reference_yaw,
            reference_yaw_rate,
        )


def main():
    rclpy.init()
    node = PositionPayloadRetrieval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.have_odom:
            node._publish_attitude_setpoint(
                np.zeros(3),
                quat_to_yaw_wxyz(node.quaternion_wxyz),
            )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
