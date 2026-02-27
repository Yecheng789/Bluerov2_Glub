#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import VehicleOdometry, VehicleThrustSetpoint, VehicleTorqueSetpoint, VehicleControlMode
from geometry_msgs.msg import Twist


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_R_wxyz(qw, qx, qy, qz):
    ww, xx, yy, zz = qw*qw, qx*qx, qy*qy, qz*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz

    r00 = ww + xx - yy - zz
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = ww - xx - yy + zz

    return (
        (r00, r01, r02),
        (r10, r11, r12),
        (r20, r21, r22),
    )


def mat3_T(R):
    return (
        (R[0][0], R[1][0], R[2][0]),
        (R[0][1], R[1][1], R[2][1]),
        (R[0][2], R[1][2], R[2][2]),
    )


def mat3_mul(A, B):
    return (
        (
            A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0],
            A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1],
            A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2],
        ),
        (
            A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0],
            A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
            A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2],
        ),
        (
            A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0],
            A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
            A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2],
        ),
    )


def mat3_sub(A, B):
    return (
        (A[0][0]-B[0][0], A[0][1]-B[0][1], A[0][2]-B[0][2]),
        (A[1][0]-B[1][0], A[1][1]-B[1][1], A[1][2]-B[1][2]),
        (A[2][0]-B[2][0], A[2][1]-B[2][1], A[2][2]-B[2][2]),
    )


def mat3_scale(A, s):
    return (
        (A[0][0]*s, A[0][1]*s, A[0][2]*s),
        (A[1][0]*s, A[1][1]*s, A[1][2]*s),
        (A[2][0]*s, A[2][1]*s, A[2][2]*s),
    )


def vee_map_skew(E):
    return (E[2][1], E[0][2], E[1][0])


def quat_to_yaw_wxyz(qw, qx, qy, qz) -> float:
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)


def euler_to_quat_wxyz(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cy*cp*cr + sy*sp*sr
    qx = cy*cp*sr - sy*sp*cr
    qy = sy*cp*sr + cy*sp*cr
    qz = sy*cp*cr - cy*sp*sr

    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n > 1e-9:
        return (qw/n, qx/n, qy/n, qz/n)
    return (1.0, 0.0, 0.0, 0.0)


class StabilizedControl(Node):
    def __init__(self):
        super().__init__("stabilized_control")

        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Topics
        self.declare_parameter("odom_topic", "/itrl_rov_1/fmu/out/vehicle_odometry")
        self.declare_parameter("control_mode_topic", "/itrl_rov_1/fmu/out/vehicle_control_mode")
        self.declare_parameter("thrust_sp_topic", "/itrl_rov_1/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_sp_topic", "/itrl_rov_1/fmu/in/vehicle_torque_setpoint")
        self.declare_parameter("cmd_vel_topic", "/itrl_rov_1/cmd_vel")

        # PX4 defaults (from uuv_att_control params)
        self.declare_parameter("UUV_ROLL_P", 4.0)
        self.declare_parameter("UUV_ROLL_D", 1.5)
        self.declare_parameter("UUV_PITCH_P", 4.0)
        self.declare_parameter("UUV_PITCH_D", 2.0)
        self.declare_parameter("UUV_YAW_P", 4.0)
        self.declare_parameter("UUV_YAW_D", 2.0)

        self.declare_parameter("UUV_SGM_YAW", 0.5)
        self.declare_parameter("UUV_SGM_THRTL", 0.1)
        self.declare_parameter("UUV_TORQUE_SAT", 0.3)
        self.declare_parameter("UUV_THRUST_SAT", 0.1)
        self.declare_parameter("UUV_STICK_MODE", 0) # 0=XYZ thrust, 1=surge-only

        # Convention knobs
        self.declare_parameter("yaw_sign", -1.0)
        self.declare_parameter("z_sign", -1.0)
        self.declare_parameter("yaw_rate_lpf_hz", 5.0)

        # Cmd hygiene
        self.declare_parameter("cmd_timeout", 0.3)
        self.declare_parameter("cmd_deadband", 0.05)

        # Loop rate (keep conservative over DDS)
        self.declare_parameter("loop_dt", 0.02)

        odom_topic = self.get_parameter("odom_topic").value
        cm_topic = self.get_parameter("control_mode_topic").value
        thrust_topic = self.get_parameter("thrust_sp_topic").value
        torque_topic = self.get_parameter("torque_sp_topic").value
        cmd_topic = self.get_parameter("cmd_vel_topic").value

        self.sub_odom = self.create_subscription(VehicleOdometry, odom_topic, self.on_odom, px4_qos)
        self.sub_cm = self.create_subscription(VehicleControlMode, cm_topic, self.on_control_mode, px4_qos)
        self.sub_cmd = self.create_subscription(Twist, cmd_topic, self.on_cmd_vel, 10)

        self.pub_thrust = self.create_publisher(VehicleThrustSetpoint, thrust_topic, px4_qos)
        self.pub_torque = self.create_publisher(VehicleTorqueSetpoint, torque_topic, px4_qos)

        # State
        self.have_odom = False
        self.q_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.omega_b = (0.0, 0.0, 0.0)

        self.enabled = False
        self.sp_yaw = 0.0
        self.wz_filt = 0.0

        # cmd (“virtual sticks”)
        self.cmd_xyz = (0.0, 0.0, 0.0)
        self.cmd_yaw = 0.0
        self.cmd_last_us = 0

        self.last_us = None

        self.timer = self.create_timer(float(self.get_parameter("loop_dt").value), self.tick)
        self.get_logger().info(f"stabilized_control listening cmd_vel={cmd_topic}")

    def on_control_mode(self, msg: VehicleControlMode):
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)

        if gate and not self.enabled:
            if self.have_odom:
                self.sp_yaw = quat_to_yaw_wxyz(*self.q_wxyz)
            self.enabled = True
            self.get_logger().info("Enabled (armed + offboard).")

        elif (not gate) and self.enabled:
            self.enabled = False
            self.get_logger().info("Disabled (not armed/offboard).")
            self.publish_zero()

    def on_odom(self, msg: VehicleOdometry):
        self.q_wxyz = (float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3]))
        self.omega_b = (
            float(msg.angular_velocity[0]),
            float(msg.angular_velocity[1]),
            float(msg.angular_velocity[2]),
        )
        self.have_odom = True
        if self.last_us is None:
            self.last_us = int(self.get_clock().now().nanoseconds / 1000)

    def on_cmd_vel(self, msg: Twist):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        yaw_sign = float(self.get_parameter("yaw_sign").value)
        z_sign = float(self.get_parameter("z_sign").value)

        surge = clamp(float(msg.linear.x), -1.0, 1.0)
        sway = clamp(-float(msg.linear.y), -1.0, 1.0)
        heave = clamp(z_sign * float(msg.linear.z), -1.0, 1.0)
        uyaw = clamp(yaw_sign * float(msg.angular.z), -1.0, 1.0)

        self.cmd_xyz = (surge, sway, heave)
        self.cmd_yaw = uyaw
        self.cmd_last_us = now_us

    def get_cmd(self, now_us: int):
        timeout_us = int(float(self.get_parameter("cmd_timeout").value) * 1e6)
        if self.cmd_last_us == 0 or (now_us - self.cmd_last_us) > timeout_us:
            return (0.0, 0.0, 0.0, 0.0)
        sx, sy, sz = self.cmd_xyz
        return (sx, sy, sz, self.cmd_yaw)

    def publish_zero(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = 0
        thr.xyz = [0.0, 0.0, 0.0]
        self.pub_thrust.publish(thr)

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = 0
        tor.xyz = [0.0, 0.0, 0.0]
        self.pub_torque.publish(tor)

    def tick(self):
        if not self.have_odom:
            return

        now_us = int(self.get_clock().now().nanoseconds / 1000)
        if self.last_us is None:
            self.last_us = now_us
            return

        dt = (now_us - self.last_us) * 1e-6
        dt = clamp(dt, 0.0002, 0.02)
        self.last_us = now_us

        if not self.enabled:
            self.sp_yaw = quat_to_yaw_wxyz(*self.q_wxyz)
            return

        deadband = float(self.get_parameter("cmd_deadband").value)
        surge_u, sway_u, heave_u, uyaw = self.get_cmd(now_us)

        # Roll/pitch leveled
        roll_sp = 0.0
        pitch_sp = 0.0

        yaw = quat_to_yaw_wxyz(*self.q_wxyz)
        sgm_yaw = float(self.get_parameter("UUV_SGM_YAW").value)

        # Explicit PX4-like yaw integration with lock when no command
        if abs(uyaw) <= deadband:
            self.sp_yaw = yaw
        else:
            self.sp_yaw = wrap_pi(self.sp_yaw + uyaw * dt * sgm_yaw)

        qw_sp, qx_sp, qy_sp, qz_sp = euler_to_quat_wxyz(roll_sp, pitch_sp, self.sp_yaw)

        # Geometric controller (same structure as PX4 control_attitude_geo)
        R = quat_to_R_wxyz(*self.q_wxyz) # body->world
        R_des = quat_to_R_wxyz(qw_sp, qx_sp, qy_sp, qz_sp)

        Rt = mat3_T(R)
        Rdt = mat3_T(R_des)

        e_R = mat3_scale(mat3_sub(mat3_mul(Rdt, R), mat3_mul(Rt, R_des)), 0.5)
        eR_x, eR_y, eR_z = vee_map_skew(e_R)

        wx, wy, wz = self.omega_b
        yaw_rate_lpf_hz = float(self.get_parameter("yaw_rate_lpf_hz").value)
        if yaw_rate_lpf_hz > 0.0:
            rc = 1.0 / (2.0 * math.pi * yaw_rate_lpf_hz)
            alpha = dt / (rc + dt)
            self.wz_filt = (1.0 - alpha) * self.wz_filt + alpha * wz
            wz_use = self.wz_filt
        else:
            wz_use = wz

        roll_p = float(self.get_parameter("UUV_ROLL_P").value)
        pitch_p = float(self.get_parameter("UUV_PITCH_P").value)
        yaw_p = float(self.get_parameter("UUV_YAW_P").value)

        roll_d = float(self.get_parameter("UUV_ROLL_D").value)
        pitch_d = float(self.get_parameter("UUV_PITCH_D").value)
        yaw_d = float(self.get_parameter("UUV_YAW_D").value)

        tau_x = -roll_p * eR_x - roll_d * wx
        tau_y = -pitch_p * eR_y - pitch_d * wy
        tau_z = -yaw_p * eR_z - yaw_d * wz_use

        torque_sat = float(self.get_parameter("UUV_TORQUE_SAT").value)
        tau_x = clamp(tau_x, -torque_sat, torque_sat)
        tau_y = clamp(tau_y, -torque_sat, torque_sat)
        tau_z = clamp(tau_z, -torque_sat, torque_sat)

        # Thrust mapping (PX4-like)
        sgm_thrtl = float(self.get_parameter("UUV_SGM_THRTL").value)
        stick_mode = int(self.get_parameter("UUV_STICK_MODE").value)

        surge = surge_u if abs(surge_u) > deadband else 0.0
        sway = sway_u if abs(sway_u) > deadband else 0.0
        heave = heave_u if abs(heave_u) > deadband else 0.0

        if stick_mode == 0:
            thrust_x = surge * sgm_thrtl
            thrust_y = sway * sgm_thrtl
            thrust_z = heave * sgm_thrtl
        else:
            thrust_x = surge * sgm_thrtl
            thrust_y = 0.0
            thrust_z = 0.0

        thrust_sat = float(self.get_parameter("UUV_THRUST_SAT").value)
        thrust_x = clamp(thrust_x, -thrust_sat, thrust_sat)
        thrust_y = clamp(thrust_y, -thrust_sat, thrust_sat)
        thrust_z = clamp(thrust_z, -thrust_sat, thrust_sat)

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = 0
        thr.xyz = [float(thrust_x), float(thrust_y), float(thrust_z)]
        self.pub_thrust.publish(thr)

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = 0
        tor.xyz = [float(tau_x), float(tau_y), float(tau_z)]
        self.pub_torque.publish(tor)


def main():
    rclpy.init()
    node = StabilizedControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.publish_zero()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()