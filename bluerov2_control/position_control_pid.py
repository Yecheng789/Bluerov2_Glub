#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint
from px4_msgs.msg import VehicleControlMode

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
    # Returns rotation matrix R_body_to_world (3x3) for quaternion (w,x,y,z)
    # Standard Hamilton convention.
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


def mat3_mul_vec(R, v):
    return (
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    )


def quat_to_yaw_wxyz(qw, qx, qy, qz) -> float:
    # Yaw extracted from quaternion; valid when msg.q is the body(FRD)->world rotation 
    # and world is the same frame as pose_frame (here NED). If mocap is ENU, convert before using.”
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PositionControlPID(Node):
    def __init__(self):
        super().__init__("position_control_pid")

        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Topics
        self.declare_parameter("odom_topic", "/itrl_rov_1/fmu/out/vehicle_odometry")
        self.declare_parameter("thrust_sp_topic", "/itrl_rov_1/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_sp_topic", "/itrl_rov_1/fmu/in/vehicle_torque_setpoint")
        self.declare_parameter("control_mode_topic", "/itrl_rov_1/fmu/out/vehicle_control_mode")

        # PX4-like parameters
        self.declare_parameter("gain_x_p", 1.0)
        self.declare_parameter("gain_y_p", 1.0)
        self.declare_parameter("gain_z_p", 1.0)
        self.declare_parameter("gain_x_d", 0.2)
        self.declare_parameter("gain_y_d", 0.2)
        self.declare_parameter("gain_z_d", 0.2)

        # Keyboard / setpoint generation (PX4-like)
        self.declare_parameter("pos_stick_db", 0.1)   # deadband on "virtual stick"
        self.declare_parameter("pgm_vel", 0.5)        # m/s per full input (like UUV_PGM_VEL)
        self.declare_parameter("sgm_yaw", 0.8)        # rad/s per full input

        # 0: move setpoint in world frame, 1: move setpoint in body frame (like UUV_POS_MODE)
        self.declare_parameter("pos_mode", 1)

        # Output limits (tune for your vehicle + px4 mixer scaling)
        self.declare_parameter("max_thrust_xy", 1.0)
        self.declare_parameter("max_thrust_z", 1.0)
        self.declare_parameter("max_torque_z", 1.0)

        # Yaw controller (external to PX4 example, because you publish torque setpoint)
        self.declare_parameter("yaw_kp", 1.5)
        self.declare_parameter("yaw_kd", 0.05)

        self.declare_parameter("cmd_vel_topic", "/itrl_rov_1/cmd_vel")
        self.declare_parameter("cmd_timeout", 0.3)   # seconds
        self.declare_parameter("yaw_sign", -1.0)     # flip if yaw turns wrong way
        self.declare_parameter("z_sign", -1.0)       # +linear.z means up in ROS; NED up is -z

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        thrust_topic = self.get_parameter("thrust_sp_topic").get_parameter_value().string_value
        torque_topic = self.get_parameter("torque_sp_topic").get_parameter_value().string_value
        cm_topic = self.get_parameter("control_mode_topic").get_parameter_value().string_value
        cmd_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value

        self.odom_sub = self.create_subscription(VehicleOdometry, odom_topic, self.on_odom, px4_qos)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, thrust_topic, px4_qos)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, torque_topic, px4_qos)
        self.cm_sub = self.create_subscription(VehicleControlMode, cm_topic, self.on_control_mode, px4_qos)
        self.cmd_sub = self.create_subscription(Twist, cmd_topic, self.on_cmd_vel, 10)

        self.is_armed = False
        self.is_offboard = False
        self.enabled = False

        self.cmd_lin = (0.0, 0.0, 0.0)  # last commanded linear (x,y,z)
        self.cmd_yaw = 0.0              # last commanded yaw rate input
        self.cmd_last_us = 0            # last cmd timestamp (us)

        self.get_logger().info(f"Listening for Twist commands on: {cmd_topic} (teleop_twist_keyboard)")

        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

        self.have_odom = False
        self.pos_w = (0.0, 0.0, 0.0)
        self.vel_v = (0.0, 0.0, 0.0)
        self.q_wxyz = (1.0, 0.0, 0.0, 0.0)

        self.sp_pos_w = None  # set on first odom
        self.sp_yaw = 0.0
        self.ang_vel_bz = 0.0

        self.last_t_us = None


    def on_control_mode(self, msg: VehicleControlMode):
        was_enabled = self.enabled

        self.is_armed = bool(msg.flag_armed)
        self.is_offboard = bool(msg.flag_control_offboard_enabled)

        gate = self.is_armed and self.is_offboard

        if gate and not was_enabled:
            # Just transitioned into (armed + offboard): reset controller setpoints to current state
            if self.have_odom:
                self.sp_pos_w = self.pos_w
                self.sp_yaw = quat_to_yaw_wxyz(*self.q_wxyz)
            self.enabled = True
            self.get_logger().info("Position PID enabled (armed + offboard).")

        elif (not gate) and was_enabled:
            # Just transitioned out of (armed + offboard)
            self.enabled = False
            self.get_logger().info("Position PID disabled (not armed/offboard).")

    def on_odom(self, msg: VehicleOdometry):
        # Treat msg.position as world position and msg.velocity as world velocity.
        # If your velocity_frame is BODY, you can rotate it with R_body_to_world.
        self.pos_w = (float(msg.position[0]), float(msg.position[1]), float(msg.position[2]))
        self.vel_v = (float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2]))
        self.q_wxyz = (float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3]))
        self.ang_vel_bz = float(msg.angular_velocity[2])
        self.have_odom = True

        if self.sp_pos_w is None:
            self.sp_pos_w = self.pos_w
            self.sp_yaw = quat_to_yaw_wxyz(*self.q_wxyz)

    
    def on_cmd_vel(self, msg: Twist):
        # We treat Twist as normalized "virtual sticks" in [-1, 1]
        # linear.x: forward (+)
        # linear.y: left (+)
        # linear.z: up (+) in ROS -> convert to NED convention with z_sign
        # angular.z: yaw rate input (+) in ROS is typically CCW about +Z -> yaw_sign may be needed for PX4/NED
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        yaw_sign = float(self.get_parameter("yaw_sign").value)
        z_sign = float(self.get_parameter("z_sign").value)

        ux = clamp(float(msg.linear.x), -1.0, 1.0)
        uy = clamp(float(msg.linear.y), -1.0, 1.0)
        uz = clamp(z_sign * float(msg.linear.z), -1.0, 1.0)
        uyaw = clamp(yaw_sign * float(msg.angular.z), -1.0, 1.0)

        self.cmd_lin = (ux, uy, uz)
        self.cmd_yaw = uyaw
        self.cmd_last_us = now_us


    def get_cmd_inputs(self, now_us: int):
        timeout_s = float(self.get_parameter("cmd_timeout").value)
        timeout_us = int(timeout_s * 1e6)

        if self.cmd_last_us == 0 or (now_us - self.cmd_last_us) > timeout_us:
            return 0.0, 0.0, 0.0, 0.0

        ux, uy, uz = self.cmd_lin
        uyaw = self.cmd_yaw
        return ux, uy, uz, uyaw


    def tick(self):
        if not self.have_odom or self.sp_pos_w is None:
            return
        
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        if not self.enabled:
            self.last_t_us = now_us
            return

        if self.last_t_us is None:
            self.last_t_us = now_us
            return

        dt = (now_us - self.last_t_us) * 1e-6
        dt = clamp(dt, 0.0002, 0.05)
        self.last_t_us = now_us

        # --- Generate trajectory setpoint (PX4-like) from keyboard ---
        pos_stick_db = float(self.get_parameter("pos_stick_db").value)
        pgm_vel = float(self.get_parameter("pgm_vel").value)
        sgm_yaw = float(self.get_parameter("sgm_yaw").value)
        pos_mode = int(self.get_parameter("pos_mode").value)

        ux, uy, uz, uyaw = self.get_cmd_inputs(now_us)

        # deadband (like PX4: only if abs(input) > db)
        vx_sp = ux * pgm_vel if abs(ux) > pos_stick_db else 0.0
        vy_sp = uy * pgm_vel if abs(uy) > pos_stick_db else 0.0
        vz_sp = uz * pgm_vel if abs(uz) > pos_stick_db else 0.0

        # update position setpoint
        R_b2w = quat_to_R_wxyz(*self.q_wxyz)

        if pos_mode == 0:
            # world frame integration
            self.sp_pos_w = (
                self.sp_pos_w[0] + vx_sp * dt,
                self.sp_pos_w[1] + vy_sp * dt,
                self.sp_pos_w[2] + vz_sp * dt,
            )
        else:
            # body -> world (as PX4 rotates velocity setpoint into world)
            v_sp_b = (vx_sp, vy_sp, vz_sp)
            v_sp_w = mat3_mul_vec(R_b2w, v_sp_b)
            self.sp_pos_w = (
                self.sp_pos_w[0] + v_sp_w[0] * dt,
                self.sp_pos_w[1] + v_sp_w[1] * dt,
                self.sp_pos_w[2] + v_sp_w[2] * dt,
            )

        # --- PD position controller in world, then rotate to body for thrust setpoint ---
        kpx = float(self.get_parameter("gain_x_p").value)
        kpy = float(self.get_parameter("gain_y_p").value)
        kpz = float(self.get_parameter("gain_z_p").value)
        kdx = float(self.get_parameter("gain_x_d").value)
        kdy = float(self.get_parameter("gain_y_d").value)
        kdz = float(self.get_parameter("gain_z_d").value)

        ex = self.sp_pos_w[0] - self.pos_w[0]
        ey = self.sp_pos_w[1] - self.pos_w[1]
        ez = self.sp_pos_w[2] - self.pos_w[2]

        # desired "force-like" vector in world coordinates
        u_w = (
            kpx * ex - kdx * self.vel_v[0],
            kpy * ey - kdy * self.vel_v[1],
            kpz * ez - kdz * self.vel_v[2],
        )

        # rotate world -> body (inverse rotation)
        R_w2b = mat3_T(R_b2w)
        u_b = mat3_mul_vec(R_w2b, u_w)

        max_xy = float(self.get_parameter("max_thrust_xy").value)
        max_z = float(self.get_parameter("max_thrust_z").value)

        thrust_b = (
            clamp(u_b[0], -max_xy, max_xy),
            clamp(u_b[1], -max_xy, max_xy),
            clamp(u_b[2], -max_z, max_z),
        )

        # --- Yaw torque ---
        yaw = quat_to_yaw_wxyz(*self.q_wxyz)

        # Lock yaw setpoint if no yaw command, otherwise integrate it
        if abs(uyaw) <= pos_stick_db:
            self.sp_yaw = yaw
        else:
            self.sp_yaw = wrap_pi(self.sp_yaw + uyaw * dt * sgm_yaw)

        yaw_err = wrap_pi(self.sp_yaw - yaw)

        yaw_kp = float(self.get_parameter("yaw_kp").value)
        yaw_kd = float(self.get_parameter("yaw_kd").value)

        # VehicleOdometry.angular_velocity is usually body rates; use z component as yaw rate approx.
        wz = 0.0
        # If your message type includes it in VehicleOdometry, use it; otherwise keep 0.
        try:
            wz = float(getattr(self, "ang_vel_bz", 0.0))
        except Exception:
            wz = 0.0

        tau_z = yaw_kp * yaw_err - yaw_kd * wz
        max_tz = float(self.get_parameter("max_torque_z").value)
        tau_z = clamp(tau_z, -max_tz, max_tz)

        # --- Publish thrust/torque setpoints ---
        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.xyz = [float(thrust_b[0]), float(thrust_b[1]), float(thrust_b[2])]
        self.thrust_pub.publish(thr)

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.xyz = [0.0, 0.0, float(tau_z)]
        self.torque_pub.publish(tor)


def main():
    rclpy.init()
    node = PositionControlPID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()