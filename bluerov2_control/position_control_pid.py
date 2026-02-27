#!/usr/bin/env python3
"""
PX4-like UUV position controller (ROS 2, Python) modeled after PX4 uuv_pos_control.

Architecture (matches PX4):
  cmd_vel (manual-like) -> generate_trajectory_setpoint() -> pose_controller_6dof()
  -> publish VehicleAttitudeSetpoint (q_d + thrust_body)

Important:
- PX4 uuv_pos_control publishes VEHICLE_ATTITUDE_SETPOINT (not thrust/torque directly).
- Therefore OffboardControlMode should have attitude=True (NOT body_rate=True).
  If you keep body_rate=True, PX4 will look for VehicleRatesSetpoint instead.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import VehicleOdometry, VehicleControlMode, VehicleAttitudeSetpoint
from geometry_msgs.msg import Twist


# ----------------- small helpers -----------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_norm_wxyz(q):
    qw, qx, qy, qz = q
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n > 1e-9:
        return (qw/n, qx/n, qy/n, qz/n)
    return (1.0, 0.0, 0.0, 0.0)


def euler_to_quat_wxyz(roll, pitch, yaw):
    # ZYX
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
    return quat_norm_wxyz((qw, qx, qy, qz))


def quat_to_euler_wxyz(qw, qx, qy, qz):
    # ZYX (roll, pitch, yaw)
    # roll
    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (qw*qy - qz*qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi/2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_to_R_wxyz(qw, qx, qy, qz):
    # body->world DCM
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


# ----------------- node -----------------

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
        self.declare_parameter("control_mode_topic", "/itrl_rov_1/fmu/out/vehicle_control_mode")
        self.declare_parameter("att_sp_topic", "/itrl_rov_1/fmu/in/vehicle_attitude_setpoint")
        self.declare_parameter("cmd_vel_topic", "/itrl_rov_1/cmd_vel")

        # PX4 uuv_pos_control params (names kept close)
        self.declare_parameter("UUV_GAIN_X_P", 1.0)
        self.declare_parameter("UUV_GAIN_Y_P", 1.0)
        self.declare_parameter("UUV_GAIN_Z_P", 1.0)
        self.declare_parameter("UUV_GAIN_X_D", 0.2)
        self.declare_parameter("UUV_GAIN_Y_D", 0.2)
        self.declare_parameter("UUV_GAIN_Z_D", 0.2)

        self.declare_parameter("UUV_STAB_MODE", 1)         # 1: roll/pitch = 0, 0: allow roll/pitch update (we keep 1 for keyboard)
        self.declare_parameter("UUV_POS_STICK_DB", 0.1)
        self.declare_parameter("UUV_PGM_VEL", 0.5)
        self.declare_parameter("UUV_POS_MODE", 1)          # 1: integrate position in BODY frame, 0: WORLD frame

        # needed by PX4 generate_trajectory_setpoint()
        self.declare_parameter("UUV_SGM_ROLL", 0.5)
        self.declare_parameter("UUV_SGM_PITCH", 0.5)
        self.declare_parameter("UUV_SGM_YAW", 0.5)
        self.declare_parameter("UUV_SP_MAX_AGE", 2.0)
        self.declare_parameter("UUV_THRUST_SAT", 0.3)

        # Command hygiene / conventions
        self.declare_parameter("cmd_timeout", 0.3)
        self.declare_parameter("yaw_sign", -1.0)
        self.declare_parameter("z_sign", -1.0)
        self.declare_parameter("sway_sign", 1.0)  # set -1 if your 'a/d' are swapped

        # Loop timing
        self.declare_parameter("loop_dt", 0.02)  # 50 Hz
        self.declare_parameter("dt_max", 0.02)   # PX4 clamps to 0.02 in these controllers

        odom_topic = self.get_parameter("odom_topic").value
        cm_topic = self.get_parameter("control_mode_topic").value
        att_sp_topic = self.get_parameter("att_sp_topic").value
        cmd_topic = self.get_parameter("cmd_vel_topic").value

        self.sub_odom = self.create_subscription(VehicleOdometry, odom_topic, self.on_odom, px4_qos)
        self.sub_cm = self.create_subscription(VehicleControlMode, cm_topic, self.on_control_mode, px4_qos)
        self.sub_cmd = self.create_subscription(Twist, cmd_topic, self.on_cmd_vel, 10)

        self.pub_att_sp = self.create_publisher(VehicleAttitudeSetpoint, att_sp_topic, px4_qos)

        # State from odom
        self.have_odom = False
        self.pos_w = (0.0, 0.0, 0.0)
        self.vel_w = (0.0, 0.0, 0.0)
        self.q_wxyz = (1.0, 0.0, 0.0, 0.0)

        # Gate
        self.enabled = False

        # Manual-like command (mapped from cmd_vel)
        self.cmd_pitch = 0.0     # maps to PX4 manual_control_setpoint.pitch (X)
        self.cmd_roll = 0.0      # maps to PX4 manual_control_setpoint.roll  (Y)
        self.cmd_throttle = 0.0  # maps to PX4 manual_control_setpoint.throttle (Z, note PX4 uses -throttle for vz)
        self.cmd_yaw = 0.0       # yaw input
        self.cmd_last_us = 0

        # PX4 trajectory_setpoint6dof-like internal state
        self.traj_pos = None         # (x,y,z) in world frame
        self.traj_q = None           # (w,x,y,z) desired orientation
        self.traj_ts_us = 0

        self.last_us = None

        self.timer = self.create_timer(float(self.get_parameter("loop_dt").value), self.tick)

    # -------- callbacks --------

    def on_control_mode(self, msg: VehicleControlMode):
        # Similar to "armed gate" behavior you already used
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)

        if gate and not self.enabled:
            self.enabled = True
            # PX4 resets trajectory setpoint on entry conditions via validity checks; do it explicitly here
            if self.have_odom:
                self.reset_trajectory_setpoint()
            self.get_logger().info("Position controller enabled (armed + offboard).")

        elif (not gate) and self.enabled:
            self.enabled = False
            self.get_logger().info("Position controller disabled (not armed/offboard).")
            self.publish_attitude_setpoint(thrust_b=(0.0, 0.0, 0.0), q_sp=self.q_wxyz)

    def on_odom(self, msg: VehicleOdometry):
        self.pos_w = (float(msg.position[0]), float(msg.position[1]), float(msg.position[2]))
        self.vel_w = (float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2]))
        self.q_wxyz = quat_norm_wxyz((float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])))
        self.have_odom = True

        if self.traj_pos is None:
            self.reset_trajectory_setpoint()

        if self.last_us is None:
            self.last_us = int(self.get_clock().now().nanoseconds / 1000)

    def on_cmd_vel(self, msg: Twist):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        yaw_sign = float(self.get_parameter("yaw_sign").value)
        z_sign = float(self.get_parameter("z_sign").value)
        sway_sign = float(self.get_parameter("sway_sign").value)

        # Map to PX4 manual conventions used in uuv_pos_control::generate_trajectory_setpoint():
        # - pitch is X (forward)
        # - roll  is Y (sway)
        # - throttle is Z (vertical)
        self.cmd_pitch = clamp(float(msg.linear.x), -1.0, 1.0)
        self.cmd_roll = clamp(sway_sign * float(msg.linear.y), -1.0, 1.0)
        self.cmd_throttle = clamp(z_sign * float(msg.linear.z), -1.0, 1.0)
        self.cmd_yaw = clamp(yaw_sign * float(msg.angular.z), -1.0, 1.0)

        self.cmd_last_us = now_us

    # -------- internal (PX4-like) --------

    def cmd_valid(self, now_us: int) -> bool:
        timeout_us = int(float(self.get_parameter("cmd_timeout").value) * 1e6)
        return (self.cmd_last_us != 0) and ((now_us - self.cmd_last_us) <= timeout_us)

    def reset_trajectory_setpoint(self):
        # Equivalent to PX4 reset_trajectory_setpoint(vlocal_pos)
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pos = self.pos_w
        self.traj_q = self.q_wxyz
        self.traj_ts_us = now_us

    def check_setpoint_validity(self, now_us: int):
        max_age_s = float(self.get_parameter("UUV_SP_MAX_AGE").value)
        age_s = (now_us - self.traj_ts_us) * 1e-6
        if age_s < 0.0 or age_s > max_age_s:
            self.reset_trajectory_setpoint()
            return

        # finite checks
        if self.traj_pos is None or self.traj_q is None:
            self.reset_trajectory_setpoint()
            return

        if not all(math.isfinite(v) for v in self.traj_pos):
            self.reset_trajectory_setpoint()
            return

        if not all(math.isfinite(v) for v in self.traj_q):
            self.reset_trajectory_setpoint()
            return

    def generate_trajectory_setpoint(self, dt: float):
        # Mirrors PX4 generate_trajectory_setpoint() but using cmd_vel inputs (manual-like)

        pos_stick_db = float(self.get_parameter("UUV_POS_STICK_DB").value)
        pgm_vel = float(self.get_parameter("UUV_PGM_VEL").value)
        pos_mode = int(self.get_parameter("UUV_POS_MODE").value)
        stab_mode = int(self.get_parameter("UUV_STAB_MODE").value)

        sgm_roll = float(self.get_parameter("UUV_SGM_ROLL").value)
        sgm_pitch = float(self.get_parameter("UUV_SGM_PITCH").value)
        sgm_yaw = float(self.get_parameter("UUV_SGM_YAW").value)

        # current desired euler from trajectory quaternion
        roll, pitch, yaw = quat_to_euler_wxyz(*self.traj_q)

        # roll/pitch setpoints
        roll_sp = roll
        pitch_sp = pitch
        if stab_mode == 1:
            roll_sp = 0.0
            pitch_sp = 0.0
        else:
            # No D-pad/buttons in cmd_vel; keep as-is (or expose separate inputs if needed)
            # This preserves PX4 structure without inventing new behavior.
            pass

        # yaw integration (PX4: yaw_setpoint = yaw + manual_yaw * dt * sgm_yaw)
        yaw_sp = wrap_pi(yaw + self.cmd_yaw * dt * sgm_yaw)

        # Translational velocity setpoint from manual sticks with deadband
        vx_sp = self.cmd_pitch * pgm_vel if abs(self.cmd_pitch) > pos_stick_db else 0.0
        vy_sp = self.cmd_roll * pgm_vel if abs(self.cmd_roll) > pos_stick_db else 0.0
        vz_sp = (-self.cmd_throttle) * pgm_vel if abs(self.cmd_throttle) > pos_stick_db else 0.0  # PX4 uses -throttle

        # Rotate velocity setpoint body->world using current vehicle attitude (PX4 uses vehicle_attitude.q)
        R_b2w = quat_to_R_wxyz(*self.q_wxyz)
        v_sp_b = (vx_sp, vy_sp, vz_sp)
        v_sp_w = mat3_mul_vec(R_b2w, v_sp_b)

        # Update trajectory quaternion
        self.traj_q = euler_to_quat_wxyz(roll_sp, pitch_sp, yaw_sp)

        # Update trajectory position (world or body mode)
        x, y, z = self.traj_pos
        if pos_mode == 0:
            # world integration
            self.traj_pos = (x + vx_sp * dt, y + vy_sp * dt, z + vz_sp * dt)
        else:
            # body integration -> integrate rotated velocity in world
            self.traj_pos = (x + v_sp_w[0] * dt, y + v_sp_w[1] * dt, z + v_sp_w[2] * dt)

        now_us = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_ts_us = now_us

    def pose_controller_6dof(self, altitude_only: bool):
        # Mirrors PX4 pose_controller_6dof(), using odom position/velocity as vlocal_pos
        kpx = float(self.get_parameter("UUV_GAIN_X_P").value)
        kpy = float(self.get_parameter("UUV_GAIN_Y_P").value)
        kpz = float(self.get_parameter("UUV_GAIN_Z_P").value)
        kdx = float(self.get_parameter("UUV_GAIN_X_D").value)
        kdy = float(self.get_parameter("UUV_GAIN_Y_D").value)
        kdz = float(self.get_parameter("UUV_GAIN_Z_D").value)
        thrust_sat = float(self.get_parameter("UUV_THRUST_SAT").value)

        x_d, y_d, z_d = self.traj_pos
        x, y, z = self.pos_w
        vx, vy, vz = self.vel_w

        # P-D, assuming target 0 velocity (same as PX4 code)
        u_w = (
            kpx * (x_d - x) - kdx * vx,
            kpy * (y_d - y) - kdy * vy,
            kpz * (z_d - z) - kdz * vz,
        )

        if altitude_only:
            u_w = (0.0, 0.0, u_w[2])

        # Rotate global->body using inverse rotation of CURRENT attitude (PX4: rotateVectorInverse)
        R_b2w = quat_to_R_wxyz(*self.q_wxyz)
        R_w2b = mat3_T(R_b2w)
        u_b = mat3_mul_vec(R_w2b, u_w)

        u_b = (
            clamp(u_b[0], -thrust_sat, thrust_sat),
            clamp(u_b[1], -thrust_sat, thrust_sat),
            clamp(u_b[2], -thrust_sat, thrust_sat),
        )

        return u_b  # thrust_body

    def publish_attitude_setpoint(self, thrust_b, q_sp):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        msg = VehicleAttitudeSetpoint()
        msg.timestamp = now_us

        qw, qx, qy, qz = q_sp
        msg.q_d = [float(qw), float(qx), float(qy), float(qz)]
        msg.thrust_body = [float(thrust_b[0]), float(thrust_b[1]), float(thrust_b[2])]

        self.pub_att_sp.publish(msg)

    # -------- main loop --------

    def tick(self):
        if not self.have_odom or self.traj_pos is None or self.traj_q is None:
            return

        now_us = int(self.get_clock().now().nanoseconds / 1000)
        if self.last_us is None:
            self.last_us = now_us
            return

        dt = (now_us - self.last_us) * 1e-6
        dt = clamp(dt, 0.0002, float(self.get_parameter("dt_max").value))
        self.last_us = now_us

        if not self.enabled:
            # Track current state while disabled (prevents jumps when re-enabled)
            self.reset_trajectory_setpoint()
            return

        # If command stream times out, hold position/yaw where we are (PX4-like: setpoint validity/reset handles this)
        if not self.cmd_valid(now_us):
            # do not integrate trajectory; just keep it valid and control to it
            self.check_setpoint_validity(now_us)
        else:
            self.check_setpoint_validity(now_us)
            self.generate_trajectory_setpoint(dt)

        # altitude_only flag: you can wire this to control_mode flags if you actually use them;
        # with offboard+cmd_vel it’s typically full position.
        altitude_only = False

        thrust_b = self.pose_controller_6dof(altitude_only=altitude_only)

        # publish attitude setpoint using trajectory quaternion + computed thrust_body
        self.publish_attitude_setpoint(thrust_b=thrust_b, q_sp=self.traj_q)


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