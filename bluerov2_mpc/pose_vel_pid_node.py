#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_euler_xyz(qw, qx, qy, qz):
    # ZYX -> (phi, theta, psi)
    t0 = +2.0 * (qw*qx + qy*qz)
    t1 = +1.0 - 2.0 * (qx*qx + qy*qy)
    phi = math.atan2(t0, t1)
    t2 = +2.0 * (qw*qy - qz*qx)
    t2 = max(min(t2, 1.0), -1.0)
    theta = math.asin(t2)
    t3 = +2.0 * (qw*qz + qx*qy)
    t4 = +1.0 - 2.0 * (qy*qy + qz*qz)
    psi = math.atan2(t3, t4)
    return phi, theta, psi


def euler_R_b2n(phi, theta, psi):
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    Rz = np.array([[cps, -sps, 0.0],
                   [sps,  cps, 0.0],
                   [0.0,  0.0,  1.0]])
    Ry = np.array([[ cth, 0.0, sth],
                   [ 0.0, 1.0, 0.0],
                   [-sth, 0.0, cth]])
    Rx = np.array([[1.0, 0.0,  0.0],
                   [0.0, cph, -sph],
                   [0.0, sph,  cph]])
    return Rz @ Ry @ Rx


@dataclass
class PID1D:
    kp: float
    ki: float
    kd: float


class PoseToBodyVelPID(Node):
    """
    Outer loop:

      Inputs:
        - mocap odometry (NED)
        - reference trajectory (internally generated)
        - optional Bool start signal

      Outputs:
        - desired body velocities/rates as geometry_msgs/Twist on /bluerov2_mpc/body_vel_cmd

    Inner loop:
        body_vel_pid_wrench node (from previous code) consumes /body_vel_cmd
        and publishes VehicleThrustSetpoint + VehicleTorqueSetpoint.
    """

    def __init__(self):
        super().__init__("pose_vel_pid")

        # Parameters
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("odom_topic", "/mocap/itrl_rov_1/odom")
        self.declare_parameter("cmd_vel_topic", "/bluerov2_mpc/body_vel_cmd")

        self.declare_parameter("traj_mode", "waypoint")  # "waypoint" | "lissajous"
        self.declare_parameter("ref_x", 4.5)
        self.declare_parameter("ref_y", 0.0)
        self.declare_parameter("ref_z", 1.2)

        # Start gating
        self.declare_parameter("require_start_signal", True)
        self.declare_parameter("start_signal_topic", "/bluerov2_mpc/start_mpc")

        # Position PID (xyz in world frame)
        self.declare_parameter("kp_pos", 0.6)
        self.declare_parameter("ki_pos", 0.05)
        self.declare_parameter("kd_pos", 0.0)

        # Yaw PID (only psi for now)
        self.declare_parameter("kp_yaw", 1.5)
        self.declare_parameter("ki_yaw", 0.05)
        self.declare_parameter("kd_yaw", 0.0)

        # Velocity saturation
        self.declare_parameter("v_max_xy", 0.3)   # m/s
        self.declare_parameter("v_max_z", 0.3)    # m/s
        self.declare_parameter("w_max_yaw", 0.5)  # rad/s

        rate = float(self.get_parameter("rate").value)
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value

        self.traj_mode = self.get_parameter("traj_mode").get_parameter_value().string_value
        self.ref_xyz = np.array([
            float(self.get_parameter("ref_x").value),
            float(self.get_parameter("ref_y").value),
            float(self.get_parameter("ref_z").value),
        ], dtype=float)

        self.require_start_signal = bool(self.get_parameter("require_start_signal").value)
        self.start_signal_topic = self.get_parameter("start_signal_topic").get_parameter_value().string_value
        self.started = not self.require_start_signal

        kp_pos = float(self.get_parameter("kp_pos").value)
        ki_pos = float(self.get_parameter("ki_pos").value)
        kd_pos = float(self.get_parameter("kd_pos").value)

        kp_yaw = float(self.get_parameter("kp_yaw").value)
        ki_yaw = float(self.get_parameter("ki_yaw").value)
        kd_yaw = float(self.get_parameter("kd_yaw").value)

        self.v_max_xy = float(self.get_parameter("v_max_xy").value)
        self.v_max_z = float(self.get_parameter("v_max_z").value)
        self.w_max_yaw = float(self.get_parameter("w_max_yaw").value)

        # PIDs (world xyz + yaw)
        self.pid_pos = PID1D(kp_pos, ki_pos, kd_pos)
        self.pid_yaw = PID1D(kp_yaw, ki_yaw, kd_yaw)

        # Internal state
        self.pose = np.zeros(6, dtype=float)  # [x,y,z,phi,theta,psi]
        self.have_pose = False

        self.int_pos = np.zeros(3, dtype=float)
        self.prev_err_pos = np.zeros(3, dtype=float)

        self.int_yaw = 0.0
        self.prev_err_yaw = 0.0

        self.prev_time = None
        self._t0 = time.time()

        # Subscriptions and publishers
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.cb_odom, 10
        )
        self.pub_cmd_vel = self.create_publisher(
            Twist, cmd_vel_topic, 10
        )
        if self.require_start_signal:
            self.create_subscription(Bool, self.start_signal_topic, self.cb_start, 10)

        self.timer = self.create_timer(1.0 / rate, self.cb_control)

        self.get_logger().info(
            f"PoseToBodyVelPID started. rate={rate}, traj_mode={self.traj_mode}, "
            f"require_start_signal={self.require_start_signal}"
        )

    # --------- Callbacks ----------
    def cb_start(self, msg: Bool):
        if msg.data and not self.started:
            self.started = True
            self.get_logger().info("Received start signal -> cascaded PID engaged.")

    def cb_odom(self, odom: Odometry):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        phi, theta, psi = quat_to_euler_xyz(q.w, q.x, q.y, q.z)
        self.pose[0:3] = np.array([p.x, p.y, p.z], dtype=float)
        self.pose[3:6] = np.array([phi, theta, psi], dtype=float)
        self.have_pose = True

    def gen_reference(self):
        if self.traj_mode == "lissajous":
            tt = time.time() - self._t0
            xr = 4.5 + 2.0 * math.sin(0.20 * tt)
            yr = 0.0 + 1.2 * math.sin(0.31 * tt + 0.7)
            zr = 1.2 + 0.6 * math.sin(0.27 * tt + 1.1)
            phi_r, theta_r, psi_r = 0.0, 0.0, 0.0
        else:
            xr, yr, zr = self.ref_xyz
            phi_r, theta_r, psi_r = 0.0, 0.0, 0.0
        return np.array([xr, yr, zr, phi_r, theta_r, psi_r], dtype=float)

    def cb_control(self):
        if not self.started or not self.have_pose:
            return

        t_now = time.time()
        if self.prev_time is None:
            self.prev_time = t_now
            return
        dt = max(1e-3, min(0.5, t_now - self.prev_time))
        self.prev_time = t_now

        x, y, z, phi, theta, psi = self.pose
        ref = self.gen_reference()
        xr, yr, zr, phi_r, theta_r, psi_r = ref

        # Position error in world (NED)
        e_pos = np.array([xr - x, yr - y, zr - z], dtype=float)

        # Yaw error
        e_yaw = wrap_angle(psi_r - psi)

        # Integrators
        self.int_pos += e_pos * dt
        self.int_yaw += e_yaw * dt

        # Derivatives
        d_pos = (e_pos - self.prev_err_pos) / dt
        d_yaw = (e_yaw - self.prev_err_yaw) / dt

        self.prev_err_pos = e_pos.copy()
        self.prev_err_yaw = float(e_yaw)

        # Position PID -> desired world-frame velocity
        v_des_n = (
            self.pid_pos.kp * e_pos
            + self.pid_pos.ki * self.int_pos
            + self.pid_pos.kd * d_pos
        )

        # Saturate world velocities
        v_des_n[0] = float(np.clip(v_des_n[0], -self.v_max_xy, self.v_max_xy))
        v_des_n[1] = float(np.clip(v_des_n[1], -self.v_max_xy, self.v_max_xy))
        v_des_n[2] = float(np.clip(v_des_n[2], -self.v_max_z,  self.v_max_z))

        # Yaw PID -> desired yaw rate (body frame z)
        w_des_yaw = (
            self.pid_yaw.kp * e_yaw
            + self.pid_yaw.ki * self.int_yaw
            + self.pid_yaw.kd * d_yaw
        )
        w_des_yaw = float(np.clip(w_des_yaw, -self.w_max_yaw, self.w_max_yaw))

        # Convert desired linear velocity to body frame
        Rb2n = euler_R_b2n(phi, theta, psi)
        v_des_b = Rb2n.T @ v_des_n  # body frame

        # For now only yaw rate is used; roll/pitch rates 0
        w_des_b = np.array([0.0, 0.0, w_des_yaw], dtype=float)

        # Publish Twist command for inner PID
        msg = Twist()
        msg.linear.x = float(v_des_b[0])
        msg.linear.y = float(v_des_b[1])
        msg.linear.z = float(v_des_b[2])
        msg.angular.x = float(w_des_b[0])
        msg.angular.y = float(w_des_b[1])
        msg.angular.z = float(w_des_b[2])
        self.pub_cmd_vel.publish(msg)


def main():
    rclpy.init()
    node = PoseToBodyVelPID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()