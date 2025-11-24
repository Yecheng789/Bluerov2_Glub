#!/usr/bin/env python3
import math
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


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
class PIDGains:
    kp: float
    ki: float
    kd: float


class BodyVelPIDToWrench(Node):
    """
    Cascaded inner loop:

      input:
        - mocap odom  (to estimate body v, w)
        - desired body velocities from MPC (geometry_msgs/Twist)

      output:
        - VehicleThrustSetpoint (NED body forces)
        - VehicleTorqueSetpoint (body torques)

    PX4 must be in OffboardControlMode with thrust_and_torque=True.
    """

    def __init__(self):
        super().__init__("body_vel_pid_wrench")

        # Parameters
        self.declare_parameter("odom_topic", "/mocap/itrl_rov_1/odom")
        self.declare_parameter("cmd_vel_topic", "/bluerov2_mpc/body_vel_cmd")
        self.declare_parameter("thrust_topic", "/itrl_rov_1/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_topic", "/itrl_rov_1/fmu/in/vehicle_torque_setpoint")

        # PID gains (linear)
        self.declare_parameter("kp_lin", 40.0)
        self.declare_parameter("ki_lin", 5.0)
        self.declare_parameter("kd_lin", 0.0)
        # PID gains (angular)
        self.declare_parameter("kp_ang", 10.0)
        self.declare_parameter("ki_ang", 2.0)
        self.declare_parameter("kd_ang", 0.0)

        # Force / torque saturation
        self.declare_parameter("f_max", 40.0)   # N
        self.declare_parameter("tau_max", 8.0)  # Nm

        # Get parameter values
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        thrust_topic = self.get_parameter("thrust_topic").get_parameter_value().string_value
        torque_topic = self.get_parameter("torque_topic").get_parameter_value().string_value

        kp_lin = float(self.get_parameter("kp_lin").value)
        ki_lin = float(self.get_parameter("ki_lin").value)
        kd_lin = float(self.get_parameter("kd_lin").value)

        kp_ang = float(self.get_parameter("kp_ang").value)
        ki_ang = float(self.get_parameter("ki_ang").value)
        kd_ang = float(self.get_parameter("kd_ang").value)

        self.f_max = float(self.get_parameter("f_max").value)
        self.tau_max = float(self.get_parameter("tau_max").value)

        self.pid_lin = PIDGains(kp_lin, ki_lin, kd_lin)
        self.pid_ang = PIDGains(kp_ang, ki_ang, kd_ang)

        # Internal state
        self.prev_pose_time = None
        self.prev_p = None
        self.prev_psi = None
        self.v_body = np.zeros(3, dtype=float)
        self.w_body = np.zeros(3, dtype=float)

        self.v_des = np.zeros(3, dtype=float)
        self.w_des = np.zeros(3, dtype=float)

        self.int_lin = np.zeros(3, dtype=float)
        self.int_ang = np.zeros(3, dtype=float)
        self.prev_err_lin = np.zeros(3, dtype=float)
        self.prev_err_ang = np.zeros(3, dtype=float)

        # Subscribers
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.cb_odom, 10
        )
        self.sub_cmd = self.create_subscription(
            Twist, cmd_vel_topic, self.cb_cmd_vel, 10
        )

        # Publishers
        self.pub_thrust = self.create_publisher(
            VehicleThrustSetpoint, thrust_topic, 10
        )
        self.pub_torque = self.create_publisher(
            VehicleTorqueSetpoint, torque_topic, 10
        )

        # Control loop timer
        self.timer = self.create_timer(0.02, self.cb_control)  # 50 Hz

        self.get_logger().info(
            f"BodyVelPIDToWrench started. odom={odom_topic}, cmd_vel={cmd_vel_topic}"
        )

    # ---------- Callbacks ----------
    def cb_cmd_vel(self, msg: Twist):
        # Desired body velocities (assumed body frame)
        self.v_des = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z],
            dtype=float
        )
        self.w_des = np.array(
            [msg.angular.x, msg.angular.y, msg.angular.z],
            dtype=float
        )

    def cb_odom(self, odom: Odometry):
        p = np.array(
            [odom.pose.pose.position.x,
             odom.pose.pose.position.y,
             odom.pose.pose.position.z],
            dtype=float
        )
        q = odom.pose.pose.orientation
        # Quaternion to ZYX Euler
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        t0 = +2.0 * (qw*qx + qy*qz)
        t1 = +1.0 - 2.0 * (qx*qx + qy*qy)
        phi = math.atan2(t0, t1)
        t2 = +2.0 * (qw*qy - qz*qx)
        t2 = max(min(t2, 1.0), -1.0)
        theta = math.asin(t2)
        t3 = +2.0 * (qw*qz + qx*qy)
        t4 = +1.0 - 2.0 * (qy*qy + qz*qz)
        psi = math.atan2(t3, t4)

        t = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        if self.prev_pose_time is None:
            self.v_body[:] = 0.0
            self.w_body[:] = 0.0
        else:
            dt = max(1e-3, t - self.prev_pose_time)
            dp = (p - self.prev_p) / dt  # NED (world) linear vel
            dpsi = wrap_angle(psi - self.prev_psi) / dt
            Rb2n = euler_R_b2n(phi, theta, psi)
            self.v_body = Rb2n.T @ dp
            self.w_body = np.array([0.0, 0.0, dpsi], dtype=float)

        self.prev_pose_time = t
        self.prev_p = p
        self.prev_psi = psi

    def cb_control(self):
        if self.prev_pose_time is None:
            return

        dt = 0.02  # match timer

        # Errors
        e_lin = self.v_des - self.v_body
        e_ang = self.w_des - self.w_body

        # Integrators
        self.int_lin += e_lin * dt
        self.int_ang += e_ang * dt

        # Derivatives
        d_lin = (e_lin - self.prev_err_lin) / dt
        d_ang = (e_ang - self.prev_err_ang) / dt

        self.prev_err_lin = e_lin.copy()
        self.prev_err_ang = e_ang.copy()

        # PID output -> acceleration proxy -> directly as force/torque
        F = (self.pid_lin.kp * e_lin
             + self.pid_lin.ki * self.int_lin
             + self.pid_lin.kd * d_lin)
        Tau = (self.pid_ang.kp * e_ang
               + self.pid_ang.ki * self.int_ang
               + self.pid_ang.kd * d_ang)

        # Saturation
        F = np.clip(F, -self.f_max, self.f_max)
        Tau = np.clip(Tau, -self.tau_max, self.tau_max)

        # Publish thrust setpoint
        now_us = int(Clock().now().nanoseconds / 1000)
        tmsg = VehicleThrustSetpoint()
        tmsg.timestamp = now_us
        tmsg.xyz[0] = float(F[0])
        tmsg.xyz[1] = float(F[1])
        tmsg.xyz[2] = float(F[2])
        self.pub_thrust.publish(tmsg)

        # Publish torque setpoint
        qmsg = VehicleTorqueSetpoint()
        qmsg.timestamp = now_us
        qmsg.xyz[0] = float(Tau[0])
        qmsg.xyz[1] = float(Tau[1])
        qmsg.xyz[2] = float(Tau[2])
        self.pub_torque.publish(qmsg)


def main():
    rclpy.init()
    node = BodyVelPIDToWrench()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()