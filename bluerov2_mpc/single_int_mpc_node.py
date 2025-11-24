#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass

import numpy as np
import casadi as cs

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

from bluerov2_mpc.models.single_int_pose_casadi import SingleIntegratorPoseCasadi


def wrap_angle_cs(a):
    return cs.atan2(cs.sin(a), cs.cos(a))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


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


@dataclass
class Tank:
    x_min: float = 0.0
    x_max: float = 9.0
    y_min: float = -2.5
    y_max: float = 2.5
    z_min: float = 0.0   # surface
    z_max: float = 3.0   # down positive
        

class SingleIntMPC:
    """
    Outer-loop MPC:

        x = [x,y,z, phi,theta,psi]
        u = [v_x, v_y, v_z, p, q, r] (body velocities/rates)

    Dynamics:
        x_{k+1} = x_k + dt * f(x_k, u_k)

    Cost: track pose, regularize velocities.
    """

    def __init__(self, dt: float, N: int):
        self.dt = dt
        self.N = N
        self.nx = 6
        self.nu = 6

        self.model = SingleIntegratorPoseCasadi(dt)

        # Opti
        self.opti = cs.Opti()
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        self.X0 = self.opti.parameter(self.nx)
        self.Xref = self.opti.parameter(self.nx)  # pose ref

        # Weighting
        self.Wp = np.diag([30.0, 30.0, 40.0])
        self.Wa = np.diag([3.0, 3.0, 8.0])
        self.Rv = np.diag([2.0, 2.0, 4.0])   # linear vel penalty
        self.Rw = np.diag([1.0, 1.0, 2.0])   # angular rate penalty

        # Input bounds
        v_max = 0.4   # m/s
        w_max = 0.6   # rad/s
        self.v_bound = v_max
        self.w_bound = w_max

        # Dynamics
        for k in range(self.N):
            xk = self.X[:, k]
            uk = self.U[:, k]
            xk1 = self.X[:, k+1]
            f_next = self.model.f_disc(xk, uk, self.dt)
            self.opti.subject_to(xk1 == f_next)

        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.X0)

        # Bounds on U
        self.opti.subject_to(
            self.opti.bounded(-self.v_bound, self.U[0:3, :], self.v_bound)
        )
        self.opti.subject_to(
            self.opti.bounded(-self.w_bound, self.U[3:6, :], self.w_bound)
        )

        # Tank position bounds (optional, can comment)
        tank = Tank()
        self.opti.subject_to(
            self.opti.bounded(tank.x_min, self.X[0, :], tank.x_max)
        )
        self.opti.subject_to(
            self.opti.bounded(tank.y_min, self.X[1, :], tank.y_max)
        )
        self.opti.subject_to(
            self.opti.bounded(tank.z_min, self.X[2, :], tank.z_max)
        )

        # Cost
        cost = 0
        for k in range(self.N + 1):
            xk = self.X[:, k]
            e_p = xk[0:3] - self.Xref[0:3]
            e_a = cs.vertcat(
                wrap_angle_cs(xk[3] - self.Xref[3]),
                wrap_angle_cs(xk[4] - self.Xref[4]),
                wrap_angle_cs(xk[5] - self.Xref[5]),
            )
            cost += cs.mtimes([e_p.T, self.Wp, e_p]) \
                    + cs.mtimes([e_a.T, self.Wa, e_a])
            if k < self.N:
                uk = self.U[:, k]
                v = uk[0:3]
                w = uk[3:6]
                cost += cs.mtimes([v.T, self.Rv, v]) \
                        + cs.mtimes([w.T, self.Rw, w])
        self.opti.minimize(cost)

        # Solver
        p_opts = {"print_time": False}
        s_opts = {"max_iter": 80, "tol": 1e-4}
        self.opti.solver("ipopt", p_opts, s_opts)

        self._u_last = np.zeros((self.nu,))

    def solve(self, x0: np.ndarray, xref: np.ndarray):
        try:
            self.opti.set_initial(self.X[:, 0], x0)
            for k in range(self.N):
                self.opti.set_initial(self.U[:, k], self._u_last)
        except Exception:
            pass

        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.Xref, xref)

        sol = self.opti.solve()
        Uopt = np.array(sol.value(self.U))
        Xpred = np.array(sol.value(self.X))
        u0 = Uopt[:, 0].copy()
        self._u_last = u0
        return u0, Xpred


class SingleIntMPCNode(Node):
    def __init__(self):
        super().__init__("single_int_mpc")

        # Params
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("horizon", 15)
        self.declare_parameter("dt", 0.10)
        self.declare_parameter("traj_mode", "waypoint")
        self.declare_parameter("ref_x", 4.5)
        self.declare_parameter("ref_y", 0.0)
        self.declare_parameter("ref_z", 1.2)
        self.declare_parameter("require_start_signal", True)
        self.declare_parameter("start_signal_topic", "/bluerov2_mpc/start_mpc")
        self.declare_parameter("odom_topic", "/mocap/itrl_rov_1/odom")
        self.declare_parameter("cmd_vel_topic", "/bluerov2_mpc/body_vel_cmd")

        rate = float(self.get_parameter("rate").value)
        N = int(self.get_parameter("horizon").value)
        dt = float(self.get_parameter("dt").value)

        self.traj_mode = self.get_parameter("traj_mode").get_parameter_value().string_value
        self.ref_xyz = np.array([
            float(self.get_parameter("ref_x").value),
            float(self.get_parameter("ref_y").value),
            float(self.get_parameter("ref_z").value),
        ])

        self.require_start_signal = bool(self.get_parameter("require_start_signal").value)
        self.start_signal_topic = self.get_parameter("start_signal_topic").get_parameter_value().string_value
        self.started = not self.require_start_signal

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value

        # MPC
        self.mpc = SingleIntMPC(dt=dt, N=N)

        # State: pose only
        self.x_pose = np.zeros(6, dtype=float)
        self.have_pose = False

        # IO
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.cb_odom, 10
        )
        self.pub_cmd_vel = self.create_publisher(
            Twist, cmd_vel_topic, 10
        )

        if self.require_start_signal:
            self.create_subscription(Bool, self.start_signal_topic, self.cb_start, 10)

        self.timer_ctrl = self.create_timer(1.0 / rate, self.cb_control)

        self._t0 = time.time()
        self.get_logger().info(
            f"SingleIntMPC ready. N={N}, dt={dt:.3f}, rate={rate}, require_start_signal={self.require_start_signal}"
        )

    # ---------- Callbacks ----------
    def cb_start(self, msg: Bool):
        if msg.data and not self.started:
            self.started = True
            self.get_logger().info("Received start signal -> MPC engaged.")

    def cb_odom(self, odom: Odometry):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        phi, theta, psi = quat_to_euler_xyz(q.w, q.x, q.y, q.z)
        self.x_pose[0:3] = np.array([p.x, p.y, p.z], dtype=float)
        self.x_pose[3:6] = np.array([phi, theta, psi], dtype=float)
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

        xref = self.gen_reference()
        x0 = self.x_pose.copy()

        try:
            u, _ = self.mpc.solve(x0, xref)
        except Exception as e:
            self.get_logger().warn(f"MPC solve failed: {e}")
            u = np.zeros(6, dtype=float)

        # Publish body-velocity setpoint (for inner PID)
        msg = Twist()
        msg.linear.x = float(u[0])
        msg.linear.y = float(u[1])
        msg.linear.z = float(u[2])
        msg.angular.x = float(u[3])
        msg.angular.y = float(u[4])
        msg.angular.z = float(u[5])
        self.pub_cmd_vel.publish(msg)


def main():
    rclpy.init()
    node = SingleIntMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()