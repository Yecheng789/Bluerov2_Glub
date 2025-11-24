#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import casadi as cs

import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from importlib.resources import files as pkg_files

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from px4_msgs.msg import OffboardControlMode, ActuatorMotors

# ---- import your three models ----
from bluerov2_mpc.models.fossen_casadi import FossenCasadi
from bluerov2_mpc.models.di_casadi import DoubleIntegratorCasadi
from bluerov2_mpc.models.koopman_casadi import KoopmanEDMDcCasadi


# -------- path resolvers for weights (share -> pkgdata -> user) --------
def _path_in_share(filename: str) -> str:
    try:
        share = Path(get_package_share_directory('bluerov2_mpc'))
        p = share / 'models' / 'weights' / filename
        if p.is_file():
            return str(p)
    except Exception:
        pass
    return ''

def _path_in_pkgdata(filename: str) -> str:
    try:
        p = pkg_files('bluerov2_mpc') / 'models' / 'weights' / filename
        p = Path(str(p))
        if p.is_file():
            return str(p)
    except Exception:
        pass
    return ''

def resolve_weight_path(param_val: str, filename: str) -> str:
    if param_val and os.path.isfile(param_val):
        return param_val
    p = _path_in_share(filename)
    if p:
        return p
    p = _path_in_pkgdata(filename)
    if p:
        return p
    return param_val


def wrap_angle(a):
    # CasADi-safe angle wrap to (-pi, pi]
    if isinstance(a, (cs.SX, cs.MX)):
        return cs.atan2(cs.sin(a), cs.cos(a))
    else:
        return (a + math.pi) % (2*math.pi) - math.pi


def quat_to_euler_xyz(qw, qx, qy, qz):
    # ZYX -> (phi, theta, psi)
    t0 = +2.0 * (qw*qx + qy*qz)
    t1 = +1.0 - 2.0 * (qx*qx + qy*qy)
    phi = math.atan2(t0, t1)
    t2 = +2.0 * (qw*qy - qz*qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
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
class Tank:
    x_min: float = 0.0
    x_max: float = 9.0
    y_min: float = -2.5
    y_max: float = 2.5
    z_min: float = 0.0   # surface
    z_max: float = 3.0   # down is positive in your mocap


class SimpleMPC:
    """Tiny NMPC: minimize position/orientation/vel error + input regularization."""
    def __init__(self, model_name: str, dt: float, N: int,
                 di_weights_path: Optional[str] = None,
                 koop_weights_path: Optional[str] = None):
        self.model_name = model_name
        self.dt = dt
        self.N  = N

        # Build step function f(x,u) -> x_next
        if model_name == 'fossen':
            self.model = FossenCasadi()
            def step(x,u): return self.model.f_disc_rk4(x,u,dt)
            self.step = step
            self.nu = 8
        elif model_name == 'di':
            assert di_weights_path is not None and os.path.isfile(di_weights_path), \
                f"Double Integrator weights not found: {di_weights_path}"
            self.model = DoubleIntegratorCasadi(di_weights_path)
            def step(x,u): return self.model.f_disc(x,u,dt)
            self.step = step
            self.nu = 8
        elif model_name == 'koopman':
            assert koop_weights_path is not None and os.path.isfile(koop_weights_path), \
                f"Koopman weights not found: {koop_weights_path}"
            self.model = KoopmanEDMDcCasadi(koop_weights_path)
            def step(x,u): return self.model.f_disc(x,u)  # one-step discrete
            self.step = step
            self.nu = 8
        else:
            raise ValueError("model_name must be one of {'fossen','di','koopman'}")

        self.nx = 12

        # Build Opti
        self.opti = cs.Opti()
        self.X = self.opti.variable(self.nx, self.N+1)
        self.U = self.opti.variable(self.nu, self.N)
        self.X0 = self.opti.parameter(self.nx)
        self.Xref = self.opti.parameter(6)   # [x,y,z, phi,theta,psi]

        # Cost weights
        self.Wp = np.diag([30.0, 30.0, 40.0])
        self.Wa = np.diag([2.0, 2.0, 8.0])
        self.Wv = 0.5*np.eye(3)
        self.Ww = 0.2*np.eye(3)
        self.R  = 5e-2*np.eye(self.nu)

        # Dynamics constraints
        for k in range(self.N):
            xk = self.X[:,k]
            uk = self.U[:,k]
            xk1 = self.X[:,k+1]
            f_next = self.step(xk, uk)
            self.opti.subject_to(xk1 == f_next)

        # Initial condition
        self.opti.subject_to(self.X[:,0] == self.X0)

        # Input bounds (thrusters) in [-1,1]
        self.opti.subject_to(self.opti.bounded(-1.0, self.U, 1.0))

        # Tank bounds on position (hard, but easy to relax later)
        # tank = Tank()
        # self.opti.subject_to(self.opti.bounded(tank.x_min, self.X[0,:], tank.x_max))
        # self.opti.subject_to(self.opti.bounded(tank.y_min, self.X[1,:], tank.y_max))
        # self.opti.subject_to(self.opti.bounded(tank.z_min, self.X[2,:], tank.z_max))

        # Build cost
        cost = 0
        for k in range(self.N+1):
            xk = self.X[:,k]
            e_p = xk[0:3] - self.Xref[0:3]
            e_a = cs.vertcat(
                wrap_angle(xk[3]-self.Xref[3]),
                wrap_angle(xk[4]-self.Xref[4]),
                wrap_angle(xk[5]-self.Xref[5]),
            )
            e_v = xk[6:9]
            e_w = xk[9:12]
            cost += cs.mtimes([e_p.T, self.Wp, e_p]) \
                    + cs.mtimes([e_a.T, self.Wa, e_a]) \
                    + cs.mtimes([e_v.T, self.Wv, e_v]) \
                    + cs.mtimes([e_w.T, self.Ww, e_w])
            if k < self.N:
                uk = self.U[:,k]
                cost += cs.mtimes([uk.T, self.R, uk])
        self.opti.minimize(cost)

        # Solver
        p_opts = {"print_time": False}
        s_opts = {"max_iter": 80, "tol": 1e-4}
        self.opti.solver("ipopt", p_opts, s_opts)

        # Warm start
        self._u_last = np.zeros((self.nu,))

    def solve(self, x0: np.ndarray, xref: np.ndarray):
        try:
            self.opti.set_initial(self.X[:,0], x0)
            for k in range(self.N):
                self.opti.set_initial(self.U[:,k], self._u_last)
        except Exception:
            pass

        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.Xref, xref)

        sol = self.opti.solve()
        Uopt = np.array(sol.value(self.U))
        Xpred = np.array(sol.value(self.X))
        u0 = Uopt[:,0].copy()
        self._u_last = u0
        return u0, Xpred


class BlueROV2MPCNode(Node):
    def __init__(self):
        super().__init__("bluerov2_mpc")

        # ---- Parameters ----
        self.declare_parameter("model", "koopman")          # 'fossen' | 'di' | 'koopman'
        self.declare_parameter("di_weights", "")            # resolve internally
        self.declare_parameter("koopman_weights", "")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("horizon", 15)
        self.declare_parameter("dt", 0.10)
        self.declare_parameter("traj_mode", "waypoint")     # 'waypoint' | 'lissajous'
        self.declare_parameter("ref_x", 4.5)
        self.declare_parameter("ref_y", 0.0)
        self.declare_parameter("ref_z", 1.2)                # positive down in your setup

        # NEW: wait for PID “ready” signal before starting MPC
        self.declare_parameter("require_start_signal", True)
        self.declare_parameter("start_signal_topic", "/bluerov2_mpc/start_mpc")

        model = self.get_parameter("model").get_parameter_value().string_value
        di_w_param  = self.get_parameter("di_weights").get_parameter_value().string_value
        kk_w_param  = self.get_parameter("koopman_weights").get_parameter_value().string_value
        rate  = float(self.get_parameter("rate").value)
        N     = int(self.get_parameter("horizon").value)
        dt    = float(self.get_parameter("dt").value)

        self.traj_mode = self.get_parameter("traj_mode").get_parameter_value().string_value
        self.ref_xyz = np.array([
            float(self.get_parameter("ref_x").value),
            float(self.get_parameter("ref_y").value),
            float(self.get_parameter("ref_z").value),
        ])

        self.require_start_signal = bool(self.get_parameter("require_start_signal").value)
        self.start_signal_topic = self.get_parameter("start_signal_topic").get_parameter_value().string_value
        self.started = not self.require_start_signal  # if not required, start immediately

        # ---- Resolve weight paths ----
        di_w = resolve_weight_path(di_w_param, 'double_integrator_weights.npz')
        kk_w = resolve_weight_path(kk_w_param, 'koopman_edmdc_weights.npz')
        self.get_logger().info(f"DI weights resolved: {di_w or '(not used)'}")
        self.get_logger().info(f"Koopman weights resolved: {kk_w or '(not used)'}")

        # ---- MPC ----
        self.mpc = SimpleMPC(model, dt, N, di_weights_path=di_w, koop_weights_path=kk_w)

        # ---- State ----
        self.x = np.zeros(12, dtype=float)
        self.prev_pose_time = None
        self.prev_p = None
        self.prev_psi = None

        # ---- IO ----
        self.sub_odom = self.create_subscription(
            Odometry, "/mocap/itrl_rov_1/odom", self.cb_odom, 10
        )
        # Heartbeat is external; MPC won’t send Offboard
        self.pub_motors = self.create_publisher(
            ActuatorMotors, "/itrl_rov_1/fmu/in/actuator_motors", 10
        )

        # Start signal subscription
        if self.require_start_signal:
            self.create_subscription(Bool, self.start_signal_topic, self.cb_start, 10)

        # Timer
        self.timer_ctrl = self.create_timer(1.0/rate, self.cb_control)

        self.get_logger().info(f"MPC ready. model={model}, N={N}, dt={dt:.3f}s, require_start_signal={self.require_start_signal}")

        # reference phase
        self._t0 = time.time()

    # ---------- Callbacks ----------
    def cb_start(self, msg: Bool):
        if msg.data and not self.started:
            self.started = True
            self.get_logger().info("Received start signal -> MPC engaged.")

    def cb_odom(self, odom: Odometry):
        p = np.array([odom.pose.pose.position.x,
                      odom.pose.pose.position.y,
                      odom.pose.pose.position.z], dtype=float)
        q = odom.pose.pose.orientation
        phi, theta, psi = quat_to_euler_xyz(q.w, q.x, q.y, q.z)

        # Estimate body velocities from finite differences
        t = odom.header.stamp.sec + odom.header.stamp.nanosec*1e-9
        if self.prev_pose_time is None:
            v_body = np.zeros(3)
            w_body = np.zeros(3)
        else:
            dt = max(1e-3, t - self.prev_pose_time)
            dp = (p - self.prev_p) / dt  # NED linear vel
            dpsi = wrap_angle(psi - self.prev_psi) / dt
            Rb2n = euler_R_b2n(phi, theta, psi)
            v_body = Rb2n.T @ dp
            w_body = np.array([0.0, 0.0, dpsi], dtype=float)

        self.prev_pose_time = t
        self.prev_p = p.copy()
        self.prev_psi = psi

        self.x[0:3] = p
        self.x[3:6] = np.array([phi, theta, psi])
        self.x[6:9] = v_body
        self.x[9:12] = w_body

    def gen_reference(self):
        if self.traj_mode == "lissajous":
            tt = time.time() - self._t0
            xr = 4.5 + 2.0*math.sin(0.20*tt)
            yr = 0.0 + 1.2*math.sin(0.31*tt + 0.7)
            zr = 1.2 + 0.6*math.sin(0.27*tt + 1.1)  # positive down
            phi_r, theta_r, psi_r = 0.0, 0.0, 0.0
        else:
            xr, yr, zr = self.ref_xyz
            phi_r, theta_r, psi_r = 0.0, 0.0, 0.0
        return np.array([xr, yr, zr, phi_r, theta_r, psi_r], dtype=float)

    def cb_control(self):
        # Wait until PID signaled “HOLD” (and stop publishing anything while waiting)
        if not self.started:
            return

        xref = self.gen_reference()
        x0 = self.x.copy()

        try:
            u, _ = self.mpc.solve(x0, xref)
        except Exception as e:
            self.get_logger().warn(f"MPC solve failed: {e}")
            u = np.zeros(8)

        # Map [-1,1] -> [0,1] for PX4 motors
        motors = (np.clip(u, -1.0, 1.0) + 1.0) * 0.5
        msg = ActuatorMotors()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.timestamp_sample = 0
        msg.reversible_flags = 0
        arr = np.zeros(12, dtype=np.float32)
        arr[:8] = motors.astype(np.float32)
        msg.control = arr.tolist()
        self.pub_motors.publish(msg)


def main():
    rclpy.init()
    node = BlueROV2MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()