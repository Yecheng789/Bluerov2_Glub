"""
MPCHoldPosition (double integrator dynamics):

Uses tank center in PX4-local NED coordinates as goal by default:
     world ENU center from SDF water box:
       (E, N, U) = (-2.175, -1.15, -95.7)
     converted to PX4 NED:
       (N, E, D) = (-1.15, -2.175, 95.7)

Explicit physical->normalized scaling:
     - MPC decides in Newtons (Fx,Fy,Fz)
     - we publish normalized thrust setpoints = F / F_max_N
   and clamp to [-thrust_sat, +thrust_sat].
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import (
    VehicleOdometry,
    VehicleControlMode,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
)

import casadi as ca


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_norm_wxyz(q):
    qw, qx, qy, qz = q
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 1e-12:
        return (qw / n, qx / n, qy / n, qz / n)
    return (1.0, 0.0, 0.0, 0.0)


def quat_to_yaw_wxyz(q):
    qw, qx, qy, qz = q
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class MPCHoldPosition(Node):
    def __init__(self):
        super().__init__("mpc_hold_position")

        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------------- Topics ----------------
        self.declare_parameter("odom_topic", "/itrl_rov_1/fmu/out/vehicle_odometry")
        self.declare_parameter("control_mode_topic", "/itrl_rov_1/fmu/out/vehicle_control_mode")
        self.declare_parameter("thrust_sp_topic", "/itrl_rov_1/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_sp_topic", "/itrl_rov_1/fmu/in/vehicle_torque_setpoint")

        # ---------------- Goal (in PX4-local NED) ----------------
        # Default = tank center converted from world ENU water box center:
        # (E,N,U)=(-2.175,-1.15,-95.7) -> (N,E,D)=(-1.15,-2.175,95.7)
        self.declare_parameter("goal_x", -1.15)    # N
        self.declare_parameter("goal_y", -2.175)   # E
        self.declare_parameter("goal_z", 95.7)     # D (down-positive)
        self.declare_parameter("hold_yaw", False)  # start with False (safer)
        self.declare_parameter("yaw_goal", 0.0)

        # ---------------- MPC settings ----------------
        self.declare_parameter("Ts", 0.10)
        self.declare_parameter("N", 15)
        self.declare_parameter("solve_rate_hz", 10.0)

        # ---------------- Model params (free-flyer baseline) ----------------
        self.declare_parameter("mass", 13.5)
        self.declare_parameter("Ix", 0.26)
        self.declare_parameter("Iy", 0.23)
        self.declare_parameter("Iz", 0.37)

        # ---------------- Weights ----------------
        self.declare_parameter("w_pos", 50.0)
        self.declare_parameter("w_vel", 5.0)
        self.declare_parameter("w_yaw", 5.0)
        self.declare_parameter("w_omega", 0.2)
        self.declare_parameter("w_u_force", 0.1)
        self.declare_parameter("w_u_torque", 0.05)

        # ---------------- Physical bounds (Newtons / N*m) ----------------
        # MPC decides forces in N/Nm. We later normalize for PX4.
        self.declare_parameter("Fx_max_N", 20.0)
        self.declare_parameter("Fy_max_N", 20.0)
        self.declare_parameter("Fz_max_N", 30.0)
        self.declare_parameter("Mz_max_Nm", 0.2)   # start small
        self.declare_parameter("Mx_max_Nm", 0.0)   # keep roll torque disabled initially
        self.declare_parameter("My_max_Nm", 0.0)   # keep pitch torque disabled initially

        # ---------------- Publish scaling / saturation ----------------
        # Convert Newtons -> normalized:
        #   thrust_norm = F_N / F_max_N
        # then clamp to +-thrust_sat (like your stabilized controller ranges)
        self.declare_parameter("thrust_sat", 0.15)
        self.declare_parameter("torque_sat_Nm", 0.2)

        # Torque command holder
        self.u_tau_cmd_Nm = np.zeros(3)   # [Mx, My, Mz]

        # Safety / publishing
        self.declare_parameter("publish_dt", 0.02)

        # ---------------- ROS wiring ----------------
        odom_topic = self.get_parameter("odom_topic").value
        cm_topic = self.get_parameter("control_mode_topic").value
        thrust_topic = self.get_parameter("thrust_sp_topic").value
        torque_topic = self.get_parameter("torque_sp_topic").value

        self.sub_odom = self.create_subscription(VehicleOdometry, odom_topic, self.on_odom, px4_qos)
        self.sub_cm = self.create_subscription(VehicleControlMode, cm_topic, self.on_control_mode, px4_qos)

        self.pub_thrust = self.create_publisher(VehicleThrustSetpoint, thrust_topic, px4_qos)
        self.pub_torque = self.create_publisher(VehicleTorqueSetpoint, torque_topic, px4_qos)

        # ---------------- State ----------------
        self.have_odom = False
        self.p_w = np.zeros(3)
        self.q_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.v_b = np.zeros(3)
        self.w_b = np.zeros(3)

        self.enabled = False

        # ---------------- MPC internals ----------------
        self.solver = None
        self.lbx = None
        self.ubx = None
        self.lbg = None
        self.ubg = None
        self.w0 = None  # warm start
        self.u_force_cmd_N = np.zeros(3)  # hold between solves (Newtons)

        self._build_mpc()

        self.solve_timer = self.create_timer(
            1.0 / float(self.get_parameter("solve_rate_hz").value), self.solve_tick
        )
        self.pub_timer = self.create_timer(float(self.get_parameter("publish_dt").value), self.publish_tick)

    # ---------------- callbacks ----------------

    def on_control_mode(self, msg: VehicleControlMode):
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)
        if gate and not self.enabled:
            self.enabled = True
            if self.have_odom and bool(self.get_parameter("hold_yaw").value):
                # optional: lock yaw at current yaw if yaw_goal not set explicitly
                pass
            self.get_logger().info("MPC enabled (armed + offboard).")
        elif (not gate) and self.enabled:
            self.enabled = False
            self.get_logger().info("MPC disabled.")
            self.u_force_cmd_N[:] = 0.0
            self.publish_zero()

    def on_odom(self, msg: VehicleOdometry):
        self.p_w = np.array([float(msg.position[0]), float(msg.position[1]), float(msg.position[2])], dtype=float)
        self.q_wxyz = quat_norm_wxyz((float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])))
        self.v_b = np.array([float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2])], dtype=float)
        self.w_b = np.array(
            [float(msg.angular_velocity[0]), float(msg.angular_velocity[1]), float(msg.angular_velocity[2])],
            dtype=float,
        )
        self.have_odom = True

        # Helpful one-time log to confirm frames
        if not hasattr(self, "_logged_frame_once"):
            self._logged_frame_once = True
            yaw = quat_to_yaw_wxyz(self.q_wxyz)
            self.get_logger().info(
                f"ODOM init: p=[{self.p_w[0]:.3f},{self.p_w[1]:.3f},{self.p_w[2]:.3f}] "
                f"v_b=[{self.v_b[0]:.3f},{self.v_b[1]:.3f},{self.v_b[2]:.3f}] yaw={yaw:.3f} rad"
            )

    # -------------------- MPC build --------------------

    def _build_mpc(self):
        Ts = float(self.get_parameter("Ts").value)
        N = int(self.get_parameter("N").value)

        m = float(self.get_parameter("mass").value)
        Ix = float(self.get_parameter("Ix").value)
        Iy = float(self.get_parameter("Iy").value)
        Iz = float(self.get_parameter("Iz").value)

        # Decision variables: X(13,N+1), F(3,N)
        x = ca.SX.sym("x", 13)
        p = x[0:3]
        q = x[3:7]
        v = x[7:10]
        w = x[10:13]

        u = ca.SX.sym("u", 6)
        F = u[0:3]
        tau = u[3:6]

        # Rotation matrix R(q) body->world (scalar-first)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        R = ca.SX(3, 3)
        R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
        R[0, 1] = 2 * (qx * qy - qz * qw)
        R[0, 2] = 2 * (qx * qz + qy * qw)
        R[1, 0] = 2 * (qx * qy + qz * qw)
        R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
        R[1, 2] = 2 * (qy * qz - qx * qw)
        R[2, 0] = 2 * (qx * qz - qy * qw)
        R[2, 1] = 2 * (qy * qz + qx * qw)
        R[2, 2] = 1 - 2 * (qx * qx + qy * qy)

        # Quaternion derivative using body rates w
        wx, wy, wz = w[0], w[1], w[2]
        qdot = ca.vertcat(
            0.5 * (-qx * wx - qy * wy - qz * wz),
            0.5 * (qw * wx + qy * wz - qz * wy),
            0.5 * (qw * wy - qx * wz + qz * wx),
            0.5 * (qw * wz + qx * wy - qy * wx),
        )

        # Translation dynamics
        pdot = R @ v
        vdot = (1.0 / m) * F

        # Rotation dynamics
        J = ca.diag(ca.vertcat(Ix, Iy, Iz))
        Jinv = ca.diag(ca.vertcat(1.0/Ix, 1.0/Iy, 1.0/Iz))

        Jw = J @ w  # 3x1

        # cross(w, Jw)
        w_cross_Jw = ca.vertcat(
            w[1]*Jw[2] - w[2]*Jw[1],
            w[2]*Jw[0] - w[0]*Jw[2],
            w[0]*Jw[1] - w[1]*Jw[0],
        )

        wdot = Jinv @ (tau - w_cross_Jw)

        xdot = ca.vertcat(pdot, qdot, vdot, wdot)
        xdot_fun = ca.Function("xdot", [x, u], [xdot])

        def rk4(xk, uk):
            k1 = xdot_fun(xk, uk)
            k2 = xdot_fun(xk + 0.5 * Ts * k1, uk)
            k3 = xdot_fun(xk + 0.5 * Ts * k2, uk)
            k4 = xdot_fun(xk + Ts * k3, uk)
            xkp1 = xk + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            qn = xkp1[3:7]
            qn = qn / ca.sqrt(ca.dot(qn, qn) + 1e-12)
            return ca.vertcat(xkp1[0:3], qn, xkp1[7:13])

        # Parameters:
        #   x0(13) + pref(3) + yaw_ref(1) + hold_yaw_flag(1)
        P = ca.SX.sym("P", 18)
        x0 = P[0:13]
        pref = P[13:16]
        yaw_ref = P[16]
        hold_yaw_flag = P[17]

        X = ca.SX.sym("X", 13, N + 1)
        U = ca.SX.sym("U", 6, N) # forces and torques

        w_pos = float(self.get_parameter("w_pos").value)
        w_vel = float(self.get_parameter("w_vel").value)
        w_yaw = float(self.get_parameter("w_yaw").value)
        w_u_force = float(self.get_parameter("w_u_force").value)
        w_u_torque = float(self.get_parameter("w_u_torque").value)

        cost = 0
        g = []

        g.append(X[:, 0] - x0)

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            Fk = uk[0:3]
            tauk = uk[3:6]
            xk1 = X[:, k + 1]
            g.append(xk1 - rk4(xk, uk))

            pk = xk[0:3]
            vk = xk[7:10]
            wk = xk[10:13]

            pos_err = pk - pref
            cost += w_pos * ca.dot(pos_err, pos_err)
            cost += w_vel * ca.dot(vk, vk)
            cost += w_u_force * ca.dot(Fk, Fk)
            cost += float(self.get_parameter("w_omega").value) * ca.dot(wk, wk)
            cost += w_u_torque * ca.dot(tauk, tauk)

            # optional yaw hold (still cheap, doesn’t command torque though)
            qk = xk[3:7]
            qw_k, qx_k, qy_k, qz_k = qk[0], qk[1], qk[2], qk[3]
            siny = 2 * (qw_k * qz_k + qx_k * qy_k)
            cosy = 1 - 2 * (qy_k * qy_k + qz_k * qz_k)
            yaw_k = ca.atan2(siny, cosy)
            yaw_err = ca.atan2(ca.sin(yaw_k - yaw_ref), ca.cos(yaw_k - yaw_ref))
            cost += (hold_yaw_flag * w_yaw) * (yaw_err * yaw_err)

        pN = X[0:3, N]
        pos_err_N = pN - pref
        cost += (2.0 * w_pos) * ca.dot(pos_err_N, pos_err_N)

        w_dec = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g_dec = ca.vertcat(*g)
        nlp = {"x": w_dec, "f": cost, "g": g_dec, "p": P}

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 40,
            "ipopt.tol": 1e-3,
            "print_time": 0,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Bounds
        Fx_max_N = float(self.get_parameter("Fx_max_N").value)
        Fy_max_N = float(self.get_parameter("Fy_max_N").value)
        Fz_max_N = float(self.get_parameter("Fz_max_N").value)

        Mx_max_Nm = float(self.get_parameter("Mx_max_Nm").value)
        My_max_Nm = float(self.get_parameter("My_max_Nm").value)
        Mz_max_Nm = float(self.get_parameter("Mz_max_Nm").value)

        nx = 13 * (N + 1)
        nu = 6 * N
        lbx = -1e9 * np.ones(nx + nu)
        ubx =  1e9 * np.ones(nx + nu)

        for k in range(N):
            idx = nx + 6*k
            # Forces
            lbx[idx+0] = -Fx_max_N;  ubx[idx+0] = Fx_max_N
            lbx[idx+1] = -Fy_max_N;  ubx[idx+1] = Fy_max_N
            lbx[idx+2] = -Fz_max_N;  ubx[idx+2] = Fz_max_N
            # Torques
            lbx[idx+3] = -Mx_max_Nm; ubx[idx+3] = Mx_max_Nm
            lbx[idx+4] = -My_max_Nm; ubx[idx+4] = My_max_Nm
            lbx[idx+5] = -Mz_max_Nm; ubx[idx+5] = Mz_max_Nm

        ng = int(g_dec.size1())
        self.lbg = np.zeros(ng)
        self.ubg = np.zeros(ng)

        self.lbx = lbx
        self.ubx = ubx

        self.w0 = np.zeros(nx + nu)

    # ---------------- helpers ----------------

    def _x_meas(self):
        x = np.zeros(13, dtype=float)
        x[0:3] = self.p_w
        x[3:7] = np.array(self.q_wxyz, dtype=float)
        x[7:10] = self.v_b
        x[10:13] = self.w_b
        return x

    def _p_vec(self):
        goal = np.array(
            [
                float(self.get_parameter("goal_x").value),
                float(self.get_parameter("goal_y").value),
                float(self.get_parameter("goal_z").value),
            ],
            dtype=float,
        )
        yaw_goal = float(self.get_parameter("yaw_goal").value)
        hold_yaw = bool(self.get_parameter("hold_yaw").value)
        hold_yaw_flag = 1.0 if hold_yaw else 0.0

        x0 = self._x_meas()
        return np.concatenate([x0, goal, np.array([yaw_goal, hold_yaw_flag], dtype=float)])

    def _forceN_to_thrust_norm(self, F_N):
        # Newtons -> normalized using per-axis maxima
        Fx_max_N = float(self.get_parameter("Fx_max_N").value)
        Fy_max_N = float(self.get_parameter("Fy_max_N").value)
        Fz_max_N = float(self.get_parameter("Fz_max_N").value)

        # avoid div0
        Fx_max_N = max(Fx_max_N, 1e-6)
        Fy_max_N = max(Fy_max_N, 1e-6)
        Fz_max_N = max(Fz_max_N, 1e-6)

        return np.array([F_N[0] / Fx_max_N, F_N[1] / Fy_max_N, F_N[2] / Fz_max_N], dtype=float)

    # ---------------- runtime ----------------

    def solve_tick(self):
        if not self.enabled or not self.have_odom or self.solver is None:
            return

        P = self._p_vec()

        try:
            sol = self.solver(
                x0=self.w0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=P,
            )
        except Exception as e:
            self.get_logger().warn(f"MPC solve failed: {e}")
            return

        w_opt = np.array(sol["x"]).reshape(-1)
        self.w0 = w_opt.copy()

        N = int(self.get_parameter("N").value)
        nx = 13 * (N + 1)

        u0 = w_opt[nx : nx + 6]
        self.u_force_cmd_N = np.array(u0[0:3], dtype=float)
        self.u_tau_cmd_Nm  = np.array(u0[3:6], dtype=float)

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

    def publish_tick(self):
        if not self.enabled or not self.have_odom:
            return

        now_us = int(self.get_clock().now().nanoseconds / 1000)

        # Convert N -> normalized thrust
        thr_norm = self._forceN_to_thrust_norm(self.u_force_cmd_N)

        thr_norm = np.array([thr_norm[0], thr_norm[1], thr_norm[2]], dtype=float)

        # Clamp to "stabilized-like" saturation
        thrust_sat = float(self.get_parameter("thrust_sat").value)
        thr_norm = np.array(
            [
                clamp(thr_norm[0], -thrust_sat, thrust_sat),
                clamp(thr_norm[1], -thrust_sat, thrust_sat),
                clamp(thr_norm[2], -thrust_sat, thrust_sat),
            ],
            dtype=float,
        )

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = 0
        thr.xyz = [float(thr_norm[0]), float(thr_norm[1]), float(thr_norm[2])]
        self.pub_thrust.publish(thr)

        torque_sat = float(self.get_parameter("torque_sat_Nm").value)
        Mx = float(clamp(self.u_tau_cmd_Nm[0], -torque_sat, torque_sat))
        My = float(clamp(self.u_tau_cmd_Nm[1], -torque_sat, torque_sat))
        Mz = float(clamp(self.u_tau_cmd_Nm[2], -torque_sat, torque_sat))

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = 0
        tor.xyz = [Mx, My, Mz]
        self.pub_torque.publish(tor)


def main():
    rclpy.init()
    node = MPCHoldPosition()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.publish_zero()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()