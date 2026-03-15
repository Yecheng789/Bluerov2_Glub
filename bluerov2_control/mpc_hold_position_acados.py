import math
import os
import shutil
from pathlib import Path

import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import (
    VehicleOdometry,
    VehicleControlMode,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
)

from bluerov2_control.models.fossen_bluerov2_model import build_bluerov2_fossen_model

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as e:
    raise ImportError(
        "acados_template is not installed or not visible in this Python environment. "
        "Install acados + the Python interface first."
    ) from e


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


def euler_to_quat_wxyz(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return quat_norm_wxyz((qw, qx, qy, qz))


def build_rigid_body_explicit_model(m, Ix, Iy, Iz):
    x = ca.SX.sym("x", 13)
    q = x[3:7]
    v = x[7:10]
    w = x[10:13]

    u = ca.SX.sym("u", 6)
    F = u[0:3]
    tau = u[3:6]

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

    wx, wy, wz = w[0], w[1], w[2]
    qdot = ca.vertcat(
        0.5 * (-qx * wx - qy * wy - qz * wz),
        0.5 * (qw * wx + qy * wz - qz * wy),
        0.5 * (qw * wy - qx * wz + qz * wx),
        0.5 * (qw * wz + qx * wy - qy * wx),
    )

    pdot = R @ v
    vdot = (1.0 / m) * F

    J = ca.diag(ca.vertcat(Ix, Iy, Iz))
    Jinv = ca.diag(ca.vertcat(1.0 / Ix, 1.0 / Iy, 1.0 / Iz))
    Jw = J @ w
    w_cross_Jw = ca.vertcat(
        w[1] * Jw[2] - w[2] * Jw[1],
        w[2] * Jw[0] - w[0] * Jw[2],
        w[0] * Jw[1] - w[1] * Jw[0],
    )
    wdot = Jinv @ (tau - w_cross_Jw)

    xdot = ca.vertcat(pdot, qdot, vdot, wdot)
    return x, u, xdot


class MPCHoldPositionAcados(Node):
    def __init__(self):
        super().__init__("mpc_hold_position_acados")

        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.declare_parameter("odom_topic", "/itrl_rov_1/fmu/out/vehicle_odometry")
        self.declare_parameter("control_mode_topic", "/itrl_rov_1/fmu/out/vehicle_control_mode")
        self.declare_parameter("thrust_sp_topic", "/itrl_rov_1/fmu/in/vehicle_thrust_setpoint")
        self.declare_parameter("torque_sp_topic", "/itrl_rov_1/fmu/in/vehicle_torque_setpoint")

        self.declare_parameter("goal_x", -1.15)
        self.declare_parameter("goal_y", -2.175)
        self.declare_parameter("goal_z", 95.7)
        self.declare_parameter("goal_roll", 0.0)
        self.declare_parameter("goal_pitch", 0.0)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("hold_attitude", True)

        self.declare_parameter("Ts", 0.04)
        self.declare_parameter("N", 25)
        self.declare_parameter("solve_rate_hz", 25.0)

        self.declare_parameter("model_type", "fossen")
        self.declare_parameter("mass", 13.5)
        self.declare_parameter("Ix", 0.26)
        self.declare_parameter("Iy", 0.23)
        self.declare_parameter("Iz", 0.37)

        self.declare_parameter("w_pos", 50.0)
        self.declare_parameter("w_vel", 15.0)
        self.declare_parameter("w_att", 8.0)
        self.declare_parameter("w_omega", 1.0)
        self.declare_parameter("w_u_force", 0.1)
        self.declare_parameter("w_u_torque", 0.05)

        self.declare_parameter("Fx_max_N", 88.0)
        self.declare_parameter("Fy_max_N", 88.0)
        self.declare_parameter("Fz_max_N", 137.0)
        self.declare_parameter("Mx_max_Nm", 30.0)
        self.declare_parameter("My_max_Nm", 16.5)
        self.declare_parameter("Mz_max_Nm", 21.0)

        self.declare_parameter("thrust_sat", 0.2)
        self.declare_parameter("torque_sat", 0.2)
        self.declare_parameter("publish_dt", 0.02)

        self.declare_parameter("codegen_dir", "/tmp/bluerov2_acados_codegen")
        self.declare_parameter("rebuild_solver", False)

        odom_topic = self.get_parameter("odom_topic").value
        cm_topic = self.get_parameter("control_mode_topic").value
        thrust_topic = self.get_parameter("thrust_sp_topic").value
        torque_topic = self.get_parameter("torque_sp_topic").value

        self.sub_odom = self.create_subscription(VehicleOdometry, odom_topic, self.on_odom, px4_qos)
        self.sub_cm = self.create_subscription(VehicleControlMode, cm_topic, self.on_control_mode, px4_qos)
        self.pub_thrust = self.create_publisher(VehicleThrustSetpoint, thrust_topic, px4_qos)
        self.pub_torque = self.create_publisher(VehicleTorqueSetpoint, torque_topic, px4_qos)

        self.have_odom = False
        self.p_w = np.zeros(3)
        self.q_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.v_b = np.zeros(3)
        self.w_b = np.zeros(3)
        self.enabled = False

        self.q_goal = euler_to_quat_wxyz(
            float(self.get_parameter("goal_roll").value),
            float(self.get_parameter("goal_pitch").value),
            float(self.get_parameter("goal_yaw").value),
        )

        self.ocp_solver = None
        self.u_force_cmd_N = np.zeros(3)
        self.u_tau_cmd_Nm = np.zeros(3)
        self.x_guess = None
        self.u_guess = None
        self.N_horizon = int(self.get_parameter("N").value)

        self._build_mpc()

        self.get_logger().info(
            f"acados MPC model_type = {str(self.get_parameter('model_type').value)}"
        )

        self.solve_timer = self.create_timer(
            1.0 / float(self.get_parameter("solve_rate_hz").value), self.solve_tick
        )
        self.pub_timer = self.create_timer(float(self.get_parameter("publish_dt").value), self.publish_tick)

    def on_control_mode(self, msg: VehicleControlMode):
        gate = bool(msg.flag_armed) and bool(msg.flag_control_offboard_enabled)
        if gate and not self.enabled:
            self.enabled = True
            self.get_logger().info("acados MPC enabled (armed + offboard).")
        elif (not gate) and self.enabled:
            self.enabled = False
            self.get_logger().info("acados MPC disabled.")
            self.u_force_cmd_N[:] = 0.0
            self.u_tau_cmd_Nm[:] = 0.0
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

        if not hasattr(self, "_logged_frame_once"):
            self._logged_frame_once = True
            yaw = quat_to_yaw_wxyz(self.q_wxyz)
            self.get_logger().info(
                f"ODOM init: p=[{self.p_w[0]:.3f},{self.p_w[1]:.3f},{self.p_w[2]:.3f}] "
                f"v_b=[{self.v_b[0]:.3f},{self.v_b[1]:.3f},{self.v_b[2]:.3f}] yaw={yaw:.3f} rad"
            )

    def _build_mpc(self):
        Ts = float(self.get_parameter("Ts").value)
        N = int(self.get_parameter("N").value)
        self.N_horizon = N
        model_type = str(self.get_parameter("model_type").value).strip().lower()

        if model_type == "fossen":
            x_sym, u_sym, xdot_fun, _ = build_bluerov2_fossen_model(Ts)
            xdot_expr = xdot_fun(x_sym, u_sym)
        else:
            m = float(self.get_parameter("mass").value)
            Ix = float(self.get_parameter("Ix").value)
            Iy = float(self.get_parameter("Iy").value)
            Iz = float(self.get_parameter("Iz").value)
            x_sym, u_sym, xdot_expr = build_rigid_body_explicit_model(m, Ix, Iy, Iz)

        p_sym = ca.SX.sym("p", 8)
        pref = p_sym[0:3]
        qref = p_sym[3:7]
        hold_att_flag = p_sym[7]

        model = AcadosModel()
        model.name = f"bluerov2_{model_type}_hold"
        model.x = x_sym
        model.u = u_sym
        model.p = p_sym
        model.xdot = ca.SX.sym("xdot", 13)
        model.f_expl_expr = xdot_expr
        model.f_impl_expr = model.xdot - xdot_expr

        x = model.x
        u = model.u
        q = x[3:7]
        pos = x[0:3]
        vel = x[7:10]
        omega = x[10:13]
        F = u[0:3]
        tau = u[3:6]

        w_pos = float(self.get_parameter("w_pos").value)
        w_vel = float(self.get_parameter("w_vel").value)
        w_att = float(self.get_parameter("w_att").value)
        w_omega = float(self.get_parameter("w_omega").value)
        w_u_force = float(self.get_parameter("w_u_force").value)
        w_u_torque = float(self.get_parameter("w_u_torque").value)

        pos_err = pos - pref
        dot_q = ca.dot(q, qref)
        att_err_cost = 1.0 - dot_q * dot_q

        stage_cost = (
            w_pos * ca.dot(pos_err, pos_err)
            + w_vel * ca.dot(vel, vel)
            + w_omega * ca.dot(omega, omega)
            + w_u_force * ca.dot(F, F)
            + w_u_torque * ca.dot(tau, tau)
            + hold_att_flag * w_att * att_err_cost
        )
        terminal_cost = (
            2.0 * w_pos * ca.dot(pos_err, pos_err)
            + hold_att_flag * 2.0 * w_att * att_err_cost
        )

        model.cost_expr_ext_cost = stage_cost
        model.cost_expr_ext_cost_e = terminal_cost

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = N
        ocp.solver_options.tf = N * Ts

        ocp.parameter_values = np.zeros(8)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        Fx_max_N = float(self.get_parameter("Fx_max_N").value)
        Fy_max_N = float(self.get_parameter("Fy_max_N").value)
        Fz_max_N = float(self.get_parameter("Fz_max_N").value)
        Mx_max_Nm = float(self.get_parameter("Mx_max_Nm").value)
        My_max_Nm = float(self.get_parameter("My_max_Nm").value)
        Mz_max_Nm = float(self.get_parameter("Mz_max_Nm").value)

        lbu = np.array([-Fx_max_N, -Fy_max_N, -Fz_max_N, -Mx_max_Nm, -My_max_Nm, -Mz_max_Nm], dtype=float)
        ubu = np.array([ Fx_max_N,  Fy_max_N,  Fz_max_N,  Mx_max_Nm,  My_max_Nm,  Mz_max_Nm], dtype=float)
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu

        x0 = np.zeros(13)
        x0[3] = 1.0
        ocp.constraints.x0 = x0

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.print_level = 0
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.qp_solver_cond_N = min(10, N)
        ocp.solver_options.tol = 1e-3

        codegen_dir = Path(str(self.get_parameter("codegen_dir").value)).expanduser().resolve()
        if bool(self.get_parameter("rebuild_solver").value) and codegen_dir.exists():
            shutil.rmtree(codegen_dir)
        codegen_dir.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(codegen_dir / model.name)

        json_file = str(codegen_dir / f"{model.name}_ocp.json")
        self.ocp_solver = AcadosOcpSolver(ocp, json_file=json_file, build=True, generate=True, verbose=False)

        self.x_guess = np.tile(x0.reshape(1, -1), (N + 1, 1))
        self.u_guess = np.zeros((N, 6), dtype=float)

    def _x_meas(self):
        x = np.zeros(13, dtype=float)
        x[0:3] = self.p_w
        x[3:7] = np.array(self.q_wxyz, dtype=float)
        x[7:10] = self.v_b
        x[10:13] = self.w_b
        return x

    def _stage_param(self):
        goal = np.array([
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
            float(self.get_parameter("goal_z").value),
        ], dtype=float)
        att_goal = np.array(self.q_goal, dtype=float)
        hold_att_flag = 1.0 if bool(self.get_parameter("hold_attitude").value) else 0.0
        return np.concatenate([goal, att_goal, np.array([hold_att_flag], dtype=float)])

    def _forceN_to_thrust_norm(self, F_N):
        Fx_max_N = max(float(self.get_parameter("Fx_max_N").value), 1e-6)
        Fy_max_N = max(float(self.get_parameter("Fy_max_N").value), 1e-6)
        Fz_max_N = max(float(self.get_parameter("Fz_max_N").value), 1e-6)
        return np.array([F_N[0] / Fx_max_N, F_N[1] / Fy_max_N, F_N[2] / Fz_max_N], dtype=float)

    def _torqueNm_to_torque_norm(self, tau_Nm):
        Mx_max_Nm = max(float(self.get_parameter("Mx_max_Nm").value), 1e-6)
        My_max_Nm = max(float(self.get_parameter("My_max_Nm").value), 1e-6)
        Mz_max_Nm = max(float(self.get_parameter("Mz_max_Nm").value), 1e-6)
        return np.array([
            tau_Nm[0] / Mx_max_Nm,
            tau_Nm[1] / My_max_Nm,
            tau_Nm[2] / Mz_max_Nm,
        ], dtype=float)

    def solve_tick(self):
        if not self.enabled or not self.have_odom or self.ocp_solver is None:
            return

        x0 = self._x_meas()
        p_stage = self._stage_param()

        qn = np.linalg.norm(x0[3:7])
        if qn > 1e-12:
            x0[3:7] = x0[3:7] / qn
        else:
            x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        for k in range(self.N_horizon):
            self.ocp_solver.set(k, "x", self.x_guess[k])
            self.ocp_solver.set(k, "u", self.u_guess[k])
            self.ocp_solver.set(k, "p", p_stage)
        self.ocp_solver.set(self.N_horizon, "x", self.x_guess[self.N_horizon])
        self.ocp_solver.set(self.N_horizon, "p", p_stage)

        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        status = self.ocp_solver.solve()
        if status != 0:
            self.get_logger().warn(f"acados solve failed, status={status}")
            return

        u0 = np.array(self.ocp_solver.get(0, "u"), dtype=float).reshape(-1)
        self.u_force_cmd_N = u0[0:3].copy()
        self.u_tau_cmd_Nm = u0[3:6].copy()

        x_new = np.zeros_like(self.x_guess)
        u_new = np.zeros_like(self.u_guess)
        for k in range(self.N_horizon + 1):
            x_new[k, :] = np.array(self.ocp_solver.get(k, "x"), dtype=float).reshape(-1)
            qk = x_new[k, 3:7]
            nq = np.linalg.norm(qk)
            if nq > 1e-12:
                x_new[k, 3:7] = qk / nq
        for k in range(self.N_horizon):
            u_new[k, :] = np.array(self.ocp_solver.get(k, "u"), dtype=float).reshape(-1)

        self.x_guess[:-1, :] = x_new[1:, :]
        self.x_guess[-1, :] = x_new[-1, :]
        self.u_guess[:-1, :] = u_new[1:, :]
        self.u_guess[-1, :] = u_new[-1, :]

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

        thr_norm = self._forceN_to_thrust_norm(self.u_force_cmd_N)
        thrust_sat = float(self.get_parameter("thrust_sat").value)
        thr_norm = np.array([
            clamp(thr_norm[0], -thrust_sat, thrust_sat),
            clamp(thr_norm[1], -thrust_sat, thrust_sat),
            clamp(thr_norm[2], -thrust_sat, thrust_sat),
        ], dtype=float)

        thr = VehicleThrustSetpoint()
        thr.timestamp = now_us
        thr.timestamp_sample = 0
        thr.xyz = [float(thr_norm[0]), float(thr_norm[1]), float(thr_norm[2])]
        self.pub_thrust.publish(thr)

        tau_norm = self._torqueNm_to_torque_norm(self.u_tau_cmd_Nm)
        torque_sat = float(self.get_parameter("torque_sat").value)
        tau_norm = np.array([
            clamp(tau_norm[0], -torque_sat, torque_sat),
            clamp(tau_norm[1], -torque_sat, torque_sat),
            clamp(tau_norm[2], -torque_sat, torque_sat),
        ], dtype=float)

        tor = VehicleTorqueSetpoint()
        tor.timestamp = now_us
        tor.timestamp_sample = 0
        tor.xyz = [float(tau_norm[0]), float(tau_norm[1]), float(tau_norm[2])]
        self.pub_torque.publish(tor)


def main():
    rclpy.init()
    node = MPCHoldPositionAcados()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.publish_zero()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
