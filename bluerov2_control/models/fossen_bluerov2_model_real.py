#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CasADi-ready BlueROV2 real-robot configuration model with direct 6D body-wrench input.

State:
    x = [pn, pe, pd, qw, qx, qy, qz, u, v, w, p, q, r]
        position in world/PX4-local frame + unit quaternion + body velocities

Input:
    tau = [Fx, Fy, Fz, Mx, My, Mz] in body frame

This is adapted from https://github.com/ViktorNfa/bluerov2_dynamics/blob/main/fossen/BlueROV2_wrench.py

Robot types:
    - "standard"   : real robot close to the official BlueROV2 heavy standard configuration,
                     but lighter (12.5 kg). Volume is kept equal to the SITL model.
    - "heavy_tube" : real robot with extra 300 mm 4-inch tube on top, total mass 15.6 kg,
                     increased displaced volume, and nearly neutral / very slightly positive buoyancy.
"""

import casadi as ca


def _skew3(a):
    return ca.vertcat(
        ca.horzcat(0, -a[2], a[1]),
        ca.horzcat(a[2], 0, -a[0]),
        ca.horzcat(-a[1], a[0], 0),
    )


def quat_normalize_sym(q, eps=1e-12):
    return q / ca.sqrt(ca.dot(q, q) + eps)


def quat_to_rotation_matrix_sym(q):
    """
    Quaternion convention: q = [qw, qx, qy, qz] (scalar-first).
    Returns R_{b->n}.
    """
    qn = quat_normalize_sym(q)
    qw, qx, qy, qz = qn[0], qn[1], qn[2], qn[3]

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
    return R


def quat_derivative_sym(q, omega_body):
    """
    q_dot = 0.5 * (q ⊗ [0, p, q, r])
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    p, qrate, r = omega_body[0], omega_body[1], omega_body[2]
    return 0.5 * ca.vertcat(
        -qx * p - qy * qrate - qz * r,
         qw * p + qy * r - qz * qrate,
         qw * qrate - qx * r + qz * p,
         qw * r + qx * qrate - qy * p,
    )


def build_bluerov2_fossen_model(Ts, current_speed_ned=(0.0, 0.0, 0.0), robot_type="standard"):
    """
    Build symbolic continuous and RK4-discrete dynamics for the BlueROV2 Fossen model.

    Args:
        Ts: MPC sampling time [s]
        current_speed_ned: constant irrotational current in navigation/world frame [m/s]
        robot_type: "standard" or "heavy_tube"

    Returns:
        x:    SX(13) state symbol
        u:    SX(6)  input symbol
        xdot_fun(x,u) -> continuous dynamics
        rk4_step(x,u) -> one-step discrete dynamics with quaternion renormalization
    """

    # ---------------- Physical parameters from source model ----------------
    rho = 1000.0
    g = 9.82

    # Real robot selector
    if robot_type == "standard":
        # Same overall geometry as the official heavy standard SITL model,
        # but with lower measured mass.
        m = 12.5
        volume = 0.0135

        # CG / CB
        xb = 0.0
        yb = 0.0
        zb = -0.01

        # Inertias
        # First-pass scaling from the SITL model by mass ratio.
        mass_scale = m / 13.0
        Ix = 0.25 * mass_scale
        Iy = 0.221 * mass_scale
        Iz = 0.356 * mass_scale

    elif robot_type == "heavy_tube":
        # Heavier real robot with extra tube on top.
        m = 15.6

        # Nearly neutral / very slightly positive buoyancy in water.
        # Neutral buoyancy would be volume = m / rho = 0.0156 m^3.
        # Use a very small margin above neutral.
        volume = 0.01565

        # Keep the same CB offset as the working model for the first pass.
        xb = 0.0
        yb = 0.0
        zb = -0.01

        # Inertias
        # First-pass scaling from the SITL model by mass ratio.
        # This captures the mass increase without adding uncertain geometry assumptions.
        mass_scale = m / 13.0
        Ix = 0.25 * mass_scale
        Iy = 0.221 * mass_scale
        Iz = 0.356 * mass_scale

    else:
        raise ValueError(f"Unknown robot_type '{robot_type}'. Expected 'standard' or 'heavy_tube'.")

    W = m * g
    B = rho * g * volume

    # Added mass derivatives from source model
    Xu_dot = -1.272
    Yv_dot = -1.424
    Zw_dot = -3.736
    Kp_dot = -0.0378
    Mq_dot = -0.027
    Nr_dot = -0.044

    # Linear + quadratic damping coefficients from source model
    Xu = -13.7
    Xu_abs = -14.1
    Yv = -1.0
    Yv_abs = -21.7
    Zw = -23.0
    Zw_abs = -19.0
    Kp = -0.5
    Kp_abs = -0.119
    Mq = -0.8
    Mq_abs = -0.047
    Nr = -0.1
    Nr_abs = -0.15

    # ---------------- State / input symbols ----------------
    x = ca.SX.sym("x", 13)
    u = ca.SX.sym("u", 6)

    pos = x[0:3]
    quat = x[3:7]
    nu = x[7:13]      # [u, v, w, p, q, r]
    tau_body = u      # [Fx, Fy, Fz, Mx, My, Mz]

    vel_lin = nu[0:3]
    vel_ang = nu[3:6]

    # ---------------- Constant matrices ----------------
    MRB = ca.diag(ca.vertcat(m, m, m, Ix, Iy, Iz))
    MA = ca.diag(ca.vertcat(-Xu_dot, -Yv_dot, -Zw_dot, -Kp_dot, -Mq_dot, -Nr_dot))
    M = MRB + MA
    Minv = ca.inv(M)

    # ---------------- Kinematics ----------------
    quat_n = quat_normalize_sym(quat)
    R_b2n = quat_to_rotation_matrix_sym(quat_n)
    R_n2b = R_b2n.T

    pos_dot = R_b2n @ vel_lin
    quat_dot = quat_derivative_sym(quat_n, vel_ang)

    # ---------------- Relative velocity (current compensation) ----------------
    current_speed_ned = ca.DM(current_speed_ned).reshape((3, 1))
    v_c_b = R_n2b @ current_speed_ned
    nu_r = ca.vertcat(vel_lin - v_c_b, vel_ang)

    u_b, v_b, w_b, p_b, q_b, r_b = nu[0], nu[1], nu[2], nu[3], nu[4], nu[5]
    u_r, v_r, w_r, p_r, q_r, r_r = nu_r[0], nu_r[1], nu_r[2], nu_r[3], nu_r[4], nu_r[5]

    # ---------------- Coriolis / centripetal ----------------
    CRB = ca.SX.zeros(6, 6)
    CRB[0, 4] =  m * w_b
    CRB[0, 5] = -m * v_b
    CRB[1, 3] = -m * w_b
    CRB[1, 5] =  m * u_b
    CRB[2, 3] =  m * v_b
    CRB[2, 4] = -m * u_b
    CRB[3, 1] =  m * w_b
    CRB[3, 2] = -m * v_b
    CRB[3, 4] =  Iz * r_b
    CRB[3, 5] = -Iy * q_b
    CRB[4, 0] = -m * w_b
    CRB[4, 2] =  m * u_b
    CRB[4, 3] = -Iz * r_b
    CRB[4, 5] =  Ix * p_b
    CRB[5, 0] =  m * v_b
    CRB[5, 1] = -m * u_b
    CRB[5, 3] =  Iy * q_b
    CRB[5, 4] = -Ix * p_b

    CA = ca.SX.zeros(6, 6)
    CA[0, 4] = -Zw_dot * w_b
    CA[0, 5] =  Yv_dot * v_b
    CA[1, 3] =  Zw_dot * w_b
    CA[1, 5] = -Xu_dot * u_b
    CA[2, 3] = -Yv_dot * v_b
    CA[2, 4] =  Xu_dot * u_b
    CA[3, 1] = -Zw_dot * w_b
    CA[3, 2] =  Yv_dot * v_b
    CA[3, 4] = -Nr_dot * r_b
    CA[3, 5] =  Mq_dot * q_b
    CA[4, 0] =  Zw_dot * w_b
    CA[4, 2] = -Xu_dot * u_b
    CA[4, 3] =  Nr_dot * r_b
    CA[4, 5] = -Kp_dot * p_b
    CA[5, 0] = -Yv_dot * v_b
    CA[5, 1] =  Xu_dot * u_b
    CA[5, 3] = -Mq_dot * q_b
    CA[5, 4] =  Kp_dot * p_b

    C = CRB + CA

    # ---------------- Damping ----------------
    D = ca.SX.zeros(6, 6)
    D[0, 0] = -Xu - Xu_abs * ca.fabs(u_r)
    D[1, 1] = -Yv - Yv_abs * ca.fabs(v_r)
    D[2, 2] = -Zw - Zw_abs * ca.fabs(w_r)
    D[3, 3] = -Kp - Kp_abs * ca.fabs(p_r)
    D[4, 4] = -Mq - Mq_abs * ca.fabs(q_r)
    D[5, 5] = -Nr - Nr_abs * ca.fabs(r_r)

    # ---------------- Hydrostatic restoring wrench ----------------
    W_minus_B = W - B
    sth = -R_b2n[2, 0]
    cth_sphi = R_b2n[2, 1]
    cth_cphi = R_b2n[2, 2]

    gvec = ca.vertcat(
        W_minus_B * sth,
        -W_minus_B * cth_sphi,
        -W_minus_B * cth_cphi,
        (yb * B) * cth_cphi - (zb * B) * cth_sphi,
        -(zb * B) * sth - (xb * B) * cth_cphi,
        (xb * B) * cth_sphi + (yb * B) * sth,
    )

    # ---------------- Dynamics ----------------
    rhs = tau_body - C @ nu - D @ nu_r - gvec
    nu_dot = Minv @ rhs

    xdot = ca.vertcat(pos_dot, quat_dot, nu_dot)
    xdot_fun = ca.Function("bluerov2_fossen_xdot", [x, u], [xdot])

    # ---------------- RK4 discretization ----------------
    xk = ca.SX.sym("xk", 13)
    uk = ca.SX.sym("uk", 6)

    k1 = xdot_fun(xk, uk)
    k2 = xdot_fun(xk + 0.5 * Ts * k1, uk)
    k3 = xdot_fun(xk + 0.5 * Ts * k2, uk)
    k4 = xdot_fun(xk + Ts * k3, uk)
    xkp1 = xk + (Ts / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    qn = quat_normalize_sym(xkp1[3:7])
    xkp1 = ca.vertcat(xkp1[0:3], qn, xkp1[7:13])
    rk4_step = ca.Function("bluerov2_fossen_rk4", [xk, uk], [xkp1])

    return x, u, xdot_fun, rk4_step