#!/usr/bin/env python3
import casadi as cs
import numpy as np

class DoubleIntegratorWrenchCasadi:
    """
    Discrete-time learned DI on body velocities & rates.
    Uses weights saved by the wrench training script:
        K_lin (6x3), K_ang (6x3)

    state x = [x,y,z, phi,theta,psi,  u,v,w, p,q,r]
    input u = [Fx,Fy,Fz,Mx,My,Mz]  (6D body-frame wrench)
    """

    def __init__(self, weights_path: str):
        W = np.load(weights_path)
        self.K_lin = cs.DM(W["K_lin"])  # (6,3)
        self.K_ang = cs.DM(W["K_ang"])  # (6,3)
        self.dt_nom = float(W["dt"][0]) # nominal logging dt (optional)
        self.create()

    # ---- rotation helpers ----
    @staticmethod
    def _rz(psi):
        c, s = cs.cos(psi), cs.sin(psi)
        return cs.vertcat(
            cs.horzcat(c, -s, 0),
            cs.horzcat(s,  c, 0),
            cs.horzcat(0,  0, 1),
        )

    @staticmethod
    def _ry(theta):
        c, s = cs.cos(theta), cs.sin(theta)
        return cs.vertcat(
            cs.horzcat( c, 0, s),
            cs.horzcat( 0, 1, 0),
            cs.horzcat(-s, 0, c),
        )

    @staticmethod
    def _rx(phi):
        c, s = cs.cos(phi), cs.sin(phi)
        return cs.vertcat(
            cs.horzcat(1, 0, 0),
            cs.horzcat(0, c,-s),
            cs.horzcat(0, s, c),
        )

    def _R_b2n(self, phi, theta, psi):
        return self._rz(psi) @ self._ry(theta) @ self._rx(phi)

    # ---- build CasADi graph ----
    def create(self):
        x   = cs.SX.sym('x', 12)  # state
        tau = cs.SX.sym('tau', 6) # wrench [Fx,Fy,Fz,Mx,My,Mz]
        dt  = cs.SX.sym('dt', 1)

        pos = x[0:3]
        phi, theta, psi = x[3], x[4], x[5]
        v   = x[6:9]   # body linear vel
        w   = x[9:12]  # body angular rates

        # Learned mapping from wrench -> body accelerations
        # (1x6)(6x3) -> (1x3)
        a_body = tau.T @ self.K_lin   # linear accel
        alpha  = tau.T @ self.K_ang   # angular accel

        v_next = v + dt * a_body.T
        w_next = w + dt * alpha.T

        Rb2n = self._R_b2n(phi, theta, psi)
        pos_next = pos + dt * (Rb2n @ v)
        ang_next = cs.vertcat(
            phi   + dt*w[0],
            theta + dt*w[1],
            psi   + dt*w[2]
        )

        x_next = cs.vertcat(pos_next, ang_next, v_next, w_next)
        self.f_disc = cs.Function('di_wrench_step', [x, tau, dt], [x_next])

    # Convenience
    def calculate_f_disc(self, x, tau, dt):
        return np.array(self.f_disc(x, tau, dt)).reshape((12,))