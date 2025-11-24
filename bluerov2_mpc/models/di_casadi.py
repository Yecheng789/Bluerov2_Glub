#!/usr/bin/env python3
import casadi as cs
import numpy as np

class DoubleIntegratorCasadi:
    """
    Discrete-time learned DI on body velocities & rates.
    Uses weights saved by the training script: K_lin (8x3), K_ang (8x3).

    x = [x,y,z, phi,theta,psi,  u,v,w, p,q,r]
    u = [u1..u8] in [-1,1]
    """

    def __init__(self, weights_path: str):
        W = np.load(weights_path)
        self.K_lin = cs.DM(W["K_lin"])  # (8,3)
        self.K_ang = cs.DM(W["K_ang"])  # (8,3)
        self.dt_nom = float(W["dt"][0]) # nominal logging dt (optional)
        self.create()

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
    def _R_b2n(self, phi,theta,psi):
        return self._rz(psi) @ self._ry(theta) @ self._rx(phi)

    def create(self):
        x  = cs.SX.sym('x', 12)
        u8 = cs.SX.sym('u', 8)
        dt = cs.SX.sym('dt', 1)

        pos   = x[0:3]
        phi,theta,psi = x[3],x[4],x[5]
        v     = x[6:9]
        w     = x[9:12]

        a_body = u8.T @ self.K_lin  # (1x8)(8x3) -> (1x3)
        alpha  = u8.T @ self.K_ang  # -> (1x3)

        v_next = v + dt * a_body.T
        w_next = w + dt * alpha.T

        Rb2n = self._R_b2n(phi,theta,psi)
        pos_next = pos + dt * (Rb2n @ v)
        ang_next = cs.vertcat(
            phi + dt*w[0],
            theta + dt*w[1],
            psi + dt*w[2]
        )

        x_next = cs.vertcat(pos_next, ang_next, v_next, w_next)
        self.f_disc = cs.Function('di_step', [x,u8,dt], [x_next])

    # Convenience
    def calculate_f_disc(self, x, u, dt):
        return np.array(self.f_disc(x,u,dt)).reshape((12,))