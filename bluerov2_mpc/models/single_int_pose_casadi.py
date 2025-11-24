#!/usr/bin/env python3
import casadi as cs
import numpy as np


class SingleIntegratorPoseCasadi:
    """
    Simple kinematic model:

      x = [x, y, z, phi, theta, psi]
      u = [v_x, v_y, v_z, p, q, r]   (body-frame linear/angular velocities)

    Continuous-time:
        p_dot   = R_b2n(phi,theta,psi) * v_body
        ang_dot = [p, q, r]

    Discrete-time:
        x_{k+1} = x_k + dt * f(x_k, u_k)
    """

    def __init__(self, dt_nom: float):
        self.dt_nom = float(dt_nom)
        self._build()

    # ---- rotation helpers (same convention as your euler_R_b2n) ----
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

    def _build(self):
        x = cs.SX.sym('x', 6)   # [x,y,z,phi,theta,psi]
        u = cs.SX.sym('u', 6)   # [vx,vy,vz,p,q,r]
        dt = cs.SX.sym('dt', 1)

        pos = x[0:3]
        phi, theta, psi = x[3], x[4], x[5]
        v_body = u[0:3]
        w_body = u[3:6]

        Rb2n = self._R_b2n(phi, theta, psi)

        pos_dot = Rb2n @ v_body
        ang_dot = w_body

        xdot = cs.vertcat(pos_dot, ang_dot)
        x_next = x + dt * xdot

        self.f_disc = cs.Function('single_int_step', [x, u, dt], [x_next])

    # Convenience wrapper
    def step(self, x, u, dt=None):
        if dt is None:
            dt = self.dt_nom
        return np.array(self.f_disc(x, u, dt)).reshape((6,))