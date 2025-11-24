#!/usr/bin/env python3
import casadi as cs
import numpy as np

class FossenWrenchCasadi:
    """
    Continuous-time Fossen BlueROV2-heavy model with direct 6D wrench input.

    state x = [x,y,z, phi,theta,psi,  u,v,w, p,q,r]
    input tau = [Fx,Fy,Fz,Mx,My,Mz]  (body-frame wrench)

    Thruster mapping is removed; you give the net body wrench directly.
    """

    def __init__(self):
        # Physical params
        self.g = 9.82
        self.rho = 1000.0
        self.m = 13.5
        self.volume = 0.0134
        self.W = self.m*self.g
        self.B = self.rho*self.g*self.volume

        # CG / CB
        self.xg = 0.0; self.yg = 0.0; self.zg = 0.0
        self.xb = 0.0; self.yb = 0.0; self.zb = -0.01

        # Inertias
        self.Ix = 0.26; self.Iy = 0.23; self.Iz = 0.37

        # Added mass (negative in Fossen; paper signs corrected)
        Xu_dot, Yv_dot, Zw_dot = -6.36, -7.12, -18.68
        Kp_dot, Mq_dot, Nr_dot = -0.189, -0.135, -0.222

        # Damping (linear + abs(.) terms)
        self.Xu, self.Xu_abs = -13.7, -141.0
        self.Yv, self.Yv_abs = -0.0,  -217.0
        self.Zw, self.Zw_abs = -33.0, -190.0
        self.Kp, self.Kp_abs = -0.0,  -1.19
        self.Mq, self.Mq_abs = -0.8,  -0.47
        self.Nr, self.Nr_abs = -0.0,  -1.5

        # Mass matrices (SX)
        MRB = cs.diag(cs.vcat([self.m, self.m, self.m, self.Ix, self.Iy, self.Iz]))
        MA  = cs.diag(cs.vcat([-Xu_dot, -Yv_dot, -Zw_dot, -Kp_dot, -Mq_dot, -Nr_dot]))
        self.M = MRB + MA
        self.Minv = cs.inv(self.M)

        # build CasADi functions
        self.create()

    # ---- helpers ----
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

    def _J2(self, phi, theta, eps=1e-7):
        sphi, cphi = cs.sin(phi), cs.cos(phi)
        sth,  cth  = cs.sin(theta), cs.cos(theta)
        cth_safe = cth + 0*eps
        tth = sth / cth_safe
        return cs.vertcat(
            cs.horzcat(1, sphi*tth,  cphi*tth),
            cs.horzcat(0, cphi,     -sphi),
            cs.horzcat(0, sphi/cth_safe, cphi/cth_safe),
        )

    def _C(self, nu):
        u,v,w,p,q,r = nu[0],nu[1],nu[2],nu[3],nu[4],nu[5]
        m, Ix, Iy, Iz = self.m, self.Ix, self.Iy, self.Iz
        CRB = cs.SX.zeros(6,6)
        CRB[0,4] =  m*w;  CRB[0,5] = -m*v
        CRB[1,3] = -m*w;  CRB[1,5] =  m*u
        CRB[2,3] =  m*v;  CRB[2,4] = -m*u
        CRB[3,1] =  m*w;  CRB[3,2] = -m*v
        CRB[3,4] =  Iz*r; CRB[3,5] = -Iy*q
        CRB[4,0] = -m*w;  CRB[4,2] =  m*u
        CRB[4,3] = -Iz*r; CRB[4,5] =  Ix*p
        CRB[5,0] =  m*v;  CRB[5,1] = -m*u
        CRB[5,3] =  Iy*q; CRB[5,4] = -Ix*p

        Xu_dot, Yv_dot, Zw_dot = -6.36, -7.12, -18.68
        Kp_dot, Mq_dot, Nr_dot = -0.189, -0.135, -0.222
        CA = cs.SX.zeros(6,6)
        CA[0,4] = -Zw_dot*w;  CA[0,5] =  Yv_dot*v
        CA[1,3] =  Zw_dot*w;  CA[1,5] = -Xu_dot*u
        CA[2,3] = -Yv_dot*v;  CA[2,4] =  Xu_dot*u
        CA[3,1] = -Zw_dot*w;  CA[3,2] =  Yv_dot*v
        CA[3,4] = -Nr_dot*r;  CA[3,5] =  Mq_dot*q
        CA[4,0] =  Zw_dot*w;  CA[4,2] = -Xu_dot*u
        CA[4,3] =  Nr_dot*r;  CA[4,5] = -Kp_dot*p
        CA[5,0] = -Yv_dot*v;  CA[5,1] =  Xu_dot*u
        CA[5,3] = -Mq_dot*q;  CA[5,4] =  Kp_dot*p
        return CRB + CA

    def _D(self, nu_r):
        u,v,w,p,q,r = [nu_r[i] for i in range(6)]
        D = cs.SX.zeros(6,6)
        D[0,0] = -(self.Xu + self.Xu_abs*cs.fabs(u))
        D[1,1] = -(self.Yv + self.Yv_abs*cs.fabs(v))
        D[2,2] = -(self.Zw + self.Zw_abs*cs.fabs(w))
        D[3,3] = -(self.Kp + self.Kp_abs*cs.fabs(p))
        D[4,4] = -(self.Mq + self.Mq_abs*cs.fabs(q))
        D[5,5] = -(self.Nr + self.Nr_abs*cs.fabs(r))
        return D

    def _gvec(self, phi, theta, psi):
        WmB = self.W - self.B
        sphi, cphi = cs.sin(phi), cs.cos(phi)
        sth,  cth  = cs.sin(theta), cs.cos(theta)
        gvec = cs.SX.zeros(6,1)
        gvec[0] =  WmB * sth
        gvec[1] = -WmB * cth*sphi
        gvec[2] = -WmB * cth*cphi
        gvec[3] =  (self.yb*self.B)*cth*cphi - (self.zb*self.B)*cth*sphi
        gvec[4] = -(self.zb*self.B)*sth - (self.xb*self.B)*cth*cphi
        gvec[5] =  (self.xb*self.B)*cth*sphi + (self.yb*self.B)*sth
        return gvec

    # ---- public API ----
    def create(self):
        x   = cs.SX.sym('x', 12)  # [p(3), euler(3), v(3), w(3)]
        tau = cs.SX.sym('tau', 6) # wrench input
        dt  = cs.SX.sym('dt', 1)

        p = x[0:3]
        phi, theta, psi = x[3], x[4], x[5]
        v = x[6:9]
        w = x[9:12]

        Rb2n = self._R_b2n(phi, theta, psi)
        J2   = self._J2(phi, theta)

        nu   = cs.vertcat(v, w)
        nu_r = nu   # no current

        C = self._C(nu)
        D = self._D(nu_r)
        g = self._gvec(phi, theta, psi)

        # Direct wrench input
        tau_vec = tau  # (6x1)

        rhs_nu  = self.Minv @ (tau_vec - C@nu - D@nu_r - g)
        eta_dot = cs.vertcat(Rb2n @ v, J2 @ w)
        xdot    = cs.vertcat(eta_dot, rhs_nu)

        self.f_expl = cs.Function('f_expl_wrench', [x, tau], [xdot])

        # RK4 stepper
        k1 = xdot
        k2 = self.f_expl(x + 0.5*dt*k1, tau)
        k3 = self.f_expl(x + 0.5*dt*k2, tau)
        k4 = self.f_expl(x + dt*k3, tau)
        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        self.f_disc_rk4 = cs.Function('f_disc_wrench_rk4', [x, tau, dt], [x_next])

    # Convenience
    def calculate_f_expl(self, x, tau):
        return np.array(self.f_expl(x, tau)).reshape((12,))

    def calculate_f_disc(self, x, tau, dt):
        return np.array(self.f_disc_rk4(x, tau, dt)).reshape((12,))