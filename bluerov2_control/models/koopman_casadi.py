#!/usr/bin/env python3
import casadi as cs
import numpy as np

class KoopmanEDMDcCasadi:
    """
    Discrete-time Koopman EDMDc with RBF lifting:
        z_{k+1} = A z_k + B u_k
        x_k     = first 12 components of z_k  (identity decoder)
    Weights loaded from training .npz file.
    """

    def __init__(self, weights_path: str):
        W = np.load(weights_path)
        self.A     = cs.DM(W["A"])         # (d,d)
        self.B     = cs.DM(W["B"])         # (d,8)
        self.Cent  = cs.DM(W["centers"])   # (k,12)
        self.gamma = float(W["gamma"][0])
        self.n     = int(W["state_dim"][0])  # 12
        self.r     = int(W["input_dim"][0])  # 8
        self.k     = int(self.Cent.shape[0]) # #RBFs
        self.d     = int(self.n + self.k)    # lift dim
        self.create()

    def _rbf_vec(self, x):
        """
        Robust RBF feature vector for a single state x (12x1 SX/DM) -> (kx1 SX).
        Avoids row/column broadcasting issues.
        """
        # Ensure column shape
        if x.size2() != 1 or x.size1() != self.n:
            x = cs.reshape(x, self.n, 1)
        feats = []
        # centers stored as (k,12); take each as column (12x1)
        for j in range(self.k):
            cj = cs.reshape(self.Cent[j, :].T, self.n, 1)  # (12x1) DM
            d  = x - cj                                    # (12x1) SX
            # scalar squared norm
            s  = cs.mtimes(d.T, d)                         # (1x1)
            feats.append(cs.exp(-self.gamma * s))          # (1x1)
        return cs.vertcat(*feats)                           # (kx1)

    def _lift(self, x):
        # x: (12x1) → z: (d x 1)
        phi = self._rbf_vec(x)            # (k x 1)
        return cs.vertcat(x, phi)         # (12+k x 1)

    def create(self):
        x  = cs.SX.sym('x', 12)           # column (12x1)
        u8 = cs.SX.sym('u', 8)            # column (8x1)

        z   = self._lift(x)               # (d x 1)
        zn  = self.A @ z + self.B @ u8    # (d x 1)
        x_n = zn[0:self.n]                # (12 x 1)

        self.f_disc = cs.Function('koopman_step', [x, u8], [x_n])

    # Convenience
    def calculate_f_disc(self, x, u):
        return np.array(self.f_disc(x, u)).reshape((12,))