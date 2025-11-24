#!/usr/bin/env python3
"""
test_casadi.py (thruster + wrench variants, robust paths)

Models tested:

  Thruster-based (8D inputs):
    - FossenCasadi                    (continuous + RK4)
    - DoubleIntegratorCasadi          (learned DI, 8D u)
    - KoopmanEDMDcCasadi              (Koopman, 8D u)

  Wrench-based (6D inputs: [Fx,Fy,Fz,Mx,My,Mz]):
    - FossenWrenchCasadi              (continuous + RK4, direct wrench)
    - DoubleIntegratorWrenchCasadi    (learned DI on wrench)
    - AnalyticDoubleIntegratorWrenchCasadi  (rigid-body DI, no damping)
    - KoopmanEDMDcWrenchCasadi        (Koopman, 6D wrench)

Search order for weight files:
  1) CLI path as given
  2) <this_file>/models/weights/<name>.npz
  3) CWD/models/weights/<name>.npz
  4) Any parent of this_file containing models/weights/<name>.npz
  5) <BLUEROV2_MPC_ROOT>/models/weights/<name>.npz  (env var)
  6) ros_ws/src/bluerov2_mpc/models/weights/<name>.npz  (from any parent)
"""

import argparse
import os
from pathlib import Path
import numpy as np

from models.fossen_casadi import FossenCasadi
from models.di_casadi import DoubleIntegratorCasadi
from models.koopman_casadi import KoopmanEDMDcCasadi

from models.fossen_wrench_casadi import FossenWrenchCasadi
from models.di_wrench_casadi import DoubleIntegratorWrenchCasadi
from models.di_ana_wrench_casadi import AnalyticDoubleIntegratorWrenchCasadi
from models.koopman_wrench_casadi import KoopmanEDMDcWrenchCasadi


# ---------- Small helpers ----------
def finite(arr: np.ndarray) -> bool:
    return np.all(np.isfinite(arr))


def random_state_in_tank(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a plausible 12D state inside the tank, with small velocities/rates.
    NED: x forward [0,9], y right [-2.5,2.5], z down [0,-3].
    """
    x = np.zeros(12, dtype=float)
    x[0] = rng.uniform(0.5, 8.5)
    x[1] = rng.uniform(-2.0, 2.0)
    x[2] = rng.uniform(-2.5, -0.2)  # NED z (down)
    x[3] = rng.uniform(-0.1, 0.1)
    x[4] = rng.uniform(-0.1, 0.1)
    x[5] = rng.uniform(-np.pi, np.pi)
    x[6:9] = rng.uniform(-0.05, 0.05, size=3)
    x[9:12] = rng.uniform(-0.05, 0.05, size=3)
    return x


def random_thruster_controls(rng: np.random.Generator,
                             steps: int,
                             scale: float = 0.5) -> np.ndarray:
    """
    Random 8D thruster commands in [-scale, scale].
    """
    return rng.uniform(-scale, scale, size=(steps, 8)).astype(float)


def random_wrench_controls(rng: np.random.Generator,
                           steps: int,
                           F_scale: float = 20.0,
                           M_scale: float = 5.0) -> np.ndarray:
    """
    Random 6D body-frame wrench [Fx,Fy,Fz,Mx,My,Mz].

    The scales are arbitrary but chosen so that accelerations are moderate.
    """
    tau = np.zeros((steps, 6), dtype=float)
    tau[:, 0:3] = rng.uniform(-F_scale, F_scale, size=(steps, 3))  # forces
    tau[:, 3:6] = rng.uniform(-M_scale, M_scale, size=(steps, 3))  # moments
    return tau


# ---------- Path resolution ----------
def find_upwards(start: Path, relative: Path) -> Path | None:
    for p in [start, *start.parents]:
        cand = p / relative
        if cand.exists():
            return cand
    return None


def resolve_weight(path_str: str, filename: str) -> Path:
    """
    Resolve a weight file, given a user hint path and a default filename.
    """
    # 0) direct path
    p = Path(path_str)
    if p.exists():
        return p

    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    env_root = os.environ.get("BLUEROV2_MPC_ROOT")

    candidates = [
        here / "models" / "weights" / filename,
        cwd / "models" / "weights" / filename,
    ]

    # Any parent containing models/weights/<filename>
    up = find_upwards(here, Path("models/weights") / filename)
    if up is not None:
        candidates.append(up)

    # Env root
    if env_root:
        candidates.append(Path(env_root) / "models" / "weights" / filename)

    # ROS-style tree anywhere above
    ros_like = find_upwards(here, Path("ros_ws/src/bluerov2_mpc/models/weights") / filename)
    if ros_like is not None:
        candidates.append(ros_like)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not locate weights '{filename}'. Tried:\n"
        + "\n".join(str(c) for c in candidates)
        + "\nSet BLUEROV2_MPC_ROOT or pass an absolute path."
    )


# ---------- Tests: thruster-based ----------
def test_fossen_thruster(dt: float, steps: int, rng: np.random.Generator) -> None:
    print("\n[1/7] FossenCasadi (thruster, 8D)")
    model = FossenCasadi()
    x = random_state_in_tank(rng)
    u_seq = random_thruster_controls(rng, steps)
    dx = np.array(model.f_expl(x, u_seq[0])).reshape((12,))
    assert dx.shape == (12,) and finite(dx)
    for k in range(steps):
        x = np.array(model.f_disc_rk4(x, u_seq[k], dt)).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ f_expl and RK4 steps OK")


def test_di_thruster(di_path: Path,
                     dt: float,
                     steps: int,
                     rng: np.random.Generator) -> None:
    print("\n[2/7] DoubleIntegratorCasadi (thruster, learned 8D)")
    print(f"  using: {di_path}")
    model = DoubleIntegratorCasadi(str(di_path))
    x = random_state_in_tank(rng)
    u_seq = random_thruster_controls(rng, steps)
    x1 = np.array(model.f_disc(x, u_seq[0], dt)).reshape((12,))
    assert x1.shape == (12,) and finite(x1)
    for k in range(steps):
        x = np.array(model.f_disc(x, u_seq[k], dt)).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ discrete steps OK")


def test_koopman_thruster(koop_path: Path,
                          steps: int,
                          rng: np.random.Generator) -> None:
    print("\n[3/7] KoopmanEDMDcCasadi (thruster, 8D)")
    print(f"  using: {koop_path}")
    model = KoopmanEDMDcCasadi(str(koop_path))
    x = random_state_in_tank(rng)
    u_seq = random_thruster_controls(rng, steps)
    x1 = np.array(model.f_disc(x, u_seq[0])).reshape((12,))
    assert x1.shape == (12,) and finite(x1)
    for k in range(steps):
        x = np.array(model.f_disc(x, u_seq[k])).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ discrete steps OK")


# ---------- Tests: wrench-based ----------
def test_fossen_wrench(dt: float,
                       steps: int,
                       rng: np.random.Generator) -> None:
    print("\n[4/7] FossenWrenchCasadi (direct wrench, 6D)")
    model = FossenWrenchCasadi()
    x = random_state_in_tank(rng)
    tau_seq = random_wrench_controls(rng, steps)
    dx = np.array(model.f_expl(x, tau_seq[0])).reshape((12,))
    assert dx.shape == (12,) and finite(dx)
    for k in range(steps):
        x = np.array(model.f_disc_rk4(x, tau_seq[k], dt)).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ f_expl_wrench and RK4 steps OK")


def test_di_wrench_learned(di_wrench_path: Path,
                           dt: float,
                           steps: int,
                           rng: np.random.Generator) -> None:
    print("\n[5/7] DoubleIntegratorWrenchCasadi (learned, 6D)")
    print(f"  using: {di_wrench_path}")
    model = DoubleIntegratorWrenchCasadi(str(di_wrench_path))
    x = random_state_in_tank(rng)
    tau_seq = random_wrench_controls(rng, steps)
    x1 = np.array(model.f_disc(x, tau_seq[0], dt)).reshape((12,))
    assert x1.shape == (12,) and finite(x1)
    for k in range(steps):
        x = np.array(model.f_disc(x, tau_seq[k], dt)).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ discrete steps OK")


def test_di_wrench_analytic(dt: float,
                            steps: int,
                            rng: np.random.Generator) -> None:
    print("\n[6/7] AnalyticDoubleIntegratorWrenchCasadi (rigid-body DI, 6D)")
    model = AnalyticDoubleIntegratorWrenchCasadi()
    x = random_state_in_tank(rng)
    tau_seq = random_wrench_controls(rng, steps)
    x1 = np.array(model.f_disc(x, tau_seq[0], dt)).reshape((12,))
    assert x1.shape == (12,) and finite(x1)
    for k in range(steps):
        x = np.array(model.f_disc(x, tau_seq[k], dt)).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ discrete steps OK")


def test_koopman_wrench(koop_wrench_path: Path,
                        steps: int,
                        rng: np.random.Generator) -> None:
    print("\n[7/7] KoopmanEDMDcWrenchCasadi (Koopman, 6D)")
    print(f"  using: {koop_wrench_path}")
    model = KoopmanEDMDcWrenchCasadi(str(koop_wrench_path))
    x = random_state_in_tank(rng)
    tau_seq = random_wrench_controls(rng, steps)
    x1 = np.array(model.f_disc(x, tau_seq[0])).reshape((12,))
    assert x1.shape == (12,) and finite(x1)
    for k in range(steps):
        x = np.array(model.f_disc(x, tau_seq[k])).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ discrete steps OK")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test CasADi BlueROV2 models (thruster + wrench variants, robust paths)."
    )
    # Thruster-based weights
    parser.add_argument(
        "--koopman",
        type=str,
        default="models/weights/koopman_edmdc_weights.npz",
        help="Koopman 8D (thruster) weights (.npz) or hint path.",
    )
    parser.add_argument(
        "--di",
        type=str,
        default="models/weights/double_integrator_weights.npz",
        help="DI 8D (thruster) weights (.npz) or hint path.",
    )

    # Wrench-based weights
    parser.add_argument(
        "--koopman-wrench",
        type=str,
        default="models/weights/koopman_edmdc_wrench_weights.npz",
        help="Koopman 6D (wrench) weights (.npz) or hint path.",
    )
    parser.add_argument(
        "--di-wrench",
        type=str,
        default="models/weights/double_integrator_wrench_weights.npz",
        help="DI 6D (wrench) weights (.npz) or hint path.",
    )

    parser.add_argument("--steps", type=int, default=20, help="Horizon length for test rollouts.")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step for discrete tests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # Resolve weight files
    koop_thr_file = resolve_weight(args.koopman, "koopman_edmdc_weights.npz")
    di_thr_file = resolve_weight(args.di, "double_integrator_weights.npz")

    koop_wrench_file = resolve_weight(
        args.koopman_wrench, "koopman_edmdc_wrench_weights.npz"
    )
    di_wrench_file = resolve_weight(
        args.di_wrench, "double_integrator_wrench_weights.npz"
    )

    rng = np.random.default_rng(args.seed)

    # Thruster-based models
    test_fossen_thruster(dt=args.dt, steps=args.steps, rng=rng)
    test_di_thruster(di_path=di_thr_file, dt=args.dt, steps=args.steps, rng=rng)
    test_koopman_thruster(koop_path=koop_thr_file, steps=args.steps, rng=rng)

    # Wrench-based models
    test_fossen_wrench(dt=args.dt, steps=args.steps, rng=rng)
    test_di_wrench_learned(
        di_wrench_path=di_wrench_file, dt=args.dt, steps=args.steps, rng=rng
    )
    test_di_wrench_analytic(dt=args.dt, steps=args.steps, rng=rng)
    test_koopman_wrench(
        koop_wrench_path=koop_wrench_file, steps=args.steps, rng=rng
    )

    print("\nAll seven CasADi models executed successfully.")


if __name__ == "__main__":
    main()