#!/usr/bin/env python3
"""
test_casadi.py (robust paths)

Search order for weights:
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
import casadi as cs

from models.fossen_casadi import FossenCasadi
from models.di_casadi import DoubleIntegratorCasadi
from models.koopman_casadi import KoopmanEDMDcCasadi


def finite(arr: np.ndarray) -> bool:
    return np.all(np.isfinite(arr))

def random_state_in_tank(rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(12, dtype=float)
    x[0] = rng.uniform(0.5, 8.5)
    x[1] = rng.uniform(-2.0, 2.0)
    x[2] = rng.uniform(-2.5, -0.2)  # NED
    x[3] = rng.uniform(-0.1, 0.1)
    x[4] = rng.uniform(-0.1, 0.1)
    x[5] = rng.uniform(-np.pi, np.pi)
    x[6:9]  = rng.uniform(-0.05, 0.05, size=3)
    x[9:12] = rng.uniform(-0.05, 0.05, size=3)
    return x

def random_controls(rng: np.random.Generator, steps: int, scale: float = 0.5) -> np.ndarray:
    return rng.uniform(-scale, scale, size=(steps, 8)).astype(float)

# ---------- Path resolution ----------
def find_upwards(start: Path, relative: Path) -> Path | None:
    for p in [start, *start.parents]:
        cand = p / relative
        if cand.exists():
            return cand
    return None

def resolve_weight(path_str: str, filename: str) -> Path:
    # 0) direct
    p = Path(path_str)
    if p.exists():
        return p

    here = Path(__file__).resolve().parent
    cwd  = Path.cwd()
    env_root = os.environ.get("BLUEROV2_MPC_ROOT")

    candidates = [
        here / "models" / "weights" / filename,
        cwd  / "models" / "weights" / filename,
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

    # Pick first existing
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not locate weights '{filename}'. Tried:\n" +
        "\n".join(str(c) for c in candidates) +
        ("\nSet BLUEROV2_MPC_ROOT or pass --koopman/--di with an absolute path.")
    )

# ---------- Tests ----------
def test_fossen(dt: float, steps: int, rng: np.random.Generator) -> None:
    print("\n[1/3] FossenCasadi")
    model = FossenCasadi()
    x = random_state_in_tank(rng)
    u_seq = random_controls(rng, steps)
    dx = np.array(model.f_expl(x, u_seq[0])).reshape((12,))
    assert dx.shape == (12,) and finite(dx)
    for k in range(steps):
        x = np.array(model.f_disc_rk4(x, u_seq[k], dt)).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ f_expl and RK4 steps OK")

def test_di(di_path: Path, dt: float, steps: int, rng: np.random.Generator) -> None:
    print("\n[2/3] DoubleIntegratorCasadi")
    print(f"  using: {di_path}")
    model = DoubleIntegratorCasadi(str(di_path))
    x = random_state_in_tank(rng)
    u_seq = random_controls(rng, steps)
    x1 = np.array(model.f_disc(x, u_seq[0], dt)).reshape((12,))
    assert x1.shape == (12,) and finite(x1)
    for k in range(steps):
        x = np.array(model.f_disc(x, u_seq[k], dt)).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ discrete steps OK")

def test_koopman(koop_path: Path, steps: int, rng: np.random.Generator) -> None:
    print("\n[3/3] KoopmanEDMDcCasadi")
    print(f"  using: {koop_path}")
    model = KoopmanEDMDcCasadi(str(KOOP := koop_path))
    x = random_state_in_tank(rng)
    u_seq = random_controls(rng, steps)
    x1 = np.array(model.f_disc(x, u_seq[0])).reshape((12,))
    assert x1.shape == (12,) and finite(x1)
    for k in range(steps):
        x = np.array(model.f_disc(x, u_seq[k])).reshape((12,))
        assert x.shape == (12,) and finite(x)
    print("  ✓ discrete steps OK")

def main():
    parser = argparse.ArgumentParser(description="Smoke-test CasADi BlueROV2 models (robust paths).")
    parser.add_argument("--koopman", type=str, default="models/weights/koopman_edmdc_weights.npz",
                        help="Koopman weights (.npz) or hint path.")
    parser.add_argument("--di", type=str, default="models/weights/double_integrator_weights.npz",
                        help="DI weights (.npz) or hint path.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve actual files
    koop_file = resolve_weight(args.koopman, "koopman_edmdc_weights.npz")
    di_file   = resolve_weight(args.di,      "double_integrator_weights.npz")

    rng = np.random.default_rng(args.seed)

    test_fossen(dt=args.dt, steps=args.steps, rng=rng)
    test_di(di_path=di_file, dt=args.dt, steps=args.steps, rng=rng)
    test_koopman(koop_path=koop_file, steps=args.steps, rng=rng)

    print("\nAll three CasADi models executed successfully.")

if __name__ == "__main__":
    main()