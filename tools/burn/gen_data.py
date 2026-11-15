"""Training trajectories for the burning surrogate: self-heating burns (burner.py) from many
starting states, each recorded at log-spaced times from 1e-9 s to 1 s. Any two times along
a trajectory make a training pair (the burn is a function of the state alone), so each one
gives many pairs of every step length.

Starts:
  fuel      C/O of any ratio, pure He, He on C/O, O/Ne; any density and temperature
  mixtures  states from fuel trajectories mixed with fresh fuel and shifted in temperature,
            as hydrodynamics mixes ash and fuel at a front

    tools/burn/.venv/Scripts/python tools/burn/gen_data.py fuel 4000
    tools/burn/.venv/Scripts/python tools/burn/gen_data.py mix 4000
Writes tools/burn/data/<kind>_<seed>.npz.
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
TIMES = np.concatenate([[0.0], np.logspace(-9, 0, 73)])
RHO = (1e4, 1e10)
T_FUEL = (1e8, 6e9)


def workers():
    """Processes for a pool of burns: the CPUs this process may use, less two. In a container
    os.cpu_count() is the host's (96 on a Runpod pod with a quota of 16): that many workers,
    each compiling the burner first (about 35 s of CPU), held the whole quota for minutes."""
    n = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        n = min(n, len(os.sched_getaffinity(0)))
    try:  # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1 (some hosts): a quota of -1 is none
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            n = min(n, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, n - 2)


def fuel(rng):
    import burner
    kind = rng.choice(["co", "co", "co", "he", "heco", "one"])
    if kind == "co":
        c = rng.uniform(0.2, 0.8)
        return burner.composition(c12=c, o16=1 - c)
    if kind == "he":
        return burner.composition(he4=1.0)
    if kind == "heco":
        h = 10 ** rng.uniform(-3, 0)
        return burner.composition(he4=h, c12=(1 - h) / 2, o16=(1 - h) / 2)
    return burner.composition(o16=0.6, ne20=0.35, mg24=0.05)


def log_uniform(rng, lo, hi):
    return 10 ** rng.uniform(np.log10(lo), np.log10(hi))


def run(args):
    kind, seed, pool_file = args
    import burner
    rng = np.random.default_rng(seed)
    if kind == "fuel":
        X0, rho, T0 = fuel(rng), log_uniform(rng, *RHO), log_uniform(rng, *T_FUEL)
    else:
        p = np.load(pool_file)
        i = rng.integers(len(p["rho"]))
        k = rng.integers(1, p["X"].shape[1])
        w = rng.uniform(0, 1)
        X0 = w * p["X"][i, k] + (1 - w) * fuel(rng)
        X0 /= X0.sum()
        rho = p["rho"][i] * 10 ** rng.uniform(-0.5, 0.5)
        T0 = float(np.clip(p["T"][i, k] * 10 ** rng.uniform(-0.3, 0.1), 1e8, 9.5e9))
    try:
        X, T, _ = burner.burn(X0, rho, T0, TIMES[-1], t_eval=TIMES)
    except Exception as e:  # noqa: BLE001
        return None
    return rho, X.T.astype(np.float64), T


def main():
    kind, n = sys.argv[1], int(sys.argv[2])
    DATA.mkdir(exist_ok=True)
    pool_file = DATA / "fuel_pool.npz"
    base = {"fuel": 0, "mix": 10_000_000}[kind] + int(time.time()) % 1_000_000 * 0
    seeds = [base + i for i in range(n)]
    t0 = time.time()
    out = []
    with Pool(workers()) as pool:
        for k, r in enumerate(pool.imap_unordered(run, [(kind, s, pool_file) for s in seeds], chunksize=4)):
            if r is not None:
                out.append(r)
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{n} in {time.time() - t0:.0f} s, {len(out)} ok", flush=True)
    rho = np.array([r[0] for r in out])
    X = np.stack([r[1] for r in out])
    T = np.stack([r[2] for r in out])
    np.savez_compressed(DATA / f"{kind}.npz", rho=rho, X=X, T=T, t=TIMES)
    if kind == "fuel":
        np.savez_compressed(pool_file, rho=rho, X=X, T=T)
    print(f"{kind}: {len(out)} trajectories in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
