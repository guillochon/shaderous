"""Training data for the electron-positron network: h and its two first derivatives
(electron_gas.target) on a grid in log10(rho Ye) and log10 T, and at random points between,
over the domain the network covers (DOMAIN). Below-normal priority, all cores but two.

    tools/burn/.venv/Scripts/python tools/eos/gen_data.py [step_dex] [random_points]

Writes tools/eos/data/eos_grid.npz (columns: log10 rho Ye, log10 T, h, dh/dln n, dh/dln T,
eta, positrons per net electron) and eos_random.npz.
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psutil

import electron_gas as eg

HERE = Path(__file__).parent
DOMAIN = {"log_rho_ye": (0.0, 11.0), "log_T": (4.0, 11.0)}


def one(p):
    lr, lt = p
    try:
        return (lr, lt) + eg.target(10.0 ** lr / eg.MU, 10.0 ** lt)
    except Exception as e:   # noqa: BLE001 - a failed point is dropped, and counted
        return (lr, lt) + (np.nan,) * 5


def init():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def run(points, name):
    t0 = time.time()
    with Pool(max(1, os.cpu_count() - 2), initializer=init) as pool:
        rows = pool.map(one, points, chunksize=16)
    a = np.array(rows, dtype=np.float64)
    bad = ~np.isfinite(a).all(axis=1)
    (HERE / "data").mkdir(exist_ok=True)
    np.savez(HERE / "data" / name, data=a[~bad])
    print(f"{name}: {len(a) - bad.sum()} points ({bad.sum()} failed) in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    init()
    step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    nrand = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    (r0, r1), (t0, t1) = DOMAIN["log_rho_ye"], DOMAIN["log_T"]
    lr = np.arange(r0, r1 + 1e-9, step)
    lt = np.arange(t0, t1 + 1e-9, step)
    run([(a, b) for a in lr for b in lt], "eos_grid.npz")
    rng = np.random.default_rng(1)
    run(list(zip(rng.uniform(r0, r1, nrand), rng.uniform(t0, t1, nrand))), "eos_random.npz")
