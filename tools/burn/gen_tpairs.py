"""Pairs of reference trajectories from the same state at two temperatures 1% apart, for
training how a step responds to temperature (patch.py): a model can fit the steps and still
get that slope wrong, and the slope is what sets when simmering gas runs away. Starts are
states along the training trajectories (and the targeted ones, gen_targets.py).

    tools/burn/.venv/Scripts/python tools/burn/gen_tpairs.py [pairs]
Writes tools/burn/tpairs.npz (rho, and X and T for the cooler and the hotter start; not in
data/, where every training run would read each as an ordinary trajectory).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import train
from dagger import burn_one
from gen_data import TIMES

EPS = 0.01


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    rng = np.random.default_rng(13)
    general = train.load({f.name for f in train.DATA.glob("*.npz")} - {"targets.npz", "fuel_pool.npz"})
    targets = train.load({"targets.npz"})
    starts = []
    for _ in range(n):
        rho, X, T, _ = targets if rng.random() < 0.25 else general
        i, k = rng.integers(len(rho)), rng.integers(X.shape[1] - 1)
        x = np.clip(X[i, k], 0.0, None)
        starts.append((rho[i], T[i, k], x / x.sum()))
    jobs = [(r, T * f, x) for r, T, x in starts for f in (1.0, 1.0 + EPS)]
    t0 = time.time()
    out = []
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for k, res in enumerate(pool.imap(burn_one, jobs, chunksize=4)):
            out.append(res)
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
    ok = [k for k in range(n) if out[2 * k] is not None and out[2 * k + 1] is not None]
    np.savez_compressed(
        train.HERE / "tpairs.npz", rho=np.array([out[2 * k][0] for k in ok]), t=TIMES,
        X0=np.stack([out[2 * k][1] for k in ok]), T0=np.stack([out[2 * k][2] for k in ok]),
        X1=np.stack([out[2 * k + 1][1] for k in ok]), T1=np.stack([out[2 * k + 1][2] for k in ok]),
    )
    print(f"{len(ok)} pairs in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
