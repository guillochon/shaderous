"""Reference trajectories from fuel with a trace of He: C/O of any ratio (half of it with a trace
of Ne20 as well) and O/Ne with a little C12, with He from 3e-5 to 3%, at 1e5 to 5e9 g/cm^3 and
3.2e8 to 1.4e9 K. Early models burned that He into Ne20 too fast and overheated the gas; the
general fuel set (gen_data.py) has too few of these starts to teach them.

    tools/burn/.venv/Scripts/python tools/burn/gen_tracehe.py [count]
Writes tools/burn/data/tracehe.npz.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import train
from dagger import burn_one
from gen_data import TIMES


def starts(n, rng):
    jobs = []
    for _ in range(n):
        X = np.zeros(13)
        if rng.random() < 0.75:  # C/O
            c = rng.uniform(0.2, 0.8)
            X[1], X[2] = c, 1 - c
            if rng.random() < 0.5:
                X[3] = 10 ** rng.uniform(-7, -2)  # trace Ne20
        else:  # O/Ne
            o = rng.uniform(0.5, 0.8)
            X[2], X[3] = o, 1 - o
            X[1] = 10 ** rng.uniform(-4, -1.3)
        X[0] = 10 ** rng.uniform(-4.5, -1.5)
        X /= X.sum()
        jobs.append((10 ** rng.uniform(5, 9.7), 10 ** rng.uniform(8.5, 9.15), X))
    return jobs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    jobs = starts(n, np.random.default_rng(7))
    t0 = time.time()
    out = []
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for k, res in enumerate(pool.imap_unordered(burn_one, jobs, chunksize=4)):
            if res is not None:
                out.append(res)
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{n} in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(
        train.DATA / "tracehe.npz",
        rho=np.array([o[0] for o in out]), X=np.stack([o[1] for o in out]), T=np.stack([o[2] for o in out]), t=TIMES,
    )
    print(f"{len(out)} trajectories in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
