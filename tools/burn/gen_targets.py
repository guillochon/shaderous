"""Reference trajectories around the states every surrogate so far gets wrong, for patching a
trained model (patch.py):

  oneb  dense O/Ne (a little C12, Mg24 and He) at 3e9 to 2e10 g/cm^3 and 7e8 to 2e9 K: the He
        goes into Mg24 by Ne20 capture and the gas simmers near 1.9e9 K before running away;
        models put the He into Ne20 instead and never ignite.
  sis   Si/S ash with some O, C and Ar/Ca at 1e6 to 1e8 g/cm^3 and 3e9 to 4.5e9 K, which
        models burn to Ni56 three times too fast.
  hotco C/O (some He, Si and S) at 1e4 to 3e5 g/cm^3 and 3.5e9 to 5e9 K, where photo-
        disintegration keeps a third of the mass in He; models make too much He and Ni56.
  nse   iron-group mixtures with Si to Ca and some He at 1e7 to 1e10 g/cm^3 and 4.5e9 to
        7e9 K, near equilibrium.

    tools/burn/.venv/Scripts/python tools/burn/gen_targets.py [count per kind]
Writes tools/burn/data/targets.npz (training) and tools/burn/targets_test.npz (every tenth
start, held out; not in data/, so no training run reads it).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import train
from dagger import burn_one
from gen_data import TIMES

HE, C, O, NE, MG, SI, S, AR, CA, TI, CR, FE, NI = range(13)
KINDS = ["oneb", "sis", "hotco", "nse"]


def mix(rng, parts):
    """Mass fractions with each (species, (lo, hi)) drawn log-uniform, normalized."""
    X = np.zeros(13)
    for i, (lo, hi) in parts:
        X[i] += 10 ** rng.uniform(np.log10(lo), np.log10(hi))
    return X / X.sum()


def start(kind, rng):
    if kind == "oneb":
        X = mix(rng, [(O, (0.4, 0.7)), (NE, (0.2, 0.45)), (MG, (1e-3, 0.08)), (C, (1e-5, 0.03)), (HE, (1e-5, 0.06))])
        return 10 ** rng.uniform(9.5, 10.3), 10 ** rng.uniform(8.85, 9.3), X
    if kind == "sis":
        X = mix(rng, [(SI, (0.2, 0.6)), (S, (0.1, 0.35)), (AR, (0.01, 0.1)), (CA, (0.01, 0.1)), (O, (1e-3, 0.15)),
                      (C, (1e-4, 0.1)), (HE, (1e-5, 0.02)), (NI, (1e-5, 0.1))])
        return 10 ** rng.uniform(6.0, 8.0), 10 ** rng.uniform(9.48, 9.65), X
    if kind == "hotco":
        c = rng.uniform(0.3, 0.7)
        X = mix(rng, [(C, (c, c)), (O, (1 - c, 1 - c)), (HE, (1e-5, 0.3)), (SI, (1e-4, 0.1)), (S, (1e-4, 0.06))])
        return 10 ** rng.uniform(4.0, 5.5), 10 ** rng.uniform(9.54, 9.7), X
    X = mix(rng, [(NI, (0.1, 0.8)), (FE, (0.01, 0.2)), (CR, (1e-3, 0.05)), (SI, (1e-3, 0.3)), (S, (1e-3, 0.2)),
                  (AR, (1e-3, 0.08)), (CA, (1e-3, 0.1)), (HE, (1e-4, 0.3))])
    return 10 ** rng.uniform(7.0, 10.0), 10 ** rng.uniform(9.65, 9.85), X


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(11)
    jobs = [start(kind, rng) for _ in range(n) for kind in KINDS]
    t0 = time.time()
    out = [None] * len(jobs)
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for k, res in enumerate(pool.imap(burn_one, jobs, chunksize=4)):
            out[k] = res
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
    for name, keep in (("data/targets.npz", lambda k: k % 10 != 9), ("targets_test.npz", lambda k: k % 10 == 9)):
        sel = [(k, o) for k, o in enumerate(out) if o is not None and keep(k // len(KINDS))]
        np.savez_compressed(
            train.HERE / name,
            rho=np.array([o[0] for _, o in sel]), X=np.stack([o[1] for _, o in sel]), T=np.stack([o[2] for _, o in sel]),
            t=TIMES, kind=np.array([k % len(KINDS) for k, _ in sel]),
        )
        print(f"{name}: {len(sel)} trajectories")
    print(f"{sum(o is None for o in out)} failed; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
