"""Reference trajectories in the regime of Kushnir's step-size test (2026-11): hot carbon-oxygen
behind a detonation front and its silicon-sulfur ash, where the surrogate's X(Ni56) depends on
the step (stepscore.py).

  co   C/O of carbon fraction 0.3 to 0.7, a little He, at 10^6.5 to 10^8 g/cm^3 and 2.5e9 to 5e9 K
  sis  Si/S ash with some O, C, Ar, Ca and Ni at 10^6.5 to 10^8.5 g/cm^3 and 3.5e9 to 5.5e9 K

    tools/burn/.venv/Scripts/python tools/burn/gen_kushnir.py [count per kind]
Writes tools/burn/data/kushnir.npz (training; stepscore.py's held-out starts come from another
seed).
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

import train
from dagger import burn_one, init
from gen_data import TIMES, workers
from gen_targets import mix

HE, C, O, NE, MG, SI, S, AR, CA, TI, CR, FE, NI = range(13)
KINDS = ["co", "sis"]


def start(kind, rng):
    if kind == "co":
        c = rng.uniform(0.3, 0.7)
        X = mix(rng, [(C, (c, c)), (O, (1 - c, 1 - c)), (HE, (1e-6, 1e-2))])
        return 10 ** rng.uniform(6.5, 8.0), 10 ** rng.uniform(np.log10(2.5e9), np.log10(5e9)), X
    X = mix(rng, [(SI, (0.2, 0.6)), (S, (0.1, 0.35)), (AR, (0.01, 0.1)), (CA, (0.01, 0.1)), (O, (1e-3, 0.2)),
                  (C, (1e-4, 0.1)), (HE, (1e-5, 0.02)), (NI, (1e-4, 0.2))])
    return 10 ** rng.uniform(6.5, 8.5), 10 ** rng.uniform(np.log10(3.5e9), np.log10(5.5e9)), X


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    rng = np.random.default_rng(23)
    jobs = [start(kind, rng) for _ in range(n) for kind in KINDS]
    t0 = time.time()
    out = [None] * len(jobs)
    with Pool(workers(), initializer=init) as pool:
        for k, res in enumerate(pool.imap(burn_one, jobs, chunksize=4)):
            out[k] = res
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
    sel = [(k, o) for k, o in enumerate(out) if o is not None]
    np.savez_compressed(train.HERE / "data" / "kushnir.npz",
                        rho=np.array([o[0] for _, o in sel]), X=np.stack([o[1] for _, o in sel]),
                        T=np.stack([o[2] for _, o in sel]), t=TIMES, kind=np.array([k % len(KINDS) for k, _ in sel]))
    print(f"data/kushnir.npz: {len(sel)} of {len(jobs)} trajectories; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
