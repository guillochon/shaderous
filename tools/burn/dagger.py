"""One round of dataset aggregation (DAgger, Ross et al. 2011) for the burning surrogate.

A model trained only on reference trajectories, applied over and over, drifts off them into
states it has never seen, where it errs more and drifts further. This rolls the current
model out the way the simulator would (train.roll: steps of a fixed dt, the temperature
following the energy released), from the starts of the training trajectories with steps of
1e-8 to 1e-2 s, collects states it visits, and burns each with the reference network
(burner.py). The new trajectories, written to data/dagger_<round>.npz, teach the next model
the way back from its own mistakes.

    tools/burn/.venv/Scripts/python tools/burn/dagger.py <model.pt> <round> [states]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

import train
from gen_data import TIMES


def burn_one(args):
    import burner
    rho, T, X = args
    try:
        Xs, Ts, _ = burner.burn(X, rho, T, TIMES[-1], t_eval=TIMES)
    except Exception:  # noqa: BLE001
        return None
    return rho, Xs.T, Ts


def main():
    ck = torch.load(train.HERE / sys.argv[1])
    rnd = int(sys.argv[2])
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    model = train.Burner(ck["width"], ck.get("depth", 3)).to(train.dev)
    model.load_state_dict(ck["state"])
    model.eval()
    rho, X, T, t = train.load()
    rng = np.random.default_rng(100 + rnd)
    pick = rng.choice(len(rho), size=min(want, len(rho)), replace=False)
    # Start anywhere along the trajectory, with a step from 1e-8 to 1e-2 s, for up to 400 steps.
    k0 = rng.integers(0, X.shape[1] - 1, size=len(pick))
    dt = 10 ** rng.uniform(-8, -2, size=len(pick))
    t0 = time.time()
    _, _, seen = train.roll(model, rho[pick], X[pick, k0], T[pick, k0], dt, 400, record=8)
    r = torch.cat([s[0] for s in seen]).cpu().numpy()
    Ts = torch.cat([s[1] for s in seen]).cpu().numpy()
    Xs = torch.cat([s[2] for s in seen]).cpu().numpy().astype(np.float64)
    # Keep states the reference can burn (its rates end at 1e10 K), one per rollout on average.
    ok = (Ts > 1e8) & (Ts < 1.2e10) & np.isfinite(Xs).all(axis=1)
    idx = np.flatnonzero(ok)
    idx = rng.choice(idx, size=min(want, len(idx)), replace=False)
    Xs = np.clip(Xs[idx], 0, None)
    Xs /= Xs.sum(axis=1, keepdims=True)
    jobs = list(zip(r[idx].astype(float), Ts[idx].astype(float), Xs))
    print(f"rolled out in {time.time() - t0:.0f} s; burning {len(jobs)} visited states", flush=True)
    out = []
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for k, res in enumerate(pool.imap_unordered(burn_one, jobs, chunksize=4)):
            if res is not None:
                out.append(res)
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(
        train.DATA / f"dagger_{rnd}.npz",
        rho=np.array([o[0] for o in out]), X=np.stack([o[1] for o in out]), T=np.stack([o[2] for o in out]), t=TIMES,
    )
    print(f"round {rnd}: {len(out)} trajectories in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
