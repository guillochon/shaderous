"""One round of dataset aggregation (DAgger, Ross et al. 2011) for the burning surrogate.

A model trained only on reference trajectories, applied over and over, drifts off them into
states it has never seen, where it errs more and drifts further. This rolls the current
model out the way the simulator would (train.roll: steps of a fixed dt, the temperature
following the energy released), from the starts of the training trajectories with steps of
1e-8 to 1e-2 s, collects states it visits (most of them from rollouts that ended far from the reference),
and burns each with the reference network
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
    rnd = int(sys.argv[2])
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    model = train.load_model(sys.argv[1])
    rho, X, T, t = train.load()
    rng = np.random.default_rng(100 + rnd)
    # Many rollouts, cheap on the GPU: from a trajectory's start (as fresh gas meets the
    # simulator) or anywhere along it, with a step from 1e-8 to 1e-2 s, for 400 steps.
    pick = rng.choice(len(rho), size=min(5 * want, len(rho)), replace=False)
    k0 = np.where(rng.random(len(pick)) < 0.5, 0, rng.integers(0, X.shape[1] - 1, size=len(pick)))
    dt = 10 ** rng.uniform(-8, -2, size=len(pick))
    t0 = time.time()
    Xend, _, seen = train.roll(model, rho[pick], X[pick, k0], T[pick, k0], dt, 400, record=8)
    r = torch.cat([s[0] for s in seen]).cpu().numpy()
    Ts = torch.cat([s[1] for s in seen]).cpu().numpy()
    Xs = torch.cat([s[2] for s in seen]).cpu().numpy().astype(np.float64)
    row = np.tile(np.arange(len(pick)), len(seen))
    # How far each rollout ended from the reference at the nearest recorded time: the states
    # of the rollouts that went wrong are the ones worth burning.
    k_end = np.clip(np.searchsorted(t, t[k0] + 400 * dt), 0, len(t) - 1)
    err = np.abs(Xend.cpu().numpy() - X[pick, k_end]).sum(axis=1)
    # Keep states the reference can burn (its rates end at 1e10 K): every state of a rollout
    # that ended more than 0.1 off (up to half of them), the rest at random.
    ok = (Ts > 1e8) & (Ts < 1.2e10) & np.isfinite(Xs).all(axis=1)
    bad = np.flatnonzero(ok & (err[row] > 0.1))
    bad = rng.choice(bad, size=min(want // 2, len(bad)), replace=False)
    rest = np.setdiff1d(np.flatnonzero(ok), bad)
    idx = np.concatenate([bad, rng.choice(rest, size=min(want - len(bad), len(rest)), replace=False)])
    print(f"{(err > 0.1).mean():.2%} of {len(pick)} rollouts ended more than 0.1 off; {len(bad)} of their states kept", flush=True)
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
