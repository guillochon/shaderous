"""Paired comparison of burner models on the same trajectories, with bootstrap intervals: the
dev trajectories of lpedit3.split() and the held-out test set (targets_test.npz) pooled (600
per regime), and general trajectories (lpedit3's 1000 and score.py's 600). Per trajectory the
worst L1 error over score.py's check times (regimes) or every check time (general, rows).
A p90 of 200 trajectories rests on its worst 20; comparing two models trajectory by
trajectory and resampling the trajectories says whether a difference is more than that.

    tools/burn/.venv/Scripts/python tools/burn/weightprog/paired.py base.pt candidate.pt ...
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lpedit3 as L  # noqa: E402
import score  # noqa: E402
import train  # noqa: E402


def sets():
    _, devset = L.split()
    tt = np.load(train.HERE / "targets_test.npz")
    out = {}
    for i, kind in enumerate(L.KINDS):
        s = tt["kind"] == i
        d = devset[kind]
        out[kind] = (np.concatenate([d[0], tt["rho"][s]]), np.concatenate([d[1], tt["X"][s]]),
                     np.concatenate([d[2], tt["T"][s]]), d[3])
    rho, X, T, t = train.load({"fuel.npz", "mix.npz"})
    va = np.random.default_rng(7).choice(len(rho), size=600, replace=False)
    g = devset["general"]
    out["general"] = (np.concatenate([g[0], rho[va]]), np.concatenate([g[1], X[va]]), np.concatenate([g[2], T[va]]), t)
    return out


def worst(name, S):
    m = train.load_model(name)
    m.parts = None
    m.net = score.Half(m.net)
    res = {}
    with torch.no_grad():
        for k, (rho, X, T, t) in S.items():
            res[k] = score.errs(m, rho, X, T, t)       # (check times, trajectories)
    return res


def main():
    names = sys.argv[1:]
    S = sets()
    R = {n: worst(n, S) for n in names}
    rng = np.random.default_rng(0)
    base = names[0]
    print("per regime: p50 / p90 of each trajectory's worst error; for candidates the change in p90 with its 90% "
          "bootstrap interval and the share of trajectories improved (worst error lower by > 10%)")
    for n in names:
        line = [f"{n:28s}"]
        tot = []
        for k in L.KINDS + ["general"]:
            e = R[n][k] if k == "general" else R[n][k].max(axis=0)
            b = R[base][k] if k == "general" else R[base][k].max(axis=0)
            q = 0.99 if k == "general" else 0.9
            if k == "general":
                ev, bv = e.reshape(-1), b.reshape(-1)
            else:
                ev, bv = e, b
            idx = rng.integers(0, len(ev), (400, len(ev)))
            diff = np.quantile(ev[idx], q, axis=1) - np.quantile(bv[idx], q, axis=1)
            lo, hi = np.quantile(diff, [0.05, 0.95])
            better = np.mean(ev < 0.9 * bv)
            worse = np.mean(ev > 1.1 * bv + 1e-4)
            line.append(f"{k} {np.median(ev):.3f}/{np.quantile(ev, q):.3f}"
                        + ("" if n == base else f" ({np.quantile(ev, q) - np.quantile(bv, q):+.3f} [{lo:+.3f},{hi:+.3f}] +{better:.0%}/-{worse:.0%})"))
            if k != "general":
                tot.append(diff)
        t = np.sum(tot, axis=0)
        if n != base:
            line.append(f"| sum of p90 change {np.median(t):+.3f} [{np.quantile(t, 0.05):+.3f},{np.quantile(t, 0.95):+.3f}]")
        print(" ".join(line), flush=True)


if __name__ == "__main__":
    main()
