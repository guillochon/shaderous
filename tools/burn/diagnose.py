"""Where a trained surrogate fails: roll it out along held-out trajectories as the simulator
would (train.roll: fixed steps, the temperature following the energy released) and compare
with the reference at several times, listing the worst cases with their starting state.

A case that is right at 0.03 s and 1 s but wrong at 0.1 s ignited at a slightly different
time; one that ends wrong at 1 s burned to the wrong place.

    tools/burn/.venv/Scripts/python tools/burn/diagnose.py model.pt [dt]
"""
import sys

import numpy as np
import torch

import train

NAMES = ["he4", "c12", "o16", "ne20", "mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48", "fe52", "ni56"]
CHECK = (0.003, 0.03, 0.1, 1.0)


def main():
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    model = train.load_model(sys.argv[1])
    # A fixed set: fuel and mixture starts only (gen_data.py). The DAgger files hold states
    # chosen because some model went wrong there, and grow with every round.
    rho, X, T, t = train.load({"fuel.npz", "mix.npz"})
    rng = np.random.default_rng(7)
    va = rng.choice(len(rho), size=600, replace=False)
    rho, X, T = rho[va], X[va], T[va]
    errs = {}
    Xc, Tc = X[:, 0], T[:, 0]
    done = 0
    for tc in CHECK:
        # To the reference's own time nearest tc, exactly (the step stretched to fit): a cell
        # that ignites between tc and the nearest output would otherwise count as wrong.
        k = int(np.argmin(np.abs(t - tc)))
        steps = max(1, int(round((t[k] - done) / dt)))
        h = (t[k] - done) / steps
        Xm, Tm, _ = train.roll(model, rho, Xc, Tc, np.full(len(rho), h), steps)
        Xc, Tc, done = Xm.cpu().numpy(), Tm.cpu().numpy(), t[k]
        errs[tc] = (np.abs(Xc - X[:, k]).sum(axis=1), Xc.copy(), k)
    print(f"dt = {dt:g}; L1 at " + ", ".join(f"{tc:g} s: p50 {np.quantile(e[0], 0.5):.2g} p99 {np.quantile(e[0], 0.99):.2g}" for tc, e in errs.items()))
    top = lambda x: ", ".join(f"{NAMES[i]} {x[i]:.2f}" for i in np.argsort(-x)[:3])
    worst = np.argsort(-np.max([e[0] for e in errs.values()], axis=0))[:10]
    for i in worst:
        line = " | ".join(f"{tc:g}s {errs[tc][0][i]:.2f}" for tc in CHECK)
        k = errs[1.0][2]
        print(f"rho {rho[i]:.1e} T0 {T[i, 0]:.1e} start [{top(X[i, 0])}]  L1: {line}")
        print(f"    at 1 s ref [{top(X[i, k])}] T {T[i, k]:.2e}   model [{top(errs[1.0][1][i])}]")


if __name__ == "__main__":
    main()
