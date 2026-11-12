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
    ck = torch.load(train.HERE / sys.argv[1])
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    model = train.Burner(ck["width"], ck.get("depth", 3)).to(train.dev)
    model.load_state_dict(ck["state"])
    model.eval()
    rho, X, T, t = train.load()
    # The fuel and mixture trajectories' own validation split (the first files loaded).
    rng = np.random.default_rng(7)
    va = rng.choice(len(rho), size=600, replace=False)
    rho, X, T = rho[va], X[va], T[va]
    errs = {}
    Xc, Tc = X[:, 0], T[:, 0]
    done = 0
    for tc in CHECK:
        steps = int(round((tc - done) / dt))
        Xm, Tm, _ = train.roll(model, rho, Xc, Tc, np.full(len(rho), dt), steps)
        Xc, Tc, done = Xm.cpu().numpy(), Tm.cpu().numpy(), done + steps * dt
        k = int(np.argmin(np.abs(t - tc)))
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
