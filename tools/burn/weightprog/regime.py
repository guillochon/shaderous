"""Where and how round 30 goes wrong on the held-out targeted trajectories: per kind, roll the
model (fp32, 1 ms steps) and compare with the reference at each check time, per species
(signed: model minus reference), with the start's density and temperature for the worst."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train
NAMES = ["he4","c12","o16","ne20","mg24","si28","s32","ar36","ca40","ti44","cr48","fe52","ni56"]
KINDS = ["oneb","sis","hotco","nse"]
m = train.load_model(sys.argv[1] if len(sys.argv) > 1 else "model_dagger_30.pt")
tt = np.load(train.HERE / "targets_test.npz"); t = tt["t"]
for ki, kind in enumerate(KINDS):
    s = tt["kind"] == ki
    rho, X, T = tt["rho"][s], tt["X"][s], tt["T"][s]
    Xc, Tc, done = X[:, 0], T[:, 0], 0.0
    print(f"== {kind}: {s.sum()} trajectories, rho {rho.min():.1e}-{rho.max():.1e}, T0 {T[:,0].min():.2e}-{T[:,0].max():.2e}")
    for tc in (0.001, 0.01, 0.1, 1.0):
        k = int(np.argmin(np.abs(t - tc)))
        steps = max(1, int(round((t[k] - done) / 1e-3)))
        with torch.no_grad():
            Xm, Tm, _ = train.roll(m, rho, Xc, Tc, np.full(len(rho), (t[k] - done) / steps), steps)
        Xc, Tc, done = Xm.cpu().numpy(), Tm.cpu().numpy(), t[k]
        d = Xc - X[:, k]
        l1 = np.abs(d).sum(1)
        bad = l1 > np.quantile(l1, 0.75)
        top = np.argsort(-np.abs(d[bad].mean(0)))[:4]
        print(f"  t {tc:g}: L1 p50 {np.median(l1):.3f} p90 {np.quantile(l1,0.9):.3f}; worst quarter mean model-ref: "
              + ", ".join(f"{NAMES[i]} {d[bad].mean(0)[i]:+.3f}" for i in top)
              + f"; T model/ref {np.median(Tc[bad]/T[bad,k]):.3f}")
    l1 = np.abs(Xc - X[:, -1]).sum(1)
    w = np.argsort(-l1)[:5]
    for i in w:
        top = np.argsort(-X[i, 0])[:3]
        print(f"    worst: rho {rho[i]:.1e} T0 {T[i,0]:.2e} start [{', '.join(f'{NAMES[j]} {X[i,0,j]:.2f}' for j in top)}] L1 {l1[i]:.2f}"
              f" ref end [{', '.join(f'{NAMES[j]} {X[i,-1,j]:.2f}' for j in np.argsort(-X[i,-1])[:3])}] T {T[i,-1]:.2e}"
              f" model [{', '.join(f'{NAMES[j]} {Xc[i,j]:.2f}' for j in np.argsort(-Xc[i])[:3])}] T {Tc[i]:.2e}")
