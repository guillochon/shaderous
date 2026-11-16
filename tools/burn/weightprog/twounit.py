"""Two gated rate units (rateedit.py with_unit, twice): hot C/O wants a mild slowing and the
Si/S runaway a strong one, which one gate serves only at a compromise (tunegate.py). Unit A
is the relu3 gate with a high threshold (offset -0.2: it fires on hot C/O and helps it);
unit B is fitted to the same targets with every C/O-rich state's (X_C12 + X_O16 > 0.2) set to
zero, so that it can slow the Si/S states hard and leave the C/O ones alone.

    tools/burn/.venv/Scripts/python tools/burn/weightprog/twounit.py [T cut]

With a T cut, unit B is also held at zero above that temperature: hot C/O burns into Si-rich
ash that looks like the Si/S states to a composition test, but is hotter than the runaway
(T ~ 3.8-4.3e9 K). Result on model_dagger_35b (2026-11-16), T cut 4.5e9, unit A alpha 2
(offset -0.2), unit B alpha 1.5 (two_t4.5e+09_a2.0_b1.5.pt, hidden layers 130 wide), paired
over dev and test (paired.py, 600 trajectories a regime): Si/S p90 1.576 -> 0.958 (-0.62
[-0.84, -0.43], 38% of trajectories better by >10%, 21% worse), hot C/O, oneb, NSE and general
flat; sum of p90 -0.61 [-0.82, -0.42], about -26%. score.py's test set alone: Si/S p90 1.53
-> 1.20, hot C/O 0.35 -> 0.35, NSE 0.22 -> 0.21; costs 1e-5 s steps p99 0.60 -> 0.65 and
general p50 at 1 s 3.8e-4 -> 5.5e-4 (general p99 at 1 s 0.44 -> 0.53 on that set, flat on the
pooled paired set). Without the T cut unit B also slowed hot C/O ash (p90 0.32 -> 0.37-0.43). Unit B alone
(arguments 4.5e9 0,1.5; one unit, 129 wide) paired: Si/S p90 0.869 (-0.71), sum -0.66
[-0.87, -0.42], hot C/O and general flat, oneb +0.011 [0.000, +0.037]; on score.py's test set
hot C/O p90 0.39 and 1e-5 s steps p99 0.70, against 0.35 and 0.65 with unit A as well.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lpedit3 as L  # noqa: E402
import rateedit as R  # noqa: E402
import train  # noqa: E402
from tunegate import gate_of  # noqa: E402

dev = train.dev
TCUT = float(sys.argv[1]) if len(sys.argv) > 1 else 1e12      # unit B also shut above this T
PAIRS = tuple(tuple(float(v) for v in p.split(",")) for p in sys.argv[2:]) or ((2.0, 1.5), (2.0, 2.0)) if len(sys.argv) > 1 else ((2.0, 1.5), (2.0, 2.0), (2.0, 3.0))


def fit_gate(a, t, fit):
    mu, sd = a[fit].mean(dim=0), a[fit].std(dim=0) + 1e-6
    an = (a - mu) / sd
    A1 = torch.cat([an[fit], torch.ones_like(an[fit, :1])], dim=1).double()
    sol = torch.linalg.solve(A1.T @ A1 / len(A1) + 1e-3 * torch.eye(A1.shape[1], device=dev, dtype=torch.float64),
                             (A1.T @ t[fit].double()) / len(A1)).float()
    w = sol[:-1].clone().requires_grad_(True)
    b = sol[-1].clone().requires_grad_(True)
    opt = torch.optim.Adam([w, b], lr=0.01)
    for _ in range(3000):
        opt.zero_grad()
        loss = ((torch.relu(an[fit] @ w + b) - t[fit]) ** 2).mean() + 1e-3 * w.abs().sum()
        loss.backward()
        opt.step()
    return (w / sd).detach(), (b - (w * mu / sd).sum()).detach()


def main():
    R.EXTRA_DTS = (1e-5, 1e-4)
    base = L.load_base("model_dagger_35b.pt")
    base.name = "model_dagger_35b.pt"
    _, devset = L.split()
    f, a, s, _ = R.exact_targets(base)
    files = [("refms.npz", 0), ("rate_model_dagger_35b.npz", 0), ("refms_1e-05.npz", 0), ("refms_0.0001.npz", 0)]
    X0 = np.concatenate([np.load(L.OUT / fn)["X0"] for fn, _ in files])
    T0 = np.concatenate([np.load(L.OUT / fn)["T"] for fn, _ in files])
    build, _ = L.split()
    am, sm = R.manifold_targets(base, build)
    # the manifold states' compositions, drawn as manifold_targets draws them
    r_, T_, Xm, _ = R.manifold_states(base, build)
    pick = np.random.default_rng(4).choice(len(r_), min(60_000, len(r_)), replace=False)
    Xm, Tm = Xm[pick], T_[pick]
    a = torch.cat([a, am])
    t = torch.cat([(-s).clamp_min(0.0), (-sm).clamp_min(0.0)])
    X = np.concatenate([X0, Xm])
    TT = np.concatenate([T0, Tm])
    co = torch.as_tensor((X[:, 1] + X[:, 2] > 0.2) | (TT > TCUT), device=dev)
    tB = torch.where(co, torch.zeros_like(t), t)
    fit = torch.arange(len(t), device=dev)
    print(f"{len(t)} states, {co.float().mean():.1%} C/O-rich; unit B wants slowing on {(tB > 0.05).float().mean():.1%}", flush=True)
    wB, bB = fit_gate(a, tB, fit)
    wA, bA = gate_of("rateu3_l0.001_a1.0.pt")
    print("base " + L.summary(L.dev_score(base, devset)), flush=True)
    for alA, alB in PAIRS:
        m = R.with_unit(base, wA, bA - 0.2, alA)
        m2 = R.with_unit(m, torch.cat([wB, torch.zeros(1, device=dev)]), bB, alB)
        print(f"A alpha {alA} (offset -0.2), B alpha {alB}: " + L.summary(L.dev_score(m2, devset)), flush=True)
        torch.save({"state": m2.state_dict(), "width": 130, "depth": 4, "arch": m2.arch},
                   train.HERE / f"two_t{TCUT:g}_a{alA}_b{alB}.pt")


if __name__ == "__main__":
    main()
