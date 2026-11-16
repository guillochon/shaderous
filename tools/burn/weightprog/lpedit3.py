"""Closed-form, many-direction weight edits of a trained Burner (third round of weight-space
programming; lpedit.py and lpedit2.py are the rank-one rounds).

lpedit2.py found that the hidden-state fixes of the bad steps do not share a direction (mean
cosine with their mean 0.31-0.41), so one rank-one push cannot carry them. Here every state
gets its own fix and one linear map carries them all, as the "locate and edit" methods for
language models do (ROME, Meng et al. 2022; MEMIT, Meng et al. 2023): at linear layer k with
inputs a (keys) and outputs z,

  values   dz_i: the nearest change to z_i from which the rest of the network makes the
           reference's step from state i (gradient descent on dz alone, weights fixed), for
           every state of the build set: bad steps get large values, good ones about zero,
           which protects them
  edit     dW = argmin sum_i |dW a_i - dz_i|^2 + lam |dW|^2  =  DZ A^T (A A^T + lam I)^-1,
           a with a 1 appended so the bias moves too; applied as W += s dW

The build set is half the targeted training trajectories of every regime (targets.npz) and
general trajectories (fuel, mix and DAgger rounds) outside score.py's general sample; the
other half of targets.npz is the dev set, judged by score.py's own metric (per trajectory
the worst L1 error over the check times, p50 and p90 by regime; general p99). The test set
(targets_test.npz) is left for the end.

    tools/burn/.venv/Scripts/python tools/burn/weightprog/lpedit3.py fixes [base.pt]
    tools/burn/.venv/Scripts/python tools/burn/weightprog/lpedit3.py sweep [base.pt]

Result on model_dagger_35b (2026-11-16): no gain. The per-state fixes exist at every layer
(200,000 build steps, error mean 0.0041 -> 0.0005), but no linear map of a layer's inputs
carries them: the fit residual stayed at 1.03-1.08 of the fixes' size for every layer and
ridge, and dev rollouts moved by noise (the sum of p90 2.23-2.38 against 2.35). The split()
and dev_score() here are what rateedit.py, tunegate.py and paired.py judge by.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import score  # noqa: E402
import train  # noqa: E402

KINDS = ["oneb", "sis", "hotco", "nse"]
dev = train.dev
NDEV, NGEN = 400, 1000
OUT = train.HERE / "weightprog" / "cache"


def tt(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)


def steps_of(rho, X, T, t, lo=1e-6, hi=1.0):
    """Every reference step (state at t_k, change to t_{k+1}) with t_k in [lo, hi]."""
    ks = [k for k in range(len(t) - 1) if lo <= t[k] <= hi]
    r = np.repeat(rho, len(ks))
    x0 = X[:, ks].reshape(-1, 13)
    d = (X[:, [k + 1 for k in ks]] - X[:, ks]).reshape(-1, 13)
    T0 = T[:, ks].reshape(-1)
    dt = np.tile([t[k + 1] - t[k] for k in ks], len(rho))
    return r, T0, dt, x0, d


def split():
    """(build trajectories, dev trajectories) as dicts of arrays."""
    rng = np.random.default_rng(0)
    tg = np.load(train.HERE / "data" / "targets.npz")
    t = tg["t"]
    build, devset = [], {}
    for i, kind in enumerate(KINDS):
        idx = np.where(tg["kind"] == i)[0]
        rng.shuffle(idx)
        devset[kind] = (tg["rho"][idx[:NDEV]], tg["X"][idx[:NDEV]], tg["T"][idx[:NDEV]], t)
        b = idx[NDEV:]
        build.append((tg["rho"][b], tg["X"][b], tg["T"][b], t))
    rho, X, T, tg2 = train.load({"fuel.npz", "mix.npz"})
    va = np.random.default_rng(7).choice(len(rho), size=600, replace=False)   # score.py's
    rest = np.setdiff1d(np.arange(len(rho)), va)
    gi = rng.choice(rest, NGEN, replace=False)
    devset["general"] = (rho[gi], X[gi], T[gi], tg2)
    rest = np.setdiff1d(rest, gi)
    gb = rng.choice(rest, min(3000, len(rest)), replace=False)
    build.append((rho[gb], X[gb], T[gb], tg2))
    dg = train.load({f"dagger_{k}.npz" for k in range(30, 39)})
    di = rng.choice(len(dg[0]), min(3000, len(dg[0])), replace=False)
    build.append((dg[0][di], dg[1][di], dg[2][di], dg[3]))
    return build, devset


def linears(model):
    return [i for i, m in enumerate(model.net) if isinstance(m, torch.nn.Linear)]


def step_error(model, zout, X, ld, d):
    P = model.width
    D1 = model.relax(zout[:, :P], X, ld)
    D2 = model.relax(zout[:, P:], (X + D1).clamp_min(0.0), ld)
    return (D1 + D2 - d).abs().sum(dim=1)


def fixes(model, k, f, X, ld, d, lam=1e-3, iters=400, batch=32768):
    """Keys a (input to linear k), values dz, and the errors before and after the fix."""
    idx = linears(model)[k]
    A, DZ, E0, E1 = [], [], [], []
    for s in range(0, len(f), batch):
        sl = slice(s, s + batch)
        with torch.no_grad():
            a = model.net[:idx](f[sl])
            z = model.net[idx](a)
        dz = torch.zeros_like(z, requires_grad=True)
        opt = torch.optim.Adam([dz], lr=0.02)
        tail = model.net[idx + 1:]
        for _ in range(iters):
            opt.zero_grad()
            e = step_error(model, tail(z + dz), X[sl], ld[sl], d[sl])
            (e.sum() + lam * (dz**2).sum()).backward()
            opt.step()
        with torch.no_grad():
            E0.append(step_error(model, tail(z), X[sl], ld[sl], d[sl]))
            E1.append(step_error(model, tail(z + dz), X[sl], ld[sl], d[sl]))
        A.append(a)
        DZ.append(dz.detach())
    return torch.cat(A), torch.cat(DZ), torch.cat(E0), torch.cat(E1)


def solve(A, DZ, lam, w=None):
    """dW (out x in+1) minimizing sum_i w_i |dW [a_i, 1] - dz_i|^2 + lam |dW|^2."""
    A1 = torch.cat([A, torch.ones_like(A[:, :1])], dim=1).double()
    W = torch.ones(len(A), device=dev, dtype=torch.float64) if w is None else w.double()
    G = (A1 * W[:, None]).T @ A1
    R = (DZ.double() * W[:, None]).T @ A1
    n = len(A)
    return torch.linalg.solve(G / n + lam * torch.eye(G.shape[0], device=dev, dtype=torch.float64), (R / n).T).T.float()


def apply(model, k, dW, s):
    m = copy.deepcopy(model)
    lin = m.net[linears(m)[k]]
    with torch.no_grad():
        lin.weight += s * dW[:, :-1]
        lin.bias += s * dW[:, -1]
    return m


def dev_score(model, devset):
    """score.py's metric on the dev trajectories: p50/p90 of each trajectory's worst error over
    the check times by regime, and the mean of general's p99 over the check times."""
    m = copy.deepcopy(model)
    m.parts = None
    m.net = score.Half(m.net)
    out = {}
    with torch.no_grad():
        for name, (rho, X, T, t) in devset.items():
            e = score.errs(m, rho, X, T, t)
            if name == "general":
                out[name] = float(np.mean(np.quantile(e, 0.99, axis=1)))
                out["general90"] = float(np.mean(np.quantile(e, 0.9, axis=1)))
            else:
                w = e.max(axis=0)
                out[name] = (float(np.median(w)), float(np.quantile(w, 0.9)))
    return out


def summary(r):
    s = " ".join(f"{k} {r[k][0]:.3f}/{r[k][1]:.3f}" for k in KINDS if k in r)
    return s + f" general p90 {r['general90']:.3f} p99 {r['general']:.3f} | p90 sum {sum(r[k][1] for k in KINDS if k in r):.3f}"


def load_base(name):
    m = train.load_model(name)
    m.parts = None
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_states(model, build):
    parts = [steps_of(*b) for b in build]
    r, T0, dt, x0, d = (np.concatenate([p[i] for p in parts]) for i in range(5))
    ok = np.isfinite(T0) & np.isfinite(x0).all(axis=1) & np.isfinite(d).all(axis=1) & (dt > 0)
    keep = np.flatnonzero(ok)
    keep = np.sort(np.random.default_rng(1).choice(keep, min(200_000, len(keep)), replace=False))
    r, T0, dt, x0, d = r[keep], np.clip(T0[keep], 1e7, 2e10), dt[keep], x0[keep], d[keep]
    f = train.features(torch.log10(tt(r)), torch.log10(tt(T0)), torch.log10(tt(dt)), tt(x0).clamp_min(0.0), model.rates)
    return f, tt(x0).clamp_min(0.0), torch.log10(tt(dt)), tt(d)


def cmd_fixes(base):
    model = load_base(base)
    build, _ = split()
    f, X, ld, d = build_states(model, build)
    print(f"{len(f)} build steps", flush=True)
    OUT.mkdir(exist_ok=True)
    for k in range(len(linears(model))):
        t0 = time.time()
        A, DZ, E0, E1 = fixes(model, k, f, X, ld, d)
        torch.save({"A": A.cpu(), "DZ": DZ.cpu(), "E0": E0.cpu(), "E1": E1.cpu()}, OUT / f"fix_{Path(base).stem}_L{k}.pt")
        print(f"layer {k}: error mean {E0.mean():.4f} -> {E1.mean():.4f}, p99 {torch.quantile(E0, 0.99):.4f} -> "
              f"{torch.quantile(E1, 0.99):.4f}; |dz| p50 {DZ.norm(dim=1).median():.3f} ({time.time() - t0:.0f} s)", flush=True)


def cmd_sweep(base, layers=(1, 2, 3, 4), lams=(1e-4, 1e-3, 1e-2), ss=(0.5, 1.0)):
    model = load_base(base)
    _, devset = split()
    print("base", summary(dev_score(model, devset)), flush=True)
    for k in layers:
        c = torch.load(OUT / f"fix_{Path(base).stem}_L{k}.pt")
        A, DZ = c["A"].to(dev), c["DZ"].to(dev)
        for lam in lams:
            dW = solve(A, DZ, lam)
            res = (A @ dW[:, :-1].T + dW[:, -1] - DZ).norm(dim=1).mean() / DZ.norm(dim=1).mean()
            for s in ss:
                m = apply(model, k, dW, s)
                print(f"layer {k} lam {lam:g} s {s}: fit residual {res:.2f}; " + summary(dev_score(m, devset)), flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "model_dagger_35b.pt"
    {"fixes": cmd_fixes, "sweep": cmd_sweep}[what](base)
