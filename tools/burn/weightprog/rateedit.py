"""A rate edit of a trained flow2f Burner: weight-space programming with the direction chosen
by the physics, the gate by a linear program, and the states by the model itself.

The worst held-out failures (Si/S quasi-equilibrium, T ~ 3.8e9 K) are a runaway: the model
turns Si28 into Ni56 5-10 times faster than the reference at 1 ms steps, the energy heats the
gas, and by 0.3 s it has burned everything the reference keeps for seconds. The network's
burn rate is two outputs, z0 and z27 (log10 k = 5 + 5 z of each stage), so:

  states   the model's own 1 ms rollouts from the build trajectories (as AHA takes its
           model's own generations), plus reference states; each burned 1 ms with the
           reference network (burner.py)
  target   per state, the shift s of z0 and z27 together that best fixes the model's 1 ms
           step (a grid search); zero where it fixes little
  gate     v on the last hidden layer (and a bias) with v . h ~ s: ridge (dense) or an L1
           linear program (sparse: least |v|_1 + C |v . h + b - s|_1)
  edit     W[0] += v, W[27] += v (and the biases): rank one, u = e0 + e27

Judged on lpedit3's dev set (half of targets.npz by regime, score.py's metric), then on the
test set by score.py.

    tools/burn/.venv/Scripts/python tools/burn/weightprog/rateedit.py data [base.pt]
    tools/burn/.venv/Scripts/python tools/burn/weightprog/rateedit.py sweep [base.pt]

The steps that worked, in order (2026-11-16, on model_dagger_35b):

    refdata, refdt     reference states of every regime and general, burned exactly for
                       1e-3, 1e-4 and 1e-5 s (15,077 each, about 5 min of CPU each)
    relu3              a new unit in the last hidden layer, g = relu(w . a + b) (a the layer
                       before; with_unit), lowering both stages' log rate by alpha g; w, b by
                       least squares on the slowing each state wants, from the exact burns at
                       all three steps, the model's own 1 ms rollout states (data) and slow
                       reference states with interval-average targets (manifold)
    tunegate.py        the gate's threshold and alpha by dev rollouts
    paired.py          the verdict: models compared trajectory by trajectory on dev and test
                       pooled, with bootstrap intervals

Findings. The worst failures (Si/S quasi-equilibrium) are a runaway: about 10x too fast a
burn at T ~ 4.3e9 K, rho ~ 2e7, heating the gas until everything burns. The rate alone
fixes most slow-state error (exact 1 ms burns: relative error p50 0.25 -> 0.13), and the
model overshoots at 1e-5 s steps as well. Ridge gates (linear in the last hidden layer)
cut Si/S hard but always slowed general fuel too (paired general p99 +0.09); the relu unit
leaves general alone. Gates fitted at 1 ms only hurt 1e-5 s steps (score.py small-step p99
0.60 -> 0.86); adding the short-step burns fixed most of that. Hot C/O wants a mild slowing and
Si/S a strong one, and one gate serves both only at a compromise: offset -0.1, alpha 1.5
(tune_o0.1_a1.5.pt), paired over 600 trajectories a regime: sis p90 1.576 -> 1.346 (-0.23
[-0.34, -0.12]), hotco 0.331 -> 0.296 (median 0.161 -> 0.126), oneb and nse slightly better,
general flat; sum of p90 -0.26 [-0.39, -0.17], about -11%. On score.py's test set alone: sis
p90 1.53 -> 1.45, hotco 0.35 -> 0.30, but 1e-5 s steps p99 0.60 -> 0.66 and general p50 at
1 s 3.8e-4 -> 7.5e-4. Stronger gates (alpha 2, no offset) reach sis p90 0.84 (-47%) at a
cost in hot C/O (p90 +0.08). The p90 of a regime's 200 test trajectories rests on 20 of
them: single-set p90 changes under about 0.1 are noise, hence paired.py.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lpedit3 as L  # noqa: E402
import train  # noqa: E402

dev = train.dev
DT = 1e-3
SHIFTS = np.linspace(-0.6, 0.6, 121)
WPS = (0.0, 1.0, 5.0)
EXTRA_DTS = ()
SCALES = (0.5, 0.7)


def burn_one(args):
    import burner
    rho, T, X = args[:3]
    dt = args[3] if len(args) > 3 else DT
    try:
        X1, T1, _ = burner.burn(X, rho, T, dt)
    except Exception:  # noqa: BLE001
        return None
    return X1, T1


def cmd_data(base, n_roll=6000, n_ref=2000):
    from gen_data import workers
    model = L.load_base(base)
    build, _ = L.split()
    rng = np.random.default_rng(3)
    R, TT, XX = [], [], []
    for rho, X, T, t in build:
        k0 = np.where(rng.random(len(rho)) < 0.5, 0, rng.integers(0, X.shape[1] // 2, size=len(rho)))
        _, _, seen = train.roll(model, rho, X[np.arange(len(rho)), k0], T[np.arange(len(rho)), k0],
                                np.full(len(rho), DT), 600, record=6)
        R.append(torch.cat([s[0] for s in seen]).cpu().numpy())
        TT.append(torch.cat([s[1] for s in seen]).cpu().numpy())
        XX.append(torch.cat([s[2] for s in seen]).cpu().numpy().astype(np.float64))
    R, TT, XX = np.concatenate(R), np.concatenate(TT), np.concatenate(XX)
    ok = np.flatnonzero((TT > 1e8) & (TT < 1.2e10) & np.isfinite(XX).all(axis=1))
    pick = rng.choice(ok, min(n_roll, len(ok)), replace=False)
    r, T0, X0 = [R[pick]], [TT[pick]], [XX[pick]]
    # reference states too
    for rho, X, T, t in build[:4]:
        i = rng.integers(0, len(rho), n_ref // 4)
        k = rng.integers(0, X.shape[1], n_ref // 4)
        r.append(rho[i]); T0.append(T[i, k]); X0.append(X[i, k])
    r, T0, X0 = np.concatenate(r), np.concatenate(T0), np.concatenate(X0)
    ok = (T0 > 1e8) & (T0 < 1.2e10)
    r, T0, X0 = r[ok], T0[ok], np.clip(X0[ok], 0, None)
    X0 /= X0.sum(axis=1, keepdims=True)
    t0 = time.time()
    with Pool(workers()) as pool:
        out = pool.map(burn_one, list(zip(r.astype(float), T0.astype(float), X0)), chunksize=8)
    keep = np.array([o is not None for o in out])
    X1 = np.stack([o[0] for o in out if o is not None])
    np.savez_compressed(L.OUT / f"rate_{Path(base).stem}.npz", rho=r[keep], T=T0[keep], X0=X0[keep], X1=X1)
    print(f"{keep.sum()} states burned in {time.time() - t0:.0f} s", flush=True)


def shifts(model, h, X, ld, d):
    """Per state, the error at each shift of z0 and z27, and the best shift."""
    head = model.net[-1]
    z = head(h)
    P = model.width
    E = []
    for s in SHIFTS:
        zs = z.clone()
        zs[:, 0] += s
        zs[:, P] += s
        E.append(L.step_error(model, zs, X, ld, d))
    E = torch.stack(E, dim=1)
    return E


def ridge(H, s, w, lam):
    H1 = torch.cat([H, torch.ones_like(H[:, :1])], dim=1).double()
    G = (H1 * w[:, None]).T @ H1 / w.sum()
    b = (H1 * (w * s)[:, None]).sum(dim=0) / w.sum()
    return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=dev, dtype=torch.float64), b).float()


def apply(model, vb, scale=1.0):
    import copy
    m = copy.deepcopy(model)
    P = m.width
    with torch.no_grad():
        for o in (0, P):
            m.net[-1].weight[o] += scale * vb[:-1]
            m.net[-1].bias[o] += scale * vb[-1]
    return m


def cmd_sweep(base):
    model = L.load_base(base)
    c = np.load(L.OUT / f"rate_{Path(base).stem}.npz")
    tt = L.tt
    f = train.features(torch.log10(tt(c["rho"])), torch.log10(tt(c["T"])), torch.full((len(c["rho"]),), np.log10(DT), device=dev),
                       tt(c["X0"]), model.rates)
    X, d = tt(c["X0"]), tt(c["X1"] - c["X0"])
    ld = torch.full((len(X),), np.log10(DT), device=dev)
    with torch.no_grad():
        h = model.net[:-1](f)
        E = shifts(model, h, X, ld, d)
    e0 = E[:, len(SHIFTS) // 2]
    best = E.argmin(dim=1)
    s = torch.as_tensor(SHIFTS, device=dev, dtype=torch.float32)[best]
    gain = 1 - E.min(dim=1).values / e0.clamp_min(1e-12)
    print(f"{len(X)} states; 1 ms error mean {e0.mean():.4f} p99 {torch.quantile(e0, 0.99):.4f}; "
          f"best shift fixes to mean {E.min(dim=1).values.mean():.4f}; shift p10/p50/p90 "
          f"{torch.quantile(s, 0.1):.2f}/{s.median():.2f}/{torch.quantile(s, 0.9):.2f}", flush=True)
    sig = gain > 0.1
    s = torch.where(sig, s, torch.zeros_like(s))
    # weight by how much the error at the best shift is worth
    w = (e0 - E.min(dim=1).values).clamp_min(0) + 1e-3
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
    fit, hold = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    _, devset = L.split()
    print("base", L.summary(L.dev_score(model, devset)), flush=True)
    for lam in (1e-3, 1e-2, 1e-1):
        vb = ridge(h[fit], s[fit], w[fit].double(), lam)
        pred = h[hold] @ vb[:-1] + vb[-1]
        with torch.no_grad():
            Eh = []
            for sc in (0.5, 1.0):
                m = apply(model, vb, sc)
                zz = m.net(f[hold])
                Eh.append(L.step_error(m, zz, X[hold], ld[hold], d[hold]).mean().item())
        corr = torch.corrcoef(torch.stack([pred, s[hold]]))[0, 1].item()
        print(f"lam {lam:g}: held-out corr(pred, shift) {corr:.2f}; held-out 1 ms error {e0[hold].mean():.4f} -> "
              f"x0.5 {Eh[0]:.4f}, x1 {Eh[1]:.4f}", flush=True)
        for sc in (0.5, 1.0):
            m = apply(model, vb, sc)
            print(f"  lam {lam:g} x{sc}: " + L.summary(L.dev_score(m, devset)), flush=True)
            torch.save({"state": m.state_dict(), "width": 128, "depth": 4, "arch": m.arch},
                       train.HERE / f"rate_l{lam:g}_x{sc}.pt")


def fast_states(build, lo=1e-6, hi=1.0, slow=0.02):
    """Reference states that change fast (the ignitions, the burns): the gate must stay shut
    there (AHA's band on productive tokens), target shift zero."""
    R, TT, X0 = [], [], []
    for rho, X, T, t in build:
        for k in range(len(t) - 1):
            if not lo <= t[k] <= hi:
                continue
            ch = np.abs(X[:, k + 1] - X[:, k]).sum(axis=1)
            ok = (ch >= slow) & (T[:, k] > 1e7) & np.isfinite(ch)
            R.append(rho[ok]); TT.append(T[ok, k]); X0.append(X[ok, k])
    return [np.concatenate(a) for a in (R, TT, X0)]


def manifold_states(model, build, lo=1e-3, hi=1.0, slow=0.02):
    """Reference states where the composition changes slowly (L1 change over the interval to
    the next saved state below `slow`), with the reference's change per DT from that interval:
    on the slow quasi-equilibrium manifold the 1 ms change is far below the L1 errors the
    other states make, but its rate sets the heating and the runaway."""
    R, TT, X0, DD = [], [], [], []
    for rho, X, T, t in build:
        for k in range(len(t) - 1):
            if not lo <= t[k] <= hi:
                continue
            ch = X[:, k + 1] - X[:, k]
            ok = (np.abs(ch).sum(axis=1) < slow) & (T[:, k] > 1e8) & np.isfinite(ch).all(axis=1)
            R.append(rho[ok]); TT.append(T[ok, k]); X0.append(X[ok, k])
            DD.append(ch[ok] * DT / (t[k + 1] - t[k]))
    return [np.concatenate(a) for a in (R, TT, X0, DD)]


def cmd_manifold(base, nmax=200_000):
    model = L.load_base(base)
    build, devset = L.split()
    r, T0, X0, d0 = manifold_states(model, build)
    rng = np.random.default_rng(4)
    pick = rng.choice(len(r), min(nmax, len(r)), replace=False)
    r, T0, X0, d0 = r[pick], T0[pick], X0[pick], d0[pick]
    tt = L.tt
    n = len(r)
    ld = torch.full((n,), np.log10(DT), device=dev)
    X, d = tt(X0), tt(d0)
    with torch.no_grad():
        f = train.features(torch.log10(tt(r)), torch.log10(tt(T0)), ld, X, model.rates)
        h = model.net[:-1](f)
        z = model.net[-1](h)
        P = model.width
        scale = d.abs().sum(dim=1) + 1e-7
        E = []
        for s in SHIFTS:
            zs = z.clone()
            zs[:, 0] += s
            zs[:, P] += s
            E.append(L.step_error(model, zs, X, ld, d) / scale)
        E = torch.stack(E, dim=1)
    e0 = E[:, len(SHIFTS) // 2]
    emin, best = E.min(dim=1)
    s = torch.as_tensor(SHIFTS, device=dev, dtype=torch.float32)[best]
    print(f"{n} slow reference states; relative 1 ms error p50 {e0.median():.2f} p90 {torch.quantile(e0, 0.9):.2f}; "
          f"at the best shift p50 {emin.median():.2f} p90 {torch.quantile(emin, 0.9):.2f}; shift p10/p50/p90 "
          f"{torch.quantile(s, 0.1):.2f}/{s.median():.2f}/{torch.quantile(s, 0.9):.2f}", flush=True)
    s = torch.where(emin < 0.7 * e0, s, torch.zeros_like(s))
    w = (e0 - emin).clamp_min(0) + 0.05
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
    fit, hold = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    fr, fT, fX = fast_states(build)
    fp = rng.choice(len(fr), min(100_000, len(fr)), replace=False)
    with torch.no_grad():
        hf = model.net[:-1](train.features(torch.log10(tt(fr[fp])), torch.log10(tt(np.clip(fT[fp], 1e7, 2e10))),
                                            torch.full((len(fp),), np.log10(DT), device=dev), tt(fX[fp]), model.rates))
    print(f"{len(fp)} fast reference states protected", flush=True)
    print("base", L.summary(L.dev_score(model, devset)), flush=True)
    for wp in WPS:
        lam = 1e-2
        H = torch.cat([h[fit], hf])
        S = torch.cat([s[fit], torch.zeros(len(hf), device=dev)])
        W = torch.cat([w[fit], torch.full((len(hf),), wp * float(w[fit].mean()), device=dev)])
        vb = ridge(H, S, W.double(), lam)
        pf = hf @ vb[:-1] + vb[-1]
        print(f"protect weight {wp}: gate on fast states |shift| p90 {torch.quantile(pf.abs(), 0.9):.3f} p99 {torch.quantile(pf.abs(), 0.99):.3f}", flush=True)
        pred = h[hold] @ vb[:-1] + vb[-1]
        corr = torch.corrcoef(torch.stack([pred, s[hold]]))[0, 1].item()
        line = []
        for sc in SCALES:
            m = apply(model, vb, sc)
            with torch.no_grad():
                eh = (L.step_error(m, m.net(f[hold]), X[hold], ld[hold], d[hold]) / scale[hold]).median().item()
            line.append(f"x{sc} {eh:.3f}")
        print(f"lam {lam:g}: held-out corr(pred, shift) {corr:.2f}; held-out relative error p50 {e0[hold].median():.3f} -> "
              + ", ".join(line), flush=True)
        for sc in SCALES:
            m = apply(model, vb, sc)
            print(f"  protect {wp} x{sc}: " + L.summary(L.dev_score(m, devset)), flush=True)
            torch.save({"state": m.state_dict(), "width": 128, "depth": 4, "arch": m.arch},
                       train.HERE / f"ratem_p{wp}_x{sc}.pt")


def cmd_refdata(base, n=16000):
    """Reference states from the build trajectories (every regime and general), t from 1e-4 to
    1 s spread evenly in log t, each burned exactly DT by the reference network."""
    from gen_data import workers
    build, _ = L.split()
    rng = np.random.default_rng(5)
    r, T0, X0 = [], [], []
    per = n // len(build)
    for rho, X, T, t in build:
        ks = np.flatnonzero((t >= 1e-4) & (t <= 1.0))
        i = rng.integers(0, len(rho), per)
        k = rng.choice(ks, per)
        r.append(rho[i]); T0.append(T[i, k]); X0.append(X[i, k])
    r, T0, X0 = np.concatenate(r), np.concatenate(T0), np.concatenate(X0)
    ok = (T0 > 1e8) & (T0 < 1.2e10) & np.isfinite(X0).all(axis=1)
    r, T0, X0 = r[ok], T0[ok], np.clip(X0[ok], 0, None)
    X0 /= X0.sum(axis=1, keepdims=True)
    t0 = time.time()
    with Pool(workers()) as pool:
        out = pool.map(burn_one, list(zip(r.astype(float), T0.astype(float), X0)), chunksize=8)
    keep = np.array([o is not None for o in out])
    X1 = np.stack([o[0] for o in out if o is not None])
    np.savez_compressed(L.OUT / "refms.npz", rho=r[keep], T=T0[keep], X0=X0[keep], X1=X1)
    print(f"{keep.sum()} reference states burned in {time.time() - t0:.0f} s", flush=True)


def cmd_exact(base, lams=(1e-2, 1e-1), scales=(0.5, 0.7, 1.0)):
    """The rate edit fitted to exact 1 ms burns: reference states (refdata) and the model's
    own rollout states (data), each with the shift that best fixes its step (relative error),
    fast states included, so no separate protection is needed."""
    model = L.load_base(base)
    _, devset = L.split()
    tt = L.tt
    parts = [np.load(L.OUT / "refms.npz"), np.load(L.OUT / f"rate_{Path(base).stem}.npz")]
    r, T0, X0, X1 = (np.concatenate([q[k] for q in parts]) for k in ("rho", "T", "X0", "X1"))
    n = len(r)
    ld = torch.full((n,), np.log10(DT), device=dev)
    X, d = tt(X0), tt(X1 - X0)
    with torch.no_grad():
        f = train.features(torch.log10(tt(r)), torch.log10(tt(T0)), ld, X, model.rates)
        h = model.net[:-1](f)
        z = model.net[-1](h)
        P = model.width
        scale = d.abs().sum(dim=1) + 1e-6
        E = []
        for s_ in SHIFTS:
            zs = z.clone()
            zs[:, 0] += s_
            zs[:, P] += s_
            E.append(L.step_error(model, zs, X, ld, d) / scale)
        E = torch.stack(E, dim=1)
    e0 = E[:, len(SHIFTS) // 2]
    emin, best = E.min(dim=1)
    s = torch.as_tensor(SHIFTS, device=dev, dtype=torch.float32)[best]
    s = torch.where(emin < 0.7 * e0, s, torch.zeros_like(s))
    print(f"{n} exactly burned states; relative error p50 {e0.median():.3f} p90 {torch.quantile(e0, 0.9):.2f}; at the best "
          f"shift p50 {emin.median():.3f} p90 {torch.quantile(emin, 0.9):.2f}; shift p10/p50/p90 {torch.quantile(s, 0.1):.2f}/"
          f"{s.median():.2f}/{torch.quantile(s, 0.9):.2f}", flush=True)
    w = torch.ones(n, device=dev, dtype=torch.float64)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
    fit, hold = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    print("base", L.summary(L.dev_score(model, devset)), flush=True)
    for lam in lams:
        vb = ridge(h[fit], s[fit], w[fit], lam)
        pred = h[hold] @ vb[:-1] + vb[-1]
        corr = torch.corrcoef(torch.stack([pred, s[hold]]))[0, 1].item()
        line = []
        for sc in scales:
            m = apply(model, vb, sc)
            with torch.no_grad():
                eh = L.step_error(m, m.net(f[hold]), X[hold], ld[hold], d[hold]) / scale[hold]
            line.append(f"x{sc} p50 {eh.median():.3f} p90 {torch.quantile(eh, 0.9):.2f}")
        print(f"lam {lam:g}: held-out corr {corr:.2f}; held-out relative error p50 {e0[hold].median():.3f} p90 "
              f"{torch.quantile(e0[hold], 0.9):.2f} -> " + ", ".join(line), flush=True)
        for sc in scales:
            m = apply(model, vb, sc)
            print(f"  exact lam {lam:g} x{sc}: " + L.summary(L.dev_score(m, devset)), flush=True)
            torch.save({"state": m.state_dict(), "width": 128, "depth": 4, "arch": m.arch},
                       train.HERE / f"ratex_l{lam:g}_x{sc}.pt")
            torch.save(vb.cpu(), L.OUT / f"ratex_vb_l{lam:g}.pt")


def exact_targets(model):
    """Features, the last two hidden layers' activations and the best shift of every exactly
    burned state (as cmd_exact)."""
    tt = L.tt
    files = [("refms.npz", DT), (f"rate_{Path(model.name).stem}.npz", DT)]
    files += [(f"refms_{dt:g}.npz", dt) for dt in EXTRA_DTS if (L.OUT / f"refms_{dt:g}.npz").exists()]
    parts = [np.load(L.OUT / fn) for fn, _ in files]
    r, T0, X0, X1 = (np.concatenate([q[k] for q in parts]) for k in ("rho", "T", "X0", "X1"))
    n = len(r)
    ld = torch.as_tensor(np.concatenate([np.full(len(q["rho"]), np.log10(dt)) for q, (_, dt) in zip(parts, files)]),
                         dtype=torch.float32, device=dev)
    print("exact sets: " + ", ".join(f"{fn} ({len(q['rho'])})" for q, (fn, _) in zip(parts, files)), flush=True)
    X, d = tt(X0), tt(X1 - X0)
    with torch.no_grad():
        f = train.features(torch.log10(tt(r)), torch.log10(tt(T0)), ld, X, model.rates)
        a = model.net[:-3](f)                    # input to the last hidden layer's linear
        h = model.net[:-1](f)
        z = model.net[-1](h)
        P = model.width
        scale = d.abs().sum(dim=1) + 1e-6
        E = []
        for s_ in SHIFTS:
            zs = z.clone()
            zs[:, 0] += s_
            zs[:, P] += s_
            E.append(L.step_error(model, zs, X, ld, d) / scale)
        E = torch.stack(E, dim=1)
    e0 = E[:, len(SHIFTS) // 2]
    emin, best = E.min(dim=1)
    s = torch.as_tensor(SHIFTS, device=dev, dtype=torch.float32)[best]
    s = torch.where(emin < 0.7 * e0, s, torch.zeros_like(s))
    return f, a, s, (X, ld, d, scale, e0)


RELU_A = 20.0


def with_unit(model, w, b, alpha):
    """The model with one hidden unit more in its last hidden layer: g = relu(w . a + b), as
    silu(A (w . a + b)) / A, a the previous layer's output, and the rate outputs of both stages
    (z0, z27) lowered by alpha g. Every other unit and output is unchanged."""
    old = [x for x in model.net if isinstance(x, torch.nn.Linear)]
    depth = len(old) - 1
    width = old[0].out_features
    new = train.Burner(width + 1, depth, model.arch).to(dev)
    Ln = [x for x in new.net if isinstance(x, torch.nn.Linear)]
    with torch.no_grad():
        for m_ in Ln:
            m_.weight.zero_(); m_.bias.zero_()
        for k in range(depth + 1):
            o, i = old[k].weight.shape
            Ln[k].weight[:o, :i] = old[k].weight
            Ln[k].bias[:o] = old[k].bias
        Ln[depth - 1].weight[width, :width] = RELU_A * w
        Ln[depth - 1].bias[width] = RELU_A * b
        for o in (0, model.width):
            Ln[depth].weight[o, width] = -alpha / RELU_A
    new.eval()
    new.parts = None
    for q in new.parameters():
        q.requires_grad_(False)
    return new


def manifold_targets(model, build, n=60_000):
    """The previous layer's activations and best shift of slow reference states with their
    interval-average 1 ms change (manifold_states, as cmd_manifold)."""
    tt = L.tt
    r, T0, X0, d0 = manifold_states(model, build)
    pick = np.random.default_rng(4).choice(len(r), min(n, len(r)), replace=False)
    r, T0, X0, d0 = r[pick], T0[pick], X0[pick], d0[pick]
    ld = torch.full((len(r),), np.log10(DT), device=dev)
    X, d = tt(X0), tt(d0)
    with torch.no_grad():
        f = train.features(torch.log10(tt(r)), torch.log10(tt(T0)), ld, X, model.rates)
        a = model.net[:-3](f)
        z = model.net(f)
        P = model.width
        scale = d.abs().sum(dim=1) + 1e-7
        E = []
        for s_ in SHIFTS:
            zs = z.clone()
            zs[:, 0] += s_
            zs[:, P] += s_
            E.append(L.step_error(model, zs, X, ld, d) / scale)
        E = torch.stack(E, dim=1)
    e0 = E[:, len(SHIFTS) // 2]
    emin, best = E.min(dim=1)
    s = torch.as_tensor(SHIFTS, device=dev, dtype=torch.float32)[best]
    return a, torch.where(emin < 0.7 * e0, s, torch.zeros_like(s))


def cmd_relu2(base):
    cmd_relu(base, alphas=(1.0, 1.5, 2.0), l1s=(1e-3,), manifold=True, tag="rateu2")


def cmd_relu3(base):
    """As relu2, with the short-step burns (refdt) in the fit: the gate must stay shut where
    short steps are right."""
    global EXTRA_DTS
    EXTRA_DTS = (1e-5, 1e-4)
    cmd_relu(base, alphas=(1.0, 1.5, 2.0), l1s=(1e-3,), manifold=True, tag="rateu3")


def cmd_relu(base, alphas=(0.5, 0.7, 1.0), l1s=(1e-3, 1e-2), manifold=False, tag="rateu"):
    """A rate edit through a new gated unit (with_unit): its gate fitted to the slowing each
    exactly burned state wants, t = max(-shift, 0), by least squares on relu(w . a + b) (129
    numbers, a few hundred Adam steps; the network's weights are untouched)."""
    model = L.load_base(base)
    model.name = base
    _, devset = L.split()
    f, a, s, (X, ld, d, scale, e0) = exact_targets(model)
    t = (-s).clamp_min(0.0)
    n = len(t)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
    fit, hold = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    if manifold:
        build, _ = L.split()
        am, sm = manifold_targets(model, build)
        print(f"{len(am)} slow states added; wanting a slower rate: {(sm < -0.05).float().mean():.1%}", flush=True)
        a = torch.cat([a, am])
        t = torch.cat([t, (-sm).clamp_min(0.0)])
        fit = torch.cat([fit, torch.arange(n, n + len(am), device=dev)])
    mu, sd = a[fit].mean(dim=0), a[fit].std(dim=0) + 1e-6
    an = (a - mu) / sd
    print(f"{n} states; wanting a slower rate (shift < -0.05): {(s < -0.05).float().mean():.1%}; "
          f"base dev: " + L.summary(L.dev_score(model, devset)), flush=True)
    for l1 in l1s:
        # start from the least-squares linear fit (a relu started at zero never moves)
        A1 = torch.cat([an[fit], torch.ones_like(an[fit, :1])], dim=1).double()
        sol = torch.linalg.solve(A1.T @ A1 / len(A1) + 1e-3 * torch.eye(A1.shape[1], device=dev, dtype=torch.float64),
                                 (A1.T @ t[fit].double()) / len(A1)).float()
        w = sol[:-1].clone().requires_grad_(True)
        b = sol[-1].clone().requires_grad_(True)
        opt = torch.optim.Adam([w, b], lr=0.01)
        for it in range(3000):
            opt.zero_grad()
            g = torch.relu(an[fit] @ w + b)
            loss = ((g - t[fit]) ** 2).mean() + l1 * w.abs().sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            gh = torch.relu(an[hold] @ w + b)
            fires = (gh > 0.02).float()
            want = (t[hold] > 0.05).float()
            print(f"l1 {l1:g}: held-out corr {torch.corrcoef(torch.stack([gh, t[hold]]))[0, 1]:.2f}; fires on "
                  f"{(fires * want).sum() / want.sum():.0%} of states wanting it, {(fires * (1 - want)).sum() / (1 - want).sum():.1%} of the rest; "
                  f"{int((w.abs() > 1e-3).sum())} inputs", flush=True)
            wr, br = (w / sd).detach(), (b - (w * mu / sd).sum()).detach()
        for alpha in alphas:
            m = with_unit(model, wr, br, alpha)
            with torch.no_grad():
                eh = L.step_error(m, m.net(f[hold]), X[hold], ld[hold], d[hold]) / scale[hold]
            print(f"  unit l1 {l1:g} alpha {alpha}: held-out relative error p50 {e0[hold].median():.3f} p90 "
                  f"{torch.quantile(e0[hold], 0.9):.2f} -> p50 {eh.median():.3f} p90 {torch.quantile(eh, 0.9):.2f}; dev "
                  + L.summary(L.dev_score(m, devset)), flush=True)
            torch.save({"state": m.state_dict(), "width": 129, "depth": 4, "arch": m.arch},
                       train.HERE / f"{tag}_l{l1:g}_a{alpha}.pt")


def cmd_refdt(base, dts=(1e-5, 1e-4)):
    """The refdata states burned again for shorter steps: the edit must leave steps alone where
    the model already gets them right (the 1 ms correction is partly the error of a long step
    through stiff burning, which a short step does not make)."""
    from gen_data import workers
    c = np.load(L.OUT / "refms.npz")
    for dt in dts:
        t0 = time.time()
        with Pool(workers()) as pool:
            out = pool.map(burn_one, [(float(a), float(b), x, dt) for a, b, x in zip(c["rho"], c["T"], c["X0"])], chunksize=8)
        keep = np.array([o is not None for o in out])
        X1 = np.stack([o[0] for o in out if o is not None])
        np.savez_compressed(L.OUT / f"refms_{dt:g}.npz", rho=c["rho"][keep], T=c["T"][keep], X0=c["X0"][keep], X1=X1)
        print(f"dt {dt:g}: {keep.sum()} states burned in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "model_dagger_35b.pt"
    {"data": cmd_data, "sweep": cmd_sweep, "manifold": cmd_manifold, "refdata": cmd_refdata, "exact": cmd_exact, "relu": cmd_relu, "relu2": cmd_relu2, "refdt": cmd_refdt, "relu3": cmd_relu3}[what](base)
