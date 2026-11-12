"""Train the burning surrogate: a small network that steps the alpha chain's mass fractions
forward by dt at fixed density, with the self-heating that burner.py integrates.

    tools/burn/.venv/Scripts/python tools/burn/train.py [steps] [width] [depth] [hard] [out.pt]

Input  log10 rho, log10 T, log10 dt (all scaled to about [-1, 1]) and the 13 mass fractions
       (as X^(1/4), which spreads out the trace species).
Output 13 logits, added to log(X + 1e-10); the new fractions are their softmax, so they are
       positive and sum to one whatever the weights are. The energy released is computed from
       the change in composition and the nuclear masses, never predicted, so energy and
       composition always agree.

Pairs come from the trajectories (gen_data.py): any two of a trajectory's times, the earlier
as the input and the later as the target, so every step length is represented.

Writes tools/burn/model.pt and prints validation errors, including a rollout check: the
model applied over and over with hydro-like steps along held-out trajectories.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).parent
DATA = HERE / "data"
torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_A = 6.02214076e23
A = np.array([4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56], dtype=np.float64)
MASS = None  # set from aprox13.py below
LOG_RHO = (4.0, 10.0)
LOG_T = (8.0, 10.5)
LOG_DT = (-12.0, 0.0)


def nuclear_q():
    """Energy per gram released per unit mass fraction moved into each species, relative to
    C12, in units of 1e18 erg/g."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("aprox13", HERE / "aprox13.py")
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    q = -N_A * net.mass / net.A  # erg/g per unit X (rest-mass energy, negated)
    return (q - q[1]) / 1e18


Q = torch.tensor(nuclear_q(), dtype=torch.float32, device=dev)


def scale(v, lo_hi):
    lo, hi = lo_hi
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def features(log_rho, log_T, log_dt, X):
    return torch.cat([
        scale(log_rho, LOG_RHO)[:, None],
        scale(log_T, LOG_T)[:, None],
        scale(log_dt, LOG_DT)[:, None],
        X.clamp_min(0.0) ** 0.25,
    ], dim=1)


class Burner(nn.Module):
    """The step as a relaxation toward a target: X' = X + (1 - exp(-k dt)) (Y - X), with the
    target Y (a softmax over 13 logits added to log X) and the rate k both from the network.
    As dt goes to zero the change does too, whatever the weights, so a cell that is not
    burning keeps its composition to rounding however many steps it takes; a step much longer
    than the burn reaches Y. (Predicting X' directly leaves a floor of about 1e-4 in every
    step that adds up over thousands of them.)"""

    def __init__(self, width=64, depth=3):
        super().__init__()
        layers, n = [], 16
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.SiLU()]
            n = width
        layers.append(nn.Linear(n, 14))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, log_rho, log_T, log_dt, X):
        z = self.net(features(log_rho, log_T, log_dt, X))
        Y = torch.softmax(torch.log(X.clamp_min(0.0) + 1e-10) + z[:, :13], dim=1)
        log_k = 5.0 + 5.0 * z[:, 13]                       # log10 of the rate, 1/s
        kdt = torch.pow(10.0, (log_k + log_dt).clamp(max=6.0))
        c = -torch.expm1(-kdt)
        return X + c[:, None] * (Y - X)


def load():
    """Every trajectory file in data/: fuel and mixture starts (gen_data.py) and the states
    earlier models wandered into (dagger.py)."""
    files = sorted(f for f in DATA.glob("*.npz") if f.name != "fuel_pool.npz")
    parts = [np.load(f) for f in files]
    rho = np.concatenate([p["rho"] for p in parts])
    X = np.concatenate([p["X"] for p in parts]).astype(np.float32)
    T = np.concatenate([p["T"] for p in parts])
    t = parts[0]["t"]
    keep = np.isfinite(T).all(axis=1) & np.isfinite(X).all(axis=(1, 2))
    return rho[keep], X[keep], T[keep], t


class Pairs:
    def __init__(self, rho, X, T, t):
        self.log_rho = torch.tensor(np.log10(rho), dtype=torch.float32, device=dev)
        self.X = torch.tensor(X, device=dev)
        self.log_T = torch.tensor(np.log10(np.clip(T, 1e7, 2e10)), dtype=torch.float32, device=dev)
        self.t = torch.tensor(t, dtype=torch.float64, device=dev)
        self.n, self.k = X.shape[0], X.shape[1]

    def sample(self, b):
        tr = torch.randint(0, self.n, (b,), device=dev)
        i = torch.randint(0, self.k - 1, (b,), device=dev)
        j = i + 1 + (torch.rand(b, device=dev) * (self.k - 1 - i).float()).long()
        dt = (self.t[j] - self.t[i]).float()
        return (self.log_rho[tr], self.log_T[tr, i], torch.log10(dt), self.X[tr, i]), self.X[tr, j]


def loss_fn(Xp, Xt, X0, hard=0.0):
    l_x = (Xp - Xt).abs().sum(dim=1)                       # composition, absolute
    e_p = ((Xp - X0) * Q).sum(dim=1)
    e_t = ((Xt - X0) * Q).sum(dim=1)
    l_e = (e_p - e_t).abs()                                # energy released, 1e18 erg/g
    # Pairs over which the composition changes a lot (ignition, the march to the iron group)
    # are few and the hardest; `hard` weights them up.
    w = 1.0 + hard * (Xt - X0).abs().sum(dim=1)
    return (w * (l_x + l_e)).mean() / w.mean(), l_x, l_e


@torch.no_grad()
def evaluate(model, pairs, n=200_000):
    model.eval()
    (r, T, dt, X0), Xt = pairs.sample(n)
    Xp = model(r, T, dt, X0)
    _, l_x, l_e = loss_fn(Xp, Xt, X0)
    model.train()
    q = torch.tensor([0.5, 0.99, 0.999], device=dev)
    return {"X_l1": torch.quantile(l_x, q).tolist(), "e_err": torch.quantile(l_e, q).tolist(), "X_l1_max": l_x.max().item(), "e_err_max": l_e.max().item()}


KB, MU, ARAD = 1.380649e-16, 1.66053906660e-24, 7.5657e-15
A_T = torch.tensor(A, dtype=torch.float32, device=dev)
Z_T = torch.tensor([2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=torch.float32, device=dev)


def heat(X0, X1, rho, T):
    """Temperature after the composition changes from X0 to X1 at density rho, the energy it
    released going into heat (one explicit step of the specific heat, as burner.py's)."""
    de = ((X1 - X0) * Q).sum(dim=1) * 1e18
    cv = 1.5 * KB / MU * (X1 / A_T * (Z_T + 1)).sum(dim=1) + 4 * ARAD * T**3 / rho
    return (T + de / cv).clamp(1e7, 3e10)


@torch.no_grad()
def roll(model, rho, X0, T0, dt, steps, record=0):
    """Apply the model `steps` times with step dt (per row), the temperature following the
    energy released, as the simulator's would. Returns the final (X, T) and, if record > 0,
    that many states visited along the way (rho, T, X) at random steps."""
    X = torch.as_tensor(X0, dtype=torch.float32, device=dev).clone()
    T = torch.as_tensor(T0, dtype=torch.float32, device=dev).clone()
    r = torch.as_tensor(rho, dtype=torch.float32, device=dev)
    lr, ld = torch.log10(r), torch.log10(torch.as_tensor(dt, dtype=torch.float32, device=dev))
    seen = []
    at = set(np.random.default_rng(2).choice(steps, size=min(record, steps), replace=False).tolist()) if record else set()
    for s in range(steps):
        X1 = model(lr, torch.log10(T), ld, X)
        T = heat(X, X1, r, T)
        X = X1
        if s in at:
            seen.append((r.clone(), T.clone(), X.clone()))
    return X, T, seen


@torch.no_grad()
def rollout(model, rho, X, T, t, dt_hydro):
    """Apply the model with steps of dt_hydro from each trajectory's start to t = 0.1 s
    (composing it many times, the temperature following the energy released), and compare
    with the reference there."""
    model.eval()
    k_end = int(np.searchsorted(t, 0.1))
    steps = int(round(t[k_end] / dt_hydro))
    Xc, _, _ = roll(model, rho, X[:, 0], T[:, 0], np.full(len(rho), dt_hydro), steps)
    Xt = torch.tensor(X[:, k_end], device=dev)
    X0 = torch.tensor(X[:, 0], device=dev)
    _, l_x, l_e = loss_fn(Xc, Xt, X0)
    model.train()
    return {"steps": steps, "X_l1_p50": torch.quantile(l_x, 0.5).item(), "X_l1_p99": torch.quantile(l_x, 0.99).item(), "X_l1_max": l_x.max().item(), "e_err_p99": torch.quantile(l_e, 0.99).item(), "e_err_max": l_e.max().item()}


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    hard = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    tag = f"w{width}d{depth}h{hard:g}"
    rho, X, T, t = load()
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(rho))
    nval = len(rho) // 20
    va, tr = idx[:nval], idx[nval:]
    train, val = Pairs(rho[tr], X[tr], T[tr], t), Pairs(rho[va], X[va], T[va], t)
    print(f"{len(tr)} training and {nval} validation trajectories, {X.shape[1]} times each, on {dev}")

    model = Burner(width, depth).to(dev)
    print(sum(p.numel() for p in model.parameters()), "parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for s in range(steps):
        (r, Tl, dt, X0), Xt = train.sample(8192)
        loss, _, _ = loss_fn(model(r, Tl, dt, X0), Xt, X0, hard)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (s + 1) % 5000 == 0 or s == 0:
            print(f"step {s + 1}: loss {loss.item():.4g}  val {evaluate(model, val)}  {time.time() - t0:.0f} s", flush=True)
    out = sys.argv[5] if len(sys.argv) > 5 else f"model_{tag}.pt"
    torch.save({"state": model.state_dict(), "width": width, "depth": depth}, HERE / out)
    for dt in (1e-5, 1e-3, 1e-2):
        print(f"rollout dt={dt:g}:", rollout(model, rho[va], X[va], T[va], t, dt), flush=True)


if __name__ == "__main__":
    main()
