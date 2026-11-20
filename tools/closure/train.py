"""Train the closure through the solver: unrolled, not per-step regression.

A sample is a window of the restricted reference (gen_data.py): a coarse state at one output
time and the reference at the next `m` outputs, each 10 to 50 coarse steps on. The coarse
solver (solver.py) is run from the first across the window with the closure in it, and the
loss compares what it reaches at each output with the reference:

  state      the primitives, point by point (each over the reference's own variance)
  spectrum   the kinetic energy and the scalar's variance per wavenumber shell, in the log
  mixing     the scalar's mixedness 4 <s (1 - s)> and the mixing layers' thickness
  range      the scalar leaving the range it has in the reference (the price of `b`)

Each of the first three is divided by what the solver scores without the closure on the same
window, so a loss of 1 per term is the baseline and the weights mean what they say.

Its own drift. Half of the samples (after the first phase) start not from the reference but
from where the model itself has carried the reference over one to three earlier outputs,
without gradient: the states it must then correct are the ones it produces, as in DAgger, and
the targets stay the reference's. The unrolled window lengthens over the phases (1, 2, then 3
outputs: up to about 150 steps differentiated).

    tools/closure/.venv/bin/python tools/closure/train.py --name full --heads abnk
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from closure import A_MAX, B_MAX, Closure
from solver import Config, advance, primitives

HERE = Path(__file__).resolve().parent


def load_set(name, device="cpu"):
    out = []
    for f in sorted((HERE / "data" / name).glob("*.npz")):
        d = np.load(f)
        out.append({"name": f.stem, "t": torch.from_numpy(d["t"]), "U": torch.from_numpy(d["U"]).to(device), "case": json.loads(str(d["case"]))})
    return out


class Shells:
    """Sums over shells of whole wavenumbers, for a grid of n x n."""

    def __init__(self, n, dtype):
        k = torch.fft.fftfreq(n, 1.0 / n)
        self.idx = torch.round(torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)).long().flatten()
        self.n = n
        self.kmax = n // 2
        self.dtype = dtype

    def __call__(self, power):
        """power[B, n, n] -> [B, kmax + 1]"""
        B = power.shape[0]
        out = torch.zeros(B, int(self.idx.max()) + 1, dtype=power.dtype)
        out = out.index_add(1, self.idx, power.reshape(B, -1))
        return out[:, : self.kmax + 1]


def stats(U, cfg, shells):
    """What the losses compare: the primitives, the two spectra and the mixing numbers."""
    rho, vx, vy, p, s, cs = primitives(U, cfg)
    w = torch.sqrt(rho)
    n = U.shape[-1]
    ke = 0.5 * (torch.fft.fft2(w * vx).abs() ** 2 + torch.fft.fft2(w * vy).abs() ** 2) / n**4
    sv = torch.fft.fft2(s - s.mean(dim=(-1, -2), keepdim=True)).abs() ** 2 / n**4
    sbar = s.mean(dim=-1)
    return {
        "prim": torch.stack([rho, vx, vy, p, s], dim=1),
        "E": shells(ke), "S": shells(sv),
        "mixed": (4 * s * (1 - s)).mean(dim=(-1, -2)), "thick": (4 * sbar * (1 - sbar)).mean(dim=-1),
        "s": s,
    }


def terms(a, r):
    """The loss terms of states with statistics `a` against the reference's `r`, each [B]."""
    d = a["prim"] - r["prim"]
    var = r["prim"].var(dim=(-1, -2)) + torch.tensor([1e-6, 0, 0, 1e-6, 1e-6]).view(1, 5)
    v2 = (r["prim"][:, 1] ** 2 + r["prim"][:, 2] ** 2).mean(dim=(-1, -2))
    mse = (d * d).mean(dim=(-1, -2))
    state = (mse[:, 0] / var[:, 0] + (mse[:, 1] + mse[:, 2]) / v2 + 0.25 * mse[:, 3] / var[:, 3] + mse[:, 4] / var[:, 4]) / 3.25

    def logspec(x, y):
        eps = 1e-9 * y[:, 1:].sum(dim=1, keepdim=True) + 1e-30
        return ((torch.log(x[:, 2:] + eps) - torch.log(y[:, 2:] + eps)) ** 2).mean(dim=1)

    spec = 0.5 * (logspec(a["E"], r["E"]) + logspec(a["S"], r["S"]))
    mix = (a["mixed"] - r["mixed"]) ** 2 + (a["thick"] - r["thick"]) ** 2
    lo = r["s"].amin(dim=(-1, -2), keepdim=True)
    hi = r["s"].amax(dim=(-1, -2), keepdim=True)
    rng = (torch.relu(a["s"] - hi) ** 2 + torch.relu(lo - a["s"]) ** 2).mean(dim=(-1, -2)) / (r["s"].var(dim=(-1, -2)) + 1e-6)
    return {"state": state, "spec": spec, "mix": mix, "range": rng}


def sample(data, batch, m, lead, rng):
    """Windows: a batch of starting states, their references at the next lead + m outputs and
    the time between outputs."""
    U0, refs, span = [], [], []
    for _ in range(batch):
        d = data[rng.integers(len(data))]
        k = int(rng.integers(0, len(d["t"]) - m - lead))
        U0.append(d["U"][k])
        refs.append(d["U"][k + 1 : k + 1 + lead + m])
        span.append(float(d["t"][k + 1] - d["t"][k]))
    return torch.stack(U0), torch.stack(refs, dim=1), torch.tensor(span)


def window_loss(model, cfg, shells, U0, refs, span, m, lead, weights):
    """The loss of one batch of windows, and its terms for the log."""
    U = U0
    with torch.no_grad():
        for j in range(lead):
            U, _ = advance(U, cfg, span, model)
    V = U.clone()  # the solver without the closure, from the same start
    total = 0.0
    log = {k: 0.0 for k in ("state", "spec", "mix", "range")}
    for j in range(m):
        U, _ = advance(U, cfg, span, model)
        with torch.no_grad():
            V, _ = advance(V, cfg, span)
            r = stats(refs[lead + j], cfg, shells)
            base = terms(stats(V, cfg, shells), r)
        got = terms(stats(U, cfg, shells), r)
        for k in ("state", "spec", "mix"):
            rel = got[k] / (base[k] + 1e-12 * (1 + base[k].mean()))
            total = total + weights[k] * rel.mean() / m
            log[k] += float(rel.mean().detach()) / m
        total = total + weights["range"] * got["range"].mean() / m
        log["range"] += float(got["range"].mean().detach()) / m
    return total, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full")
    ap.add_argument("--heads", default="abnk")
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=900)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--w-state", type=float, default=1.0)
    ap.add_argument("--w-spec", type=float, default=1.0)
    ap.add_argument("--w-mix", type=float, default=0.5)
    ap.add_argument("--w-range", type=float, default=200.0)
    ap.add_argument("--init", default=None, help="a model to start from")
    ap.add_argument("--fixed-ab", action="store_true", help="hold a and b at their limits (the constants) and learn only the other heads")
    ap.add_argument("--phases", default="0.3:1:0,0.35:2:2,0.35:3:3",
                    help="share of the iterations : outputs unrolled : most outputs of the model's own drift, per phase")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cfg = Config()
    data = load_set("train")
    n = data[0]["U"].shape[-1]
    shells = Shells(n, torch.float32)
    model = Closure(a.hidden, a.layers, a.heads, (A_MAX, B_MAX) if a.fixed_ab else None)
    if a.init:
        model.load_state_dict(torch.load(HERE / "models" / f"{a.init}.pt")["state"])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    weights = {"state": a.w_state, "spec": a.w_spec, "mix": a.w_mix, "range": a.w_range}
    out = HERE / "models"
    out.mkdir(exist_ok=True)
    log = open(HERE / f"train_{a.name}.log", "a")
    phases = [(float(x), int(y), int(z)) for x, y, z in (p.split(":") for p in a.phases.split(","))]
    it = 0
    t0 = time.time()
    for share, m, max_lead in phases:
        for _ in range(int(round(share * a.iters))):
            lr = a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / a.iters)))
            for g in opt.param_groups:
                g["lr"] = lr
            lead = int(rng.integers(1, max_lead + 1)) if max_lead and rng.random() < 0.5 else 0
            U0, refs, span = sample(data, a.batch, m, lead, rng)
            loss, terms_ = window_loss(model, cfg, shells, U0, refs, span, m, lead, weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                opt.step()
            it += 1
            line = f"{it:5d} m={m} lead={lead} loss {float(loss.detach()):.4f} " + " ".join(f"{k} {v:.4f}" for k, v in terms_.items()) + f" lr {lr:.1e} {time.time() - t0:.0f}s"
            print(line, file=log, flush=True)
            if it % 10 == 0:
                print(line, flush=True)
            if it % 50 == 0 or it == a.iters:
                torch.save({"state": model.state_dict(), "hidden": a.hidden, "layers": a.layers, "heads": a.heads, "fixed": a.fixed_ab, "iters": it, "args": vars(a)}, out / f"{a.name}.pt")


if __name__ == "__main__":
    main()
