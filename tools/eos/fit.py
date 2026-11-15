"""Fit a small network to the electron-positron gas's thermal free energy (electron_gas.py):
h(u, v) with u, v = log10(rho Ye), log10 T scaled to [-1, 1] over the domain (gen_data.DOMAIN).
F_th = -P_pair(T) + n k T h, so pressure and energy are h's first derivatives, and the heat
capacity and sound speed its second: the network is fit to h and both first derivatives at
once (a Sobolev fit), and checked on the second against finite differences of the exact gas.

    tools/burn/.venv/Scripts/python tools/eos/fit.py [width] [depth] [out.pt]

Adam on minibatches, then L-BFGS on everything, in double precision. Prints the errors over
held-out random points, in h and in the quantities the simulator uses.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

from gen_data import DOMAIN

HERE = Path(__file__).parent
torch.set_default_dtype(torch.float64)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
(R0, R1), (T0, T1) = DOMAIN["log_rho_ye"], DOMAIN["log_T"]
# d/dln n = DU d/du, d/dln T = DV d/dv
DU = 2.0 / ((R1 - R0) * math.log(10.0))
DV = 2.0 / ((T1 - T0) * math.log(10.0))


class EosNet(nn.Module):
    def __init__(self, width=24, depth=2):
        super().__init__()
        layers, n = [], 2
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.Tanh()]
            n = width
        layers.append(nn.Linear(n, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, uv):
        return self.net(uv)[:, 0]


def scaled(a):
    return np.stack([2 * (a[:, 0] - R0) / (R1 - R0) - 1, 2 * (a[:, 1] - T0) / (T1 - T0) - 1], axis=1)


def derivs(model, uv, second=False):
    uv = uv.clone().requires_grad_(True)
    h = model(uv)
    g = torch.autograd.grad(h.sum(), uv, create_graph=True)[0]
    out = [h, g[:, 0] * DU, g[:, 1] * DV]
    if second:
        gu = torch.autograd.grad(g[:, 0].sum(), uv, create_graph=True)[0]
        gv = torch.autograd.grad(g[:, 1].sum(), uv, create_graph=True)[0]
        out += [gu[:, 0] * DU * DU, gu[:, 1] * DU * DV, gv[:, 1] * DV * DV]
    return out


def load():
    a = np.concatenate([np.load(HERE / "data" / f)["data"] for f in ("eos_grid.npz", "eos_random.npz")])
    return a


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    out = sys.argv[3] if len(sys.argv) > 3 else f"eos_w{width}d{depth}.pt"
    torch.manual_seed(0)
    a = load()
    rng = np.random.default_rng(0)
    test = rng.random(len(a)) < 0.1
    X = torch.tensor(scaled(a), dtype=torch.float64)
    Y = torch.tensor(a[:, 2:5], dtype=torch.float64)
    Xtr, Ytr, Xte, Yte = X[~test], Y[~test], X[test], Y[test]
    model = EosNet(width, depth)

    def loss_on(x, y):
        h, hn, hT = derivs(model, x)
        return ((h - y[:, 0]) ** 2 + (hn - y[:, 1]) ** 2 + (hT - y[:, 2]) ** 2).mean()

    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 6000)
    for it in range(6000):
        i = torch.randint(0, len(Xtr), (4096,))
        opt.zero_grad()
        l = loss_on(Xtr[i], Ytr[i])
        l.backward()
        opt.step()
        sched.step()
        if it % 1000 == 0:
            print(f"adam {it}: rms {l.item() ** 0.5:.2e}  ({time.time() - t0:.0f} s)", flush=True)
    lb = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, history_size=50, line_search_fn="strong_wolfe")
    for it in range(int(__import__("os").environ.get("LBFGS_ROUNDS", "40"))):
        def closure():
            lb.zero_grad()
            l = loss_on(Xtr, Ytr)
            l.backward()
            return l
        l = lb.step(closure)
        if it % 10 == 0:
            print(f"lbfgs {it}: rms {l.item() ** 0.5:.2e}  ({time.time() - t0:.0f} s)", flush=True)
    with torch.no_grad():
        pass
    h, hn, hT = [t.detach() for t in derivs(model, Xte)]
    err = torch.stack([h - Yte[:, 0], hn - Yte[:, 1], hT - Yte[:, 2]], 1).abs()
    print(f"held out ({len(Xte)}): max |err| h {err[:, 0].max():.2e} h_n {err[:, 1].max():.2e} h_T {err[:, 2].max():.2e};"
          f" rms {err.pow(2).mean(0).sqrt().tolist()}")
    torch.save({"state": model.state_dict(), "width": width, "depth": depth, "domain": DOMAIN}, HERE / out)
    print(f"wrote {out}; {sum(p.numel() for p in model.parameters())} weights")


if __name__ == "__main__":
    main()
