"""Fit the inverse network: (log rho, log e_th, log Abar) -> (log T, p_th / (rho e_th), Gamma_th)
with Gamma_th = (c_s^2 - c0^2) rho / p_th, c0^2 the zero-temperature electrons' share (exact,
cheap). One forward pass gives the hydro everything it needs, with no Newton iteration.
Data from gen_inverse.py.

    tools/burn/.venv/Scripts/python tools/eos/fit_inverse.py [width] [depth] [out.pt]

Errors are weighted by what matters: log T to 4e-4 dex (0.1%), the pressure ratio to 1e-4, and
Gamma_th to 1e-3. Adam, then L-BFGS, in double precision.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

HERE = Path(__file__).parent
torch.set_default_dtype(torch.float64)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
TOL = torch.tensor([4e-4, 1e-4, 1e-3])


def prepare(a):
    """Inputs and targets from the data's columns; returns them with the input scaling."""
    lr, lt, la, le, p, cs2, c0 = a.T
    rho = 10.0 ** lr
    x = np.stack([lr, le, la], 1)
    y = np.stack([lt, p / (rho * 10.0 ** le), (cs2 - c0) * rho / p], 1)
    return x, y


class InvNet(nn.Module):
    def __init__(self, lo, hi, width=64, depth=2):
        super().__init__()
        self.register_buffer("lo", torch.tensor(lo))
        self.register_buffer("hi", torch.tensor(hi))
        layers, n = [], 3
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.Tanh()]
            n = width
        layers.append(nn.Linear(n, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(2 * (x - self.lo) / (self.hi - self.lo) - 1)


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    out = sys.argv[3] if len(sys.argv) > 3 else f"inv_w{width}d{depth}.pt"
    torch.manual_seed(0)
    a = np.load(HERE / "data" / "inverse.npz")["data"]
    x, y = prepare(a)
    test = np.random.default_rng(0).random(len(x)) < 0.1
    # The targets' own offsets (log T spans 7 dex): the network learns them scaled.
    ylo, yhi = y.min(0), y.max(0)
    X, Y = torch.tensor(x), torch.tensor((y - ylo) / (yhi - ylo))
    tol = TOL / torch.tensor(yhi - ylo)
    model = InvNet(x.min(0), x.max(0), width, depth)
    Xtr, Ytr, Xte, Yte = X[~test], Y[~test], X[test], Y[test]
    loss_on = lambda xx, yy: (((model(xx) - yy) / tol) ** 2).mean()
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 8000)
    for it in range(8000):
        i = torch.randint(0, len(Xtr), (4096,))
        opt.zero_grad()
        l = loss_on(Xtr[i], Ytr[i])
        l.backward()
        opt.step()
        sched.step()
        if it % 2000 == 0:
            print(f"adam {it}: rms in tolerances {l.item() ** 0.5:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    lb = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, history_size=50, line_search_fn="strong_wolfe")
    for it in range(int(os.environ.get("LBFGS_ROUNDS", "60"))):
        def closure():
            lb.zero_grad()
            l = loss_on(Xtr, Ytr)
            l.backward()
            return l
        l = lb.step(closure)
        if it % 20 == 0:
            print(f"lbfgs {it}: rms in tolerances {l.item() ** 0.5:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    with torch.no_grad():
        e = ((model(Xte) - Yte) * torch.tensor(yhi - ylo)).abs()
    names = ["log T (dex)", "p_th/(rho e_th)", "Gamma_th"]
    for k, nm in enumerate(names):
        print(f"held out {nm}: rms {e[:, k].pow(2).mean().sqrt():.1e}  p99 {e[:, k].quantile(0.99):.1e}  max {e[:, k].max():.1e}")
    torch.save({"state": model.state_dict(), "width": width, "depth": depth, "ylo": ylo, "yhi": yhi}, HERE / out)
    print(f"wrote {out}; {sum(p.numel() for p in model.parameters())} weights")


if __name__ == "__main__":
    main()
