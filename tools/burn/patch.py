"""Patch a trained surrogate where it fails without undoing what it does well: fine-tune it
on the targeted trajectories (gen_targets.py, data/targets.npz) while it keeps fitting all
the other data and, on states near those, keeps matching its own steps from before the patch.

    tools/burn/.venv/Scripts/python tools/burn/patch.py init.pt out.pt [mode] [steps] [lr] [tw] [anchor] [rank] [cw] [jw] [ow]

mode: which weights change.
  head  the output layer only (the last hidden layer's features are kept: does the network
        already compute what the fix needs?)
  lora  a rank-`rank` change to every layer, merged into the weights at the end
  full  every weight
tw: weight of the targeted batch against the general one. anchor: weight of matching the
unpatched model near the general batch's states.

Three optional terms, each off at weight 0:
  cw  consistency: one step of 2 dt against two steps of dt (the temperature following the
      first's energy, as in the simulator), at steps of 1e-8 to 1e-2 s. Needs no reference.
      Over many short steps a small bias in each adds up (rollouts at 1e-5 s steps end far
      worse than at 1e-3 s); a model whose steps compose has none to add.
  jw  temperature slope: the difference between the steps from one state at two
      temperatures 1% apart against the reference's (tpairs.npz, gen_tpairs.py). Fitting the
      steps does not fit their slope in T, and that slope sets when simmering gas runs away.
  ow  overshoot: on trajectories that end at (quasi-)equilibrium, any change in a species
      beyond where the reference ends, in the direction it is going (Si/S ash burned to
      Ni56 past the Si-group equilibrium the reference stops at).
"""
import copy
import sys
import time

import numpy as np
import torch
from torch import nn

import train
from train import HERE, Q, Pairs, dev, heat, load, load_model, log_loss, loss_fn, nearby, reservoir_loss

HARD, REL, FRESH, AUX, LOGW, RESW = 4.0, 0.3, 0.2, 1.0, 0.3, 1.0   # as run_dagger.ps1 trains


class LoRA(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T

    def merged(self):
        with torch.no_grad():
            self.base.weight += self.B @ self.A
        return self.base


def sample_ends(p, b):
    """As Pairs.sample, with each pair's trajectory's last state and whether the trajectory
    has settled there (changed by less than 1e-3 over its last five times)."""
    tr = torch.randint(0, p.n, (b,), device=dev)
    i = torch.randint(0, p.k - 1, (b,), device=dev)
    i = torch.where(torch.rand(b, device=dev) < FRESH, torch.zeros_like(i), i)
    j = i + 1 + (torch.rand(b, device=dev) * (p.k - 1 - i).float()).long()
    D = p.X[tr, j] - p.X[tr, i]
    end = p.X[tr, -1]
    settled = (end - p.X[tr, -6]).abs().sum(dim=1) < 1e-3
    inputs = (p.log_rho[tr], p.log_T[tr, i], torch.log10((p.t[j] - p.t[i]).float()), p.X[tr, i].float())
    return inputs, D.float(), (D * train.Q64).sum(dim=1).float(), (end - p.X[tr, i]).float(), settled


class TPairs:
    def __init__(self, path):
        d = np.load(path)
        self.log_rho = torch.tensor(np.log10(d["rho"]), dtype=torch.float32, device=dev)
        self.t = torch.tensor(d["t"], dtype=torch.float64, device=dev)
        self.X = [torch.tensor(d[k], dtype=torch.float64, device=dev) for k in ("X0", "X1")]
        self.log_T = [torch.tensor(np.log10(d[k][:, 0]), dtype=torch.float32, device=dev) for k in ("T0", "T1")]
        self.n = len(self.log_rho)

    def sample(self, b):
        tr = torch.randint(0, self.n, (b,), device=dev)
        j = torch.randint(1, len(self.t), (b,), device=dev)
        diff = (self.X[1][tr, j] - self.X[1][tr, 0]) - (self.X[0][tr, j] - self.X[0][tr, 0])
        return self.log_rho[tr], self.log_T[0][tr], self.log_T[1][tr], torch.log10(self.t[j].float()), \
            self.X[0][tr, 0].float(), diff.float(), (diff * train.Q64).sum(dim=1).float()


def main():
    a = sys.argv
    init, out = a[1], a[2]
    mode = a[3] if len(a) > 3 else "lora"
    steps = int(a[4]) if len(a) > 4 else 10000
    lr = float(a[5]) if len(a) > 5 else 3e-4
    tw = float(a[6]) if len(a) > 6 else 1.0
    anchor = float(a[7]) if len(a) > 7 else 1.0
    rank = int(a[8]) if len(a) > 8 else 8
    cw = float(a[9]) if len(a) > 9 else 0.0
    jw = float(a[10]) if len(a) > 10 else 0.0
    ow = float(a[11]) if len(a) > 11 else 0.0
    torch.manual_seed(0)
    names = {f.name for f in train.DATA.glob("*.npz")} - {"targets.npz", "fuel_pool.npz"}
    general = Pairs(*load(names))
    targets = Pairs(*load({"targets.npz"}))
    tpairs = TPairs(HERE / "tpairs.npz") if jw > 0 else None
    print(f"{general.n} general and {targets.n} targeted trajectories; mode {mode}, {steps} steps, lr {lr:g}, "
          f"tw {tw:g}, anchor {anchor:g}, consistency {cw:g}, T slope {jw:g}, overshoot {ow:g}")

    model = load_model(init)
    model.train()
    before = copy.deepcopy(model).eval()
    for p in model.parameters():
        p.requires_grad_(mode == "full")
    if mode == "head":
        for p in model.net[-1].parameters():
            p.requires_grad_(True)
    elif mode == "lora":
        for i, m in enumerate(model.net):
            if isinstance(m, nn.Linear):
                model.net[i] = LoRA(m, rank)
    params = [p for p in model.parameters() if p.requires_grad]
    print(sum(p.numel() for p in params), "trainable parameters")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    def step_loss(r, Tl, dt, X0, Dt, et):
        Dp = model(r, Tl, dt, X0)
        loss, _, _ = loss_fn(Dp, Dt, et, HARD, REL, model.parts, AUX)
        return loss + LOGW * log_loss(Dp, Dt, et) + RESW * reservoir_loss(Dp, Dt, X0), Dp

    def overshoot(Dp, rem, settled):
        beyond = torch.relu(Dp * torch.sign(rem) - rem.abs()).sum(dim=1)
        return (beyond * settled).sum() / settled.sum().clamp_min(1)

    def consistency(r, Tl, X0):
        ld = -8.0 + 6.0 * torch.rand(len(r), device=dev)
        # The long step is the target, held fixed (the data train it at every step length):
        # with gradients through both sides the term pulled both into a state where they
        # agree and fit nothing (the general loss from 1.5 to 41 in 300 steps, twice).
        with torch.no_grad():
            D2 = model(r, Tl, ld + np.log10(2.0), X0)
        Da = model(r, Tl, ld, X0)
        X1 = (X0 + Da).clamp_min(0.0)
        # The second step's state goes in without its gradient: the inputs' X^(1/4) and log X
        # have infinite slope at a species' zero (every step was skipped), and through the
        # temperature (heat's Newton iterations) one batch's gradient threw the model into a
        # state it never left (the general loss from 1.5 to 34 within 1000 steps).
        with torch.no_grad():
            T1 = heat(Da, X1, torch.pow(10.0, r), torch.pow(10.0, Tl))
        Db = model(r, torch.log10(T1), ld, X1.detach())
        Dab = (X1 + Db).clamp_min(0.0) - X0
        return log_loss(D2, Dab, (Dab * Q).sum(dim=1))

    def slope():
        r, T0, T1, ld, X0, diff, ediff = tpairs.sample(4096)
        Dd = model(r, T1, ld, X0) - model(r, T0, ld, X0)
        return log_loss(Dd, diff, ediff)

    t0 = time.time()
    extra = {}
    for s in range(steps):
        (r, Tl, dt, X0), Dt, et, rem, settled = sample_ends(general, 8192)
        loss_g, Dp = step_loss(r, Tl, dt, X0, Dt, et)
        (tr_, tT, tdt, tX), tD, te, trem, tset = sample_ends(targets, 4096)
        loss_t, tDp = step_loss(tr_, tT, tdt, tX, tD, te)
        loss = loss_g + tw * loss_t
        if anchor > 0:
            near = nearby(r[:4096], Tl[:4096], dt[:4096], X0[:4096])
            with torch.no_grad():
                Dq = before(*near)
            loss = loss + anchor * step_loss(*near, Dq, (Dq * Q).sum(dim=1))[0]
        if ow > 0:
            extra["overshoot"] = overshoot(Dp, rem, settled) + overshoot(tDp, trem, tset)
            loss = loss + ow * extra["overshoot"]
        if cw > 0:
            extra["consistency"] = consistency(torch.cat([r[:2048], tr_[:1024]]), torch.cat([Tl[:2048], tT[:1024]]),
                                               torch.cat([X0[:2048], tX[:1024]]))
            loss = loss + cw * extra["consistency"]
        if jw > 0:
            extra["slope"] = slope()
            loss = loss + jw * extra["slope"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if all(p.grad is None or torch.isfinite(p.grad).all() for p in params) and torch.isfinite(loss):
            # Tighter than train.py's 100: a patch should move the weights a little at a time.
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
        sched.step()
        if (s + 1) % 1000 == 0 or s == 0 or (s < 1000 and (s + 1) % 100 == 0):
            more = "".join(f"  {k} {v.item():.4g}" for k, v in extra.items())
            print(f"step {s + 1}: general {loss_g.item():.4g}  targeted {loss_t.item():.4g}{more}  {time.time() - t0:.0f} s", flush=True)
    if mode == "lora":
        for i, m in enumerate(model.net):
            if isinstance(m, LoRA):
                model.net[i] = m.merged()
    ck = torch.load(HERE / init)
    torch.save({"state": model.state_dict(), "width": ck["width"], "depth": ck["depth"], "arch": ck["arch"]}, HERE / out)
    print("wrote", out)


if __name__ == "__main__":
    main()
