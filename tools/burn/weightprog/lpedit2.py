"""A second round of rank-one, linear-program weight edits, closer to "AHA! Weight-Space
Programming" (haithamb.github.io/aha, code github.com/Eppy-EpHard/AHA) and Barry's linear
program (nathan.rs/posts/weight-space-lp) than lpedit.py was:

  matched pairs  AHA contrasts the model's own residuals on two branches of the same prefix,
                 one looping and one not, so that the difference holds the behaviour and not
                 the input. The analogue here, at the same input: the model's own hidden
                 pre-activation z at layer k, and the nearest z + dz from which the rest of
                 the network makes the reference's step (found by gradient descent on dz
                 alone, weights fixed). u_k = normalized mean of dz / |dz| over the bad states.
  edit inside    the edit goes into a hidden layer's weights (AHA edits down_proj, writing
                 into the residual stream), W_k += u v^T, so the push passes through the
                 layers after it, rather than into the output layer.
  push size      from the data: delta = the median of u . dz over the bad states (how far
                 along u the fixes go), times a multiplier (AHA: delta from the gap the layer
                 writes; Barry: 0.5x of the right push brought the loops back, 2x degraded).
  program        minimize |v|_1 + slacks with v . a >= delta on bad states, |v . a| <= eps
                 on good ones (eps = 0.1 delta), the 8 largest |v_j| std(a_j) kept, solved
                 again on those (HiGHS; the layers are 128 wide, so no GPU solver is needed).
  layer choice   every layer k is solved; ranked by hit rate on held-out bad states (the
                 share whose one-step error falls) and the held-out good states' change,
                 before any rollout is looked at; rollouts judge only the top layers.
  constant push  Barry found a constant steering vector nearly as good as the edit: the
                 bias += delta u at the chosen layer is scored as a baseline.

Bad states are the 20% of one regime's reference steps with the largest one-step error, from
the training set's targeted trajectories (targets.npz); the held-out test set is left for
score.py.

    tools/burn/.venv/Scripts/python tools/burn/weightprog/lpedit2.py [base.pt] [kind]

Result on model_dagger_35b, sis (2026-11-16): the fixes are not shared. A small dz fixes each
bad step (error p50 0.017 -> 0.001), but the mean cosine of the fixes with their mean is
0.31-0.41 at every layer. At half the push, the gated edits on layers 2-4 lowered held-out
bad steps' error by 11-12% and left good states within 1-2%. Dev rollouts (the regime's
held-out training trajectories): base sis 0.490 / general 0.331; layer 4 0.405 / 0.332,
layer 3 0.455 / 0.334, layer 2 0.430 / 0.418. Constant pushes broke general (0.50-0.52):
unlike Barry's loops, the gate is needed here. On score.py's held-out test set the gain
did not carry over: sis p90 1.53 -> 1.51, hotco p90 0.35 -> 0.38, general p50 at 1 s
3.8e-4 -> 5.5e-4 (layer 4). Not shipped; 35b stays.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402
from lpedit import KINDS, pairs, quick_eval, solve_gate, tt  # noqa: E402

TOPK = 8
dev = train.dev


def linears(model):
    return [i for i, m in enumerate(model.net) if isinstance(m, torch.nn.Linear)]


def at_layer(model, f, k):
    """a (input to linear k) and z (its output) for features f."""
    idx = linears(model)[k]
    a = model.net[:idx](f)
    return a, model.net[idx](a)


def tail(model, k, z):
    idx = linears(model)[k]
    return model.net[idx + 1:](z)


def step_error(model, zout, X, ld, d):
    P = model.width
    D1 = model.relax(zout[:, :P], X, ld)
    D2 = model.relax(zout[:, P:], (X + D1).clamp_min(0.0), ld)
    return (D1 + D2 - d).abs().sum(dim=1)


def states(model, r, T0, dt, x0, d):
    f = train.features(torch.log10(tt(r)), torch.log10(tt(T0)), torch.log10(tt(dt)), tt(x0).clamp_min(0.0), model.rates)
    return f.detach(), tt(x0).clamp_min(0.0), torch.log10(tt(dt)), tt(d)


def errors(model, f, X, ld, d):
    with torch.no_grad():
        return step_error(model, model.net(f), X, ld, d)


def fixes(model, k, f, X, ld, d, lam=1e-3, iters=300):
    """For each state, dz: the change to linear k's output that brings the step to the
    reference's, kept small (an L2 penalty), the weights untouched."""
    with torch.no_grad():
        _, z = at_layer(model, f, k)
    dz = torch.zeros_like(z, requires_grad=True)
    opt = torch.optim.Adam([dz], lr=0.02)
    for _ in range(iters):
        opt.zero_grad()
        e = step_error(model, tail(model, k, z + dz), X, ld, d)
        (e.sum() + lam * (dz**2).sum()).backward()
        opt.step()
    with torch.no_grad():
        e = step_error(model, tail(model, k, z + dz), X, ld, d)
    return dz.detach(), e


def edited(model, k, u, v, bias=None):
    m = copy.deepcopy(model)
    lin = m.net[linears(m)[k]]
    with torch.no_grad():
        if v is not None:
            lin.weight += torch.outer(tt(u), tt(v))
        if bias is not None:
            lin.bias += tt(bias)
    return m


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "model_dagger_35b.pt"
    kind = KINDS.index(sys.argv[2] if len(sys.argv) > 2 else "sis")
    model = train.load_model(base)
    model.parts = None
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(0)
    tg = np.load(train.HERE / "data" / "targets.npz")
    t = tg["t"]
    sel = tg["kind"] == kind
    idx = np.where(sel)[0]
    rng.shuffle(idx)
    make, judge = idx[: len(idx) // 2], idx[len(idx) // 2:]
    f, X, ld, d = states(model, *pairs(tg["rho"][make], tg["X"][make], tg["T"][make], t))
    err = errors(model, f, X, ld, d)
    bad = torch.where(err > torch.quantile(err, 0.8))[0]
    bad = bad[torch.randperm(len(bad), generator=torch.Generator().manual_seed(0)).to(dev)]
    nb = int(0.8 * len(bad))
    bfit, bhold = bad[:nb], bad[nb:]
    print(f"{KINDS[kind]}: {len(err)} steps, error p50 {err.median():.4f}; {len(bad)} bad "
          f"(error p50 {err[bad].median():.4f}), {len(bhold)} held out", flush=True)

    gen = train.load({"fuel.npz", "mix.npz"} | {f"dagger_{k}.npz" for k in range(30, 39)})
    gi = rng.choice(len(gen[0]), 1500, replace=False)
    rg = pairs(gen[0][gi], gen[1][gi], gen[2][gi], gen[3])
    others = np.where(~sel)[0][:600]
    ro = pairs(tg["rho"][others], tg["X"][others], tg["T"][others], t)
    goodown = np.setdiff1d(np.arange(len(err)), bad.cpu().numpy())
    own = pairs(tg["rho"][make], tg["X"][make], tg["T"][make], t)
    cat = [np.concatenate([a, b, c[goodown]]) for a, b, c in zip(rg, ro, own)]
    pick = rng.permutation(len(cat[0]))[:24000]
    gf, gX, gld, gd = states(model, *[c[pick] for c in cat])
    gfit, ghold = slice(0, 20000), slice(20000, None)
    gerr = errors(model, gf, gX, gld, gd)

    gj = rng.choice(len(gen[0]), 300, replace=False)
    sets = {KINDS[kind]: (tg["rho"][judge], tg["X"][judge], tg["T"][judge], t),
            "general": (gen[0][gj], gen[1][gj], gen[2][gj], gen[3])}
    m0 = copy.deepcopy(model)
    m0.name = base
    print("base rollouts", {k: round(v, 3) for k, v in quick_eval(m0, sets).items()}, flush=True)

    found = []
    for k in range(len(linears(model))):
        dz, efix = fixes(model, k, f[bfit], X[bfit], ld[bfit], d[bfit])
        n = dz.norm(dim=1, keepdim=True)
        u = (dz / n).mean(dim=0)
        u = u / u.norm()
        cons = float(((dz / n) @ u).mean())
        reach = (dz @ u).median().item()
        with torch.no_grad():
            a_b, _ = at_layer(model, f[bfit], k)
            a_g, _ = at_layer(model, gf[gfit], k)
        ab, ag = a_b.cpu().numpy(), a_g.cpu().numpy()
        v1, _ = solve_gate(ab, ag, True, 0.1)
        imp = np.abs(v1) * np.concatenate([ab, ag]).std(axis=0)
        cols = np.argsort(-imp)[:TOPK]
        v2, _ = solve_gate(ab, ag, True, 0.1, cols=cols)
        v = np.zeros(ab.shape[1])
        v[cols] = v2
        print(f"layer {k}: fixes reach error p50 {efix.median():.4f}; u: mean cosine {cons:.2f}, "
              f"push u.dz p50 {reach:.2f}; units {sorted(cols.tolist())}", flush=True)
        for mult in (0.5, 1.0, 2.0):
            m = edited(model, k, u.cpu().numpy(), v * reach * mult)
            eb = errors(m, f[bhold], X[bhold], ld[bhold], d[bhold])
            eg = errors(m, gf[ghold], gX[ghold], gld[ghold], gd[ghold])
            hit = float((eb < err[bhold]).float().mean())
            gain = float(eb.mean() / err[bhold].mean())
            hurt = float(eg.mean() / gerr[ghold].mean())
            g99 = float(torch.quantile(eg, 0.99) / torch.quantile(gerr[ghold], 0.99))
            print(f"  x{mult}: held-out bad: hit rate {hit:.2f}, mean error x{gain:.3f}; "
                  f"held-out good: mean error x{hurt:.3f}, p99 x{g99:.3f}", flush=True)
            found.append((hit - 2 * max(hurt - 1, 0), k, mult, u, v * reach * mult, reach))

    found.sort(key=lambda x: -x[0])
    for score_, k, mult, u, v, reach in found[:3]:
        m = edited(model, k, u.cpu().numpy(), v)
        m.name = base
        res = quick_eval(m, sets)
        print(f"rollouts, edit layer {k} x{mult} (rank score {score_:.2f}): "
              + ", ".join(f"{a} {b:.3f}" for a, b in res.items()), flush=True)
        torch.save({"state": m.state_dict(), "width": 128, "depth": 4, "arch": m.arch},
                   train.HERE / f"lp2_L{k}_x{mult}.pt")
        mc = edited(model, k, u.cpu().numpy(), None, bias=(u * reach * mult).cpu().numpy())
        mc.name = base
        res = quick_eval(mc, sets)
        print(f"rollouts, constant push layer {k} x{mult}: "
              + ", ".join(f"{a} {b:.3f}" for a, b in res.items()), flush=True)


if __name__ == "__main__":
    main()
