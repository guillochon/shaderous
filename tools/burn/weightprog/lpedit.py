"""Weight-space edits of a trained Burner, fusing our gated paths (weightprog.py) with the
rank-one, linear-program edits of "AHA! Weight-Space Programming" (haithamb.github.io/aha):

  bad states   reference steps where the model's own one-step error is large, in one regime
               (Si/S quasi-equilibrium by default), from the training set's targeted
               trajectories (targets.npz); the held-out test set is left for scoring
  direction u  in output space, from the reference: minus the mean of the normalized gradients
               of the step error with respect to the network's outputs, over the bad states
  gate v       a sparse linear program (scipy HiGHS): minimize |v|_1 + C (slacks) so that the
               gate reaches 1 on bad states and stays near 0 on a large sample of good states
               from every regime; the k largest |v_j| std(x_j) kept, and the program solved again
               on those alone
  A (rank-one) the output layer's weights W += alpha u v^T, v reading the last hidden layer;
               near zero (within eps) on good states
  B (path)     a parallel path as weightprog.py builds them, its gate g = relu(v . x - b) on the
               network's inputs, carried through the hidden layers, adding alpha g u to the outputs

    tools/burn/.venv/Scripts/python tools/burn/weightprog/lpedit.py base.pt [kind]

Writes lp_A_<alpha>.pt and lp_B_<alpha>.pt for a few pushes alpha, and prints a quick dev
evaluation of each (rollouts of training trajectories, float16 as score.py) beside the base.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import score  # noqa: E402
import train  # noqa: E402

KINDS = ["oneb", "sis", "hotco", "nse"]
A, B = 20.0, 20.0           # as weightprog.py: relu(y) as silu(A y) / A; carried as g + B
TOPK = 8
dev = train.dev


def pairs(rho, X, T, t, lo=3e-4, hi=3e-2):
    """Reference steps (state at t_k, change to t_{k+1}) with t_k in [lo, hi]."""
    ks = [k for k in range(len(t) - 1) if lo <= t[k] <= hi]
    r = np.repeat(rho, len(ks))
    x0 = X[:, ks].reshape(-1, 13)
    d = (X[:, [k + 1 for k in ks]] - X[:, ks]).reshape(-1, 13)
    T0 = T[:, ks].reshape(-1)
    dt = np.tile([t[k + 1] - t[k] for k in ks], len(rho))
    return r, T0, dt, x0, d


def tt(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)


def step_parts(model, r, T0, dt, x0):
    """Inputs, last hidden layer and outputs of the model at these states."""
    f = train.features(torch.log10(tt(r)), torch.log10(tt(T0)), torch.log10(tt(dt)), tt(x0).clamp_min(0.0), model.rates)
    h = model.net[:-1](f)
    return f, h


def errors_and_grads(model, r, T0, dt, x0, d):
    """Each state's one-step L1 error against the reference, and its gradient with respect to the
    network's outputs z."""
    f, h = step_parts(model, r, T0, dt, x0)
    z = model.net[-1](h).detach().requires_grad_(True)
    X = tt(x0).clamp_min(0.0)
    ld = torch.log10(tt(dt))
    P = model.width
    D1 = model.relax(z[:, :P], X, ld)
    D2 = model.relax(z[:, P:], (X + D1).clamp_min(0.0), ld)
    err = (D1 + D2 - tt(d)).abs().sum(dim=1)
    g, = torch.autograd.grad(err.sum(), z)
    return f.detach(), h.detach(), err.detach(), g


def solve_gate(xb, xg, two_sided, eps, C=10.0, cols=None):
    """Sparse v (and bias b for the one-sided gate) with v.x - b >= 1 on bad rows and
    |v.x| <= eps (two-sided) or v.x - b <= 0 (one-sided) on good rows; slacks penalized by C."""
    if cols is not None:
        xb, xg = xb[:, cols], xg[:, cols]
    nb, ng, n = len(xb), len(xg), xb.shape[1]
    # variables: p (n), q (n), b+ , b-, slacks for bad (nb), good-upper (ng), good-lower (ng)
    nsl = nb + ng * (2 if two_sided else 1)
    c = np.concatenate([np.ones(2 * n), [0.0, 0.0], np.full(nsl, C / max(nb, 1))])
    c[2 * n + 2 + nb:] = C / max(ng, 1)
    rows = []
    rhs = []
    one = np.ones((nb, 1))
    # bad: -(v.x - b) - s <= -1
    Ab = hstack([csr_matrix(-xb), csr_matrix(xb), csr_matrix(one), csr_matrix(-one),
                 -identity(nb), csr_matrix((nb, nsl - nb))])
    rows.append(Ab); rhs.append(-np.ones(nb))
    onegs = np.ones((ng, 1))
    # good upper: v.x - b - s <= eps (eps = 0 for the one-sided gate)
    Au = hstack([csr_matrix(xg), csr_matrix(-xg), csr_matrix(-onegs), csr_matrix(onegs),
                 csr_matrix((ng, nb)), -identity(ng), csr_matrix((ng, nsl - nb - ng))])
    rows.append(Au); rhs.append(np.full(ng, eps if two_sided else 0.0))
    if two_sided:
        # good lower: -(v.x - b) - s <= eps
        Al = hstack([csr_matrix(-xg), csr_matrix(xg), csr_matrix(onegs), csr_matrix(-onegs),
                     csr_matrix((ng, nb + ng)), -identity(ng)])
        rows.append(Al); rhs.append(np.full(ng, eps))
    Aub = vstack(rows).tocsr()
    bounds = [(0, None)] * (2 * n) + ([(0, 0), (0, 0)] if two_sided else [(0, None), (0, None)]) + [(0, None)] * nsl
    res = linprog(c, A_ub=Aub, b_ub=np.concatenate(rhs), bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    v = res.x[:n] - res.x[n:2 * n]
    b = res.x[2 * n] - res.x[2 * n + 1]
    return v, b


def gate_with_topk(xb, xg, two_sided, eps):
    v, b = solve_gate(xb, xg, two_sided, eps)
    allx = np.concatenate([xb, xg])
    imp = np.abs(v) * allx.std(axis=0)
    cols = np.argsort(-imp)[:TOPK]
    v2, b2 = solve_gate(xb, xg, two_sided, eps, cols=cols)
    v = np.zeros(xb.shape[1])
    v[cols] = v2
    return v, b2, cols


def edit_rank_one(model, u, v, alpha):
    m = train.load_model(model.name)
    with torch.no_grad():
        m.net[-1].weight += alpha * torch.outer(tt(u), tt(v))
    return m


def edit_path(model, u, v, b, alpha):
    """A parallel path as weightprog.py builds them: one unit more in every hidden layer."""
    old = [x for x in model.net if isinstance(x, nn.Linear)]
    depth = len(old) - 1
    w = old[0].out_features
    new = train.Burner(w + 1, depth, model.arch).to(dev)
    L = [x for x in new.net if isinstance(x, nn.Linear)]
    with torch.no_grad():
        for m in L:
            m.weight.zero_(); m.bias.zero_()
        for k in range(depth + 1):
            o, i = old[k].weight.shape
            L[k].weight[:o, :i] = old[k].weight
            L[k].bias[:o] = old[k].bias
        L[0].weight[w, :] = A * tt(v)
        L[0].bias[w] = -A * b                       # silu(A (v.x - b)) ~ A g
        L[1].weight[w, w] = 1.0 / A
        L[1].bias[w] = B                            # silu(g + B) ~ g + B
        for k in range(2, depth):
            L[k].weight[w, w] = 1.0
        L[depth].weight[:, w] += alpha * tt(u)
        L[depth].bias[:] -= alpha * B * tt(u)
    new.eval()
    return new


def quick_eval(model, sets):
    """Mean worst-1% L1 over the check times (score.py's metric) on each set, in float16."""
    m = train.load_model(model.name) if isinstance(model, str) else model
    import copy
    m.parts = None                     # the last forward pass's tensors, which do not copy
    mh = copy.deepcopy(m)
    mh.net = score.Half(mh.net)
    out = {}
    with torch.no_grad():
        for name, (rho, X, T, t) in sets.items():
            e = score.errs(mh, rho, X, T, t)
            q = 0.99 if name == "general" else 0.9
            out[name] = float(np.mean(np.quantile(e, q, axis=1)))
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "model_dagger_35b.pt"
    kind = KINDS.index(sys.argv[2] if len(sys.argv) > 2 else "sis")
    model = train.load_model(base)
    model.name = base
    rng = np.random.default_rng(0)
    tg = np.load(train.HERE / "data" / "targets.npz")
    t = tg["t"]
    sel = tg["kind"] == kind
    # Half the regime's training trajectories make the edit, the other half judge it.
    idx = np.where(sel)[0]
    rng.shuffle(idx)
    make, judge = idx[: len(idx) // 2], idx[len(idx) // 2:]
    r, T0, dt, x0, d = pairs(tg["rho"][make], tg["X"][make], tg["T"][make], t)
    f, h, err, g = errors_and_grads(model, r, T0, dt, x0, d)
    cut = torch.quantile(err, 0.8)
    bad = err > cut
    print(f"{KINDS[kind]}: {len(err)} reference steps, error p50 {err.median():.3f} p80 {cut:.3f}; {int(bad.sum())} bad", flush=True)
    gb = g[bad]
    gn = gb / gb.norm(dim=1, keepdim=True)
    u = -gn.mean(dim=0)
    cons = float((gn @ (-u / u.norm())).mean())
    u = (u / u.norm()).cpu().numpy()
    print(f"direction u: mean cosine with the bad states' own {cons:.2f}; largest outputs "
          + ", ".join(f"z{j} {u[j]:+.2f}" for j in np.argsort(-np.abs(u))[:6]), flush=True)
    # Good states: every regime's other reference steps and the general trajectories.
    gen = train.load({"fuel.npz", "mix.npz"} | {f"dagger_{k}.npz" for k in range(30, 39)})
    gi = rng.choice(len(gen[0]), 1500, replace=False)
    rg, Tg, dtg, xg, dg = pairs(gen[0][gi], gen[1][gi], gen[2][gi], gen[3])
    others = np.where(~sel)[0][:600]
    ro, To, dto, xo, do = pairs(tg["rho"][others], tg["X"][others], tg["T"][others], t)
    goodsis = ~bad.cpu().numpy()
    R = np.concatenate([rg, ro, r[goodsis]]); TT = np.concatenate([Tg, To, T0[goodsis]])
    DT = np.concatenate([dtg, dto, dt[goodsis]]); XX = np.concatenate([xg, xo, x0[goodsis]])
    pick = rng.choice(len(R), min(20000, len(R)), replace=False)
    fg, hg = step_parts(model, R[pick], TT[pick], DT[pick], XX[pick])
    fb, hb = f[bad].cpu().numpy(), h[bad].cpu().numpy()
    fg, hg = fg.detach().cpu().numpy(), hg.detach().cpu().numpy()
    print(f"good states: {len(fg)}", flush=True)
    # Dev sets: the regime's judging trajectories and a general sample, not the test set.
    gj = rng.choice(len(gen[0]), 300, replace=False)
    sets = {KINDS[kind]: (tg["rho"][judge], tg["X"][judge], tg["T"][judge], t), "general": (gen[0][gj], gen[1][gj], gen[2][gj], gen[3])}
    print("base", {k: round(v, 3) for k, v in quick_eval(model, sets).items()}, flush=True)
    for eps in (0.05, 0.2):
        vA, _, colsA = gate_with_topk(hb, hg, True, eps)
        fire = hg @ vA
        print(f"A eps {eps}: units {sorted(colsA.tolist())}; good |v.h| p99 {np.quantile(np.abs(fire), 0.99):.3f}; bad v.h p10 {np.quantile(hb @ vA, 0.1):.2f}", flush=True)
        for alpha in (0.5, 1.0, 2.0, 4.0):
            m = edit_rank_one(model, u, vA, alpha)
            res = quick_eval(m, sets)
            print(f"  A eps {eps} alpha {alpha}: " + ", ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
            torch.save({"state": m.state_dict(), "width": 128, "depth": 4, "arch": m.arch},
                       train.HERE / f"lp_A_e{eps}_a{alpha}.pt")
    vB, bB, colsB = gate_with_topk(fb, fg, False, 0.0)
    gfire = np.maximum(fg @ vB - bB, 0)
    print(f"B: inputs {sorted(colsB.tolist())}; good gate > 0.05 in {np.mean(gfire > 0.05):.4f}; bad gate p10 {np.quantile(np.maximum(fb @ vB - bB, 0), 0.1):.2f}", flush=True)
    for alpha in (0.5, 1.0, 2.0, 4.0):
        m = edit_path(model, u, vB, bB, alpha)
        res = quick_eval(m, sets)
        print(f"  B alpha {alpha}: " + ", ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
        torch.save({"state": m.state_dict(), "width": 129, "depth": 4, "arch": m.arch}, train.HERE / f"lp_B_a{alpha}.pt")


if __name__ == "__main__":
    main()
