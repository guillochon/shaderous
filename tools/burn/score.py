"""Score models as a shader would run them (the MLP in float16): diagnose.py's rollouts (600
fuel and mixture trajectories, 1 ms steps, the L1 error at 3 ms, 30 ms, 0.1 s and 1 s), the
same with 1e-5 s steps to 0.03 s (where a small bias in every step adds up), and the held-out
targeted trajectories (targets_test.npz, gen_targets.py) by kind.

    tools/burn/.venv/Scripts/python tools/burn/score.py model.pt ... [--json out.json]
"""
import json
import sys

import numpy as np
import torch

import train

CHECK = (0.003, 0.03, 0.1, 1.0)
KINDS = ["oneb", "sis", "hotco", "nse"]


class Half(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward(self, x):
        return self.net(x.half()).float()


def errs(model, rho, X, T, t, dt=1e-3):
    """L1 error at each CHECK time (rows) for each trajectory (columns)."""
    Xc, Tc, done, out = X[:, 0], T[:, 0], 0.0, []
    for tc in CHECK:
        k = int(np.argmin(np.abs(t - tc)))
        steps = max(1, int(round((t[k] - done) / dt)))
        Xm, Tm, _ = train.roll(model, rho, Xc, Tc, np.full(len(rho), (t[k] - done) / steps), steps)
        Xc, Tc, done = Xm.cpu().numpy(), Tm.cpu().numpy(), t[k]
        out.append(np.abs(Xc - X[:, k]).sum(axis=1))
    return np.array(out)


def score(name, gen, t, tt):
    m = train.load_model(name)
    m.net = Half(m.net)
    res = {"model": name}
    with torch.no_grad():
        e = errs(m, *gen, t)
        res["general_p99"] = [float(q) for q in np.quantile(e, 0.99, axis=1)]
        res["general_p50_1s"] = float(np.median(e[-1]))
        k = int(np.argmin(np.abs(t - 0.03)))
        Xm, _, _ = train.roll(m, gen[0], gen[1][:, 0], gen[2][:, 0], np.full(len(gen[0]), t[k] / 3000), 3000)
        es = np.abs(Xm.cpu().numpy() - gen[1][:, k]).sum(axis=1)
        res["small_steps"] = [float(np.median(es)), float(np.quantile(es, 0.99))]
        if tt is not None:
            for i, kind in enumerate(KINDS):
                s = tt["kind"] == i
                ek = errs(m, tt["rho"][s], tt["X"][s], tt["T"][s], tt["t"]).max(axis=0)
                res[kind] = [float(np.median(ek)), float(np.quantile(ek, 0.9))]
    return res


def line(r):
    s = f"{r['model']:28s} general p99 " + " ".join(f"{q:.2f}" for q in r["general_p99"])
    s += f" | p50@1s {r['general_p50_1s']:.1e} | 1e-5 s steps @0.03s p50 {r['small_steps'][0]:.1e} p99 {r['small_steps'][1]:.2f}"
    for kind in KINDS:
        if kind in r:
            s += f" | {kind} p50 {r[kind][0]:.2f} p90 {r[kind][1]:.2f}"
    return s


def main():
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    rho, X, T, t = train.load({"fuel.npz", "mix.npz"})
    va = np.random.default_rng(7).choice(len(rho), size=600, replace=False)
    gen = rho[va], X[va], T[va]
    path = train.HERE / "targets_test.npz"
    tt = np.load(path) if path.exists() else None
    results = []
    for name in args:
        r = score(name, gen, t, tt)
        print(line(r), flush=True)
        results.append(r)
    if out:
        json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
