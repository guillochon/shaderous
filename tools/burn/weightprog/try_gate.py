"""Evaluate gated programs on round 30 (fp16, as score.py): general rollouts and the targeted kinds.
Usage: try_gate.py name ...   (programs defined below by name)"""
import sys
from pathlib import Path
import json, os
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train, score
from weightprog import program, gate_values

fr = lambda lr: 2 * (lr - 4.0) / 6.0 - 1.0          # scaled log rho
fT = lambda lT: 2 * (lT - 8.0) / 2.5 - 1.0          # scaled log T
x4 = lambda i: 3 + i                                # X^(1/4) of species i
SI, S, AR, CA, NI, O, NE = 5, 6, 7, 8, 12, 2, 3
loss = lambda i, st: 27 * st + 1 + i                # loss logit of species i in stage st
dest = lambda i, st: 27 * st + 14 + i

def sis_gate(d, species=(SI, S, AR, CA), lr=(6.3, 8.3), lT=(9.447, 9.643), si=0.6):
    return {"boxes": [(0, fr(lr[0]), fr(lr[1])), (1, fT(lT[0]), fT(lT[1])), (x4(SI), si, 1.2)],
            "deltas": {loss(i, st): -d for i in species for st in (0, 1)}}

def sis_dest(d, lr=(6.3, 8.3), lT=(9.447, 9.643), si=0.6):
    return {"boxes": [(0, fr(lr[0]), fr(lr[1])), (1, fT(lT[0]), fT(lT[1])), (x4(SI), si, 1.2)],
            "deltas": {dest(SI, st): d for st in (0, 1)}}

def qse_hold(d, eps=0.05, lT=(9.398, 9.845)):
    # log10(fwd/rev) of Si28 + a <-> S32 = 20 (f47 - f48): within eps of balance, or reverse winning
    return {"boxes": [({47: 20.0, 48: -20.0}, -3.0, eps), (1, fT(lT[0]), fT(lT[1])), (x4(SI), 0.3, 1.2)],
            "deltas": {dest(i, st): d for i in (SI, S) for st in (0, 1)}}

def qse_hold_sis(d, eps=0.05, lr=(7.0, 8.2), lT=(9.398, 9.70)):
    return {"boxes": [({47: 20.0, 48: -20.0}, -3.0, eps), (0, fr(lr[0]), fr(lr[1])), (1, fT(lT[0]), fT(lT[1])), (x4(SI), 0.6, 1.2)],
            "deltas": {dest(i, st): d for i in (SI, S) for st in (0, 1)}}

def qse_slow_sis(d, eps=0.05, lr=(7.0, 8.2), lT=(9.398, 9.70)):
    return {"boxes": [({47: 20.0, 48: -20.0}, -3.0, eps), (0, fr(lr[0]), fr(lr[1])), (1, fT(lT[0]), fT(lT[1])), (x4(SI), 0.6, 1.2)],
            "deltas": {27 * st: -d for st in (0, 1)}}

def tight(d):
    g = qse_slow_sis(d, lT=(9.398, 9.623))
    return {**g, "boxes": g["boxes"][:3] + [(x4(SI), 0.75, 1.2)]}

PROGRAMS = {
    "qt08s": [tight(0.8)], "qt10s": [tight(1.0)], "qt14s": [tight(1.4)],
    "qt04": [qse_slow_sis(0.4, lT=(9.398, 9.623))],
    "qt04s": [{**qse_slow_sis(0.4, lT=(9.398, 9.623)), "boxes": qse_slow_sis(0.4, lT=(9.398, 9.623))["boxes"][:3] + [(x4(SI), 0.75, 1.2)]}],
    "qt06s": [{**qse_slow_sis(0.6, lT=(9.398, 9.623)), "boxes": qse_slow_sis(0.6, lT=(9.398, 9.623))["boxes"][:3] + [(x4(SI), 0.75, 1.2)]}],
    "qk02": [qse_slow_sis(0.2)], "qk04": [qse_slow_sis(0.4)],
    "qs1": [qse_hold_sis(1.0)], "qs2": [qse_hold_sis(2.0)],
    "qh1": [qse_hold(1.0)], "qh2": [qse_hold(2.0)], "qh4": [qse_hold(4.0)],
    "sd2": [sis_dest(2.0)], "sd4": [sis_dest(4.0)], "sd6": [sis_dest(6.0)],
    "base": [],
    "sis1": [sis_gate(1.0)], "sis2": [sis_gate(2.0)], "sis4": [sis_gate(4.0)],
}
if __name__ == "__main__":
    base = train.load_model("model_dagger_30.pt")
    rho, X, T, t = train.load({"fuel.npz", "mix.npz"})
    va = np.random.default_rng(7).choice(len(rho), size=600, replace=False)
    gen = rho[va], X[va], T[va]
    tt = np.load(train.HERE / "targets_test.npz")
    for name in sys.argv[1:]:
        if name.endswith(".pt"):
            m, gates = train.load_model(name), []
        else:
            gates = PROGRAMS[name]
            base = train.load_model(os.environ.get("BASE", "model_dagger_30.pt"))
            m = program(base, gates) if gates else base
        if gates:
            f = train.features(torch.log10(torch.tensor(gen[0], dtype=torch.float32, device=train.dev)),
                               torch.log10(torch.tensor(gen[2][:, 0], dtype=torch.float32, device=train.dev)),
                               torch.full((600,), -3.0, device=train.dev),
                               torch.tensor(gen[1][:, 0], dtype=torch.float32, device=train.dev), "full")
            gv = gate_values(m, gates, f)
        m.net = score.Half(m.net)
        with torch.no_grad():
            e = score.errs(m, *gen, t)
            line = f"{name:8s} general p99 " + " ".join(f"{q:.2f}" for q in np.quantile(e, 0.99, axis=1)) + f" mean {np.quantile(e,0.99,axis=1).mean():.3f}"
            for k, kind in enumerate(score.KINDS):
                s = tt["kind"] == k
                ek = score.errs(m, tt["rho"][s], tt["X"][s], tt["T"][s], tt["t"]).max(axis=0)
                line += f" | {kind} {np.median(ek):.2f}/{np.quantile(ek, 0.9):.2f}"
        if gates:
            line += f" | gate on at start in {int((gv > 0.5).sum())}/600 general"
        print(line, flush=True)
