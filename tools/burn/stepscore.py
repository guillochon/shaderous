"""Does a model burn the same at every step size? Kushnir's test (2026-11): hot carbon-oxygen
at 1e7 g/cm^3 from 3.5e9 K made X(Ni56) = 0.05 at 0.1 s with steps of 1e-5 s and 0.8 with
3e-5 s (reference 0.65), and from 4.4e9 K at 2.1e7 g/cm^3 one step of 1.2e-4 s made 0.7 where
the reference makes 0.19. The rollouts that score.py and paired.py judge use 1 ms (and 1e-5 s)
steps, and neither compares step sizes.

Starts (held out: stepref.npz beside this file, not in data/): C/O of carbon fraction 0.3 to 0.7
at 10^6.5 to 10^8 g/cm^3 and 3e9 to 5e9 K, and Si/S ash (some O, C, Ar, Ca, a little Ni) at
10^6.5 to 10^8 g/cm^3 and 3.5e9 to 5e9 K. Each is burned once by the reference (burner.py)
and recorded at CHECK times.

Each model is rolled out (train.roll, float16 as in score.py) with every step in DTS to each
check time it reaches in at most MAX_STEPS steps. Printed per kind and step: the median and 90th
percentile over starts of the worst L1 error over the check times; then the spread of X(Ni56)
at 0.01 s across the steps from 3e-6 to 1e-3 s (per start, largest minus smallest), median and
90th percentile.

    tools/burn/.venv/Scripts/python tools/burn/stepscore.py model.pt ... [--json out.json]
    tools/burn/.venv/Scripts/python tools/burn/stepscore.py --ref [count per kind]   (rebuild)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
REF = HERE / "stepref.npz"
CHECK = np.array([1e-4, 1e-3, 1e-2, 0.1])
DTS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1.2e-4, 3e-4, 1e-3]
MAX_STEPS = 20000
KINDS = ["co", "sis"]
NI = 12


def starts(n, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for kind in range(2):
        for _ in range(n):
            X = np.zeros(13)
            if kind == 0:
                c = rng.uniform(0.3, 0.7)
                X[1], X[2] = c, 1 - c
                T = rng.uniform(3.0e9, 5.0e9)
            else:
                # si28, s32, ar36, ca40, o16, c12, ni56
                w = rng.dirichlet([6, 3, 1, 1, 1.5, 0.5, 0.3])
                for i, x in zip((5, 6, 7, 8, 2, 1, 12), w):
                    X[i] = x
                T = rng.uniform(3.5e9, 5.0e9)
            out.append((10 ** rng.uniform(6.5, 8.0), X, T, kind))
    return out


def ref_one(args):
    import burner
    rho, X, T, kind = args
    try:
        Xt, Tt, _ = burner.burn(X, rho, T, CHECK[-1], t_eval=CHECK, wall=120)
        return rho, X, T, kind, Xt.T
    except Exception:  # noqa: BLE001 - dropped
        return None


def init():
    import psutil
    import burner
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    burner.burn(burner.composition(c12=0.5, o16=0.5), 1e7, 3e9, 1e-6)   # compile first


def build_ref(n):
    from gen_data import workers
    with Pool(workers(), initializer=init) as pool:
        out = [r for r in pool.map(ref_one, starts(n), chunksize=2) if r is not None]
    np.savez_compressed(REF, rho=np.array([r[0] for r in out]), X0=np.stack([r[1] for r in out]),
                        T0=np.array([r[2] for r in out]), kind=np.array([r[3] for r in out]),
                        X=np.stack([r[4] for r in out]), t=CHECK)
    print(f"{REF.name}: {len(out)} starts")


def score(name, ref):
    import train
    from score import Half
    m = train.load_model(name)
    m.net = Half(m.net)
    rho, X0, T0, Xr = ref["rho"], ref["X0"], ref["T0"], ref["X"]
    res = {"model": name, "dts": DTS}
    ni = {}
    with torch.no_grad():
        for dt in DTS:
            Xc, Tc, done, worst = X0, T0, 0.0, np.zeros(len(rho))
            for k, tc in enumerate(CHECK):
                if tc < dt:          # a check shorter than one step: not reached by this step
                    continue
                if tc / dt > MAX_STEPS:
                    break
                steps = int(round((tc - done) / dt))
                Xm, Tm, _ = train.roll(m, rho, Xc, Tc, np.full(len(rho), dt), steps)
                Xc, Tc, done = Xm.cpu().numpy(), Tm.cpu().numpy(), tc
                worst = np.maximum(worst, np.abs(Xc - Xr[:, k]).sum(axis=1))
                if k == 2:
                    ni[dt] = Xc[:, NI].copy()
            for i, kind in enumerate(KINDS):
                s = ref["kind"] == i
                res.setdefault(kind, []).append([float(np.median(worst[s])), float(np.quantile(worst[s], 0.9))])
    span = [dt for dt in DTS if 3e-6 <= dt <= 1e-3 and dt in ni]
    spread = np.ptp(np.stack([ni[dt] for dt in span]), axis=0)
    for i, kind in enumerate(KINDS):
        s = ref["kind"] == i
        res[kind + "_ni_spread"] = [float(np.median(spread[s])), float(np.quantile(spread[s], 0.9))]
    return res


def line(r):
    s = r["model"]
    for kind in KINDS:
        s += f"\n  {kind:4s} worst L1 p50/p90 by step: " + "  ".join(
            f"{dt:.1e} {a:.2f}/{b:.2f}" for dt, (a, b) in zip(r["dts"], r[kind]))
        a, b = r[kind + "_ni_spread"]
        s += f"\n  {kind:4s} X(Ni56) at 0.01 s, spread across steps 3e-6 to 1e-3 s: p50 {a:.2f} p90 {b:.2f}"
    return s


def main():
    args = sys.argv[1:]
    if args and args[0] == "--ref":
        build_ref(int(args[1]) if len(args) > 1 else 100)
        return
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    ref = np.load(REF)
    results = []
    for name in args:
        r = score(name, ref)
        print(line(r), flush=True)
        results.append(r)
    if out:
        json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
