"""Burn every stored trajectory again from its own starting state with the current reference
burner (burner.py), keeping each file's starts, times and layout: how the training data
follows a change to the burner's physics without losing the coverage the DAgger rounds built
up. Old files are read from data_old/ (and *_old.npz beside this file), new ones written in
their place.

    tools/burn/.venv/Scripts/python tools/burn/reburn.py [file ...]   (default: everything)

Below-normal priority, all cores but two (gen_data.workers). A trajectory whose burn fails is
dropped, with its row of every other array.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psutil

import burner
from gen_data import workers

HERE = Path(__file__).parent
OLD = HERE / "data_old"


def init():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def burn_one(args):
    rho, X0, T0, t = args
    try:
        X, T, _ = burner.burn(np.clip(X0, 0, None) / np.clip(X0, 0, None).sum(), rho, T0, t[-1], t_eval=t)
        return X.T, T
    except Exception:   # noqa: BLE001 - dropped, and counted
        return None


def run(pool, jobs, label):
    t0 = time.time()
    out = []
    for k, r in enumerate(pool.imap(burn_one, jobs, chunksize=4)):
        out.append(r)
        if (k + 1) % 2000 == 0:
            print(f"  {label}: {k + 1}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
    return out


def reburn(pool, src, dst):
    d = dict(np.load(src))
    t = d.get("t")
    if t is None:
        from gen_data import TIMES
        t = TIMES
    if "X0" in d:   # temperature pairs (gen_tpairs.py): both members of each pair
        jobs = [(r, d[x][i, 0], d[T][i, 0], t) for x, T in (("X0", "T0"), ("X1", "T1")) for i, r in enumerate(d["rho"])]
        out = run(pool, jobs, dst.name)
        n = len(d["rho"])
        ok = [i for i in range(n) if out[i] is not None and out[n + i] is not None]
        new = {"rho": d["rho"][ok], "t": t,
               "X0": np.stack([out[i][0] for i in ok]), "T0": np.stack([out[i][1] for i in ok]),
               "X1": np.stack([out[n + i][0] for i in ok]), "T1": np.stack([out[n + i][1] for i in ok])}
    else:
        jobs = [(r, d["X"][i, 0], d["T"][i, 0], t) for i, r in enumerate(d["rho"])]
        out = run(pool, jobs, dst.name)
        ok = [i for i, o in enumerate(out) if o is not None]
        new = {k: v[ok] for k, v in d.items() if k not in ("X", "T", "t") and len(v) == len(d["rho"])}
        new.update(X=np.stack([out[i][0] for i in ok]), T=np.stack([out[i][1] for i in ok]))
        if "t" in d:
            new["t"] = t
    np.savez_compressed(dst, **new)
    print(f"{dst.name}: {len(ok)} of {len(d['rho'])} trajectories", flush=True)


def main():
    init()
    files = sys.argv[1:]
    if not files:
        files = sorted(p.name for p in OLD.glob("*.npz")) + ["tpairs.npz", "targets_test.npz"]
    with Pool(workers(), initializer=init) as pool:
        for name in files:
            src = OLD / name if (OLD / name).exists() else HERE / name.replace(".npz", "_old.npz")
            dst = (HERE / "data" / name) if (OLD / name).exists() else HERE / name
            t0 = time.time()
            reburn(pool, src, dst)
            print(f"  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
