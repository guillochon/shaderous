"""Mass of each species (Msun: the collision's mass unit) in each .athdf file under the folders."""
import glob, os, sys
import h5py
import numpy as np

for run in sys.argv[1:]:
    for path in sorted(glob.glob(os.path.join(run, "**", "*.athdf"), recursive=True)):
        with h5py.File(path, "r") as f:
            names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["VariableNames"]]
            prim = f["prim"][()]
            xf = [f[f"x{a}f"][()] for a in (1, 2, 3)]
            t = float(f.attrs["Time"])
        dv = np.diff(xf[0], axis=1)[:, 0] * np.diff(xf[1], axis=1)[:, 0] * np.diff(xf[2], axis=1)[:, 0]
        v = {n: prim[i] for i, n in enumerate(names)}
        m = v["rho"] * dv[:, None, None, None]
        out = {n: float((m * v[n]).sum()) for n in names if n.lower() in ("ni56", "si28", "s32", "o16", "c12", "fe52")}
        print(f"{os.path.relpath(path, run):50s} t={t:6.2f} M={m.sum():.4f} " + " ".join(f"{k}={x:.3f}" for k, x in out.items()) + f"  vars={names[:6]}..." if not out else
              f"{os.path.relpath(path, run):50s} t={t:6.2f} M={m.sum():.4f} " + " ".join(f"{k}={x:.3f}" for k, x in out.items()))
