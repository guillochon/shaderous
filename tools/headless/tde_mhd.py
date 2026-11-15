#!/usr/bin/env python3
"""Magnetized tidal disruptions: the measures Guillochon & McCourt (2017) report, from .athdf outputs.

    uv run --with h5py --with numpy --with matplotlib python tools/headless/tde_mhd.py runs/tde-dipole
    uv run ... python tools/headless/tde_mhd.py runs/tde-dipole runs/tde-random -o tde_mhd.png

For each output (the runner's --athdf files) of each run:
  core       the surviving star: the gas within CORE_RADIUS stellar radii of the density peak
             whose density is above CORE_DENSITY of the peak's. Its magnetic energy, sum B^2/2 dV,
             is GM17's "magnetic energy within the star", which grows 13 to 20 times in their
             partial disruptions.
  streams    the star's gas (passive scalar r0 > 0.5) outside the core. The field is split along
             the direction of stretching, taken as that of the gas's velocity relative to the
             core's (the debris recedes from the core along the streams): B_par = B . v_hat,
             B_perp the rest. GM17: B_perp^2 V falls as the streams lengthen while B_par^2 V
             holds or grows, so B_par^2 / B_perp^2 rises and the streams' beta falls toward 1.
  least beta 2 p / B^2 at the cell where it is least, with that cell's density and scalar.
Prints a table and, with -o, plots the core's magnetic energy and the streams' field against
time. Units are code units (the presets': G = M_star = R_star = 1).
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np

CORE_RADIUS = 2.0
CORE_DENSITY = 1e-3


def load(path):
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["VariableNames"]]
        prim = f["prim"][()]
        x = [f[f"x{a}v"][()] for a in (1, 2, 3)]
        xf = [f[f"x{a}f"][()] for a in (1, 2, 3)]
        t = float(f.attrs["Time"])
    v = {n: prim[i] for i, n in enumerate(names)}
    nb, nz, ny, nx = prim.shape[1:]
    # Positions and volumes per cell, [block, z, y, x].
    X = np.broadcast_to(x[0][:, None, None, :], (nb, nz, ny, nx))
    Y = np.broadcast_to(x[1][:, None, :, None], (nb, nz, ny, nx))
    Z = np.broadcast_to(x[2][:, :, None, None], (nb, nz, ny, nx))
    dv = np.diff(xf[0], axis=1)[:, 0] * np.diff(xf[1], axis=1)[:, 0] * np.diff(xf[2], axis=1)[:, 0]
    DV = np.broadcast_to(dv[:, None, None, None], (nb, nz, ny, nx))
    return t, v, (X, Y, Z), DV


def measure(path):
    t, v, (X, Y, Z), DV = load(path)
    rho, p = v["rho"], v["press"]
    vel = np.stack([v["vel1"], v["vel2"], v["vel3"]])
    if "Bcc1" not in v:
        raise SystemExit(f"{path}: no magnetic field (Bcc1-3)")
    B = np.stack([v["Bcc1"], v["Bcc2"], v["Bcc3"]])
    b2 = (B * B).sum(0)
    s = v.get("r0", np.ones_like(rho))
    i = np.unravel_index(np.argmax(rho), rho.shape)
    c = np.array([X[i], Y[i], Z[i]])
    r = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
    core = (r < CORE_RADIUS) & (rho > CORE_DENSITY * rho[i])
    m = rho * DV
    vcore = (vel * m * core).sum((1, 2, 3, 4)) / max((m * core).sum(), 1e-300)
    stream = (s > 0.5) & ~core
    dvel = vel - vcore[:, None, None, None, None]
    speed = np.sqrt((dvel * dvel).sum(0))
    vhat = dvel / np.maximum(speed, 1e-30)
    bpar2 = (B * vhat).sum(0) ** 2
    bperp2 = np.maximum(b2 - bpar2, 0)
    beta = 2 * p / np.maximum(b2, 1e-300)
    k = np.unravel_index(np.argmin(beta), beta.shape)
    ms = (m * stream).sum()
    out = {
        "file": os.path.basename(path), "t": t,
        "rho_max": float(rho[i]), "core_mass": float((m * core).sum()),
        "emag_total": float((0.5 * b2 * DV).sum()),
        "emag_core": float((0.5 * b2 * DV * core).sum()),
        "stream_mass": float(ms),
        "stream_bpar2_V": float((0.5 * bpar2 * DV * stream).sum()),
        "stream_bperp2_V": float((0.5 * bperp2 * DV * stream).sum()),
        "stream_eint": float((1.5 * p * DV * stream).sum()),
        "stream_mass_weighted_beta": float(np.exp((np.log(np.maximum(beta, 1e-30)) * m * stream).sum() / ms)) if ms > 0 else float("nan"),
        "beta_min": float(beta[k]), "beta_min_rho": float(rho[k]), "beta_min_scalar": float(s[k]),
        "beta_min_r": float(r[k]),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories (or .athdf files)")
    ap.add_argument("-o", "--out", help="write a plot here")
    ap.add_argument("--json", help="write the measures here")
    a = ap.parse_args()
    results = {}
    for run in a.runs:
        files = sorted(glob.glob(os.path.join(run, "*.athdf"))) if os.path.isdir(run) else [run]
        rows = [measure(f) for f in files]
        results[run] = rows
        e0 = rows[0]["emag_core"] or 1
        print(f"\n{run}")
        print(f"{'t':>7} {'rho_max':>9} {'core M':>7} {'Emag core':>10} {'/start':>7} {'Emag all':>10} {'str M':>7} {'Bpar2 V':>9} {'Bperp2 V':>9} {'par/perp':>8} {'str beta':>9} {'beta min':>9} {'(rho,s,r)':>22}")
        for q in rows:
            ratio = q["stream_bpar2_V"] / q["stream_bperp2_V"] if q["stream_bperp2_V"] > 0 else float("nan")
            print(f"{q['t']:7.2f} {q['rho_max']:9.3g} {q['core_mass']:7.3f} {q['emag_core']:10.3e} {q['emag_core'] / e0:7.2f} {q['emag_total']:10.3e} {q['stream_mass']:7.3f} {q['stream_bpar2_V']:9.2e} {q['stream_bperp2_V']:9.2e} {ratio:8.2f} {q['stream_mass_weighted_beta']:9.3g} {q['beta_min']:9.2e} ({q['beta_min_rho']:.1e},{q['beta_min_scalar']:.2f},{q['beta_min_r']:.1f})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    if a.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for run, rows in results.items():
            t = [q["t"] for q in rows]
            label = os.path.basename(os.path.normpath(run))
            e0 = rows[0]["emag_core"] or 1
            ax[0].plot(t, [q["emag_core"] / e0 for q in rows], label=label)
            line = ax[1].plot(t, [q["stream_bpar2_V"] for q in rows], label=f"{label}: B_par^2 V")[0]
            ax[1].plot(t, [q["stream_bperp2_V"] for q in rows], ls="--", color=line.get_color(), label=f"{label}: B_perp^2 V")
            ax[1].plot(t, [q["stream_eint"] for q in rows], ls=":", color=line.get_color(), label=f"{label}: E_int")
        ax[0].set(xlabel="t (code units, 1593 s)", ylabel="core magnetic energy / start", yscale="log", title="Surviving core")
        ax[0].axhspan(13, 20, color="0.85", label="GM17: 13-20x")
        ax[0].legend(fontsize=8)
        ax[1].set(xlabel="t (code units, 1593 s)", ylabel="energy in the streams (code units)", yscale="log", title="Streams: field along and across stretching")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.out, dpi=130)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
