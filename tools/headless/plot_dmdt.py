#!/usr/bin/env python3
"""Plot debris energies from the headless runner (--debris): dM/de and the return rate dM/dt.

    python tools/headless/plot_dmdt.py runs/tde                 # the run's last output
    python tools/headless/plot_dmdt.py runs/tde/*-debris.json   # several outputs, to watch dM/dt settle
    python tools/headless/plot_dmdt.py runs/tde -o dmdt.png

Left: mass per unit energy, in units of the energy spread, split into what had left the box
and what was still in it (a surviving core set apart). Right: the return rate of the bound
debris against time since pericenter, in solar masses per year and years where the problem has
units (code units otherwise), with a t^(-5/3) line for reference. Needs numpy and matplotlib.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MSUN_KG = 1.98847e30
YEAR_S = 3.15576e7


def reports(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "*-debris.json")))
            if not found:
                raise SystemExit(f"no *-debris.json in {p}: run the runner with --debris")
            files.append(found[-1])
        else:
            files.append(p)
    return [(f, json.load(open(f))) for f in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="debris JSON files, or run folders (their last output)")
    ap.add_argument("-o", "--out", help="image file (default: dmdt.png beside the first input)")
    a = ap.parse_args()
    rs = reports(a.paths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for i, (f, r) in enumerate(rs):
        last = i == len(rs) - 1
        alpha = 1.0 if last else 0.35
        de, nb = r["de"], len(r["escaped"])
        e = (r["lo"] + (np.arange(nb) + 0.5) * r["width"]) / de
        per = r["width"] / de  # bin width in units of de: mass per unit (e / de)
        esc = np.array(r["escaped"]) / per
        box = np.array(r["inBox"]) / per
        label = f"t = {r['t']:.3g}"
        ax1.step(e, esc + box, where="mid", color=f"C{i}", alpha=alpha, label=f"{label}: total")
        if last:
            ax1.step(e, esc, where="mid", color="0.4", lw=0.8, ls="--", label="left the box")
            ax1.step(e, box, where="mid", color="0.4", lw=0.8, ls=":", label="in the box")

        u = r.get("units")
        t = np.array(r["dmdt"]["t"]) - r["tPeri"]
        rate = np.array(r["dmdt"]["rate"])
        if u:
            t = t * u["time"] / YEAR_S
            rate = rate * u["mass"] * YEAR_S / (u["time"] * MSUN_KG)
        ok = rate > 0
        ax2.loglog(t[ok], rate[ok], color=f"C{i}", alpha=alpha, label=label)

    _, r = rs[-1]
    u = r.get("units")
    core = r.get("core")
    title = f"bound {r['totals']['bound']:.3g}, unbound {r['totals']['unbound']:.3g}"
    if core:
        title += f", core {core['mass']:.4g}"
    ax1.set_title(title + " (code mass)", fontsize=10)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\epsilon / \Delta\epsilon$")
    ax1.set_ylabel(r"$dM / d(\epsilon/\Delta\epsilon)$")
    ax1.axvline(0, color="0.7", lw=0.6)
    ax1.legend(fontsize=8)

    # t^(-5/3) through the peak, for reference.
    t = np.array(r["dmdt"]["t"]) - r["tPeri"]
    rate = np.array(r["dmdt"]["rate"])
    if u:
        t, rate = t * u["time"] / YEAR_S, rate * u["mass"] * YEAR_S / (u["time"] * MSUN_KG)
    if rate.size and rate.max() > 0:
        k = int(np.argmax(rate))
        tt = np.geomspace(t[k], t[k] * 100, 20)
        ax2.loglog(tt, rate[k] * (tt / t[k]) ** (-5 / 3), color="0.5", ls="--", lw=0.8, label=r"$t^{-5/3}$")
        floor = rate[k] * 1e-4
        ax2.set_ylim(floor, rate[k] * 3)
    ax2.set_xlabel("time since pericenter" + (" (yr)" if u else " (code)"))
    ax2.set_ylabel(r"$dM/dt$" + (r" ($M_\odot$ / yr)" if u else " (code)"))
    ax2.set_title("return rate of bound debris", fontsize=10)
    ax2.legend(fontsize=8)

    out = a.out or os.path.join(os.path.dirname(rs[0][0]), "dmdt.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
