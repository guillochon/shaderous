"""Tidal disruption: the debris of a partial and a full disruption, rendered and as a return rate.

    uv run python paper/figures/make_tde.py          # plot from data/tde/
    uv run python paper/figures/make_tde.py --run    # first run both cases again (minutes on a GPU)

The runs are the tidal-disruption presets "Partial, gamma = 5/3" (beta = 0.6) and "Full, gamma = 5/3"
(beta = 1.5), each on its own grid, run with the code's command-line runner
(tools/headless/cli.mjs, docs/headless.md) in headless Chrome: debris energies at every output
(--debris) and an image of the volume view. --run needs Node, npm install in the code's
repository, Chrome and a GPU; the outputs are written to paper/figures/data/tde/<case>/.

(a, b) The density at t = 6, about two dynamical times after pericenter, as five isosurfaces
evenly spaced in log density from 0.01 to 10 (code units), drawn by the code's own ray-marcher. (c) Mass per unit specific orbital energy about the black hole,
in units of the spread Delta e = G M_bh R_star / r_t^2, at the last output (t = 40): what had left the box
and what was still in it, the surviving core set apart. (d) The return rate of the bound debris
to pericenter, from Kepler's third law, against time since pericenter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "tde"
CODE = HERE.parents[2]  # the code's repository (paper/ is inside it)
MSUN_KG = 1.98847e30
YEAR_S = 3.15576e7

CASES = {
    "partial": {"problem": "tidal-disruption", "label": r"Partial, $\beta = 0.6$", "color": style.PALETTE[0]},
    "full": {"problem": "tidal-disruption-full", "label": r"Full, $\beta = 1.5$", "color": style.PALETTE[1]},
}
T_END = 40
T_IMAGE = 6
EVERY = 2


def run() -> None:
    for name, case in CASES.items():
        out = DATA / name
        cmd = [
            "node", str(CODE / "tools/headless/cli.mjs"), "run", "--problem", case["problem"],
            "--until", str(T_END), "--every", str(EVERY), "--debris", "--png", "--width", "900",
            "--view", "showBlocks=false", "--view", "dist=1.3", "--view", "contours=true", "--view", "nContours=5",
            "--view", "fog=false", "--view", "surfaceOpacity=0.4", "--view", "lo=0.01", "--view", "hi=10",
            "--low-priority", "--out", str(out),
        ]
        print(" ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True, cwd=CODE)
        # Keep what the figure uses: the image at T_IMAGE, the last debris report, the manifest.
        for f in out.iterdir():
            keep = f.name in ("manifest.json", "diag.jsonl") or f.name.endswith(f"-t{T_IMAGE}.png") or f.name.endswith(f"-t{T_END}-debris.json")
            if not keep:
                f.unlink()


def load(name: str) -> tuple[dict, Path]:
    d = DATA / name
    report = json.loads(next(d.glob(f"*-t{T_END}-debris.json")).read_text())
    return report, next(d.glob(f"*-t{T_IMAGE}.png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run both cases again first")
    if ap.parse_args().run:
        run()

    style.apply()
    # One column: the two renderings above, the two plots below.
    fig = plt.figure(figsize=(style.COLUMN, 1.08 * style.COLUMN), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    img_axes = [fig.add_subplot(grid[0, i]) for i in (0, 1)]
    ax_e = fig.add_subplot(grid[1, 0])
    ax_t = fig.add_subplot(grid[1, 1])

    for i, (name, case) in enumerate(CASES.items()):
        r, png = load(name)
        ax = img_axes[i]
        ax.imshow(mpimg.imread(png), interpolation="none")
        ax.set_axis_off()
        ax.text(0.03, 0.97, f"({'ab'[i]}) {case['label']}", transform=ax.transAxes, va="top", ha="left",
                fontsize=style.LABEL_SIZE - 1, color="white")

        de, nb = r["de"], len(r["escaped"])
        e = (r["lo"] + (np.arange(nb) + 0.5) * r["width"]) / de
        per = r["width"] / de
        dm = (np.array(r["escaped"]) + np.array(r["inBox"])) / per
        ax_e.step(e, dm, where="mid", color=case["color"], label=case["label"])

        u = r["units"]
        t = (np.array(r["dmdt"]["t"]) - r["tPeri"]) * u["time"] / YEAR_S
        rate = np.array(r["dmdt"]["rate"]) * u["mass"] * YEAR_S / (u["time"] * MSUN_KG)
        ok = rate > 0
        ax_t.loglog(t[ok], rate[ok], color=case["color"], label=case["label"])
        # The numbers the text quotes.
        k = int(np.argmax(rate))
        core = r["core"]["mass"] if r["core"] else 0.0
        print(f"{name}: core {core:.3f} Msun, bound {r['totals']['bound']:.3f}, unbound {r['totals']['unbound']:.3f}; "
              f"peak {rate[k]:.2f} Msun/yr at {t[k] * 365.25:.0f} d after pericenter")
        if name == "full":
            k = int(np.argmax(rate))
            tt = np.geomspace(t[k] * 1.5, t[k] * 200, 20)
            ax_t.loglog(tt, 0.8 * rate[k] * (tt / t[k]) ** (-5 / 3), color=style.MUTED, ls="--", lw=0.8, label=r"$\propto t^{-5/3}$")

    ax_e.set_yscale("log")
    ax_e.set_xlim(-3, 3)
    ax_e.set_ylim(1e-5, 30)
    ax_e.axvline(0, color=style.MUTED, lw=0.5)
    ax_e.set_xlabel(r"Orbital energy $\epsilon / \Delta\epsilon$")
    ax_e.set_ylabel(r"$dM / d(\epsilon/\Delta\epsilon)$ ($M_\odot$)")
    ax_e.text(0.03, 0.97, "(c)", transform=ax_e.transAxes, va="top", bbox=style.LABEL_BOX)
    ax_e.legend(loc="upper right", fontsize=style.LABEL_SIZE - 1, **{k: v for k, v in style.LEGEND.items() if k != "fontsize"})

    ax_t.set_xlim(1e-2, 30)
    ax_t.set_ylim(1e-4, 300)
    ax_t.set_xlabel("Time after pericenter (yr)")
    ax_t.set_ylabel(r"$dM/dt$ ($M_\odot\,\mathrm{yr}^{-1}$)")
    ax_t.text(0.03, 0.97, "(d)", transform=ax_t.transAxes, va="top", bbox=style.LABEL_BOX)
    ax_t.legend(loc="upper right", fontsize=style.LABEL_SIZE - 1, **{k: v for k, v in style.LEGEND.items() if k != "fontsize"})

    style.save(fig, "tde")


if __name__ == "__main__":
    main()
