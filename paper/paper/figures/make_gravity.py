"""Gravity: energies in the Evrard collapse, and the potential of a uniform sphere.

    uv run python paper/figures/make_gravity.py          # plot from data/gravity/
    uv run python paper/figures/make_gravity.py --run    # run the simulations again (minutes on a GPU)

(a) The Evrard (1988) collapse run with the code's command-line runner (tools/headless/cli.mjs)
in headless Chrome, energies at every output (diag.jsonl): on a uniform 64^3 grid and on an
adaptive mesh (128 cells at the finest level, two levels), each with the gravitational energy
source in flux form (fluxwork=1, the default) and in kick form (fluxwork=0).
(b) collect.mjs runs src/validation/figure_data.ts (spherePotentialError) in the code's page: the
potential of a uniform sphere with isolated boundaries against the analytic one, along the x axis,
on a uniform 64^3 grid and with two and three levels of refinement.
--run needs Node, npm install in the code's repository, Chrome and a GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "gravity"
CODE = HERE.parents[2]
T_END = 3
EVERY = 0.05
GRIDS = {"uniform": ["--n", "64"], "adaptive": ["--n", "128", "--amr", "2"]}
FORMS = {"flux": "1", "kick": "0"}


def run() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for grid, flags in GRIDS.items():
        for form, flag in FORMS.items():
            out = DATA / f"evrard-{grid}-{form}"
            cmd = ["node", str(CODE / "tools/headless/cli.mjs"), "run", "--problem", "evrard", "--dim", "3", *flags,
                   "--set", f"fluxwork={flag}", "--until", str(T_END), "--every", str(EVERY), "--low-priority", "--quiet", "--out", str(out)]
            subprocess.run(cmd, check=True, cwd=CODE)
    subprocess.run(["node", str(HERE / "collect.mjs"), "spherePotentialError", str(DATA / "sphere.json")], check=True, cwd=CODE)


def energies(grid: str, form: str) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in (DATA / f"evrard-{grid}-{form}" / "diag.jsonl").read_text().splitlines()]
    return {k: np.array([r[k] for r in rows]) for k in ("t", "eKin", "eInt", "eGrav", "eTot")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run the simulations again first")
    if ap.parse_args().run:
        run()

    style.apply()
    fig, (a, b) = plt.subplots(2, 1, figsize=(style.COLUMN, 1.45 * style.COLUMN), layout="constrained")

    names = {"eKin": "Kinetic", "eInt": "Thermal", "eGrav": "Gravitational", "eTot": "Total"}
    colors = {"eKin": style.PALETTE[0], "eInt": style.PALETTE[1], "eGrav": style.PALETTE[2], "eTot": style.INK}
    for grid, lw in (("uniform", 1.0), ("adaptive", 0.6)):
        for form, ls in (("flux", "-"), ("kick", "--")):
            e = energies(grid, form)
            early = e["t"] <= 1.5 + 1e-6  # before the outgoing shock reaches the box's edge
            drift = np.max(np.abs(e["eTot"][early] - e["eTot"][0])) / abs(e["eTot"][0])
            peak = e["t"][np.argmax(e["eInt"])]
            print(f"Evrard {grid} {form}: largest change of the total energy {drift:.2%} up to t = 1.5; thermal energy peaks at t = {peak:.2f}")
            for k in names:
                a.plot(e["t"], e[k], color=colors[k], ls=ls, lw=lw, label=names[k] if (grid, form) == ("uniform", "flux") else None)
    a.plot([], [], color=style.MUTED, ls="-", label="Flux form")
    a.plot([], [], color=style.MUTED, ls="--", label="Kick form")
    a.set_xlabel(r"Time $t$")
    a.set_ylabel(r"Energy")
    a.set_xlim(0, T_END)
    a.set_ylim(-3.6, 2.3)
    a.legend(loc="lower right", ncol=2, **style.LEGEND)
    a.text(0.03, 0.97, "(a)", transform=a.transAxes, va="top", bbox=style.LABEL_BOX)

    for i, s in enumerate(json.loads((DATA / "sphere.json").read_text())):
        x = np.array(s["x"])
        err = np.abs(np.array(s["err"]))
        label = {"uniform, 64^3": r"Uniform, $64^3$"}.get(s["label"], s["label"].replace("^3", "$^3$"))
        b.semilogy(x, err, ".-" if i == 0 else ".", color=style.PALETTE[i], ms=2, lw=0.6, label=label)
        print(f"sphere {s['label']}: largest relative error {err.max():.2e}")
    b.axvspan(-0.25, 0.25, color=style.MUTED, alpha=0.1, lw=0)
    b.set_xlabel(r"Position along the axis $x$")
    b.set_ylabel("Relative error of the potential")
    b.set_xlim(-0.5, 0.5)
    b.legend(loc="lower left", **style.LEGEND)
    b.text(0.03, 0.97, "(b)", transform=b.transAxes, va="top", bbox=style.LABEL_BOX)

    style.save(fig, "gravity")


if __name__ == "__main__":
    main()
