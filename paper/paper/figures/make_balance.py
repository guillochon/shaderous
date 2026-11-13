"""Hydrostatic balance in single precision: a stratified column and a white dwarf.

    uv run python paper/figures/make_balance.py          # plot from data/balance/
    uv run python paper/figures/make_balance.py --run    # run the simulations again (minutes on a GPU)

collect.mjs runs src/validation/figure_data.ts in the code's page in headless Chrome:
  wellBalanced  the well-balanced test's isothermal column (density falling by e every 4 cells,
                64 cells tall, 2D) over one sound crossing, the largest Mach number at 40 times,
                with and without the well-balanced reconstruction
  whiteDwarf    a 1 solar-mass white dwarf on the adaptive mesh (256 cells at the finest level,
                two levels) over six dynamical times, its central density, settled into the
                grid's balance (the preset) and mapped from its profile alone
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
DATA = HERE / "data" / "balance"
CODE = HERE.parents[2]
RUNS = [("wellBalanced", [], "column.json"), ("whiteDwarf", [1, 6, 256, 2], "white_dwarf.json")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run the simulations again first")
    if ap.parse_args().run:
        DATA.mkdir(parents=True, exist_ok=True)
        for name, args, out in RUNS:
            subprocess.run(["node", str(HERE / "collect.mjs"), name, str(DATA / out), json.dumps(args)], check=True, cwd=CODE)

    style.apply()
    fig, (a, b) = plt.subplots(2, 1, figsize=(style.COLUMN, 1.4 * style.COLUMN), layout="constrained")

    c = json.loads((DATA / "column.json").read_text())
    for key, label, ls in (("balanced", "Well-balanced", "-"), ("plain", "Without", "--")):
        a.semilogy(c[key]["t"], c[key]["mach"], color=style.INK, ls=ls, label=label)
        print(f"column {key}: largest Mach number {max(c[key]['mach']):.2e}")
    a.set_xlabel("Time (sound crossings)")
    a.set_ylabel("Largest Mach number")
    a.set_xlim(0, 1)
    a.legend(loc="center right", **style.LEGEND)
    a.text(0.03, 0.97, "(a)", transform=a.transAxes, va="top", bbox=style.LABEL_BOX)

    w = json.loads((DATA / "white_dwarf.json").read_text())
    for key, label, ls in (("settled", "Settled", "-"), ("mapped", "Mapped", "--")):
        rho = np.array(w[key]["rho"])
        b.plot(w[key]["t"], rho / rho[0], color=style.INK, ls=ls, label=label)
        print(f"white dwarf {key}: central density from {rho[0]:.4g} to {rho[-1]:.4g}, "
              f"largest change {np.max(np.abs(rho / rho[0] - 1)):.2%}; t_dyn = {w[key]['tdyn']:.4g}")
    b.set_ylim(top=1.06)
    b.set_xlabel("Time (dynamical times)")
    b.set_ylabel("Central density / initial")
    b.legend(loc="lower left", **style.LEGEND)
    b.text(0.03, 0.97, "(b)", transform=b.transAxes, va="top", bbox=style.LABEL_BOX)

    style.save(fig, "balance")


if __name__ == "__main__":
    main()
