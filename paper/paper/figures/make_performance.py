"""Throughput: cell updates per second against the number of cells.

    uv run python paper/figures/make_performance.py          # plot from data/performance/
    uv run python paper/figures/make_performance.py --run    # measure again on this machine's GPU

collect.mjs runs src/validation/figure_data.ts (throughput) in the code's page in headless Chrome,
at normal priority (a benchmark): for each grid, a warm-up, then a fixed number of steps timed on
the GPU, counting every cell of a uniform grid and every leaf cell of an adaptive mesh as one
update per step (per-block timesteps off). A 2D Kelvin-Helmholtz instability and a 3D Sedov blast,
each on uniform grids and on adaptive meshes. The adapter and browser are recorded with the data.
This measures one GPU in one browser; the figure the paper plans needs an integrated and a
discrete GPU in Chrome and Firefox. --run needs Node, npm install in the code's repository, Chrome
and a GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "performance"
CODE = HERE.parents[2]
GRIDS = [
    {"label": "2D uniform", "problem": "kelvin-helmholtz", "ndim": 2, "runs": [[256, 0], [512, 0], [1024, 0], [2048, 0]]},
    {"label": "2D adaptive", "problem": "kelvin-helmholtz", "ndim": 2, "runs": [[1024, 2], [2048, 3], [4096, 4]]},
    {"label": "3D uniform", "problem": "sedov", "ndim": 3, "runs": [[32, 0], [64, 0], [128, 0], [256, 0]]},
    {"label": "3D adaptive", "problem": "sedov", "ndim": 3, "runs": [[128, 2], [256, 3], [512, 4]]},
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="measure again first")
    if ap.parse_args().run:
        DATA.mkdir(parents=True, exist_ok=True)
        subprocess.run(["node", str(HERE / "collect.mjs"), "throughput", str(DATA / "throughput.json"), json.dumps([GRIDS]), "--normal"],
                       check=True, cwd=CODE)

    data = json.loads((DATA / "throughput.json").read_text())
    style.apply()
    fig, ax = plt.subplots(figsize=(style.COLUMN, 0.8 * style.COLUMN), layout="constrained")
    markers = {"2D uniform": "o", "2D adaptive": "s", "3D uniform": "^", "3D adaptive": "D"}
    for i, g in enumerate(data["grids"]):
        cells = [r["cells"] for r in g["runs"]]
        rate = [r["updates_per_second"] for r in g["runs"]]
        filled = "uniform" in g["label"]
        ax.loglog(cells, rate, marker=markers[g["label"]], color=style.PALETTE[i // 2], mfc=None if filled else "white",
                  ls="-" if filled else "--", ms=3.5, label=g["label"])
        for r in g["runs"]:
            print(f"{g['label']:12s} n = {r['n']:5d}, {r['levels']} levels: {r['cells']:>10,} cells, "
                  f"{r['updates_per_second']:.3g} updates/s, {r['ms_per_step']:.2f} ms/step")
    ax.set_xlabel("Cells (leaf cells on the adaptive mesh)")
    ax.set_ylabel(r"Cell updates per second (s$^{-1}$)")
    ax.legend(loc="lower right", **style.LEGEND)
    ax.text(0.03, 0.97, data["adapter"], transform=ax.transAxes, va="top", fontsize=style.LABEL_SIZE - 1, bbox=style.LABEL_BOX)
    style.save(fig, "performance")


if __name__ == "__main__":
    main()
