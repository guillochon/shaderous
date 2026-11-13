"""The adaptive mesh, from the code's own renderer.

    uv run python paper/figures/make_amr.py          # assemble from data/amr/
    uv run python paper/figures/make_amr.py --run    # take the shots again (minutes on a GPU)

The shots are taken by shots.mjs, which runs each setup to a time in headless Chrome with the
code's scripting interface (docs/api.md) and saves the view; data/amr/spec.json lists the setups
(each a link's query string), the times and the views, and beside each image a JSON file records
the setup in force, the time reached and the GPU. --run needs Node, npm install in the code's
repository, Chrome and a GPU.

(a, b) A 2D Kelvin-Helmholtz instability on 2048^2 at the finest level, three levels: density
with the outlines of the blocks, and the refinement level. (c) Two white dwarfs at the greatest
compression of a head-on collision (t = 4, where the central density peaks), as a volume
rendering with the block outlines. (d) A slice through a 3D Sedov blast (256 cells at the finest
level, three levels) with per-block timesteps, colored by each block's time bin.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "amr"
CODE = HERE.parents[2]
# (image, label, key): key is the integer range of a field shown from lo = 0 to hi = 3.
PANELS = [
    ("kh-density", "(a) Kelvin-Helmholtz, density", None),
    ("kh-level", "(b) Refinement level", "level"),
    ("wd", "(c) White dwarfs at greatest compression", None),
    ("sedov", "(d) Sedov blast, time bin", "bin"),
]
BG = np.array([0.05, 0.058, 0.078])  # the renderer's background (render/camera.wgsl)


def trim(img):
    """A slice view's image without the background around the slice."""
    rgb = img[..., :3]
    fg = np.abs(rgb - BG).max(axis=2) > 0.02
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def key(ax, what: str) -> None:
    """Colors of the integer values 0 to 3 under the color map (inferno, as the shots used)."""
    cmap = plt.get_cmap("inferno")
    box = style.LABEL_BOX
    ax.add_patch(Rectangle((0.02, 0.02), 0.66, 0.17, transform=ax.transAxes, facecolor=box["facecolor"], alpha=box["alpha"], edgecolor="none"))
    for k in range(4):
        x = 0.04 + 0.11 * k
        ax.add_patch(Rectangle((x, 0.04), 0.09, 0.07, transform=ax.transAxes, facecolor=cmap(k / 3), edgecolor=style.INK, lw=0.3))
        ax.text(x + 0.045, 0.125, str(k), transform=ax.transAxes, ha="center", va="bottom", fontsize=style.LABEL_SIZE - 1, color=style.INK)
    ax.text(0.5, 0.075, what, transform=ax.transAxes, va="center", fontsize=style.LABEL_SIZE - 1, color=style.INK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="take the shots again first")
    if ap.parse_args().run:
        subprocess.run(["node", str(HERE / "shots.mjs"), str(DATA / "spec.json"), str(DATA)], check=True, cwd=CODE)

    style.apply()
    fig, axes = plt.subplots(1, 4, figsize=(style.PAGE, style.PAGE / 4 + 0.05), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01)
    for ax, (name, label, k) in zip(axes, PANELS, strict=True):
        meta = json.loads((DATA / f"{name}.json").read_text())
        if meta["errors"]:
            raise SystemExit(f"{name}: the page reported errors: {meta['errors']}")
        img = mpimg.imread(DATA / f"{name}.png")
        if meta["view"].get("mode") == "slice" or "kh" in name:
            img = trim(img)
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", ha="left", fontsize=style.LABEL_SIZE - 0.5, bbox=style.LABEL_BOX)
        if k:
            key(ax, k)
        print(f"{name}: {meta['setup']} at t = {meta['t']:.4g} ({meta['steps']} steps)")
    style.save(fig, "amr")


if __name__ == "__main__":
    main()
