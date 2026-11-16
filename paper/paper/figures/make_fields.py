"""Magnetic field lines, from the code's own renderer.

    uv run python paper/figures/make_fields.py          # assemble from data/fields/
    uv run python paper/figures/make_fields.py --run    # take the shots again (a minute on a GPU)

The shots are taken by shots.mjs, which runs each setup to a time in headless Chrome with the
code's scripting interface (docs/api.md) and saves the view; data/fields/spec.json lists the
setups (each a link's query string), the times and the views, and beside each image a JSON file
records the setup in force, the time reached and the GPU. --run needs Node, npm install in the
code's repository, Chrome and a GPU.

(a) The Orszag-Tang vortex on 512^2 cells at t = 0.5: the density in a slice, with field lines
that are contours of the flux function, 28 levels. (b) The partial tidal disruption of a star
threaded by a dipole field (256 cells at the finest level, three levels) at t = 6: field lines
traced in three dimensions and drawn as tubes, on a white background, with an axes triad drawn
here from the camera's angles. In both panels the renderer colors the lines by the magnetic
pressure on a logarithmic scale.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "fields"
CODE = HERE.parents[2]
PANELS = [("orszag-tang", "(a) Orszag-Tang vortex"), ("disruption", "(b) Magnetized tidal disruption")]
BG = np.array([0.05, 0.058, 0.078])  # the renderer's dark background (render/camera.wgsl)


def trim(img):
    """A slice view's image without the background around the slice."""
    fg = np.abs(img[..., :3] - BG).max(axis=2) > 0.02
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="take the shots again first")
    if ap.parse_args().run:
        subprocess.run(["node", str(HERE / "shots.mjs"), str(DATA / "spec.json"), str(DATA)], check=True, cwd=CODE)

    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE, style.PAGE / 2), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    for ax, (name, label) in zip(axes, PANELS, strict=True):
        meta = json.loads((DATA / f"{name}.json").read_text())
        if meta["errors"]:
            raise SystemExit(f"{name}: the page reported errors: {meta['errors']}")
        img = mpimg.imread(DATA / f"{name}.png")
        if name == "orszag-tang":
            img = trim(img)
        else:
            img = style.crop_to_content(img, 1.0, margin=0.06)
            style.triad(ax, meta["camera"]["yaw"], meta["camera"]["pitch"], at=(0.09, 0.1), size=0.07)
        ax.imshow(img, interpolation="none")
        ax.set_axis_off()
        ax.text(0.015, 0.975, label, transform=ax.transAxes, va="top", ha="left", fontsize=style.LABEL_SIZE, bbox=style.LABEL_BOX)
        print(f"{name}: {meta['setup']} at t = {meta['t']:.4g} ({meta['steps']} steps)")
    style.save(fig, "fields")


if __name__ == "__main__":
    main()
