"""Shaderous in the browser: the interface, and three frames from its renderer.

    uv run python paper/figures/make_gallery.py          # assemble from data/gallery/
    uv run python paper/figures/make_gallery.py --run    # take the shots again (minutes on a GPU)

The shots are taken by shots.mjs, which runs each setup to a time in headless Chrome with the
code's scripting interface (docs/api.md) and saves the view; the interface is a screenshot of the
whole page as a person sees it. data/gallery/spec.json lists the setups (each a link's query
string), the times and the views; beside each image, a JSON file records the setup in force, the
time reached and the GPU. --run needs Node, npm install in the code's repository, Chrome and a GPU.
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
DATA = HERE / "data" / "gallery"
CODE = HERE.parents[2]
PANELS = [
    ("interface", "(a) Tidal disruption, the whole interface"),
    ("wd", "(b) Colliding white dwarfs"),
    ("binary", "(c) Circumbinary disk"),
    ("thermal", "(d) Thermal instability"),
]
ASPECT = 1.6  # every panel is cropped to the interface's 16:10
SLICES = {"binary", "thermal"}  # slice views, drawn inside a margin of background
BG = np.array([0.05, 0.058, 0.078])  # the renderer's background (render/camera.wgsl)


def trim(img):
    """A slice view's image without the background around the slice."""
    fg = np.abs(img[..., :3] - BG).max(axis=2) > 0.02
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def crop(img, aspect: float):
    """The middle of an image, cut to width / height = aspect."""
    h, w = img.shape[:2]
    if w / h > aspect:
        keep = int(round(h * aspect))
        x0 = (w - keep) // 2
        return img[:, x0 : x0 + keep]
    keep = int(round(w / aspect))
    y0 = (h - keep) // 2
    return img[y0 : y0 + keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="take the shots again first")
    if ap.parse_args().run:
        subprocess.run(["node", str(HERE / "shots.mjs"), str(DATA / "spec.json"), str(DATA)], check=True, cwd=CODE)

    style.apply()
    w = style.PAGE / 2
    fig, axes = plt.subplots(2, 2, figsize=(style.PAGE, 2 * w / ASPECT + 0.1), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    for ax, (name, label) in zip(axes.flat, PANELS, strict=True):
        meta = json.loads((DATA / f"{name}.json").read_text())
        if meta["errors"]:
            raise SystemExit(f"{name}: the page reported errors: {meta['errors']}")
        img = mpimg.imread(DATA / f"{name}.png")
        if name in SLICES:
            img = trim(img)
        ax.imshow(crop(img, ASPECT), interpolation="none")
        ax.set_axis_off()
        ax.text(0.015, 0.975, label, transform=ax.transAxes, va="top", ha="left", fontsize=style.LABEL_SIZE, bbox=style.LABEL_BOX)
        print(f"{name}: {meta['setup']} at t = {meta['t']:.4g} ({meta['steps']} steps)")
    style.save(fig, "gallery")


if __name__ == "__main__":
    main()
