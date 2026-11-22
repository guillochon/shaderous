"""A magnetized double white dwarf, from a checkpoint opened for viewing only.

    uv run python paper/figures/make_merger.py          # plot from data/merger/images/
    uv run python paper/figures/make_merger.py --shots  # take the image again (about a minute on a GPU)

The run is the setup "Double white dwarf mass transfer, magnetized" (its setup is in
data/merger/images/merger.json): a 0.6 Msun carbon-oxygen donor at 0.97 of its Roche lobe and a
0.9 Msun accretor, in the frame turning with the orbit, each threaded by its own tangled field
(plasma beta 1e5 at the center), with the Helmholtz equation of state, in a box 80,000 km across
on 1024^3 cells at the finest level with five levels of refinement. It ran on an NVIDIA H100 with
the code's command-line runner. Its checkpoint at t = 46 (126 s), 3.5 GB, is not in the
repository: --shots needs it at runs/unity-dwd-1024/seg-2070200-r1/0010-t46.athdf in the code's
repository, with Node, npm install there, Chrome and a GPU. shots.mjs opens it for viewing only
(viewAthdf, with the face field for the field lines), which merges its finest level into the one
below to fit the default GPU memory budget, and saves the volume view
(data/merger/images/spec.json gives the view).

The density as fog and eight isosurfaces, in the code's inferno color map on a logarithmic scale
from 10 to 1.1e5 g/cm^3, with 300 field lines colored by the magnetic pressure, drawn by the
code's own ray-marcher on its dark background; the color bar and the axes triad are drawn here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from PIL import Image

import style

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "data" / "merger" / "images"
CODE = HERE.parents[2]  # the code's repository (paper/ is inside it)
# Code units: lengths of 10^4 km and masses of 1 Msun, so a unit of density is Msun / (10^9 cm)^3.
DENSITY_CGS = 1.98847e33 / 1e27
# and a unit of time sqrt(L^3 / (G M)).
TIME_S = np.sqrt(1e7**3 / (6.6743e-11 * 1.98847e30))
# The page's inferno (COLORMAPS in src/render/transfer.ts): control points interpolated linearly.
INFERNO = [
    (0, 0, 4), (31, 12, 72), (85, 15, 109), (136, 34, 106), (186, 54, 85),
    (227, 89, 51), (249, 140, 10), (249, 201, 50), (252, 255, 164),
]


def shots() -> None:
    subprocess.run(["node", str(HERE / "shots.mjs"), str(IMAGES / "spec.json"), str(IMAGES)], check=True, cwd=CODE)


def crop_dark(img, aspect: float, margin: float = 0.03):
    """A rendering on the dark background cut to what is drawn, plus a margin, at width / height = aspect."""
    rgb = img[..., :3]
    bg = np.median(rgb[:8, :8].reshape(-1, 3), axis=0)
    drawn = (np.abs(rgb - bg) > 0.04).any(axis=2)
    # Ignore specks: a row or column counts when enough of it is drawn.
    rows = np.where(drawn.sum(axis=1) > 6)[0]
    cols = np.where(drawn.sum(axis=0) > 6)[0]
    cy, cx = (rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2
    h = (rows[-1] - rows[0]) * (1 + 2 * margin)
    w = (cols[-1] - cols[0]) * (1 + 2 * margin)
    h, w = max(h, w / aspect), max(w, h * aspect)
    h, w = min(h, img.shape[0]), min(w, img.shape[1])
    y0 = int(np.clip(round(cy - h / 2), 0, img.shape[0] - round(h)))
    x0 = int(np.clip(round(cx - w / 2), 0, img.shape[1] - round(w)))
    return img[y0 : y0 + int(round(h)), x0 : x0 + int(round(w))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", action="store_true", help="take the image again first")
    args = ap.parse_args()
    if args.shots:
        shots()

    meta = json.loads((IMAGES / "merger.json").read_text())
    if meta["errors"]:
        raise SystemExit(f"the page reported errors: {meta['errors']}")
    view = meta["view"]
    lo, hi = view["lo"] * DENSITY_CGS, view["hi"] * DENSITY_CGS
    cmap = LinearSegmentedColormap.from_list("shaderous-inferno", [tuple(c / 255 for c in p) for p in INFERNO])

    style.apply()
    aspect = 1.55
    # The image fills the column; the color bar hangs below it, as wide as it is.
    fig = plt.figure(figsize=(style.COLUMN, style.COLUMN / aspect))
    ax = fig.add_axes([0, 0, 1, 1])
    cax = ax.inset_axes([0, -0.075, 1, 0.045])

    img = crop_dark(mpimg.imread(IMAGES / "merger.png"), aspect, margin=0.06)
    # Embedded as is, at about 330 pixels an inch of the column (the PNG at full size would be 5 MB).
    width = 1200
    img = np.asarray(Image.fromarray((img[..., :3] * 255).astype(np.uint8)).resize((width, round(width * img.shape[0] / img.shape[1])), Image.LANCZOS))
    ax.imshow(img, interpolation="none")
    ax.set_axis_off()
    style.triad(ax, meta["camera"]["yaw"], meta["camera"]["pitch"], at=(0.09, 0.13), size=0.06, color="white")
    ax.text(0.02, 0.97, f"$t = {meta['t'] * TIME_S:.0f}$ s", transform=ax.transAxes, va="top", ha="left",
            fontsize=style.LABEL_SIZE, color="white")

    sm = plt.cm.ScalarMappable(norm=LogNorm(lo, hi), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([10, 100, 1e3, 1e4, 1e5])
    cb.set_label(r"Density (g cm$^{-3}$)")
    cb.outline.set_linewidth(0.5)
    print(f"t = {meta['t']} (code), {meta['steps']} steps; color range {lo:.3g} to {hi:.3g} g/cm^3")

    style.save(fig, "merger")


if __name__ == "__main__":
    main()
