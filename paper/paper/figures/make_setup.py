"""A setup made in the Setup dialog, and the run that it yields: two magnetized stars that collide.

    uv run python paper/figures/make_setup.py          # assemble from data/setup/
    uv run python paper/figures/make_setup.py --run    # take the images again (two minutes on a GPU)

The setup is the document data/setup/collision.shaderous.json: two polytropes of a solar mass,
each with a tangled magnetic field, on a collision course. shots.mjs builds the run from that
file three times (data/setup/spec.json): once as a person sees the page, with the Setup dialog
open on its Objects page, and twice without a window, for the view after the stars have settled
and for the view as they merge. The .json beside each image records the setup in force, the time
reached and the GPU. --run needs Node, npm install in the code's repository, Chrome and a GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "setup"
CODE = HERE.parents[2]

# The part of the dialog that the figure shows, as fractions of its image (left, top, right,
# bottom): the settings and the preview, without the list of presets beside them and the row of
# buttons under them.
CROP = (0.186, 0.0, 1.0, 0.835)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="take the images again first")
    if ap.parse_args().run:
        subprocess.run(["node", str(HERE / "shots.mjs"), str(DATA / "spec.json"), str(DATA)], check=True, cwd=CODE)

    style.apply()
    meta = {name: json.loads((DATA / f"{name}.json").read_text()) for name in ("dialog", "run", "collision")}
    for name, m in meta.items():
        if m["errors"]:
            raise SystemExit(f"setup: the page reported errors in {name}: {m['errors']}")

    dialog = mpimg.imread(DATA / "dialog.png")
    h, w = dialog.shape[:2]
    x0, y0, x1, y1 = CROP
    dialog = dialog[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)]
    dh, dw = dialog.shape[:2]

    # The dialog on the left, and the two square views stacked on the right at half its height.
    width = style.PAGE
    wide = width / (1 + dh / (2 * dw))
    height = wide * dh / dw
    fig = plt.figure(figsize=(width, height))
    left = fig.add_axes((0, 0, wide / width, 1))
    left.imshow(dialog, interpolation="lanczos")
    views = []
    for i, name in enumerate(("run", "collision")):
        ax = fig.add_axes((wide / width, 0.5 * (1 - i), 1 - wide / width, 0.5))
        ax.imshow(mpimg.imread(DATA / f"{name}.png"), interpolation="lanczos")
        views.append(ax)
    for ax, label in zip([left, *views], "abc"):
        ax.set_axis_off()
        ax.text(0.012 if ax is left else 0.035, 0.985 if ax is left else 0.965, f"({label})", transform=ax.transAxes, ha="left", va="top", color="white", fontweight="bold")
    for name, m in meta.items():
        print(f"setup: {name} at t = {m['t']:.4g} ({m['steps']} steps) on {m['adapter']}")
    style.save(fig, "setup")


if __name__ == "__main__":
    main()
