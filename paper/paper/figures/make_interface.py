"""Shaderous in the browser: the whole page during a 3D run of an accreting binary.

    uv run python paper/figures/make_interface.py          # assemble from data/interface/
    uv run python paper/figures/make_interface.py --run    # take the screenshot again (minutes on a GPU)

The screenshot is taken by shots.mjs, which opens the page as a person sees it in headless
Chrome, runs the setup to a time and captures the whole window. The setup is the binary-accretion
preset in 3D with the binary's orbit tilted 90 degrees to the disk (polar), shown as isosurfaces
of the density with the fog of the volume view off; data/interface/spec.json gives the setup, the
time and the view, and interface.json beside the image records the setup in force, the time
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
DATA = HERE / "data" / "interface"
CODE = HERE.parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="take the screenshot again first")
    if ap.parse_args().run:
        subprocess.run(["node", str(HERE / "shots.mjs"), str(DATA / "spec.json"), str(DATA)], check=True, cwd=CODE)

    style.apply()
    meta = json.loads((DATA / "interface.json").read_text())
    if meta["errors"]:
        raise SystemExit(f"interface: the page reported errors: {meta['errors']}")
    img = mpimg.imread(DATA / "interface.png")
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(style.PAGE, style.PAGE * h / w), layout="constrained")
    fig.get_layout_engine().set(w_pad=0, h_pad=0)
    ax.imshow(img, interpolation="none")
    ax.set_axis_off()
    print(f"interface: {meta['setup']} at t = {meta['t']:.4g} ({meta['steps']} steps)")
    style.save(fig, "interface")


if __name__ == "__main__":
    main()
