"""Shared look for every figure in the paper (.cursor/rules/figure-design.mdc).

    import style
    style.apply()
    fig, ax = plt.subplots(figsize=(style.COLUMN, 0.75 * style.COLUMN))
    ax.plot(x, y, color=style.INK)
    ax.legend(**style.LEGEND)
    ax.text(x0, y0, "label", bbox=style.LABEL_BOX)
    style.save(fig, "name")          # writes paper/figures/name.pdf

Change the look here, never in one script, so the figures stay a set.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import matplotlib as mpl
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Approximate column and full-page widths in inches for each journal class.
WIDTHS = {
    "aastex": (3.5, 7.1),
    "mnras": (3.32, 6.97),
    "aa": (3.46, 7.09),
    "elsarticle": (3.5, 7.0),
    "revtex": (3.4, 7.0),
    "amsart": (5.0, 5.0),
}
with (ROOT / "paper.toml").open("rb") as fh:
    JOURNAL = tomllib.load(fh)["paper"].get("journal", "aastex")
COLUMN, PAGE = WIDTHS.get(JOURNAL, (3.5, 7.1))

# Okabe & Ito (2008) colorblind-safe palette, in the order series should use it.
INK = "#222222"  # data, the primary series
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
MUTED = "#7a7a7a"  # windows, references, de-emphasized series

FONT_SIZE = 8
LABEL_SIZE = 7
LEGEND = dict(frameon=True, facecolor="white", framealpha=0.8, edgecolor="none",
              fontsize=LABEL_SIZE, labelspacing=0.32)
LABEL_BOX = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.15)


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": LABEL_SIZE,
        "ytick.labelsize": LABEL_SIZE,
        "legend.fontsize": LABEL_SIZE,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # embed TrueType so journals' preflight checks pass
    })


def save(fig, name: str) -> Path:
    out = HERE / f"{name}.pdf"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT).as_posix()}")
    return out


def triad(ax, yaw: float, pitch: float, at=(0.11, 0.11), size: float = 0.075, color: str = INK) -> None:
    """Arrows along x, y and z as the renderer's orbit camera sees them, in a corner of an image.

    The camera (cameraFrame() in src/render/renderer.ts) sits at yaw and pitch about the box
    center, z up; each axis is drawn as its projection on the screen's right and up directions.
    `at` and `size` are in fractions of the axes.
    """
    fwd = -np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    right = np.array([fwd[1], -fwd[0], 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    x0, y0 = at
    for axis, name in zip(np.eye(3), "xyz", strict=True):
        dx, dy = size * axis @ right, size * axis @ up
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=5))
        ax.text(x0 + 1.35 * dx, y0 + 1.35 * dy, f"${name}$", transform=ax.transAxes, ha="center", va="center",
                fontsize=LABEL_SIZE - 1, color=color)


def crop_to_content(img, aspect: float, margin: float = 0.02):
    """A rendering on a white background cut to what is drawn, plus a margin, at width / height = aspect.

    The cut is centered on the drawn pixels and widened in one direction to the aspect; where it
    runs past the image's edge, the image is padded with white.
    """
    rgb = img[..., :3]
    drawn = (rgb < 0.97).any(axis=2)
    rows = np.where(drawn.any(axis=1))[0]
    cols = np.where(drawn.any(axis=0))[0]
    cy, cx = (rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2
    h = (rows[-1] - rows[0]) * (1 + 2 * margin)
    w = (cols[-1] - cols[0]) * (1 + 2 * margin)
    h, w = max(h, w / aspect), max(w, h * aspect)
    pad = int(max(h, w))
    big = np.ones((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, img.shape[2]), dtype=img.dtype)
    big[pad : pad + img.shape[0], pad : pad + img.shape[1]] = img
    y0, x0 = int(round(cy - h / 2)) + pad, int(round(cx - w / 2)) + pad
    return big[y0 : y0 + int(round(h)), x0 : x0 + int(round(w))]
