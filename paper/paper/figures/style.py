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
