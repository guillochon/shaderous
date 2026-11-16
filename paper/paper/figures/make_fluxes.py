"""How the hydrodynamic solver conserves without storing a face flux: a diagram.

    uv run python paper/figures/make_fluxes.py

Drawn here, from no data. (a) Two cells of one level share a face; each evaluates the face's flux
for itself, from the same inputs with the same arithmetic, so the two agree bit for bit.
(b) A refinement boundary, in two dimensions: the fine cells evaluate the fluxes of their own
faces against ghost cells that are computed as they are loaded, and the coarse cell evaluates the
same fine fluxes again and takes their mean. (c) A boundary between time bins: the slower block
takes the fluxes of the faster block's two substeps, half when its step starts and half at its
midpoint, and the faster block reads the slower one from its half-step prediction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import style

FACE = style.PALETTE[1]  # the shared face and its flux
LEFT = style.PALETTE[0]  # what the left (coarse, slow) side evaluates
RIGHT = style.PALETTE[2]  # what the right (fine, fast) side evaluates
SMALL = style.LABEL_SIZE - 1


def cell(ax, x, y, w, h, *, fill="white", edge=style.INK, ls="-", lw=0.8, z=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linestyle=ls, linewidth=lw, zorder=z))


def arrow(ax, a, b, color, *, ls="-", lw=1.1, rad=0.0, style_="-|>", z=4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style_, mutation_scale=7, color=color, linestyle=ls, linewidth=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def panel(ax, label, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(0.0, 1.0, label, transform=ax.transAxes, va="top", ha="left", fontsize=style.LABEL_SIZE)


def same_level(ax) -> None:
    panel(ax, "(a) One level", (-1.25, 3.25), (-1.7, 2.6))
    for i, name in enumerate(["$i-1$", "$i$", "$i+1$", "$i+2$"]):
        inner = i in (1, 2)
        cell(ax, i - 1, 0, 1, 1, fill="white" if inner else "#f1f1f1", edge=style.INK if inner else style.MUTED)
        ax.text(i - 0.5, 0.14, name, ha="center", va="bottom", fontsize=SMALL, color=style.INK if inner else style.MUTED)
    ax.plot([1, 1], [0, 1], color=FACE, lw=2.2, zorder=3, solid_capstyle="butt")
    # Each cell evaluates the face's flux for itself.
    arrow(ax, (0.62, 0.72), (1.38, 0.72), LEFT)
    arrow(ax, (0.62, 0.50), (1.38, 0.50), RIGHT)
    ax.text(0.5, 1.12, "evaluated\nfor cell $i$", ha="center", va="bottom", fontsize=SMALL, color=LEFT)
    ax.text(1.5, 1.12, "evaluated\nfor cell $i+1$", ha="center", va="bottom", fontsize=SMALL, color=RIGHT)
    arrow(ax, (0.5, 1.1), (0.72, 0.76), LEFT, lw=0.6, style_="-", z=2)
    arrow(ax, (1.5, 1.1), (1.28, 0.54), RIGHT, lw=0.6, style_="-", z=2)
    ax.text(1.0, -0.14, "$F_{i+1/2}$: the same bits", ha="center", va="top", fontsize=SMALL, color=FACE)
    ax.text(1.0, -0.52, "same inputs, same arithmetic,\none compiled pipeline; nothing stored", ha="center", va="top",
            fontsize=SMALL, color=style.INK)


def refinement(ax) -> None:
    panel(ax, "(b) Refinement boundary", (-0.35, 4.3), (-1.25, 3.05))
    cell(ax, 0, 0, 2, 2)
    # The fine ghost cells of the fine block, inside the coarse cell: computed as they are loaded.
    for j in (0, 1):
        cell(ax, 1, j, 1, 1, fill="#eaf3fa", edge=LEFT, ls=(0, (3, 2)), lw=0.7, z=2)
        cell(ax, 2, j, 1, 1)
        cell(ax, 3, j, 1, 1, fill="#f1f1f1", edge=style.MUTED)
        ax.plot([2, 2], [j + 0.04, j + 0.96], color=FACE, lw=2.2, zorder=3, solid_capstyle="butt")
        arrow(ax, (1.6, j + 0.62), (2.4, j + 0.62), LEFT)
        arrow(ax, (1.6, j + 0.40), (2.4, j + 0.40), RIGHT)
        ax.text(2.56, j + 0.5, f"$F_{j + 1}$", ha="left", va="center", fontsize=SMALL, color=FACE)
    ax.text(0.5, 1.0, "coarse\ncell", ha="center", va="center", fontsize=SMALL)
    ax.text(1.5, 2.1, "ghost cells,\ncomputed on load", ha="center", va="bottom", fontsize=SMALL, color=LEFT)
    ax.text(3.0, 2.1, "fine cells", ha="center", va="bottom", fontsize=SMALL)
    ax.text(1.0, -0.14, "recomputes $F_1$, $F_2$;\ntakes their mean", ha="center", va="top", fontsize=SMALL, color=LEFT)
    ax.text(3.0, -0.14, "each takes\nits own $F$", ha="center", va="top", fontsize=SMALL, color=RIGHT)


def time_bins(ax) -> None:
    panel(ax, "(c) Time-bin boundary", (-0.95, 4.35), (-1.15, 3.15))
    # Time runs to the right; the faster block above, the slower below, the shared face between.
    cell(ax, 0, 1.55, 2, 0.8)
    cell(ax, 2, 1.55, 2, 0.8)
    cell(ax, 0, 0.0, 4, 0.8)
    ax.plot([2, 2], [0, 0.8], color=style.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(1, 1.95, "substep 1", ha="center", va="center", fontsize=SMALL)
    ax.text(3, 1.95, "substep 2", ha="center", va="center", fontsize=SMALL)
    ax.text(2, 0.4, "one step", ha="center", va="center", fontsize=SMALL, bbox=style.LABEL_BOX)
    ax.text(-0.12, 1.95, "fast", ha="right", va="center", fontsize=SMALL, color=RIGHT)
    ax.text(-0.12, 0.4, "slow", ha="right", va="center", fontsize=SMALL, color=LEFT)
    # The face's flux in each substep, taken by both blocks.
    for k, x in enumerate((1, 3)):
        arrow(ax, (x, 1.17), (x, 1.53), RIGHT)
        arrow(ax, (x, 1.17), (x, 0.82), LEFT)
        ax.text(x + 0.12, 1.38, f"$F^{{({k + 1})}}\\,\\Delta t$", ha="left", va="center", fontsize=SMALL, color=FACE)
        ax.plot([x - 0.09, x + 0.09], [1.175, 1.175], color=FACE, lw=2.2, zorder=5, solid_capstyle="butt")
    # The slower block applies them when its step starts and at its midpoint.
    ax.text(0.0, -0.36, "applied at\nthe start", ha="center", va="top", fontsize=SMALL, color=LEFT)
    ax.text(2.0, -0.36, "applied at\nthe midpoint", ha="center", va="top", fontsize=SMALL, color=LEFT)
    for x in (0, 2):
        arrow(ax, (x, -0.34), (x, -0.02), LEFT, lw=0.8)
    # The faster block's second substep reads the slower block's half-step prediction.
    arrow(ax, (2.0, 0.82), (2.0, 1.53), style.MUTED, ls=(0, (3, 2)), lw=0.8)
    ax.text(2.07, 1.0, "half-step\nprediction", ha="left", va="center", fontsize=SMALL - 0.5, color=style.MUTED, linespacing=0.95)
    # The time axis.
    arrow(ax, (-0.1, 2.74), (4.25, 2.74), style.INK, lw=0.7)
    for x, name in ((0, "$t$"), (2, r"$t + \Delta t$"), (4, r"$t + 2\Delta t$")):
        ax.plot([x, x], [2.68, 2.80], color=style.INK, lw=0.7)
        ax.text(x, 2.64, name, ha="center", va="top", fontsize=SMALL)


def main() -> None:
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.PAGE, 0.3 * style.PAGE), layout="constrained",
                             gridspec_kw={"width_ratios": [4.5, 4.65, 5.3]})
    same_level(axes[0])
    refinement(axes[1])
    time_bins(axes[2])
    style.save(fig, "fluxes")


if __name__ == "__main__":
    main()
