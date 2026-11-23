"""The Moon-forming impact: four slices through the orbital plane, colored by what the gas is made of.

    uv run python paper/figures/make_impact.py                 # plot from data/impact/
    uv run python paper/figures/make_impact.py --from RUNDIR   # first make the panels and records again from a run

The run is the preset "Moon-forming impact" (moon-forming-impact) in a box 10 Earth radii across
(the preset's is 16), at 1024 cells per side at the finest level with four levels of refinement:
102 cells to an Earth radius. It ran on one NVIDIA A100 (80 GB) of the Unity cluster with the
Slurm job tools/headless/slurm/impact.sbatch, which runs impact_run.mjs in headless Chromium
(tools/headless/session.mjs): every 0.25 time units it saves slices through the orbital plane
(z at the middle of the box, 2048^2 samples, each from the finest block there) of the density,
the passive scalar (1 in the proto-Earth, 0.5 in Theia), the mass fraction of the iron alloy and
the refinement level, with the run's diagnostics and its impact() report (the planet, disk and
escaping gas and the moon the disk would make). The job was stopped at t = 12.25: from t = 12 the
bound clump left of Theia crosses the box's face and is lost, so only outputs to t = 12 (2.3 h
after contact) are used. The raw slices (5 GB) are not in the repository;
--from needs them in RUNDIR. Running the impact again needs a GPU with about 80 GB:
    node paper/paper/figures/impact_run.mjs RUNDIR --setup "problem=moon-forming-impact&n=1024&amr=4" \\
      --doc domain.box=10 --gpu-memory 74 --times 0,0.25,..,16 --every 0.25 --res 2048 --vars rho,r0,fe85si15,level

data/impact/ holds, for each panel, the composited image at the resolution embedded here
(tNN.png) and its output's record (tNN.json: setup, time, steps, GPU, diagnostics, the impact
report); the records of every output (reports/); the totals at every output (diag.jsonl); and
the time of contact measured from the slices (contact.json). The text's numbers come from these
(printed by this script).

Color: each of the four materials has its own color, the proto-Earth's mantle and core and
Theia's mantle and core, and a cell's color is the mean of the four weighted by its share of each
(the iron fraction from the species, the share from Theia from the passive scalar). The
brightness is the logarithm of the density, from black at 1e-4 g/cm^3 to full at 10 g/cm^3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from matplotlib.patches import Patch
from PIL import Image

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "impact"

PANELS = [1.5, 3, 6, 12]  # code time units of 805.5 s
# The last output used: after it, the bound clump left of Theia crosses the box's face (outflow)
# and is lost, which changes the disk and the planet from then on.
END = 12
HALF = 4.0  # half the panels' width, in Earth radii
BOX = 10.0  # the box, in Earth radii
PIXELS = 640  # each panel's width as embedded: about 370 pixels an inch of the page
CONTACT_RHO = 0.1  # code density (2.3 g/cm^3) above which gas counts as a body's, for the contact time

# Code units: Earth radii and Earth masses.
DENSITY_CGS = 5.9722e27 / 6.371e8**3
RHO_LO, RHO_HI = 1e-4, 10.0  # g/cm^3, black to full brightness
LUNAR = 0.0123  # lunar masses in Earth masses
MINUS = "\N{MINUS SIGN}"
COLORS = {  # the Okabe-Ito colors of style.PALETTE: cool for the proto-Earth, warm for Theia
    "Proto-Earth mantle": "#56B4E9",
    "Proto-Earth core": "#009E73",
    "Theia mantle": "#E69F00",
    "Theia core": "#D55E00",
}


def tag(t: float) -> str:
    return "t" + f"{t:g}".replace(".", "p")


def composite(rho, dye, iron):
    """RGB in [0, 1]: the four materials' colors mixed by share, dimmed by log density."""
    theia = np.clip(2 * (1 - dye), 0, 1)
    iron = np.clip(iron, 0, 1)
    c = [np.array(to_rgb(v)) for v in COLORS.values()]
    rgb = ((1 - theia) * (1 - iron))[..., None] * c[0] + ((1 - theia) * iron)[..., None] * c[1] \
        + (theia * (1 - iron))[..., None] * c[2] + (theia * iron)[..., None] * c[3]
    b = np.clip(np.log10(rho * DENSITY_CGS / RHO_LO) / np.log10(RHO_HI / RHO_LO), 0, 1)
    return rgb * b[..., None]


def gap(rundir: Path, k: str) -> float:
    """The least distance in the orbital plane, in Earth radii, between the two bodies' dense gas."""
    from scipy.ndimage import distance_transform_edt

    rho, dye = np.load(rundir / f"{k}-rho.npy"), np.load(rundir / f"{k}-r0.npy")
    dense = rho > CONTACT_RHO
    theia, earth = dense & (dye < 0.75), dense & (dye >= 0.75)
    return float(distance_transform_edt(~earth)[theia].min() * BOX / rho.shape[0])


def contact(rundir: Path, times: list[float]) -> dict:
    """The time of contact: the gap between the bodies, extrapolated linearly to zero from the last two outputs before it closes."""
    gaps = []
    for t in times:
        g = gap(rundir, tag(t))
        gaps.append((t, g))
        if g < 2 * BOX / 2048:
            break
    (t1, g1), (t2, g2) = gaps[-3], gaps[-2]
    return {"t": t2 + g2 * (t2 - t1) / (g1 - g2), "rho": CONTACT_RHO, "gaps": gaps}


def records(rundir: Path) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "reports").mkdir(exist_ok=True)
    times = sorted(t for f in rundir.glob("t*.json") if f.stem.count("-") == 0 and (t := json.loads(f.read_text())["t"]) <= END)
    for t in times:
        meta = json.loads((rundir / f"{tag(t)}.json").read_text())
        keep = {x: meta[x] for x in ("t", "steps", "wall", "setup", "adapter", "mesh", "gpuMemory", "diagnostics", "impact", "errors")}
        (DATA / "reports" / f"{tag(t)}.json").write_text(json.dumps(keep, indent=1) + "\n")
    diag = [x for x in (rundir / "diag.jsonl").read_text().splitlines() if json.loads(x)["t"] <= END]
    (DATA / "diag.jsonl").write_text("\n".join(diag) + "\n")
    (DATA / "contact.json").write_text(json.dumps(contact(rundir, times), indent=1) + "\n")
    for t in PANELS:
        k = tag(t)
        slices = [np.load(rundir / f"{k}-{v}.npy") for v in ("rho", "r0", "fe85si15")]
        res = slices[0].shape[0]
        # Rows are y, columns x; the panel is the middle 2 HALF of the box, y up.
        a = int(round(res * (0.5 - HALF / BOX)))
        rgb = composite(*(s[a : res - a, a : res - a] for s in slices))[::-1]
        Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).resize((PIXELS, PIXELS), Image.LANCZOS).save(DATA / f"{k}.png")
        meta = json.loads((DATA / "reports" / f"{k}.json").read_text())
        meta["panel"] = {"half": HALF, "box": BOX, "res": res, "pixels": PIXELS}
        (DATA / f"{k}.json").write_text(json.dumps(meta, indent=1) + "\n")


def numbers(minutes: float, t_contact: float) -> None:
    """The numbers the text quotes, from the run's records."""
    lines = [json.loads(x) for x in (DATA / "diag.jsonl").read_text().splitlines()]
    m0, e0 = lines[0]["mass"], lines[0]["eTot"]
    last = lines[-1]
    print(f"contact at t = {t_contact:.2f} ({t_contact * minutes:.0f} min)")
    print(f"run to t = {last['t']} ({last['t'] * minutes / 60:.1f} h), {last['steps']} steps, {last['wall'] / 3600:.1f} h of wall time")
    print(f"mass lost {1 - last['mass'] / m0:.1e}, largest energy change {max(abs(x['eTot'] / e0 - 1) for x in lines):.1e}")
    print(f"largest mesh: {max(lines, key=lambda x: int(x['mesh'].split()[0]))['mesh']}; page errors: {sum(x['errors'] for x in lines)}")
    reports = sorted((json.loads(f.read_text()) for f in (DATA / "reports").glob("*.json")), key=lambda r: r["t"])
    for r in reports:
        i = r["impact"]
        if not i or "error" in i or r["t"] < t_contact + 60 / minutes:
            continue
        d = i["disk"]
        theia = 2 * (1 - d["shares"]["dye"]) if d["mass"] else 0
        print(f"t = {r['t']:5} ({(r['t'] - t_contact) * minutes / 60:.2f} h after contact): planet {i['planet']['mass']:.3f} "
              f"({i['planet']['shares']['fe85si15']:.0%} iron), disk {d['mass']:.4f} ({theia:.0%} Theia, {d['shares']['fe85si15']:.1%} iron, "
              f"L/L_R {d['overRoche']:.2f}), escaping {i['escaping']['mass']:.1e}, moon {i['moon'] / LUNAR:.2f} lunar masses")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, metavar="RUNDIR", help="make the panels and records again from RUNDIR")
    args = ap.parse_args()
    if args.src:
        records(args.src)

    metas = [json.loads((DATA / f"{tag(t)}.json").read_text()) for t in PANELS]
    for m in metas:
        if m["errors"]:
            raise SystemExit(f"t = {m['t']}: the page reported errors: {m['errors']}")
    minutes = metas[0]["impact"]["units"]["time"] / 60
    t_contact = json.loads((DATA / "contact.json").read_text())["t"]
    numbers(minutes, t_contact)

    style.apply()
    n = len(PANELS)
    gap_w = 0.008
    w = (1 - (n - 1) * gap_w) / n
    key = 0.2  # the strip below the images for the key, as a fraction of the figure's height
    height = style.PAGE * w / (1 - key)
    fig = plt.figure(figsize=(style.PAGE, height))
    for i, (t, m) in enumerate(zip(PANELS, metas, strict=True)):
        ax = fig.add_axes([i * (w + gap_w), key, w, 1 - key])
        ax.imshow(np.asarray(Image.open(DATA / f"{tag(t)}.png")), interpolation="none", extent=[-HALF, HALF, -HALF, HALF])
        ax.set_axis_off()
        dt = (m["t"] - t_contact) * minutes
        when = (f"{dt:+.0f} min" if abs(dt) < 90 else f"{dt / 60:+.1f} h").replace("-", MINUS)
        ax.text(0.03, 0.97, f"({'abcd'[i]}) $t = {m['t'] * minutes / 60:.2f}$ h", transform=ax.transAxes,
                va="top", ha="left", fontsize=style.LABEL_SIZE, color="white")
        ax.text(0.03, 0.89, f"{when} from contact", transform=ax.transAxes, va="top", ha="left",
                fontsize=style.LABEL_SIZE - 1, color="white")
        if i == 0:
            style.triad_plane(ax, at=(0.1, 0.1), size=0.08, color="white")
            # A scale bar of two Earth radii.
            ax.plot([HALF - 2.5, HALF - 0.5], [-HALF + 0.5] * 2, color="white", lw=1.0)
            ax.text(HALF - 1.5, -HALF + 0.62, r"$2\,R_\oplus$", ha="center", va="bottom", color="white", fontsize=style.LABEL_SIZE - 1)

    # The key: the four materials, and the brightness against density.
    lax = fig.add_axes([0.0, 0.0, 0.5, key - 0.02])
    lax.set_axis_off()
    lax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=k) for k, c in COLORS.items()],
               loc="center", ncol=2, frameon=False, fontsize=style.LABEL_SIZE, handlelength=1.6, columnspacing=1.5)
    cax = fig.add_axes([0.56, 0.42 * key, 0.4, 0.2 * key])
    gray = LinearSegmentedColormap.from_list("brightness", ["black", "white"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(RHO_LO, RHO_HI), cmap=gray), cax=cax, orientation="horizontal")
    cb.set_label(r"Density (g cm$^{-3}$), as brightness")
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.5)

    style.save(fig, "impact")


if __name__ == "__main__":
    main()
