"""Throughput: cell updates per second against the number of cells.

    uv run python paper/figures/make_performance.py                      # plot from data/performance/
    uv run python paper/figures/make_performance.py --run --device NAME  # measure again on this machine's GPU

Each device measured is listed under "devices" in data/performance/reference.json, with its data
file and its peak memory bandwidth under "bandwidth_gb_s"; --device names which one this machine is.

collect.mjs runs src/validation/figure_data.ts (throughput) in the code's page in headless Chrome,
at normal priority (a benchmark): for each grid, a warm-up, then a fixed number of steps timed on
the GPU, counting every cell of a uniform grid and every leaf cell of an adaptive mesh as one
update per step (per-block timesteps off). A 2D Kelvin-Helmholtz instability and a 3D Sedov blast,
each on uniform grids and on adaptive meshes. The adapter and browser are recorded with the data.
The whole set is measured three times in fresh pages; each point is the median, with the range.
Every device is drawn in its own color; the figure the paper plans also needs Firefox. --run needs Node, npm install in the code's repository, Chrome
and a GPU.

The published throughput of AthenaK (Stone2026, hydrodynamics on one V100 and one A100) is drawn
for comparison, from data/performance/reference.json with the devices' peak memory bandwidths
from the vendors' datasheets. Panel (b) divides every throughput by its device's peak bandwidth,
since both codes are limited by memory traffic; the A100's bar spans its 40 GB and 80 GB cards.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "performance"
CODE = HERE.parents[2]
REPEATS = 3
GRIDS = [
    {"label": "2D uniform", "problem": "kelvin-helmholtz", "ndim": 2, "runs": [[256, 0], [512, 0], [1024, 0], [2048, 0]]},
    {"label": "2D adaptive", "problem": "kelvin-helmholtz", "ndim": 2, "runs": [[1024, 2], [2048, 3], [4096, 4]]},
    {"label": "3D uniform", "problem": "sedov", "ndim": 3, "runs": [[32, 0], [64, 0], [128, 0], [256, 0]]},
    {"label": "3D adaptive", "problem": "sedov", "ndim": 3, "runs": [[128, 2], [256, 3], [512, 4]]},
]


def measure(device: str) -> None:
    """Run the whole set REPEATS times, each in a fresh page, and keep the median of each point."""
    sets = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(REPEATS):
            out = Path(tmp) / f"throughput_{i}.json"
            subprocess.run(["node", str(HERE / "collect.mjs"), "throughput", str(out), json.dumps([GRIDS]), "--normal"],
                           check=True, cwd=CODE)
            sets.append(json.loads(out.read_text()))
    merge(sets, device)


def data_file(device: str) -> Path:
    ref = json.loads((DATA / "reference.json").read_text())
    for d in ref["devices"]:
        if d["name"] == device:
            return DATA / d["data"]
    raise SystemExit(f"{device!r} is not listed under devices in {DATA / 'reference.json'}: add it, with its bandwidth")


def merge(sets: list[dict], device: str) -> None:
    data = sets[0]
    for gi, g in enumerate(data["grids"]):
        for ri, r in enumerate(g["runs"]):
            each = [s["grids"][gi]["runs"][ri] for s in sets]
            rates = [e["updates_per_second"] for e in each]
            med = sorted(each, key=lambda e: e["updates_per_second"])[len(each) // 2]
            g["runs"][ri] = {**med, "updates_per_second": statistics.median(rates), "repeats": rates}
    DATA.mkdir(parents=True, exist_ok=True)
    data_file(device).write_text(json.dumps(data) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="measure again first")
    ap.add_argument("--merge", nargs="+", type=Path, help="merge sets measured separately (collect.mjs outputs)")
    ap.add_argument("--device", help="the device this machine's measurement is for, as named in reference.json")
    args = ap.parse_args()
    if (args.run or args.merge) and not args.device:
        ap.error("--run and --merge need --device")
    if args.run:
        measure(args.device)
    elif args.merge:
        merge([json.loads(p.read_text()) for p in args.merge], args.device)

    ref = json.loads((DATA / "reference.json").read_text())
    devices = [{**d, "bw": ref["bandwidth_gb_s"][d["name"]]["value"][0],
                "data": json.loads((DATA / d["data"]).read_text())} for d in ref["devices"]]
    style.apply()
    fig, axes = plt.subplots(2, 1, figsize=(style.COLUMN, 1.45 * style.COLUMN), sharex=True, layout="constrained")
    markers = {"2D uniform": "o", "2D adaptive": "s", "3D uniform": "^", "3D adaptive": "D"}
    for ax, divide in zip(axes, [False, True]):
        for k, d in enumerate(devices):
            per = d["bw"] if divide else 1.0
            for g in d["data"]["grids"]:
                cells = [r["cells"] for r in g["runs"]]
                rate = [r["updates_per_second"] / per for r in g["runs"]]
                lo = [(r["updates_per_second"] - min(r.get("repeats", [r["updates_per_second"]]))) / per for r in g["runs"]]
                hi = [(max(r.get("repeats", [r["updates_per_second"]])) - r["updates_per_second"]) / per for r in g["runs"]]
                filled = "uniform" in g["label"]
                ax.errorbar(cells, rate, yerr=[lo, hi], marker=markers[g["label"]], color=style.PALETTE[k],
                            mfc=None if filled else "white", ls="-" if filled else "--", ms=3.5, capsize=0, elinewidth=0.8)
        a = ref["athenak"]
        for k, p in enumerate(a["points"]):
            bw = p["bandwidth_gb_s"]
            if not divide:
                y, err = p["updates_per_second"], None
            else:
                ys = [p["updates_per_second"] / b for b in bw]
                y = (max(ys) + min(ys)) / 2
                err = [[y - min(ys)], [max(ys) - y]] if max(ys) > min(ys) else None
            ax.errorbar([a["cells"] * (0.85 if k == 0 else 1.18)], [y], yerr=err, marker="*" if k else "P", color=style.MUTED, ms=6, ls="none",
                        capsize=2, elinewidth=0.8, label=f"AthenaK, {p['device']}, 3D uniform")
        ax.set_xscale("log")
        ax.set_xlim(2e4, 8e7)
        ax.set_yscale("log")
    for d in devices:
        print(f"{d['name']} ({d['data']['adapter']}, {d['bw']} GB/s)")
        for g in d["data"]["grids"]:
            for r in g["runs"]:
                print(f"  {g['label']:12s} n = {r['n']:5d}, {r['levels']} levels: {r['cells']:>10,} cells, "
                      f"{r['updates_per_second']:.3g} updates/s (range {min(r.get('repeats', [0])):.3g} to "
                      f"{max(r.get('repeats', [0])):.3g}), {r['ms_per_step']:.2f} ms/step, "
                      f"{r['updates_per_second'] / d['bw']:.3g} per GB/s")
    for p in ref["athenak"]["points"]:
        print(f"AthenaK {p['device']}: {p['updates_per_second']:.3g} updates/s, "
              + ", ".join(f"{p['updates_per_second'] / b:.3g} per GB/s at {b} GB/s" for b in p["bandwidth_gb_s"]))
    axes[0].set_ylim(top=3.5e9)
    axes[1].set_ylim(top=3.5e6)
    axes[0].set_ylabel(r"Cell updates per second (s$^{-1}$)")
    axes[1].set_ylabel("Cell updates per second\nper GB/s of peak bandwidth")
    axes[1].set_xlabel("Cells (leaf cells on the adaptive mesh)")
    # Color is the device, the marker the grid: one legend entry for each, not one per pair.
    handles = [Line2D([], [], color=style.PALETTE[k], label=f"Shaderous, {d['label']}") for k, d in enumerate(devices)]
    handles += [Line2D([], [], color=style.INK, marker=m, ms=3.5, mfc=None if "uniform" in g else "white",
                       ls="-" if "uniform" in g else "--", label=g) for g, m in markers.items()]
    handles += axes[0].get_legend_handles_labels()[0][:len(ref["athenak"]["points"])]
    axes[0].legend(handles=handles, loc="lower right", **{**style.LEGEND, "fontsize": style.LABEL_SIZE - 1})
    browsers = sorted({d["data"]["adapter"].split(", ")[-1].split("/")[0] for d in devices})
    axes[0].text(0.03, 0.97, f"(a) {', '.join(browsers)}", transform=axes[0].transAxes, va="top",
                 fontsize=style.LABEL_SIZE - 1, bbox=style.LABEL_BOX)
    bws = "\n".join(f"{d['label']}: {d['bw']} GB/s" for d in devices)
    axes[1].text(0.36, 0.03, f"(b) Divided by peak memory bandwidth\n{bws}",
                 transform=axes[1].transAxes, va="bottom", fontsize=style.LABEL_SIZE - 1, bbox=style.LABEL_BOX)
    style.save(fig, "performance")


if __name__ == "__main__":
    main()
