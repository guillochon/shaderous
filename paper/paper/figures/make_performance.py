"""Throughput: cell updates per second against the number of cells.

    uv run python paper/figures/make_performance.py                      # plot from data/performance/
    uv run python paper/figures/make_performance.py --run --device NAME  # measure again on this machine's GPU
    uv run python paper/figures/make_performance.py --run --firefox --device NAME  # the same in Firefox

Each device measured is listed under "devices" in data/performance/reference.json, with its data
file and its peak memory bandwidth under "bandwidth_gb_s"; --device names which one this machine is.

collect.mjs runs src/validation/figure_data.ts (throughput) in the code's page in headless Chrome,
at normal priority (a benchmark): for each grid, a warm-up, then a fixed number of steps timed on
the GPU, counting every cell of a uniform grid and every leaf cell of an adaptive mesh as one
update per step (per-block timesteps off). A 2D Kelvin-Helmholtz instability and a 3D Sedov blast,
each on uniform grids and on adaptive meshes. The adapter and browser are recorded with the data.
The whole set is measured three times in fresh pages; each point is the median, with the range.
Every device is drawn in its own color, and Firefox on the same GPU as a device of its own (its
entry names the GPU under "gpu", for the bandwidth). --run needs Node, npm install in the code's repository, Chrome
and a GPU.

The published throughput of AthenaK (Stone2026, hydrodynamics on one V100 and one A100) is drawn
for comparison, from data/performance/reference.json with the devices' peak memory bandwidths
from the vendors' datasheets. Panel (b) divides every throughput by its device's peak bandwidth,
since both codes are limited by memory traffic; the A100's bar spans its 40 GB and 80 GB cards. The hollow
markers in (b) are AthenaK's values times four, what it would take per byte moved if its benchmark
ran as its repository's defaults do, in double precision with a two-stage integrator.
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


def measure(device: str, firefox: bool = False) -> None:
    """Run the whole set REPEATS times, each in a fresh page, and keep the median of each point."""
    sets = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(REPEATS):
            out = Path(tmp) / f"throughput_{i}.json"
            subprocess.run(["node", str(HERE / "collect.mjs"), "throughput", str(out), json.dumps([GRIDS]), "--normal",
                            *(["--firefox"] if firefox else [])],
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
    ap.add_argument("--firefox", action="store_true", help="measure in the installed Firefox, not in Chrome")
    ap.add_argument("--device", help="the device this machine's measurement is for, as named in reference.json")
    ap.add_argument("--grids", action="store_true",
                    help="print collect.mjs's argument for the set and stop (for runs elsewhere, as tools/headless/slurm/throughput.sbatch)")
    args = ap.parse_args()
    if args.grids:
        print(json.dumps([GRIDS]))
        return
    if (args.run or args.merge) and not args.device:
        ap.error("--run and --merge need --device")
    if args.run:
        measure(args.device, args.firefox)
    elif args.merge:
        merge([json.loads(p.read_text()) for p in args.merge], args.device)

    ref = json.loads((DATA / "reference.json").read_text())
    devices = [{**d, "bw": ref["bandwidth_gb_s"][d.get("gpu", d["name"])]["value"][0],
                "data": json.loads((DATA / d["data"]).read_text())} for d in ref["devices"]]
    style.apply()
    # Rows: the throughput (a) and the same per unit of bandwidth (b). Columns: two dimensions and
    # three, so that each panel holds two grids per device.
    fig, axes = plt.subplots(2, 2, figsize=(style.PAGE, 0.78 * style.PAGE), sharex="col", sharey="row", layout="constrained")
    markers = {"2D uniform": "o", "2D adaptive": "s", "3D uniform": "^", "3D adaptive": "D"}
    for row, divide in enumerate([False, True]):
        for k, d in enumerate(devices):
            per = d["bw"] if divide else 1.0
            for g in d["data"]["grids"]:
                ax = axes[row, g["ndim"] - 2]
                cells = [r["cells"] for r in g["runs"]]
                rate = [r["updates_per_second"] / per for r in g["runs"]]
                lo = [(r["updates_per_second"] - min(r.get("repeats", [r["updates_per_second"]]))) / per for r in g["runs"]]
                hi = [(max(r.get("repeats", [r["updates_per_second"]])) - r["updates_per_second"]) / per for r in g["runs"]]
                filled = "uniform" in g["label"]
                ax.errorbar(cells, rate, yerr=[lo, hi], marker=markers[g["label"]], color=style.PALETTE[k],
                            mfc=None if filled else "white", ls="-" if filled else "--", ms=3.5, capsize=0, elinewidth=0.8)
        # AthenaK's benchmark is a three-dimensional uniform grid: drawn with the 3D panels.
        ax = axes[row, 1]
        a = ref["athenak"]
        # One marker per device, drawn apart along x for legibility (all ran the same grid).
        at = {"V100": (0.72, "P"), "A100": (1.0, "*"), "H100": (1.38, "X")}
        for k, p in enumerate(a["points"]):
            shift, mk = at[p["device"]]
            bw = p["bandwidth_gb_s"]
            if not divide:
                y, err = p["updates_per_second"], None
            else:
                ys = [p["updates_per_second"] / b for b in bw]
                y = (max(ys) + min(ys)) / 2
                err = [[y - min(ys)], [max(ys) - y]] if max(ys) > min(ys) else None
            ax.errorbar([a["cells"] * shift], [y], yerr=err, marker=mk, color=style.MUTED, ms=6, ls="none",
                        capsize=2, elinewidth=0.8, label=f"AthenaK, {p['device']}, 3D uniform")
            if divide:
                # The same per byte moved, if the benchmark ran in double precision with two stages
                # per step (see "like_for_like" in reference.json): hollow.
                f = a["like_for_like"]["factor"]
                ax.errorbar([a["cells"] * shift], [f * y], yerr=None if err is None else [[f * e[0]] for e in err],
                            marker=mk, color=style.MUTED, mfc="white", ms=6, ls="none", capsize=2, elinewidth=0.8)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0, 0].set_xlim(4e4, 8e6)
    axes[0, 1].set_xlim(2e4, 8e7)
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
    axes[0, 0].set_ylabel(r"Cell updates per second (s$^{-1}$)")
    axes[1, 0].set_ylabel("Cell updates per second\nper GB/s of peak bandwidth")
    for ax in axes[1]:
        ax.set_xlabel("Cells (leaf cells on the adaptive mesh)")
    # Color is the device, the marker the grid: one legend entry for each, not one per pair.
    handles = [Line2D([], [], color=style.PALETTE[k], label=f"Shaderous, {d['label']}") for k, d in enumerate(devices)]
    handles += [Line2D([], [], color=style.INK, marker=m, ms=3.5, mfc=None if "uniform" in g else "white",
                       ls="-" if "uniform" in g else "--", label=g) for g, m in markers.items()]
    handles += axes[0, 1].get_legend_handles_labels()[0][:len(ref["athenak"]["points"])]
    handles += [Line2D([], [], color=style.MUTED, marker="*", mfc="white", ms=6, ls="none",
                       label="AthenaK, per byte moved (b)")]
    # Above the panels, where it covers no point.
    fig.legend(handles=handles, loc="outside upper center", ncols=4, **{**style.LEGEND, "fontsize": style.LABEL_SIZE - 1})
    for ax, tag in zip(axes.flat, ["(a) Two dimensions", "(a) Three dimensions", "(b) Two dimensions", "(b) Three dimensions"]):
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, va="top", fontsize=style.LABEL_SIZE - 1, bbox=style.LABEL_BOX)
    # One line per GPU: a second browser on the same GPU has the same bandwidth.
    gpus = {d.get("gpu", d["name"]): d["bw"] for d in devices}
    bws = "\n".join(f"{name}: {bw:,} GB/s" for name, bw in gpus.items())
    axes[1, 0].text(0.97, 0.03, f"Peak memory bandwidth\n{bws}", transform=axes[1, 0].transAxes, ha="right", va="bottom",
                    fontsize=style.LABEL_SIZE - 1, bbox=style.LABEL_BOX)
    style.save(fig, "performance")


if __name__ == "__main__":
    main()
