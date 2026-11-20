"""The figures of docs/closure.md, from what evaluate.py and appchecks.py wrote to eval/.

    tools/closure/.venv/bin/python tools/closure/figures.py

Writes docs/img/closure/*.png. One color per run throughout: the reference in black, the
coarse run as it is in blue, with the low-Mach Riemann solver in orange, with the closure by
constants in green and by the trained network in dashed yellow.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import metrics as M

HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval"
OUT = HERE.parents[1] / "docs" / "img" / "closure"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
STYLE = {
    "ref": ("Reference (4x finer, averaged)", INK, "-"),
    "base": ("Coarse, no closure", "#2a78d6", "-"),
    "hllclm": ("Coarse, low-Mach solver", "#eb6834", "-"),
    "constant": ("Coarse, closure (constants)", "#1baf7a", "-"),
    "closure": ("Coarse, closure (network)", "#eda100", "--"),
}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white", "lines.linewidth": 1.8, "axes.titlesize": 9.5,
})


def title(case):
    f = case["flow"]
    if case["preset"] == "decaying-turbulence":
        return f"Turbulence, Mach {f['mach']}" + (f", band {f['ratio']}x denser" if f["ratio"] != 1 else "")
    return f"Shear layer, speed {f['shear']}, density ratio {f['rhoIn']}"


def load(stem):
    return json.loads((EVAL / f"{stem}.json").read_text()), np.load(EVAL / f"{stem}.npz")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(name)


def spectra(res, snaps, names, fname, which="E"):
    """Kinetic energy (or scalar variance) spectra at the end of each run."""
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.0), sharey=False)
    for ax, name in zip(np.atleast_1d(axes), names):
        fn = M.ke_spectrum if which == "E" else M.scalar_spectrum
        for key in ("ref", "base", "hllclm", "constant", "closure"):
            if f"{name}/{key}" not in snaps:
                continue
            U = snaps[f"{name}/{key}"][-1].astype(np.float64)
            E = fn(U)
            k = np.arange(len(E))
            label, color, ls = STYLE[key]
            ax.loglog(k[1:], E[1:], color=color, ls=ls, label=label, lw=2.2 if key == "ref" else 1.8)
        ax.set_title(f"{title(res[name]['case'])}, t = {res[name]['t'][-1]:g}")
        ax.set_xlabel("wavenumber k (turns per box)")
        top = fn(snaps[f"{name}/ref"][-1].astype(np.float64))[1:].max()
        ax.set_ylim(top * 1e-8, top * 3)
    np.atleast_1d(axes)[0].set_ylabel("kinetic energy per shell" if which == "E" else "scalar variance per shell")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=7.5)
    save(fig, fname)


def series(res, names, key, ylabel, fname):
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 2.7))
    for ax, name in zip(np.atleast_1d(axes), names):
        r = res[name]
        t = r["t"]
        ax.plot(t, [x[key] for x in r["ref"]], color=INK, lw=2.2, label=STYLE["ref"][0])
        for m in ("base", "hllclm", "constant", "closure"):
            if m in r["methods"]:
                ax.plot(t, [x[key] for x in r["methods"][m]["series"]], color=STYLE[m][1], label=STYLE[m][0])
        ax.set_title(title(r["case"]))
        ax.set_xlabel("time")
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    np.atleast_1d(axes)[0].legend(fontsize=7.5)
    save(fig, fname)


def snapshots(res, snaps, name, fname, when=-1):
    """The passive scalar: reference, coarse, coarse with the closure."""
    keys = [k for k in ("ref", "base", "hllclm", "constant") if f"{name}/{k}" in snaps]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 2.9))
    for ax, key in zip(axes, keys):
        U = snaps[f"{name}/{key}"][when].astype(np.float64)
        ax.imshow(U[5] / U[0], origin="lower", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(STYLE[key][0], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    r = res[name]
    t = r["t"]
    idx = [len(t) // 4, len(t) // 2, len(t) - 1][when]
    fig.suptitle(f"{title(r['case'])}: the passive scalar at t = {t[idx]:g}", fontsize=9.5, y=1.0)
    save(fig, fname)


def bars(res, fname):
    """Errors against the reference, per case: what the closure changes."""
    names = list(res)
    rows = [("E_err", "Kinetic energy spectrum\n(mean |log10 ratio|)"), ("S_err", "Scalar spectrum\n(mean |log10 ratio|)"),
            ("mixedness_err", "Mixedness\n(mean |difference|)"), ("thickness_err", "Mixing-layer thickness\n(mean |difference|)")]
    fig, axes = plt.subplots(1, len(rows), figsize=(3.1 * len(rows), 0.42 * len(names) + 1.3), sharey=True)
    y = np.arange(len(names))
    for ax, (key, label) in zip(axes, rows):
        for i, m in enumerate(("base", "hllclm", "constant", "closure")):
            v = [res[n]["methods"][m]["summary"][key] if m in res[n]["methods"] else np.nan for n in names]
            ax.barh(y + (i - 1.5) * 0.2, v, height=0.18, color=STYLE[m][1], label=STYLE[m][0])
        ax.set_xlabel(label)
        ax.set_yticks(y)
        ax.set_yticklabels([title(res[n]["case"]) + (f" ({res[n]['case']['n']}²)" if res[n]["case"]["n"] != 128 else "") for n in names], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    save(fig, fname)


def shocks_fig():
    f = EVAL / "shocks.npz"
    if not f.exists():
        return
    d = np.load(f)
    r = json.loads((EVAL / "shocks.json").read_text())
    n = d["sod_off"].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    ax = axes[0]
    ax.plot(d["sod_x"], d["sod_exact"], color=INK, lw=1.2, label="Exact")
    ax.plot(d["sod_x"], d["sod_off"][n // 2], color=STYLE["base"][1], lw=3.2, alpha=0.55, label="No closure")
    ax.plot(d["sod_x"], d["sod_constant"][n // 2], color=STYLE["closure"][1], lw=1.2, label="Closure (constants)")
    ax.set_title(f"Sod shock tube: density (largest change {r['sod']['rho_max_change_constant']:.1e})")
    ax.set_xlabel("x")
    ax.legend(fontsize=7.5)
    x = (np.arange(n) + 0.5) / n - 0.5
    for ax, key, name in ((axes[1], "strong_shock", "Strong shock tube"), (axes[2], "sedov", "Sedov blast, along y = 0")):
        ax.plot(x, d[f"{key}_off"][n // 2], color=STYLE["base"][1], lw=3.2, alpha=0.55, label="No closure")
        ax.plot(x, d[f"{key}_constant"][n // 2], color=STYLE["closure"][1], lw=1.2, label="Closure (constants)")
        ax.set_title(f"{name}: density (largest change {r[key]['rho_max_change_constant']:.1e})")
        ax.set_xlabel("x")
    save(fig, "shocks.png")


def long_fig():
    f = EVAL / "long.json"
    if not f.exists():
        return
    r = json.loads(f.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8))
    for name, ls in (("decaying-turbulence", "-"), ("kelvin-helmholtz", "--")):
        for tag, key in (("off", "base"), ("constant", "constant")):
            rows = r[f"{name}/{tag}"]
            t = [x["t"] for x in rows]
            label = f"{'Turbulence' if name.startswith('dec') else 'Shear layer'}, {'closure (constants)' if tag == 'constant' else 'no closure'}"
            axes[0].semilogy(t, [x["eKin"] for x in rows], color=STYLE[key][1], ls=ls, label=label)
            axes[1].plot(t, [x["eTot"] / rows[0]["eTot"] - 1 for x in rows], color=STYLE[key][1], ls=ls)
            axes[2].plot(t, [x["maxMach"] for x in rows], color=STYLE[key][1], ls=ls)
    axes[0].set_title("Kinetic energy")
    axes[1].set_title("Total energy: change from the start")
    axes[2].set_title("Largest Mach number")
    for ax in axes:
        ax.set_xlabel("time")
    axes[0].legend(fontsize=7)
    save(fig, "long_rollout.png")


def main():
    for stem, tag in (("app_test", ""), ("app_res", "_res")):
        if not (EVAL / f"{stem}.json").exists():
            continue
        res, snaps = load(stem)
        names = list(res)
        bars(res, f"errors{tag}.png")
        if tag == "":
            dt = [n for n in names if n.startswith("dt")][:3]
            kh = [n for n in names if n.startswith("kh")][:3]
            spectra(res, snaps, dt, "spectra_turbulence.png")
            spectra(res, snaps, kh, "spectra_shear.png")
            spectra(res, snaps, dt, "scalar_spectra_turbulence.png", "S")
            series(res, kh, "thickness", "mixing-layer thickness", "thickness_shear.png")
            series(res, dt + kh[:1], "mixedness", "mixedness 4<s(1-s)>", "mixedness.png")
            series(res, dt, "kinetic", "kinetic energy", "kinetic_turbulence.png")
            snapshots(res, snaps, dt[0], "snapshot_turbulence.png", 1)
            snapshots(res, snaps, kh[0], "snapshot_shear.png", 1)
        else:
            spectra(res, snaps, names, "spectra_res.png")
    shocks_fig()
    long_fig()


if __name__ == "__main__":
    main()
