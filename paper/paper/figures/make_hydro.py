"""Accuracy of the hydrodynamics: the Sod tube, the Sedov blast on the adaptive mesh, convergence.

    uv run python paper/figures/make_hydro.py          # plot from data/hydro/
    uv run python paper/figures/make_hydro.py --run    # run the simulations again (minutes on a GPU)

The runs are made by collect.mjs, which loads src/validation/figure_data.ts into the code's page in
headless Chrome and runs the validation suite's setups as its tests run them:
  sod                   the Sod tube along x on 256^2 at t = 0.2, the middle row, with the exact
                        solution from the code's exact Riemann solver (src/reference/riemann_exact.ts)
  sodConvergence        its mean density error on 32^2 to 1024^2
  advectionConvergence  smooth advection of a Gaussian once around the periodic box diagonally,
                        mean density error on 32^2 to 512^2, default limiter (Koren) and MC
  sedovCells            a 3D Sedov blast on the adaptive mesh (256 cells at the finest level, three
                        levels) at t = 0.06: distance and density of the leaf cells (every k-th,
                        at most 200,000)
--run needs Node, npm install in the code's repository, Chrome and a GPU.

The Sedov similarity solution is computed here: the self-similar Euler equations integrated
inward from the strong-shock jump at the shock, normalized by the blast's energy (E = 1, rho = 1).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "hydro"
CODE = HERE.parents[2]

RUNS = [
    ("sod", [256], "sod.json"),
    ("sodConvergence", [[32, 64, 128, 256, 512, 1024]], "sod_convergence.json"),
    ("advectionConvergence", [[32, 64, 128, 256, 512]], "advection_koren.json"),
    ("advectionConvergence", [[32, 64, 128, 256, 512], "mc"], "advection_mc.json"),
    ("sedovCells", [256, 3], "sedov.json"),
]


def run() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, args, out in RUNS:
        subprocess.run(["node", str(HERE / "collect.mjs"), name, str(DATA / out), json.dumps(args)], check=True, cwd=CODE)


def load(name: str):
    return json.loads((DATA / name).read_text())


def sedov_solution(gamma: float, n: int = 4000):
    """The spherical Sedov-Taylor blast in a uniform medium: (xi, rho / rho_0) behind the shock,
    with xi = r / R(t), and the constant xi_0 in R = xi_0 (E t^2 / rho_0)^(1/5).

    With u = Rdot U(xi), rho = rho_0 Omega(xi), p = rho_0 Rdot^2 P(xi) and Rdot = 2R / (5t), the
    Euler equations become
        (U - xi) Omega' + Omega (U' + 2U / xi) = 0
        (U - xi) U' + P' / Omega = 3U / 2
        (U - xi) (P' / P - gamma Omega' / Omega) = 3,
    integrated inward from the strong-shock values at xi = 1. The energy integral then fixes xi_0.
    """
    g = gamma

    def deriv(xi, y):
        om, u, p = y
        a = u - xi
        m = np.array([[a, om, 0.0], [0.0, a, 1.0 / om], [-g * a / om, 0.0, a / p]])
        b = np.array([-2.0 * om * u / xi, 1.5 * u, 3.0])
        return np.linalg.solve(m, b)

    y = np.array([(g + 1) / (g - 1), 2 / (g + 1), 2 / (g + 1)])
    xs = np.linspace(1.0, 1e-3, n)
    out = [y.copy()]
    for x0, x1 in zip(xs[:-1], xs[1:], strict=True):
        h = x1 - x0
        k1 = deriv(x0, y)
        k2 = deriv(x0 + h / 2, y + h / 2 * k1)
        k3 = deriv(x0 + h / 2, y + h / 2 * k2)
        k4 = deriv(x1, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y.copy())
    om, u, p = np.array(out).T
    integrand = (0.5 * om * u**2 + p / (g - 1)) * xs**2
    energy = -np.trapezoid(integrand, xs)  # xs decreases
    xi0 = (25.0 / (16.0 * np.pi * energy)) ** 0.2
    return xs, om, xi0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run the simulations again first")
    if ap.parse_args().run:
        run()

    style.apply()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(style.PAGE, 0.31 * style.PAGE), layout="constrained")

    # (a) Sod tube.
    s = load("sod.json")
    x = np.array(s["x"])
    for key, label, color in (("rho", r"$\rho$", style.PALETTE[0]), ("v", r"$v_x$", style.PALETTE[1]), ("p", r"$p$", style.PALETTE[2])):
        a.plot(x, s["exact"][key], color=style.INK, lw=0.7)
        a.plot(x, s[key], ".", color=color, ms=2.2, label=label)
    a.set_xlabel(r"$x$")
    a.set_ylabel("Value")
    a.set_xlim(0, 1)
    a.legend(loc="center left", **style.LEGEND)
    a.text(0.03, 0.97, f"(a) Sod, {s['n']} cells, $t = {s['t']:g}$", transform=a.transAxes, va="top", bbox=style.LABEL_BOX)

    # (b) Sedov blast: every (sampled) leaf cell, and the similarity solution.
    d = load("sedov.json")
    xi, om, xi0 = sedov_solution(d["gamma"])
    radius = xi0 * d["t"] ** 0.4  # E = 1, rho = 1
    print(f"Sedov: xi0 = {xi0:.4f} for gamma = {d['gamma']}, shock at r = {radius:.4f}; {d['cells']} leaf cells, {len(d['r'])} plotted")
    b.plot(d["r"], d["rho"], ".", color=style.PALETTE[0], ms=0.8, alpha=0.4, rasterized=True, label="Leaf cells")
    r = np.concatenate([xi * radius, [radius, radius, 0.5]])
    rho = np.concatenate([om, [om[0], 1.0, 1.0]])
    order = np.argsort(r, kind="stable")
    b.plot(r[order], rho[order], color=style.INK, lw=0.8, label="Similarity solution")
    b.set_xlim(0, 0.5)
    b.set_xlabel(r"Distance from the center $r$")
    b.set_ylabel(r"Density $\rho$")
    b.legend(loc="center left", **style.LEGEND)
    b.text(0.03, 0.97, f"(b) Sedov, $t = {d['t']:g}$", transform=b.transAxes, va="top", bbox=style.LABEL_BOX)

    # (c) Convergence.
    series = [
        ("advection_koren.json", "Advection, Koren", style.PALETTE[0], "o"),
        ("advection_mc.json", "Advection, MC", style.PALETTE[1], "s"),
        ("sod_convergence.json", "Sod", style.PALETTE[2], "^"),
    ]
    for f, label, color, marker in series:
        e = load(f)
        n = np.array([q["n"] for q in e])
        l1 = np.array([q["l1"] for q in e])
        c.loglog(n, l1, marker=marker, color=color, label=label, ms=3)
        order = np.log(l1[-2] / l1[-1]) / np.log(n[-1] / n[-2])
        print(f"{label}: L1 {l1[-1]:.3g} at {n[-1]}, order between the last two {order:.2f}")
    nn = np.array([32, 1024])
    for k, ls in ((1, ":"), (2, "--"), (3, "-.")):
        c.loglog(nn, 0.3 * (nn / 32.0) ** -k, color=style.MUTED, lw=0.6, ls=ls)
        c.text(nn[-1] * 1.05, 0.3 * (nn[-1] / 32.0) ** -k, str(k), color=style.MUTED, fontsize=style.LABEL_SIZE - 1, va="center")
    c.set_xlabel("Cells per side")
    c.set_ylabel("Mean density error")
    c.set_xlim(24, 1500)
    c.set_ylim(1e-7, 1)
    c.legend(loc="lower left", **style.LEGEND)
    c.text(0.03, 0.97, "(c)", transform=c.transAxes, va="top", bbox=style.LABEL_BOX)

    style.save(fig, "hydro")


if __name__ == "__main__":
    main()
