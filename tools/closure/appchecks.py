"""What the closure must not do, measured in the simulator itself (tools/headless):

  shocks   shock tubes and blasts with the closure on and off: a Sod tube, the strong shock
           tube, a Sedov blast (all 2D, 256 x 256). The largest and the mean change of the
           density, and the Sod tube's error against the exact solution either way.
  oblique  Sod tubes at five angles to the grid, in the solver's mirror (see oblique()).
  long     decaying turbulence and a Kelvin-Helmholtz slab on 128 x 128 run to t = 30, some
           47,000 and 23,000 steps (the training window is at most 150): mass, total energy,
           kinetic energy, the largest Mach number and the scalar's range every half time unit.
  cost     wall time per step with the closure on and off, on 512, 1024 and 2048 cells a side.

    tools/closure/.venv/bin/python tools/closure/appchecks.py shocks|oblique|long|cost

Results go to eval/shocks.json (and .npz, the profiles for the figures), eval/long.json and
eval/cost.json.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL = HERE / "eval"
RAW = EVAL / "raw"
GAMMA = 1.4


def preset(name):
    return json.loads((ROOT / "src" / "presets" / f"{name}.json").read_text())


def run(doc, n, args, tag):
    """Run a setup document; returns (manifest, lines of diag.jsonl, the run's folder)."""
    out = RAW / tag
    if out.exists():
        shutil.rmtree(out)
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"{tag}.shaderous.json"
    doc = dict(doc)
    doc.pop("id", None)
    f.write_text(json.dumps(doc))
    cmd = ["node", "tools/headless/cli.mjs", "run", "--setup", str(f), "--dim", "2", "--n", str(n), "--out", str(out), "--quiet", *args]
    subprocess.run(cmd, cwd=ROOT, check=True)
    man = json.loads((out / "manifest.json").read_text())
    if man.get("errors"):
        raise RuntimeError(f"{tag}: the page reported {man['errors']}")
    diag = [json.loads(line) for line in (out / "diag.jsonl").read_text().splitlines()]
    return man, diag, out


def final_slices(man, out):
    return {s["variable"]: np.load(out / s["file"]).astype(np.float64) for s in man["outputs"][-1]["slices"]}


SLICES = ["--slice-format", "npy", "--slice", "rho", "--slice", "press", "--slice", "vel1", "--slice", "vel2"]


def exact_riemann(left, right, xi, g=GAMMA):
    """The exact solution of a shock tube (Toro 2009, section 4): (rho, u, p) at xi = x / t."""
    (rl, ul, pl), (rr, ur, pr) = left, right
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def f(p, r, pk, c):
        if p > pk:
            a, b = 2 / ((g + 1) * r), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(a / (p + b)), np.sqrt(a / (p + b)) * (1 - 0.5 * (p - pk) / (p + b))
        return 2 * c / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1), (p / pk) ** (-(g + 1) / (2 * g)) / (r * c)

    p = 0.5 * (pl + pr)
    for _ in range(100):
        fl, dl = f(p, rl, pl, cl)
        fr, dr = f(p, rr, pr, cr)
        step = (fl + fr + ur - ul) / (dl + dr)
        p = max(p - step, 1e-12)
        if abs(step) < 1e-14 * p:
            break
    fl, _ = f(p, rl, pl, cl)
    fr, _ = f(p, rr, pr, cr)
    u = 0.5 * (ul + ur) + 0.5 * (fr - fl)
    out = np.zeros((3, len(xi)))
    for i, s in enumerate(xi):
        if s <= u:  # left of the contact
            if p > pl:
                sl = ul - cl * np.sqrt((g + 1) / (2 * g) * p / pl + (g - 1) / (2 * g))
                out[:, i] = (rl, ul, pl) if s < sl else (rl * (p / pl + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / pl + 1), u, p)
            else:
                cs = cl * (p / pl) ** ((g - 1) / (2 * g))
                if s < ul - cl:
                    out[:, i] = (rl, ul, pl)
                elif s > u - cs:
                    out[:, i] = (rl * (p / pl) ** (1 / g), u, p)
                else:
                    c = 2 / (g + 1) * (cl + (g - 1) / 2 * (ul - s))
                    out[:, i] = (rl * (c / cl) ** (2 / (g - 1)), 2 / (g + 1) * (cl + (g - 1) / 2 * ul + s), pl * (c / cl) ** (2 * g / (g - 1)))
        else:
            if p > pr:
                sr = ur + cr * np.sqrt((g + 1) / (2 * g) * p / pr + (g - 1) / (2 * g))
                out[:, i] = (rr, ur, pr) if s > sr else (rr * (p / pr + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / pr + 1), u, p)
            else:
                cs = cr * (p / pr) ** ((g - 1) / (2 * g))
                if s > ur + cr:
                    out[:, i] = (rr, ur, pr)
                elif s < u + cs:
                    out[:, i] = (rr * (p / pr) ** (1 / g), u, p)
                else:
                    c = 2 / (g + 1) * (cr - (g - 1) / 2 * (ur - s))
                    out[:, i] = (rr * (c / cr) ** (2 / (g - 1)), 2 / (g + 1) * (-cr + (g - 1) / 2 * ur + s), pr * (c / cr) ** (2 * g / (g - 1)))
    return out


def shocks():
    n = 256
    cases = {"sod": preset("sod-x"), "strong_shock": preset("strong-shock"), "sedov": preset("sedov")}
    results, keep = {}, {}
    for name, doc in cases.items():
        fields = {}
        for tag, extra in (("off", []), ("on", ["--set", "closure=network"]), ("constant", ["--set", "closure=constant"])):
            man, diag, out = run(doc, n, ["--slice-res", str(n), *SLICES, *extra], f"{name}_{tag}")
            fields[tag] = final_slices(man, out)
            fields[tag]["steps"] = diag[-1]["steps"]
            fields[tag]["t"] = diag[-1]["t"]
        a, b = fields["off"]["rho"], fields["on"]["rho"]
        r = {"t": fields["off"]["t"], "steps_off": fields["off"]["steps"], "steps_on": fields["on"]["steps"],
             "rho_max_change": float(abs(a - b).max() / a.max()), "rho_mean_change": float(abs(a - b).mean() / a.mean()),
             "p_max_change": float(abs(fields["off"]["press"] - fields["on"]["press"]).max() / fields["off"]["press"].max()),
             "rho_max_change_constant": float(abs(a - fields["constant"]["rho"]).max() / a.max())}
        if name == "sod":
            x = (np.arange(n) + 0.5) / n - 0.5
            ex = exact_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), x / r["t"])
            r["l1_off"] = float(abs(a[n // 2] - ex[0]).mean())
            r["l1_on"] = float(abs(b[n // 2] - ex[0]).mean())
            r["l1_constant"] = float(abs(fields["constant"]["rho"][n // 2] - ex[0]).mean())
            keep["sod_x"], keep["sod_exact"] = x, ex[0]
        results[name] = r
        keep[f"{name}_off"], keep[f"{name}_on"] = a.astype(np.float32), b.astype(np.float32)
        keep[f"{name}_constant"] = fields["constant"]["rho"].astype(np.float32)
        print(name, json.dumps(r), flush=True)
    (EVAL / "shocks.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(EVAL / "shocks.npz", **keep)


def oblique(model="final"):
    """Sod tubes oblique to the grid, in the mirror of the solver (solver.py, which agrees with
    the simulator to rounding): a periodic box whose left state fills the bands where
    frac(a x + b y) < 1/2, so that the tube's waves are planar along (a, b) and nothing but the
    grid makes them otherwise. (In the simulator an oblique tube meets the box's open faces,
    which bend its waves: that flow is sheared in fact, and is no test of the gates.) With the
    trained closure, and with the largest coefficients it could ever give wherever its gates
    are open."""
    import torch

    from closure import Constant
    from export import load_model
    from solver import Config, advance, from_prim

    cfg, n, T = Config(), 256, 0.05
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    X, Y = x.view(1, n).expand(n, n), x.view(n, 1).expand(n, n)
    methods = {"trained": load_model(model)[0].double(), "largest": Constant(0.8, 0.6)}
    results = {}
    for a, b in ((1, 0), (1, 1), (2, 1), (3, 1), (3, 2)):
        left = torch.remainder(a * X + b * Y, 1.0) < 0.5
        zero = torch.zeros_like(X)
        U0 = from_prim(torch.where(left, 1.0, 0.125), zero, zero, torch.where(left, 1.0, 0.1), left.double()).unsqueeze(0)
        with torch.no_grad():
            off, _ = advance(U0, cfg, T)
            r = {}
            for name, cl in methods.items():
                on, _ = advance(U0, cfg, T, cl)
                r[name] = float((on[0, 0] - off[0, 0]).abs().max() / off[0, 0].max())
        results[f"{a},{b}"] = r
        print(f"normal ({a}, {b}):", json.dumps(r), flush=True)
    (EVAL / "oblique.json").write_text(json.dumps(results, indent=1))


def long():
    results = {}
    for name in ("decaying-turbulence", "kelvin-helmholtz"):
        for tag, extra in (("off", []), ("on", ["--set", "closure=network"]), ("constant", ["--set", "closure=constant"])):
            man, diag, _ = run(preset(name), 128, ["--until", "30", "--every", "0.5", *extra], f"long_{name}_{tag}")
            rows = [{"t": d["t"], "steps": d["steps"], "mass": d["mass"], "eTot": d["eTot"], "eKin": d["eKin"], "maxMach": d["maxMach"],
                     "floored": d["floored"], "s": d["ranges"]["Passive scalar"], "rho": d["ranges"]["Density"]} for d in diag]
            results[f"{name}/{tag}"] = rows
            last = rows[-1]
            print(f"{name} {tag}: {last['steps']} steps to t = {last['t']}; mass drift {abs(last['mass'] / rows[0]['mass'] - 1):.1e}, energy drift {abs(last['eTot'] / rows[0]['eTot'] - 1):.1e}, "
                  f"eKin {rows[0]['eKin']:.4f} -> {last['eKin']:.4f}, max Mach ever {max(r['maxMach'] for r in rows):.3f}, scalar range [{min(r['s'][0] for r in rows):.4f}, {max(r['s'][1] for r in rows):.4f}], floored {max(r['floored'] for r in rows)}", flush=True)
    (EVAL / "long.json").write_text(json.dumps(results))


def cost():
    results = {}
    doc = preset("decaying-turbulence")
    for n, steps in ((512, 6000), (1024, 2000), (2048, 600)):
        row = {}
        for tag, extra in (("off", []), ("on", ["--set", "closure=network"]), ("constant", ["--set", "closure=constant"]), ("hllclm", ["--set", "riemann=hllclm"]), ("viscous", ["--set", "nu=1e-6"])):
            best = None
            for rep in range(3):
                # Outputs at the start (pipelines compiled) and the end only: the wall time between
                # is the steps'.
                man, diag, _ = run(doc, n, ["--steps", str(steps), *extra], f"cost_{n}_{tag}")
                wall = diag[-1]["wall"] - diag[0]["wall"]
                ms = 1e3 * wall / (diag[-1]["steps"] - diag[0]["steps"])
                best = ms if best is None else min(best, ms)
            row[tag] = best
        row["overhead"] = row["on"] / row["off"] - 1
        row["overhead_constant"] = row["constant"] / row["off"] - 1
        results[str(n)] = row
        print(n, json.dumps(row), flush=True)
    (EVAL / "cost.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    EVAL.mkdir(exist_ok=True)
    {"shocks": shocks, "oblique": oblique, "long": long, "cost": cost}[sys.argv[1]](*sys.argv[2:])
