"""Judge a closure a posteriori: whole coarse runs against the restricted reference.

Every case of a held-out set (gen_data.py: `test`, and `res` for other resolutions) is run
coarse from its own start to its end, thousands of steps, far beyond the training window, and
its statistics are compared with the reference's at every tenth of a time unit:

  the kinetic energy spectrum and the scalar's (mean |log10| of the ratio per shell, and the
  energy in the bands k = 8-16, 16-32 and 32-64 over the reference's);
  the mixing layers' thickness 4 <s_bar (1 - s_bar)> and its growth rate, the mixedness
  4 <s (1 - s)> and the scalar's variance;
  the totals (mass, momentum, energy: conserved to rounding or not) and the scalar's range.

Two ways of running the coarse case:

  --mode app     in the simulator itself (tools/headless), as a user would: closure off
                 (HLLC, and HLLC with the low-Mach fix) and on, with constants and with
                 the trained network. This is the result that counts.
  --mode mirror  in the PyTorch mirror (solver.py) from the restricted reference's first
                 state: any trained variant, also those the simulator does not have (the eddy
                 viscosity heads).

Results go to eval/<mode>_<set>.json and the states at a few times to eval/<mode>_<set>.npz.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import metrics as M
from closure import Constant
from export import load_model
from gen_data import SETS, run_case
from solver import Config, advance

HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval"
EVERY = 5  # outputs of the reference (0.02 apart) between comparisons
BANDS = {"k8": (8, 16), "k16": (16, 32), "k32": (32, 64)}


def reference(set_name, case):
    d = np.load(HERE / "data" / set_name / f"{case['name']}.npz")
    return d["t"][::EVERY], d["U"][::EVERY].astype(np.float64)


def measure(U, Uref):
    """One state against the reference's at the same time."""
    n = U.shape[-1]
    E, Er = M.ke_spectrum(U), M.ke_spectrum(Uref)
    S, Sr = M.scalar_spectrum(U), M.scalar_spectrum(Uref)
    s = U[5] / np.maximum(U[0], 1e-30)
    out = {"E_err": M.spectrum_error(E, Er), "S_err": M.spectrum_error(S, Sr), **M.mixing(U), **M.totals(U),
           "enstrophy": M.enstrophy(U), "s_min": float(s.min()), "s_max": float(s.max())}
    for k, (a, b) in BANDS.items():
        b = min(b, n // 2 + 1)
        if a < b:
            out[f"E_{k}"] = M.band_energy(E, a, b) / max(M.band_energy(Er, a, b), 1e-300)
            out[f"S_{k}"] = M.band_energy(S, a, b) / max(M.band_energy(Sr, a, b), 1e-300)
    return out


def summarize(series, ref_series, t):
    """Numbers for a whole run: time averages of the errors against the reference."""
    def col(rows, k):
        return np.array([r[k] for r in rows])

    out = {"E_err": float(col(series, "E_err")[1:].mean()), "S_err": float(col(series, "S_err")[1:].mean())}
    for k in ("thickness", "mixedness", "variance", "kinetic", "enstrophy"):
        out[f"{k}_err"] = float(abs(col(series, k) - col(ref_series, k))[1:].mean())
    for k in BANDS:
        if f"E_{k}" in series[0]:
            out[f"E_{k}"] = float(np.exp(np.log(col(series, f"E_{k}")[1:]).mean()))
            out[f"S_{k}"] = float(np.exp(np.log(col(series, f"S_{k}")[1:]).mean()))
    # Growth rate of the mixing layers: the slope of the thickness over the middle half of the run.
    a, b = len(t) // 4, 3 * len(t) // 4
    slope = lambda rows: float(np.polyfit(t[a:b], col(rows, "thickness")[a:b], 1)[0])
    out["growth"], out["growth_ref"] = slope(series), slope(ref_series)
    out["s_min"], out["s_max"] = float(col(series, "s_min").min()), float(col(series, "s_max").max())
    out["s_min_ref"], out["s_max_ref"] = float(col(ref_series, "s_min").min()), float(col(ref_series, "s_max").max())
    for k in ("mass", "energy", "px", "scalar"):
        v = col(series, k)
        out[f"{k}_drift"] = float(abs(v - v[0]).max() / max(abs(v[0]), 1e-30) if k != "px" else abs(v - v[0]).max())
    out["finite"] = bool(np.isfinite(col(series, "energy")).all())
    return out


APP_METHODS = {
    "base": [],
    "hllclm": ["--set", "riemann=hllclm"],
    "constant": ["--set", "closure=constant"],
    "closure": ["--set", "closure=network"],
}


def run_app(case, method, dt_out):
    t, U, man = run_case(case, EVAL / "raw", factor=1, dt_out=dt_out, extra=APP_METHODS[method])
    return t, U.astype(np.float64), man["outputs"][-1]["steps"]


def run_mirror(case, U0, t, closure):
    cfg = Config()
    U = torch.from_numpy(U0).unsqueeze(0).float()
    out = [U0]
    steps = 0
    with torch.no_grad():
        for k in range(1, len(t)):
            U, s = advance(U, cfg, float(t[k] - t[k - 1]), closure)
            steps += s
            out.append(U[0].double().numpy())
    return t, np.stack(out), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="app", choices=["app", "mirror"])
    ap.add_argument("--set", default="test", choices=list(SETS))
    ap.add_argument("--models", default="final", help="mirror mode: trained models to run, comma separated")
    ap.add_argument("--only", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    EVAL.mkdir(exist_ok=True)
    results, snaps = {}, {}
    if a.mode == "mirror":
        methods = {"base": None, "const": Constant(0.8, 0.9)}
        for name in a.models.split(","):
            methods[name] = load_model(name)[0]
    else:
        methods = dict.fromkeys(APP_METHODS)
    for case in SETS[a.set]:
        if a.only and a.only not in case["name"]:
            continue
        t, Uref = reference(a.set, case)
        ref_series = [measure(u, u) for u in Uref]
        results[case["name"]] = {"t": t.tolist(), "case": case, "ref": ref_series, "methods": {}}
        keep = [len(t) // 4, len(t) // 2, len(t) - 1]
        snaps[f"{case['name']}/ref"] = Uref[keep].astype(np.float32)
        for name, cl in methods.items():
            if a.mode == "app":
                tt, U, steps = run_app(case, name, float(t[1] - t[0]))
            else:
                tt, U, steps = run_mirror(case, Uref[0], t, cl)
            assert len(tt) == len(t), (len(tt), len(t))
            series = [measure(u, r) for u, r in zip(U, Uref)]
            summary = summarize(series, ref_series, t)
            summary["steps"] = steps
            results[case["name"]]["methods"][name] = {"series": series, "summary": summary}
            snaps[f"{case['name']}/{name}"] = U[keep].astype(np.float32)
            print(f"{case['name']:32s} {name:8s} steps {steps:6d} E_err {summary['E_err']:.3f} S_err {summary['S_err']:.3f} "
                  f"thick {summary['thickness_err']:.4f} mixed {summary['mixedness_err']:.4f} E16 {summary.get('E_k16', float('nan')):.2f} "
                  f"s [{summary['s_min']:.3f}, {summary['s_max']:.3f}] finite {summary['finite']}", flush=True)
    stem = f"{a.mode}_{a.set}{a.tag}"
    (EVAL / f"{stem}.json").write_text(json.dumps(results))
    np.savez_compressed(EVAL / f"{stem}.npz", **snaps)


if __name__ == "__main__":
    main()
