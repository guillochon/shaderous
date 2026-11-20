"""Is the mixing layers' thickness a fair target? A well-posed shear layer, as an ensemble.

The Kelvin-Helmholtz slabs of the test set have a sharp edge and no viscosity: no resolution
converges on them, and each is one run of a chaotic flow. Here the slab's edges are a tanh of
width 0.02 and the gas has a viscosity of 2e-4, so that the flow has an answer, and every case
is run for several seeds of its kick. For each seed: references at 512 and at 1024 cells a
side (both averaged onto 128), and coarse runs at 128 without a closure, with the low-Mach
solver and with the closure by constants, all with the same viscosity. What is compared is
the ensemble mean of the thickness and the mixedness over time, against their scatter from
seed to seed.

    tools/closure/.venv/bin/python tools/closure/diagnostic.py run      (about 40 minutes of GPU)
    tools/closure/.venv/bin/python tools/closure/diagnostic.py report

(The dye has no physical diffusion in the simulator, so its mixedness is set by the grid in
the references too; the thickness is not.)
"""
import json
import sys
from pathlib import Path

import numpy as np

import metrics as M
from gen_data import run_case

HERE = Path(__file__).resolve().parent
OUT = HERE / "eval" / "diagnostic"
NU, SMOOTH, DT = 2e-4, 0.02, 0.1
SEEDS = (1, 2, 3, 4, 5, 6)
CASES = {"shear1_ratio2": {"shear": 1.0, "rhoIn": 2.0}, "shear1.4_ratio1": {"shear": 1.4, "rhoIn": 1.0}}
RUNS = {  # name: (cells a side, restriction, extra settings)
    "ref512": (512, 4, []),
    "ref1024": (1024, 8, []),
    "base": (128, 1, []),
    "hllclm": (128, 1, ["--set", "riemann=hllclm"]),
    "constant": (128, 1, ["--set", "closure=constant"]),
}


def one(case, seed, run):
    n, f, extra = RUNS[run]
    out = OUT / f"{case}_s{seed}_{run}.json"
    if out.exists():
        return
    c = {"name": f"diag_{case}_s{seed}_{run}", "preset": "kelvin-helmholtz", "n": n // f, "t_end": 4.0,
         "flow": {**CASES[case], "modes": 4, "amp": 0.05, "smooth": SMOOTH, "seed": seed}}
    t, U, man = run_case(c, OUT / "raw", factor=f, dt_out=DT, extra=["--set", f"nu={NU}", *extra])
    assert U.shape[-1] == 128, U.shape
    rows = []
    for u in U.astype(np.float64):
        E = M.ke_spectrum(u)
        rows.append({**M.mixing(u), "kinetic": M.totals(u)["kinetic"], "E_k8": float(E[8:16].sum()), "E_k16": float(E[16:32].sum()), "E_k32": float(E[32:64].sum())})
    out.write_text(json.dumps({"t": t.tolist(), "rows": rows, "steps": man["outputs"][-1]["steps"]}))
    print(f"{case} seed {seed} {run}: {man['outputs'][-1]['steps']} steps, thickness {rows[-1]['thickness']:.3f}, mixedness {rows[-1]['mixedness']:.3f}", flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        for seed in SEEDS:
            for r in RUNS:
                # The 1024 reference only for the first case: it is there to check the 512 one.
                if r == "ref1024" and case != "shear1_ratio2":
                    continue
                one(case, seed, r)
    print("DONE", flush=True)


def report():
    for case in CASES:
        data = {}
        for r in RUNS:
            runs = [json.loads(f.read_text()) for f in sorted(OUT.glob(f"{case}_s*_{r}.json"))]
            if runs:
                data[r] = runs
        t = np.array(data["ref512"][0]["t"])
        late = t >= 1.0
        print(f"\n## {case}: {len(data['ref512'])} seeds")
        for key in ("thickness", "mixedness", "E_k16"):
            arr = {r: np.array([[row[key] for row in run["rows"]] for run in runs]) for r, runs in data.items()}
            ref = arr["ref512"]
            print(f"\n{key}: mean over seeds at t = 1, 2, 3, 4 (scatter between seeds), and the error of the mean over t >= 1")
            for r, a in arr.items():
                m, sd = a.mean(0), a.std(0, ddof=1)
                at = " ".join(f"{m[i]:.3f} ({sd[i]:.3f})" for i in (10, 20, 30, 40))
                err = abs(m - ref.mean(0))[late].mean()
                # Standard error of the difference of two ensemble means.
                se = np.sqrt(sd**2 / len(a) + ref.std(0, ddof=1) ** 2 / len(ref))[late].mean()
                each = np.mean([abs(x - ref.mean(0))[late].mean() for x in a])
                print(f"  {r:9s} {at}   |mean - ref512 mean| {err:.4f} (standard error {se:.4f}); a single run's error {each:.4f}")
        slope = lambda a: np.polyfit(t[(t >= 0.5) & (t <= 3.0)], a.mean(0)[(t >= 0.5) & (t <= 3.0)], 1)[0]
        th = {r: np.array([[row["thickness"] for row in run["rows"]] for run in runs]) for r, runs in data.items()}
        print("\ngrowth rate of the mean thickness, t = 0.5 to 3: " + ", ".join(f"{r} {slope(a):.3f}" for r, a in th.items()))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
