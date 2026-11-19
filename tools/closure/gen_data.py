"""Reference data for the closure: high-resolution runs of Shaderous itself, restricted.

Each case is run with the command-line runner (tools/headless) at `factor` times the coarse
resolution, its state written every DT_OUT of simulated time, and each output averaged
conservatively onto the coarse grid (data.restrict). What is kept is the coarse trajectory,
data/<set>/<name>.npz: `t` [T], `U` [T, 6, n, n] (float32) and the case's settings. The raw
slices (a megabyte per variable per output at 512 x 512) are deleted.

    tools/closure/.venv/bin/python tools/closure/gen_data.py [--set train|test|res] [--only NAME]

The sets:
  train  decaying turbulence (ten seeds over Mach 0.2 to 0.5, peak wavenumbers 3 to 6, with and
         without a denser band) and Kelvin-Helmholtz slabs (shear 0.7 to 1.4, density ratio 1
         to 3, 1 to 4 seeded wavelengths), all on 128 x 128 from 512 x 512;
  test   conditions held out of training: new seeds, Mach 0.4 and 0.7, a band three times
         denser, shear 1.2 and 2, density ratio 2.5 and 5;
  res    two of the test cases on 256 x 256 (from 1024 x 1024).
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from data import load_run, restrict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
DT_OUT = 0.02
GAMMA = 1.4


def dt_case(seed, mach, kpeak, ratio, n=128, t_end=2.0):
    return {"name": f"dt_s{seed}_m{mach}_k{kpeak}_r{ratio}_n{n}", "preset": "decaying-turbulence", "n": n, "t_end": t_end,
            "flow": {"seed": seed, "mach": mach, "kpeak": kpeak, "ratio": ratio}}


def kh_case(shear, rho_in, modes, amp, n=128, t_end=3.0):
    return {"name": f"kh_v{shear}_r{rho_in}_m{modes}_a{amp}_n{n}", "preset": "kelvin-helmholtz", "n": n, "t_end": t_end,
            "flow": {"shear": shear, "rhoIn": rho_in, "modes": modes, "amp": amp}}


SETS = {
    "train": [
        dt_case(1, 0.3, 4, 1), dt_case(2, 0.3, 4, 1), dt_case(3, 0.2, 3, 1), dt_case(4, 0.5, 6, 1), dt_case(5, 0.3, 6, 2),
        dt_case(6, 0.5, 3, 2), dt_case(7, 0.2, 4, 2), dt_case(8, 0.3, 3, 1), dt_case(9, 0.5, 4, 1), dt_case(10, 0.2, 6, 2),
        kh_case(1, 2, 2, 0.1), kh_case(0.7, 2, 3, 0.1), kh_case(1.4, 2, 4, 0.05), kh_case(1, 1, 3, 0.1),
        kh_case(1.4, 3, 2, 0.1), kh_case(0.7, 1.5, 4, 0.05), kh_case(1, 3, 1, 0.1), kh_case(1.4, 1, 2, 0.05),
    ],
    "test": [
        dt_case(103, 0.3, 4, 1), dt_case(101, 0.4, 5, 1), dt_case(102, 0.4, 5, 3), dt_case(104, 0.7, 4, 1),
        kh_case(1.2, 2.5, 3, 0.1), kh_case(2, 2, 2, 0.1), kh_case(1, 5, 2, 0.1),
    ],
    # (The simulator's 2D grids start at 128 cells a side, so there is no coarser case.)
    "res": [dt_case(103, 0.3, 4, 1, n=256), kh_case(1.2, 2.5, 3, 0.1, n=256)],
}


def setup_document(case):
    """The preset's document with the case's flow parameters."""
    doc = json.loads((ROOT / "src" / "presets" / f"{case['preset']}.json").read_text())
    for part in doc["parts"]:
        if part["kind"] == "flow":
            part.update(case["flow"])
    doc.pop("id", None)
    return doc


def run_case(case, out_dir, factor=4, dt_out=DT_OUT, keep=False, extra=()):
    """Run a case at `factor` times its coarse resolution and return (t, U) restricted to the
    coarse grid, as float32 arrays."""
    out_dir = Path(out_dir)
    raw = out_dir / f"{case['name']}.raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    doc = raw.with_suffix(".shaderous.json")
    doc.write_text(json.dumps(setup_document(case)))
    n = case["n"] * factor
    cmd = ["node", "tools/headless/cli.mjs", "run", "--setup", str(doc), "--dim", "2", "--n", str(n),
           "--until", str(case["t_end"]), "--every", str(dt_out), "--slice-res", str(n), "--slice-format", "npy",
           "--out", str(raw), "--quiet", *extra]
    for v in ("rho", "vel1", "vel2", "press", "r0"):
        cmd += ["--slice", v]
    subprocess.run(cmd, cwd=ROOT, check=True)
    t, U, man = load_run(raw, GAMMA)
    U = restrict(U, factor)
    if man.get("errors"):
        raise RuntimeError(f"{case['name']}: the page reported {man['errors']}")
    if not keep:
        shutil.rmtree(raw)
        doc.unlink()
    return t.numpy(), U.to(torch.float32).numpy(), man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, choices=list(SETS))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    for name, cases in SETS.items():
        if a.set and a.set != name:
            continue
        for case in cases:
            if a.only and a.only not in case["name"]:
                continue
            out = DATA / name / f"{case['name']}.npz"
            if out.exists():
                continue
            t, U, _ = run_case(case, DATA / name)
            np.savez_compressed(out, t=t, U=U, case=json.dumps(case), gamma=GAMMA)
            print(f"{name}/{case['name']}: {len(t)} outputs to t = {t[-1]:g}", flush=True)


if __name__ == "__main__":
    main()
