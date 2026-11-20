"""The tables of docs/closure.md, as Markdown, from eval/*.json (evaluate.py, appchecks.py).

    tools/closure/.venv/bin/python tools/closure/summary.py
"""
import json
from pathlib import Path

import numpy as np

from figures import title

EVAL = Path(__file__).resolve().parent / "eval"
METHODS = (("base", "none"), ("hllclm", "low-Mach solver"), ("constant", "closure, constants"), ("closure", "closure, network"))


def table(stem):
    res = json.loads((EVAL / f"{stem}.json").read_text())
    cols = [("E_err", "KE spectrum", 2), ("E_k16", "KE, k 16-32", 2), ("S_err", "Scalar spectrum", 2), ("mixedness_err", "Mixedness", 3), ("thickness_err", "Thickness", 3)]
    print("| Case | Run | " + " | ".join(c[1] for c in cols) + " | Growth rate (ref.) | Scalar range |")
    print("|---|---|" + "---|" * (len(cols) + 2))
    wins = {c[0]: 0 for c in cols}
    for name, r in res.items():
        label = title(r["case"]) + (f", {r['case']['n']}²" if r["case"]["n"] != 128 else "")
        for m, mname in METHODS:
            if m not in r["methods"]:
                continue
            s = r["methods"][m]["summary"]
            cells = [f"{s[k]:.{d}f}" if k in s else "" for k, _, d in cols]
            print(f"| {label if m == 'base' else ''} | {mname} | " + " | ".join(cells) + f" | {s['growth']:.3f} ({s['growth_ref']:.3f}) | {s['s_min']:.3f} to {s['s_max']:.3f} |")
        b, c = r["methods"]["base"]["summary"], r["methods"]["constant"]["summary"]
        for k, _, _ in cols:
            if k in b:
                better = abs(np.log(c[k])) < abs(np.log(b[k])) if k == "E_k16" else c[k] < b[k]
                wins[k] += bool(better)
    n = len(res)
    print(f"\nCloser to the reference with the closure (constants) than without, of {n} cases: " + ", ".join(f"{c[1]} {wins[c[0]]}" for c in cols))
    for k, lab, _ in cols:
        if k == "E_k16":
            continue
        b = np.array([r["methods"]["base"]["summary"][k] for r in res.values()])
        h = np.array([r["methods"]["hllclm"]["summary"][k] for r in res.values()])
        c = np.array([r["methods"]["constant"]["summary"][k] for r in res.values()])
        w = np.array([r["methods"]["closure"]["summary"][k] for r in res.values()])
        print(f"  {lab}: mean error {b.mean():.3f} without, {h.mean():.3f} low-Mach, {c.mean():.3f} constants (ratio {c.mean() / b.mean():.2f}; per case {np.round(c / b, 2).tolist()}), {w.mean():.3f} network (ratio {w.mean() / b.mean():.2f})")
    g = [(abs(r["methods"][m]["summary"]["growth"] - r["methods"][m]["summary"]["growth_ref"])) for r in res.values() for m in ("base",)]
    gc = [(abs(r["methods"]["constant"]["summary"]["growth"] - r["methods"]["constant"]["summary"]["growth_ref"])) for r in res.values()]
    print(f"  Growth rate: mean |error| {np.mean(g):.3f} without, {np.mean(gc):.3f} with; per case without {np.round(g, 3).tolist()} with {np.round(gc, 3).tolist()}")
    drift = {k: max(r["methods"]["constant"]["summary"][f"{k}_drift"] for r in res.values()) for k in ("mass", "energy", "px", "scalar")}
    print("  Largest drifts with the closure:", {k: f"{v:.1e}" for k, v in drift.items()}, " steps:", [r["methods"]["constant"]["summary"]["steps"] for r in res.values()])


for stem in ("app_test", "app_res"):
    if (EVAL / f"{stem}.json").exists():
        print(f"\n## {stem}\n")
        table(stem)
for f in ("shocks", "oblique", "cost"):
    if (EVAL / f"{f}.json").exists():
        print(f"\n## {f}\n" + (EVAL / f"{f}.json").read_text())
