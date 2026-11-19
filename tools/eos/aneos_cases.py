"""Reference states for tests/unit/aneos.test.ts: ANEOS itself (pyaneos, with NOTENSION, as
tools/eos/aneos_table.py runs it) at random places off the tables' nodes, per material:
density (g/cm^3), temperature (K), pressure (dyn/cm^2), energy (erg/g, without the table's
offset) and ANEOS's phase flag.

    python tools/eos/aneos_cases.py <pyaneos checkout> <name>=<dir with ANEOS.INPUT> ...

Writes tests/unit/aneos_cases.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

import aneos_table as at

rng = np.random.default_rng(7)
out = {}
for spec in sys.argv[2:]:
    name, deck = spec.split("=", 1)
    m = at.load_material(Path(sys.argv[1]), deck)
    n = 240
    # Where bodies and their debris are: 1e-5 g/cm^3 to 30, 150 K to 1e5 K.
    lr = np.sort(rng.uniform(-5, 1.45, n))
    lt = rng.uniform(2.2, 5.0, n)
    rows = []
    for k in range(0, n, 40):
        s = slice(k, k + 40)
        f = at.evaluate(m, 10.0 ** lr[s], 10.0 ** lt[s])
        for d in range(len(lr[s])):
            rows.append([float(f"{10.0 ** lr[s][d]:.9g}"), float(f"{10.0 ** lt[s][d]:.9g}"), float(f"{f['p'][d, d]:.9g}"), float(f"{f['e'][d, d]:.9g}"), int(f["kpa"][d, d])])
    out[name] = rows
Path(__file__).parents[2].joinpath("tests/unit/aneos_cases.json").write_text(json.dumps(out) + "\n")
print({k: len(v) for k, v in out.items()})
