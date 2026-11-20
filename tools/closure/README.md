# Turbulence closure: training tools

Offline Python tools for the trained closure of `src/closure/` (docs/numerics.md section 26;
the results are in docs/closure.md). Nothing here runs in the simulator: what reaches it is
`src/closure/closure_net.ts` (weights and constants) and `src/closure/closure_cases.json`
(cells the TypeScript mirror is tested against), both written by `export.py`.

```bash
cd tools/closure
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy matplotlib
```

| File | What it does |
|---|---|
| `solver.py` | A differentiable mirror of the uniform-grid hydro step (2D, periodic, ideal gas), in PyTorch. Agrees with the simulator to rounding (1e-6 over 300 steps), with the closure in it to 1e-4 |
| `closure.py` | The closure: features, gates, network, and its limits |
| `gen_data.py` | Reference data: runs of the simulator at 4 times the resolution (tools/headless), averaged onto the coarse grid. Sets `train`, `test` (held-out conditions) and `res` (other resolutions) |
| `train.py` | Unrolled training through `solver.py`, with the model's own drift in the starts |
| `evaluate.py` | Whole coarse runs against the reference: spectra, mixing, conservation. `--mode app` runs the simulator (no closure, the low-Mach solver, `closure=constant`, `closure=network`), `--mode mirror` the PyTorch solver (any variant) |
| `appchecks.py` | In the simulator: shock tubes and blasts with the closure on (constants, network) and off, runs to t = 30, cost per step |
| `diagnostic.py` | The well-posed shear layer as an ensemble: smooth edges, viscosity, six seeds, references at 512² and 1024² |
| `figures.py` | The figures of docs/closure.md |
| `export.py` | A trained model into `src/closure/` |
| `metrics.py`, `data.py` | Statistics of a state; reading the runner's outputs |

The whole pipeline, from nothing (about 25 minutes of GPU for the data, an hour and a half of
CPU for the training):

```bash
.venv/bin/python gen_data.py
.venv/bin/python train.py --name final --heads ab --iters 400 --batch 3 --threads 2
.venv/bin/python export.py final
.venv/bin/python evaluate.py --mode app --set test
.venv/bin/python evaluate.py --mode app --set res
.venv/bin/python appchecks.py shocks && .venv/bin/python appchecks.py oblique
.venv/bin/python appchecks.py long && .venv/bin/python appchecks.py cost
.venv/bin/python figures.py
```

After changing anything in `closure.py` (a feature, a gate, a limit), change its twins,
`src/hydro/closure_net.wgsl` and `closureCell()` in `src/closure/closure.ts`, train again and
export: `tests/unit/closure.test.ts` and the `closure` test of the validation suite fail until
the three agree.

The shipped weights are `models/final.pt` (two hidden layers of 24). They came from two runs:
220 iterations with the scalar's jump scaled without its bound, then 240 more from that
checkpoint with the bound in place (`--init`, `--phases 0.25:1:0,0.35:2:2,0.4:3:3`,
`--lr 2e-3`). `--heads nk` trains the eddy viscosity and diffusivity alone, and `--heads abnk`
all four outputs: the two variants docs/closure.md reports as no better than no closure.
`appchecks.py oblique final` is the oblique shock tubes.

`data/`, `eval/`, the other models and the logs are not in the repository.
