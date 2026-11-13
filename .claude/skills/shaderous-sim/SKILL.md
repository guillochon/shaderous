---
name: shaderous-sim
description: Run, inspect and measure Shaderous simulations without a person at the page — through the shaderous MCP tools, the command-line runner (single runs, parameter sweeps, convergence studies, figures, .athdf outputs), or window.shaderous in a browser. Use whenever a task needs a simulation actually run or looked at - checking a physics change, reproducing a bug, making a figure or movie frames, a sweep, or answering "what happens if" - and when verifying a change to the solver or renderer in the real app.
---

# Driving Shaderous headlessly

Shaderous runs entirely in the browser on WebGPU. There are three ways to drive it, all built on
one page API, `window.shaderous`:

| Need | Use |
| --- | --- |
| Explore interactively in this conversation: load, step, look, adjust | **MCP tools** (`mcp__shaderous__*`), if they are loaded |
| A run or sweep to completion with files on disk; anything long | **Command-line runner**, `npm run headless -- ...` |
| Checking a code change in the dev server, or tests | **`window.shaderous`** in the browser pane or Playwright |

Full references: [docs/api.md](../../../docs/api.md) (page API, setup keys) and
[docs/headless.md](../../../docs/headless.md) (runner, MCP server). Read them for anything not
covered here.

## Setups

A setup is described with the same keys as a share link: `problem`, `dim` (2 or 3), `n` (cells per
side), `amr` (refinement levels, 0 to 5), plus overrides such as `tend`, `gamma`, `eos`,
`riemann`, `limiter`, `cfl`, `gravity`, `cool`, `nu`, `kappa`, `bc` and `p.<param>`. View keys
like `mode`, `field`, `axis` and `cmap` also work. Unknown keys are ignored and out-of-range
values clamped, so **read back `state().query`** to see what actually took effect.

- Get problem ids and parameters from `describe` (MCP), `npm run headless -- describe [id]`, or
  `shaderous.describe()`. Don't guess them.
- Some presets open on a grid of their own (`describe` gives it as `grid`): leave out `n` and
  `amr` to get it. The γ = 4/3 tidal disruptions need theirs (256³, 3 levels): on a coarser
  grid the star settles too compact and survives encounters that should disrupt it.
- Grid sizes: 2D runs take n from 128 up, 3D runs from 32 up. Above `resolutions.uniformMax`,
  a grid needs `amr` of 1 or more.
- Some problems are 3D only (`supports2d: false`). For those, `dim=2` is ignored.

## Workflow

1. **Start small.** Use 2D at n=256, or 3D at n=64. Check that the run behaves, then scale up.
   A 3D run at 256³ or more with AMR takes minutes and a lot of GPU memory.
2. **Load, then run to a time.** `run`/`runTo` lands on the requested time exactly. A run stops
   at the problem's end time (`stopped: "end"`); to go further, set `tend` in the setup.
3. **Check health after every run.**
   - `errors` must be empty.
   - `diagnostics.mass` should stay constant, unless the problem has sources, sinks or outflow
     boundaries.
   - `eTot` should stay constant without cooling, sources or open boundaries.
   - `floored` counts cells held at the density or pressure floor. Many of them mean the run is
     struggling.
4. **Measure before you look.** `diagnostics`, `stats`, `probe` and small `slice`s give
   numbers. Use `screenshot` to see structure, and pick the field and view on purpose: a slice
   of a 3D run shows more than the default volume rendering.
5. **Record provenance.** Note `state().query`, the build and the adapter. The runner writes
   them to `manifest.json` for you.

## MCP tools

Typical sequence: `describe` → `load {setup}` → `run {until}` → `diagnostics` →
`screenshot {width, view: {mode: "slice", field: 0}}`.

- The page starts on the first call, which takes a few seconds.
- A run stays paused between calls. `load` always starts from t = 0.
- `control {id, value}` changes a setting partway through a run, as the panel would: switching
  the Riemann solver or EOS, gravity, viscosity. `controls` lists the ids. Number fields are in
  the units the panel shows, so call `control {id: "units", value: "code"}` first if needed.
- `slice` returns its values inline only up to res 64. For more, pass `save_path` and read the
  file.
- If the tools are listed but fail with "could not start Chrome" or a WebGPU error, use the
  runner with `--headed`, or tell the user.

## Command-line runner

```bash
npm run headless -- run --problem sedov --dim 2 --n 512 --until 0.05 --every 0.01 --png --view field=pressure --out runs/sedov
```

```bash
npm run headless -- sweep --problem sedov --dim 2 --vary n=128,256,512 --until 0.05 --slice rho --slice-res 256 --out runs/conv
```

- Each run writes `manifest.json`, `diag.jsonl` (one line per output), and then whatever was
  asked for: PNGs with `--png`, slice JSON with `--slice VAR`, `.athdf` files with `--athdf`.
- Exit codes: 0 means success, 1 an error, 2 that the page reported errors (listed in the
  manifest).
- Run long jobs in the background and at low priority: add `--low-priority`. On Windows you
  can also prefix the command with `cmd //c start //b //wait //belownormal`, but then use flags
  instead of a query string, since `cmd` splits commands on `&`.
- `--debris` writes the debris energies and dM/dt at each output, for problems that track them
  (the tidal disruption). `python tools/headless/plot_dmdt.py RUNDIR` draws dM/dε and dM/dt.
  Read dM/dt only a few dynamical times past pericenter, once it stops changing between
  outputs. Check the budget: `diagnostics.mass` plus `totals.escaped` plus the accreted mass
  should equal the starting mass.
- Outputs go under `runs/`, which git ignores. Show the user images with the file tools; read
  numbers from `diag.jsonl` and the slices.

## `window.shaderous` in a browser

On the dev server (`preview_start` with name `shaderous`, or `npm run dev` on port 5183), open
`/?headless=1&problem=...`. In that mode the page doesn't autoplay or draw; omit `headless` to
watch the run. Then evaluate:

```js
await shaderous.ready;
await shaderous.load('problem=sedov&dim=2&n=256');
await shaderous.runTo(0.05);
await shaderous.diagnostics();
```

All results are JSON. `screenshot()` returns a PNG data URL. Use this path to verify a code
change in the real app. `tests/gpu/api.spec.ts` shows the same calls driven through
Playwright.

## Verifying changes to the tooling itself

- `npm run test:headless` drives the MCP server and the runner end to end.
- `npx playwright test tests/gpu/api.spec.ts` tests the page API.
- After changing `src/ui/api.ts`, update docs/api.md. After changing the runner or the MCP
  server, update docs/headless.md and this skill.
