---
name: shaderous-sim
description: Run, inspect and measure Shaderous simulations without a person at the page — through the shaderous MCP tools, the command-line runner (single runs, parameter sweeps, convergence studies, figures, .athdf outputs), or window.shaderous in a browser. Use whenever a task needs a simulation actually run or looked at - checking a physics change, reproducing a bug, making a figure or movie frames, a sweep, or answering "what happens if" - and when verifying a change to the solver or renderer in the real app. Also use it to build a simulation nobody has set up yet from a description in words ("collide a 0.5 Msun main-sequence star with a 1 Msun white dwarf", "a disk around a binary with a third body", "a blast next to a dense cloud"): it says how to turn that into a setup document from stars, point masses, orbits, disks and gas regions, and run it.
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

## Building a setup from a description

When no preset is the thing asked for, write a **setup document**: one JSON file that holds the
grid, the physics and a list of parts (stars, point masses, orbits, disks, gas regions,
obstacles, magnetic fields). Every preset is such a file. [docs/setups.md](../../../docs/setups.md)
is the reference for every key; read it before writing one.

**Start from the nearest preset, never from nothing.** A preset's floors, tenuous-gas limits,
boundaries and refinement were tuned; a blank document has none of that.

| The request is like | Start from |
| --- | --- |
| a star and a compact object or black hole | `tidal-disruption` (star of index 3/2) or `tidal-disruption-n3` (index 3) |
| two white dwarfs | `wd-collision` (or `wd-collision-burning`) |
| one star at rest, or with a field | `polytrope`, `polytrope-random`, `white-dwarf` |
| a disk around one or two masses | `mri-disk`, `binary` |
| blasts, clouds, shocks, walls | `sedov`, `sphere`, `forward-step`, `empty` |

Get the document with MCP `describe {problem}` (its `document`) or `shaderous.preset(id)`. From
the command line, read `src/presets/<id>.json`; if it has `extends`, lay it over that parent's
file (its keys win, and a list such as `parts` replaces the parent's whole). Change it, then run
it: MCP `load {document, setup}`,
`npm run headless -- run --setup FILE ...`, or `shaderous.load(query, document)`. The `setup`
query then only says the grid and what differs. Worked examples are in `examples/setups/`.

### Which part for which object

| In the description | Part | Notes |
| --- | --- | --- |
| Main-sequence star, brown dwarf, giant's envelope | `star`, `type: "polytrope"` | `mass`, `radius`, index `n`: 3 for a Sun-like or heavier star, 1.5 for a low-mass (under about 0.4 M☉) or fully convective one. Gas γ = 5/3. On the main sequence R/R☉ ≈ (M/M☉)^0.8 below a solar mass, ^0.57 above |
| White dwarf, resolved | `star`, `type: "white-dwarf"` | Only its `mass` (the radius follows). Needs `eos` `degenerate` or `helmholtz` and the white dwarf units of `wd-collision`. Cannot share a setup with a polytrope |
| White dwarf beside a normal star; neutron star; black hole; planet as a perturber | `mass` | A point mass: `gm` = G M, `soft` at least two finest cells, `sink: 0` unless it should swallow gas |
| Orbit, encounter, collision | `orbit` | Places two bodies: below |
| Accretion disk | `disk` | `about` masses or stars; `radialFn` and `verticalFn` take formulas |
| Cloud, bubble, slab, hot spot | `region` | A sphere, box or half-space with its own density, pressure, velocity |
| Wall, body in a wind, inlet | `obstacle`, `inflow` | |
| Magnetic field | a star's `field`, a disk's `loops`, or a `field` part (uniform) | |

**The rule of scale decides gas or point.** The finest cell is `box / n`. An object needs about
16 finest cells across its radius to hold together as gas (32 for index 3). One smaller than a
few cells cannot be gas: make it a point mass. A white dwarf is a hundredth of a main-sequence
star's size, so beside one it is always a point mass; two white dwarfs together are gas, in
their own units.

### Units and numbers

- Use the units of the preset you started from. Solar units (`length` in `Rsun`, `mass` in
  `Msun`, `"G": "units"`, which makes G = 1): masses are in M☉, lengths in R☉, speeds in
  437 km/s, and one unit of time is 1593 s. A point mass of M solar masses has `gm` = M.
- A star's dynamical time is sqrt(R³ / G M). Set `run.tDyn` to it, and plan run lengths in it.
- When you change a star, rescale what was tuned to the old one: densities (`fluff.rho`, a
  `background` that is not `relativeTo` the star, `mesh.refineDense`) go as M / R³, specific
  energies (`fluff.u`) as G M / R, speeds (`fluff.v`) as sqrt(G M / R).
- Keep `background` `relativeTo` the star: its `rho` and `cut` are then fractions of the star's
  central density, and follow it.

### Orbits

- `"type": "flyby"`: a point mass on a parabola about a star that stays at the origin. Use it
  with `"frame": "gas"` so the star stays in the box. `q` is the point mass over the star's
  mass. `beta` is the tidal radius over the pericenter distance, with tidal radius
  r_t = R q^(1/3). **A collision means the pericenter is inside the star**: β above q^(1/3).
  Nearly head-on is β of several times that; a graze is β ≈ q^(1/3). `start` is the starting
  separation in tidal radii, 2.5 to 3.
- `"type": "collision"`: two stars on a parabola about their center of mass. `impact` is the
  pericenter over the sum of the radii: 0 is head-on, 1 a graze. `starts`, in sums of radii.
- `"type": "binary"`: two point masses, `separation`, `ecc`, `q`. `"onOrbit": true` on a mass
  holds it on its orbit whatever the gas does.
- Or give `pos` and `vel` yourself, with no orbit part. Two bodies falling together from rest
  at infinity meet at relative speed sqrt(2 G (M₁ + M₂) / d) at separation d; share positions
  and velocities so the center of mass stays at the origin, at rest.

### Box, grid and physics

- Box: about twice the starting separation plus a few radii, so nothing starts near an edge.
- Self-gravitating stars need 3D, `"gravity": "isolated"`, and no periodic face; use outflow
  only (`3`) on all six so nothing flows back in.
- Use the adaptive mesh: `amr` 2 to 4 with `n` the finest resolution. `mesh.refineMasses`
  keeps the mesh fine around point masses; `mesh.refineDense` (a density) keeps stars at the
  finest level.
- Options that cannot be combined are refused or reported with the reason (the table in
  docs/setups.md). `load` rejects a malformed document and names the key.

### Then check it, and say what you assumed

0. Check the document before it runs: MCP `check_setup {document}`, or
   `shaderous.check(document)`. Fix every `error` (a rule broken, with the reason) and read every
   `warning` (a softening under two cells, bodies that overlap or leave the box, a grid past the
   GPU memory): `load` does not refuse a document over them.
1. Load it at a coarse grid first (`setup: "n=128&amr=2"`), run a few steps, and look at
   `diagnostics` (the mass should be the stars' total) and a slice through the orbital plane.
2. Check that the stars hold still before the encounter: `state.preparing` is true while they
   settle, and their central density should not swing by more than a few percent after.
3. Run to the encounter on the document's own grid, with `--low-priority` if it takes minutes.
4. Tell the user every modeling choice the description left open: the polytropic index, the
   radius you took, the impact parameter, what became a point mass and what that leaves out (a
   point-mass white dwarf has no gas of its own, so nothing of it burns or is stripped).

**Worked example**, `examples/setups/ms-wd-collision.shaderous.json`, for "collide a 0.5 M☉
main-sequence star with a 1.0 M☉ white dwarf". Started from `tidal-disruption-n3`. The star:
`mass` 0.5, `radius` 0.5, `n` 3. The white dwarf is 0.008 R☉ across, far below a cell, so it is
a `mass` with `gm` 1, `soft` 0.05 and no sink. The orbit: `flyby` with `q` 2 (so r_t = 0.63)
and `beta` 4, a pericenter of 0.16 R☉, well inside the star. Box 4 R☉ at 256³ with three
levels, `refineMasses` on. `fluff.rho` raised fourfold for the denser star (M / R³ = 4). It
runs in about a minute to t = 2, by when the white dwarf has passed through the star.

```bash
npm run headless -- run --setup examples/setups/ms-wd-collision.shaderous.json --until 3 --every 0.25 --png --view mode=slice --view sliceAxis=2 --low-priority --out runs/ms-wd
```

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
- `--gpu-memory GB` sets the GPU memory the simulation may take (default 2.5 GB on a desktop):
  it sets how many blocks the adaptive mesh can hold. A big 3D adaptive run that reports
  "block budget full" in its status wants more if the GPU has it. With the MCP tools, use
  `control {id: "gpumem", value: GB}` (it restarts the run).
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
