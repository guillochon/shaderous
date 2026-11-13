# Scripting API

The simulator page can be driven by a program instead of a person: a Playwright or Puppeteer
script, anything that speaks the Chrome DevTools Protocol, an AI agent with a browser, or the
browser console. The page puts one object on `window`, `window.shaderous`. It loads a setup,
steps the run to a step count or an exact time, reads the gas back, takes screenshots, saves
and loads `.athdf` files, and sets any control in the panel.

The code is in [src/ui/api.ts](../src/ui/api.ts), and
[tests/gpu/api.spec.ts](../tests/gpu/api.spec.ts) exercises all of it.

For runs without writing any browser code, use the command-line runner or the MCP server built
on this API. Both are in [headless.md](headless.md).

**Contents**

1. [Quick start](#quick-start)
2. [Conventions](#conventions)
3. [Headless mode](#headless-mode)
4. [Setups](#setups)
5. [Reference](#reference)
6. [Recipes](#recipes)
7. [Limits and caveats](#limits-and-caveats)

---

## Quick start

Start the dev server (`npm run dev`, which serves http://localhost:5183) or use the live site.
Then, from Node with Playwright:

```js
import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const browser = await chromium.launch({
  channel: 'chrome',
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto('http://localhost:5183/?headless=1&problem=sedov&dim=2&n=512');
await page.evaluate(() => shaderous.ready);

const S = (name, ...args) => page.evaluate(([n, a]) => shaderous[n](...a), [name, args]);

await S('runTo', 0.03);                            // lands on t = 0.03 exactly
console.log(await S('diagnostics'));               // mass, energies, momenta, ranges, ...
const { values, res } = await S('slice', { variable: 'rho', res: 256 });
await S('view', { field: 3, colormap: 'viridis' }); // pressure
const png = await S('screenshot', { width: 1024 });
writeFileSync('sedov.png', Buffer.from(png.split(',')[1], 'base64'));
const file = await S('athdf');
writeFileSync(file.name, Buffer.from(file.base64, 'base64'));
await browser.close();
```

On Windows, add `--use-angle=d3d11` to the arguments. If headless Chrome finds no GPU adapter,
launch it with `headless: false`, as [playwright.config.ts](../playwright.config.ts) does with
`PW_HEADED=1`.

In the console of an open page you can call the same methods directly:
`await shaderous.runTo(1)`.

## Conventions

- **Every method returns a promise.** The one exception is `errors()`, which returns its
  answer directly.
- **Results are plain JSON.** They hold numbers, strings, arrays and objects, and binary files
  come as base64. A result therefore survives `page.evaluate`, CDP's `Runtime.evaluate` with
  `returnByValue`, or any other bridge.
- **Calls run one at a time, in the order they were made.** A call that changes the run also
  waits for any rebuild the page has queued, such as shader compilation after a setting
  changed. Two calls fired together therefore never interleave.
- **A failed call rejects with an `Error` whose message says what to do.** Examples: an
  unknown problem names where to look them up (`describe()`), and a bad option lists the valid
  ones. A failed call does not block the calls after it.
- **Values are in code units.** Positions are given as fractions of the box: 0 to 1 along each
  axis, with z ignored in 2D.
- **The API holds the run paused.** It only advances when you call `run`, `step` or `runTo`.
  Methods that build or reset a run leave it paused at its start. `play()` hands stepping back
  to the page's own loop, which is the same as pressing Run.

## Headless mode

Add `headless=1` to the page's address (`/?headless=1&problem=...`). In this mode:

- nothing plays by itself, whatever the Autoplay setting;
- the frame loop draws nothing and reads nothing back, so the GPU only does the steps you ask
  for; the view is drawn when `screenshot()` asks for it;
- `headless=1` stays in the address as the page rewrites it.

The API also works without `headless=1`, on a page someone is watching. There, `run`, `step` and
`runTo` pause the page's own stepping while they work, and the view keeps drawing.

The page does not need to be visible. Stepping doesn't depend on `requestAnimationFrame`, so
it also works in a background tab. A screenshot taken while the page has no layout (a hidden
tab, say) is drawn at 1024 × 1024 unless you give a size.

## Setups

A setup is described with the same keys as the page's address, so a shared link is also a
script's setup. `load` takes any of these forms:

```js
shaderous.load('problem=sedov&dim=3&n=128&amr=2&gamma=1.4')
shaderous.load('https://guillochon.github.io/shaderous/?problem=binary&p.q=0.5')
shaderous.load({ problem: 'sedov', dim: 3, n: 128, amr: 2, gamma: 1.4 })
```

The main keys are listed below. [src/ui/link.ts](../src/ui/link.ts) has all of them, with the
range each value is clamped to.

| Key | Meaning |
| --- | --- |
| `problem` | Preset id (`describe().problems`) |
| `dim`, `n`, `amr` | 2 or 3 dimensions, cells per side, levels of refinement (0 to 5). Leave out `n` to get the preset's own grid (`describe().problems[i].grid`), or 3D at the default size. With `n` but no `amr`, the grid is uniform |
| `tend` | End time (`inf` for none) |
| `eos`, `gamma`, `K` | Equation of state and its constants |
| `riemann`, `limiter`, `cfl` | Riemann solver, slope limiter, Courant number |
| `gravity`, `G`, `vcycles` | Gravity mode, gravitational constant, multigrid V-cycles per step |
| `cool`, `crate`, `calpha`, ... | Cooling function and its parameters |
| `nu`, `kappa` | Viscosity and conductivity |
| `bc` | Six digits: kind of boundary at the lower x, y, z faces and then the upper ones |
| `refhi`, `reflo`, `refrho`, `tbins` | Refinement thresholds and time bins (adaptive runs) |
| `p.<id>` | One of the preset's own parameters (`describe().problems[i].params`) |
| `mode`, `field`, `axis`, `slice`, `log`, `cmap`, `lo`, `hi`, ... | The view |

Keys left out take the preset's defaults. A value that is unknown or out of range is ignored
or clamped, never trusted. `state().query` gives back the setup actually in force, as a
minimal query string. Store it next to your results, and `load` will reproduce the run.

## Reference

### `ready: Promise<void>`

Resolves once WebGPU is up and the run named in the address has been built. Rejects if the
page cannot start, for example when the browser has no WebGPU.

### `describe() → object`

Everything there is to choose from:

- `problems`: one entry per preset, with `id`, `label`, `family`, `variant`, `blurb`,
  `supports2d`, `debris` (whether `debris()` works for it), `grid` (the grid it opens on, or
  null), `tEnd`, `species`, and `params`. Each parameter has `id`, `label`, `value`
  (the default), `min`, `max`, `title`, `live` and `kind`.
- `fields`: the fields the view can show, as `{ index, name }`.
- `variables`: what `slice` and `stats` can read.
- `resolutions`: the grid sizes on offer for 2 and 3 dimensions, and `uniformMax`. A larger
  grid must be adaptive, with `amr` of 1 or more.
- `eos`, `cooling`, `riemann`, `limiter`, `gravity`, `colormaps`: the allowed values.

### `load(setup) → state`

Builds a new run from a setup (see [Setups](#setups)) and returns `state()` with the run
paused at t = 0. If the build fails, the call rejects with the page's error.

### `loadAthdf(base64) → state`

Starts a run from an `.athdf` file, paused at the file's time. A file Shaderous wrote carries
its own setup, so the run continues exactly where the file left off. A file from another code
(Athena++, or anything else that writes the same layout) is loaded into the current preset at
the nearest resolution on offer.

### `state() → object`

Returns `problem`, `label`, `ndim`, `n`, `amr`, `t`, `dt`, `steps`, `tEnd` (null if none),
`running`, `finished` and `preparing`, plus `query`, the setup as a query string, and
`adapter`, the GPU adapter the run is on.
`preparing` is true while a star in the initial conditions is still being settled into
hydrostatic balance. No steps are taken during that time, and `run` waits for it to finish.

### `run({ steps?, until?, timeoutMs? }) → result`

Advances the run, stopping at whichever limit comes first:

- `steps`: take at most this many steps;
- `until`: stop when the clock reaches this time. The last step is shortened so that the run
  lands on `until` exactly;
- `timeoutMs`: give up after this much wall time and return where the run got to.

The result is `{ t, dt, steps, taken, seconds, stopped }`. `stopped` is one of:

- `'steps'`: the step count was reached;
- `'time'`: the clock reached `until`;
- `'end'`: the run reached its end time (`tEnd`), and the state is now `finished`;
- `'timeout'`: `timeoutMs` ran out.

To run past `tEnd`, load the setup with a later `tend`, or `control('tend', '')` for no end
time.

Steps go to the GPU in batches of about a tenth of a second, with one readback per batch.

### `step(n = 1)` and `runTo(t, timeoutMs?)`

Shorthands for `run({ steps: n })` and `run({ until: t, timeoutMs })`.

### `restart() → state`

Starts the run again from its initial conditions, paused. Sources, sinks and edits are removed.

### `play()` and `pause()`

Hand stepping to the page's own loop, or take it back. You only need these on a page someone is
watching. In headless mode the page's loop is off and `play()` does nothing, so use `run`.

### `diagnostics() → object`

The run's diagnostics, read back now:

- the clock: `t`, `dt`, `steps`;
- totals: `mass`, `momentum` (a 3-vector), and the energies `eKin`, `eInt`, `eGrav`, `eTot`;
- `maxMach`;
- `floored`: cells held at the density or pressure floor;
- `jeansUnresolved`: cells where the Jeans length is under-resolved;
- `com`: the gas's center of mass relative to the box center;
- `masses`: the point masses, with position, velocity, mass and accreted mass;
- `ranges`: the least and greatest value of each field, keyed by field name.

### `debris() → object`

Debris energies, for problems that track them (`describe().problems[i].debris`; the tidal
disruption does). The gas is binned by its specific orbital energy about the problem's point
mass,

ε = ½ |v − v<sub>bh</sub>|² + φ,

where φ is the full potential at the gas: the point mass's and the gas's own. Two parts are
added up:

- **What has left the box.** The update tallies the gas crossing each open face of the box,
  binned by its energy as it crosses. Once out, the gas moves almost ballistically in the point
  mass's potential, so its energy no longer changes. The tally runs for the whole run, and
  `.athdf` files keep it, so a run continued from one carries on with it.
- **What is in the box.** This is a census taken when you call. A surviving core is set apart
  first: gas is bound to it where ½ |v − v<sub>core</sub>|² + φ<sub>self</sub> < 0, and
  v<sub>core</sub> is found by iterating from the gas's mean velocity. Without self-gravity
  there is no core.

A problem may bin only the gas its passive scalar marks. The tidal disruption does this, so
its bins hold the star's gas and not the ambient medium around it. The rest is counted apart,
as `ambient`.

The tallies are exact 64-bit fixed-point sums, so they add no rounding of their own. The mass
in the box, plus what left, plus what the sinks accreted, equals the starting mass to within
the rounding of `diagnostics().mass`.

| Key | Meaning |
| --- | --- |
| `t`, `tPeri` | The time now, and the time of pericenter |
| `gm`, `de` | G M of the point mass, and the energy scale (for a tidal disruption, the spread G M<sub>bh</sub> R<sub>★</sub> / r<sub>t</sub>²) |
| `lo`, `width` | Bin k covers energies `lo + k·width` to `lo + (k+1)·width`. There are 1000 bins across ±10 `de` |
| `escaped`, `inBox` | Mass per bin: what has left the box, and what is in it, the core apart |
| `below`, `above` | Mass below the first bin and above the last, as `[escaped, in box]` |
| `ambient` | Gas left out of the bins (the ambient medium, where the scalar marks the debris), as `[escaped, in box]` |
| `core` | `{ mass, vel }` of the surviving core, or null |
| `totals` | `escaped` and `inBox` (all the gas, ambient included), `core`; `bound` and `unbound` (the binned debris with ε < 0 and ε > 0); `ambient` |
| `dmdt` | `{ t, rate }`: the bound debris by when it returns to pericenter, earliest first |
| `units` | `{ time, mass }`: seconds and kilograms per code unit, or null |

The return times come from Kepler's third law, t = t<sub>peri</sub> + 2π G M (−2ε)<sup>−3/2</sup>,
and the rate is dM/dt = dM/dε · (2π G M)<sup>2/3</sup> / 3 · (t − t<sub>peri</sub>)<sup>−5/3</sup>.
Run a few dynamical times past pericenter before reading it, so that the energies have
stopped changing. Until then the core is not settled either: while the star is still
stretched, the gas that will fall back onto it can read as unbound. The census reads the grid
once per iteration, which takes milliseconds.

### `stats() → { t, steps, variables }`

The least, greatest and volume-weighted mean value of each variable over the whole box:
`rho`, `press`, `vel1`, `vel2`, `vel3`, `r0` (the passive scalar), and each species' mass
fraction. The whole state is read back from the GPU, which takes a moment on a large grid.

### `slice({ variable = 'rho', axis = 2, pos = 0.5, res }) → object`

A variable sampled on a `res` × `res` grid, on the plane perpendicular to `axis` at the
fraction `pos` of the box. `res` defaults to the grid's size, at most 256, and can go up to
2048.

The result is `{ t, variable, axis, pos, res, axes, min, max, values }`. `values` is a flat
array: `values[j * res + i]`, where `i` runs along `axes[0]` and `j` along `axes[1]`.

A variable can be any of those listed for `stats`, or `level`, the refinement level of the
block at each point. On an adaptive mesh each sample is taken from the finest block covering
it. In 2D the axis is always 2.

### `probe(points) → { t, points }`

The state at a list of points `[x, y, z]`. Each point returns `rho`, `press`, `vel` (a 3-vector),
`r0`, `level`, and `X`, the species' mass fractions by name, when the run has species.

### `screenshot({ width?, height? }) → string`

The view as a PNG data URL. With no size, it is drawn at the canvas's size on the page. Give
`width` alone for a square image. The overlays the page draws on top of the canvas (color bar,
markers, labels) are not included. With the range set to automatic, it is fitted to the current
data before the image is drawn.

### `view(changes) → view`

Changes what is drawn, leaving the run alone. These keys are accepted:

| Key | Meaning |
| --- | --- |
| `mode` | `'volume'`, `'slice'` or `'projection'` |
| `field`, `species` | Index into `describe().fields`; which species, for the mass-fraction field |
| `log` | Logarithmic color scale |
| `lo`, `hi` | Ends of the color range; setting either turns the automatic range off |
| `sliceAxis`, `slicePos` | Slice and projection axis, and slice position (0 to 1) |
| `minRho`, `opacity` | Volume view: least density drawn, and opacity |
| `yaw`, `pitch`, `dist` | Volume view: camera angles and distance |
| `zoom`, `cx`, `cy` | Slice view: magnification, and the point at the center of the view |
| `showBlocks`, `contours`, `nContours` | Mesh block outlines, and contour lines (isosurfaces in the volume view) |
| `fog`, `surfaceOpacity` | Volume view: draw the gas as fog (off leaves the isosurfaces alone), and how opaque the isosurfaces are |
| `light` | A white background instead of the dark one |
| `box` | Volume view: outline the box (on by default) |
| `colormap` | A name from `describe().colormaps` |
| `autoRange` | Fit the color range to the data |

### `controls() → array`

The controls the panel is showing, each as `{ id, kind, label, title, value }`. A select also
has `options`, and a number field has `min` and `max` where it sets them. It includes controls
inside collapsed sections.

### `control(id, value?)`

Sets one control, or presses a button when it is given no value, exactly as a person would.
It then waits for whatever the change starts, such as a rebuild of the shaders or a restart.

This lets you change a run partway through, the way a user can: switch the Riemann solver,
the equation of state, gravity, viscosity, or a preset parameter, without starting from t = 0.
Changes that a person's edit would restart (initial conditions, resolution) restart here too.

Number fields hold values in the units the panel shows. For a problem with physical units,
call `control('units', 'code')` first if you mean code units. The run stays paused afterwards,
unless the control was `run`.

### `athdf() → { name, time, bytes, base64 }`

The state of the gas as an `.athdf` file, base64-encoded: HDF5 in Athena++'s layout, which yt
and Athena++'s own tools read. The file also carries the setup, so `loadAthdf` can continue
the run exactly.

### `errors() → string[]`

Everything the page has reported going wrong, oldest first: exceptions, WebGPU validation
errors and shader compilation errors. This is the same report the page's error panel shows.
Check it after a sequence of calls, because a GPU error can arrive after the call that caused
it has already returned.

### `build: string`

The build of the page, for your records.

## Recipes

**A parameter sweep.** Load each setup, run it to a time, and keep a number from each:

```js
for (const gamma of [1.1, 1.4, 5 / 3]) {
  await S('load', { problem: 'sedov', dim: 2, n: 512, gamma });
  await S('runTo', 0.05);
  const { variables } = await S('stats');
  console.log(gamma, variables.press.max);
}
```

**Snapshots at regular times.** Call `runTo` at each output time and then `athdf`. Each
`runTo` lands on its time exactly, so the outputs line up across runs.

**Reproducibility.** Save `state().query` and `build` with each result. Loading that query in
the same build sets up the same run. Results may still differ slightly between GPUs and drivers.

**Watching while a script drives.** Leave out `headless=1` and open the page in a visible
window. The view keeps drawing while the script steps the run.

**The return rate of a tidal disruption.** Run past pericenter, then read the debris:

```js
await S('load', { problem: 'tidal-disruption-full', dim: 3, n: 128, amr: 2 });
await S('runTo', 8);
const { dmdt, core, totals } = await S('debris');
```

The runner does the same at every output with `--debris`, and
`tools/headless/plot_dmdt.py` plots the result ([headless.md](headless.md)).

**Changing physics partway through.**

```js
await S('runTo', 1);
await S('control', 'eos', 'gas_radiation'); // pressure carried across, run continues
await S('runTo', 2);
```

## Limits and caveats

- **It needs a browser with WebGPU.** Headless Chrome on a machine with a GPU works. Without
  a usable GPU, Chrome may fall back to a software adapter, which is very slow.
- **Large grids are slow to read back.** `stats`, `slice`, `probe` and `athdf` read the whole
  state back from the GPU, which takes a while at 512³. Sample sparingly on big runs.
- **There are no brushes or tools yet.** The editing tools (stirring, heating, adding gas,
  stars, sinks, obstacles) are only available by pointer.
- **`play()` uses the page's own loop.** In headless mode that loop is off, so step with `run`.
- **It is one page and one run.** For runs in parallel, open several pages. Each one takes its
  own GPU memory.
