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

- **Every method returns a promise**, except `describe()`, `preset()`, `check()` and
  `errors()`, which return their answer directly (and `preset()` throws rather than rejects).
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

Every key is listed below, in the order a link writes them. The settings come from the option
table ([src/setup/options.ts](../src/setup/options.ts)) and the view's from
[src/render/view_options.ts](../src/render/view_options.ts); this table is written from them
by `npm run gen`.

<!-- gen:link-keys -->
| Key | Of | Meaning |
| --- | --- | --- |
| `problem` | The run | Preset id |
| `setup` | The run | A setup document of one’s own, packed (with problem=custom) |
| `dim` | The run | 2 or 3 dimensions |
| `n` | The run | Cells per side (the finest level’s on an adaptive mesh). Left out: the preset’s own grid |
| `amr` | The run | Levels of refinement, 0 to 5. With n but no amr, the grid is uniform |
| `tend` | Settings | End time (inf for none) |
| `sym` | Settings | Mirror symmetry: none, x, y, z, xy, xz, yz, xyz. Compute only the part of a symmetric setup above its planes of symmetry, each mirrored axis’s lower face a plane of symmetry; the totals are those of the whole. With every axis mirrored (an octant in 3D, a quadrant in 2D) each runs from 0 to the box side: twice the resolution for the same cells. With only some (a half, or a quadrant in 3D) those run from 0 to half the box with half the cells: the same cube as without mirrors, at half or a quarter of the cost |
| `eos` | Settings | Equation of state: ideal, barotropic, liquid, gas_radiation, degenerate, ionizing, helmholtz, hhe, aneos. How pressure follows from density and internal energy. Changing it carries each cell’s pressure over, and the run continues |
| `material` | Settings | Material: 0, 1, 2, 3. The material of the ANEOS equation of state: rock, iron alloy or water, each from its published tables. A run whose two species are materials (a planet with a core) takes them instead |
| `gamma` | Settings | gamma. Adiabatic index γ: p = (γ - 1) ρ e for an ideal gas, p = K ρ^γ for a barotropic one |
| `K` | Settings | K (p = K rho^gamma). Barotropic constant K: p = K ρ^γ. With γ = 1 it is the square of the sound speed |
| `lc` | Settings | Sound speed. The liquid’s sound speed c₀, a constant: p = c₀² (ρ - ρ₀). It is chosen, not physical: ten times the fastest flow keeps the density within 1% of ρ₀ (a Mach number of 0.1), and the step shrinks in proportion as it grows |
| `lrho` | Settings | Reference density. The density ρ₀ at which the liquid’s pressure is zero: p = c₀² (ρ - ρ₀). The liquid should start near it |
| `arad` | Settings | Radiation constant a. Radiation pressure a T⁴ / 3 is added to the gas pressure |
| `degrho` | Settings | Degeneracy density. Density at which the electrons’ Fermi momentum reaches mₑc. For a setup with physical units it is set from the composition (about 2×10⁶ g/cm³ for C/O); in code units |
| `degp` | Settings | Degeneracy pressure scale. The scale π mₑ⁴ c⁵ / 3h³ of the degenerate electron pressure. Set from physical constants for a setup with physical units; in code units |
| `coulomb` | Settings | Coulomb corrections. The ions’ Coulomb energy, as a multiplier: 1 for all of it, 0 for none. It lowers the cold pressure by a percent or two in dense white dwarf matter (more in the envelope) and adds to the heat capacity of cold, strongly coupled ions |
| `hherad` | Settings | Radiation term. The radiation’s pressure a T⁴ / 3 and energy a T⁴ in the equation of state, as a multiplier: 1 for gas in equilibrium with its radiation (a star’s interior), 0 for gas alone (a cloud or thin gas, which its radiation leaves). Radiation transport does not run with this equation of state |
| `iont` | Settings | Ionization temperature. The ionization energy of hydrogen, 13.6 eV, over k, in code temperature. Set from physical constants for a setup with physical units |
| `ionrho` | Settings | Saha density. Density scale of the Saha equation in code density: denser gas needs a higher temperature to ionize. Set from physical constants for a setup with physical units |
| `frho` | Settings | Tenuous below density. Gas thinner than this has its heat and speed capped (the next two fields), so a few hot or fast cells of ambient gas cannot set the timestep for the whole grid. Not conservative, but such gas carries next to no mass. 0 is off |
| `fu` | Settings | Max specific energy. Largest internal energy per unit mass tenuous gas may have. 0 is no cap |
| `fv` | Settings | Max speed. Largest speed tenuous gas may have. 0 is no cap |
| `fva` | Settings | Max Alfvén speed. With a magnetic field: largest Alfvén speed \|B\| / √ρ anywhere. Where the field is stronger than that for the gas it threads, mass is added at rest until it is not. Not conservative. 0 is no cap |
| `cool` | Settings | Cooling function: off, power, ism. Optically thin radiative cooling (and heating) of the gas |
| `crate` | Settings | Coefficient Λ₀. Power law: the cooling rate per unit volume is ρ² Λ₀ T^α |
| `calpha` | Settings | Exponent of T. Power law: the exponent α of T in the cooling rate (1/2 is bremsstrahlung) |
| `cheat` | Settings | Heating rate. Heating per unit mass (power law), or the heating rate at unit density (interstellar) |
| `ctunit` | Settings | Kelvin per unit T. Interstellar: kelvin per code unit of temperature (code density is atoms per cm³) |
| `ctfloor` | Settings | Temperature floor. Gas at or below this temperature does not cool |
| `rad` | Settings | Radiation transport: off, m1. Gray radiation transport with the M1 closure: the radiation’s energy and flux, carried with the gas, streaming and diffusing between cells, and trading heat and momentum with the gas through its opacity (docs/numerics.md section 24). Cartesian coordinates, on the uniform grid or the adaptive mesh (all blocks then take one timestep), without a magnetic field |
| `kap` | Settings | Opacity: constant, electron, kramers. How opaque the gas is to the radiation, per unit mass: a constant, electron scattering, or Kramers’ free-free absorption with electron scattering. With physical units the scattering and Kramers constants come from the composition |
| `kapa` | Settings | Absorption κ_a. Absorption opacity per unit mass (the Planck mean), which exchanges heat between gas and radiation. With electron scattering, a small true absorption |
| `kaps` | Settings | Scattering κ_s. Scattering opacity per unit mass: it drags on the radiation’s flux without heating the gas (electron scattering is 0.2 (1 + X) cm²/g) |
| `kapk` | Settings | Kramers K₀. Kramers: the free-free absorption is K₀ ρ T^-3.5 per unit mass, in code units |
| `kapt` | Settings | Kramers least T. Kramers: the opacity is taken at no lower temperature than this (it grows without bound as T falls) |
| `clight` | Settings | Speed of light c. The speed of light in code units. With physical units it is theirs; a scale-free setup sets its own |
| `chat` | Settings | Reduced speed of light. The speed light crosses transparent gas at in the run (at most c): its step is about dx / chat. Ten times the fastest gas is enough; opaque cells take the true speed whatever this is |
| `clocal` | Settings | Speed of light per cell (0 or 1). Opaque cells take the true speed of light and only transparent ones the reduced one, so the radiation in a star has its true heat capacity. Off, one reduced speed everywhere |
| `radflux` | Settings | Radiation flux: hll, glf. The flux between cells: HLL with the M1 closure’s wave speeds, or global Lax-Friedrichs (the speed of light: more diffusive, more isotropic around sources) |
| `rsweeps` | Settings | Radiation sweeps per step. Radiation sweeps per hydro step (1 to 16). The hydro step is at most this many of the sweep’s own steps, so more sweeps let it be longer |
| `nu` | Settings | Viscosity ν. Kinematic viscosity: the stress is ρ ν times the shear. Obstacles hold the gas at their walls. 0 is off |
| `kappa` | Settings | Conductivity κ. Thermal conduction: the heat flux is -κ ρ grad T, so the temperature of an ideal gas diffuses at (γ - 1) κ. 0 is off |
| `eta` | Settings | Resistivity η. With a magnetic field: magnetic diffusivity, so the field diffuses as η ∇²B (the electric field gains η curl B) and the current it dissipates, η J², heats the gas. 0 is off (ideal MHD, with only the scheme’s own diffusion) |
| `drive` | Settings | Driving acceleration. Turbulence driving: the root-mean-square acceleration of a random force that stirs the gas at large scales (a band of wavelengths, correlated over a time τ). The gas reaches speeds of about (accel × box / band)^½ where it balances the dissipation. 0 is off |
| `dtau` | Settings | Driving correlation time τ. How long the random force keeps its pattern (its Ornstein-Uhlenbeck correlation time) |
| `dkmin` | Settings | Driving band, longest. The longest wavelengths driven, as wavelengths per box side |
| `dkmax` | Settings | Driving band, shortest. The shortest wavelengths driven, as wavelengths per box side (at most 64 wavevectors are kept, the longest first) |
| `dzeta` | Settings | Solenoidal share ζ. What the force stirs: 1 swirls only (solenoidal, no compression), 0 compresses only, 0.5 the natural mix |
| `dseed` | Settings | Driving seed. Which random force: each seed gives another, and the same seed the same run |
| `gravity` | Settings | Gravity: off, constant, periodic, isolated. Off, a uniform field, or the self-gravity of the gas. Point masses pull either way |
| `G` | Settings | G. Gravitational constant in code units, or the strength of the uniform field. A setup with physical units may leave it to them |
| `vcycles` | Settings | V-cycles. Multigrid V-cycles per step for self-gravity. More converge the potential further, at more cost. On the adaptive mesh this is the most per step |
| `jfloor` | Settings | Jeans floor (0 or 1). Pressure floor that keeps the Jeans length resolved (Truelove) |
| `jcells` | Settings | Cells per Jeans length. The Jeans floor raises the pressure so the Jeans length spans at least this many cells |
| `fluxwork` | Settings | Conserve energy (0 or 1). Flux-form gravitational work: conserves total energy |
| `refhi` | Settings | Refine above. Adaptive mesh: a block is split where the normalized second derivative of density or pressure exceeds this anywhere in it. The indicator is near 1 at a shock or contact and small where the gas varies smoothly. Lower refines more |
| `reflo` | Settings | Merge below. Adaptive mesh: four sibling blocks are merged where the indicator is below this in all of them. Kept below the refinement threshold |
| `refrho` | Settings | Refine above density. Adaptive mesh: gas thinner than this does not ask for refinement, so ripples in the ambient gas far from a star leave its blocks coarse. 0 is off |
| `tbins` | Settings | Local timesteps: 0, 1, 2, 3. Adaptive mesh: blocks of slow gas take longer steps than the fastest block, where that saves work |
| `riemann` | Settings | Riemann solver: hllc, hllclm, hll. How the flux between neighboring cells is worked out. HLLC resolves contact waves; its low-Mach version cuts dissipation in slow flow; HLL is the most diffusive and the most robust |
| `limiter` | Settings | Limiter: koren, mc, vanleer, minmod. How steeply the reconstruction may vary within a cell: Koren is sharpest, Minmod most diffusive and most robust |
| `cfl` | Settings | CFL number. Timestep as a fraction of the time a signal takes to cross a cell. Smaller is more stable and slower |
| `sdr` | Settings | Start-up damping rate. For a while after the start, every cell’s velocity is damped at this rate (per unit time), so that whatever settled stars are still out of balance with on the grid is taken away while it is small instead of ringing through the run. The rate falls smoothly to zero over the start-up damping time. 0 is off |
| `sdt` | Settings | Start-up damping time. Time from the start over which the start-up damping falls smoothly to zero (as a half cosine). 0 is off |
| `psplit` | Settings | Pressure: explicit, implicit. How the pressure acts. Explicit: in the fluxes between cells, with a timestep that follows the sound speed. Implicit: advection is explicit and the pressure is solved for each step, so the timestep follows the flow speed, however fast the sound: for the liquid and the ideal gas, in Cartesian coordinates, on the uniform grid or the adaptive mesh |
| `cut` | Settings | Partly filled cells (0 or 1). Obstacles cut the cells their surfaces cross: each cell holds gas in a share of its volume and each face is open in a share of its area, so a body keeps its true shape below a cell (a thin blade, its pitch), where the plain grid makes it a staircase of whole cells. For the liquid with implicit pressure, on the uniform grid, with obstacles at rest |
| `pcycles` | Settings | Pressure V-cycles. Implicit pressure: multigrid V-cycles per step (warm-started). More converge the pressure further, at more cost |
| `ptol` | Settings | Pressure tolerance. Implicit pressure: the residual of the pressure equation after a step’s V-cycles, relative to its largest right-hand side, above which the step is counted as unconverged |
| `acfl` | Settings | Acoustic steps per step. Implicit pressure: the most steps of the explicit scheme that one step may span. It is what bounds the step of a liquid at rest; sound that a step spans many times over is damped |
| `dampr` | Settings | Damping from r. Radius from the box center where the damping zone starts |
| `damprate` | Settings | Damping rate. How many times per local orbital period the gas at the box edge is relaxed toward its initial state. 0 is off |
| `wb` | Settings | Well-balanced reconstruction (0 or 1) |
| `bc` | Settings | Six digits: the boundary at the lower x, y, z faces and then the upper ones (0 periodic, 1 outflow, 2 reflecting, 3 outflow only, 4 line-tied wall, 5 moving wall) |
| `p.<id>` | Settings | One of the preset’s own parameters, by its id |
| `mode` | The view | `volume`, `slice` or `projection` |
| `field` | The view | Which field is drawn: an index into `describe().fields` |
| `species` | The view | Which species, for the mass-fraction field |
| `log` | The view | Logarithmic color scale |
| `lo` | The view | Lower end of the color range; setting either end turns the automatic range off |
| `hi` | The view | Upper end of the color range |
| `axis` | The view | Slice and projection axis (0 x, 1 y, 2 z) |
| `slice` | The view | Slice position along its axis (0 to 1) |
| `rhocut` | The view | Volume view: least density drawn, as a fraction of the way from the smallest to the largest on a log scale |
| `opacity` | The view | Volume view: opacity of the fog |
| `yaw` | The view | Volume view: camera angle about the vertical, in radians |
| `pitch` | The view | Volume view: camera angle above the horizontal, in radians |
| `dist` | The view | Volume view: camera distance, in box lengths |
| `tx` | The view | Volume view: the point the camera orbits, in fractions of the box (0.5: the center) |
| `ty` | The view | The same, in y |
| `tz` | The view | The same, in z |
| `zoom` | The view | Slice view: magnification (1 fits the box) |
| `cx` | The view | Slice view: the point at the center of the view, in fractions of the box |
| `cy` | The view | The same, along the other axis of the slice |
| `blocks` | The view | Outline the blocks of an adaptive mesh |
| `contours` | The view | Contour lines (isosurfaces in the volume view) |
| `ncont` | The view | How many contours |
| `fog` | The view | Volume view: draw the gas as fog (off leaves the isosurfaces alone) |
| `isoa` | The view | Volume view: how opaque the isosurfaces are |
| `exact` | The view | Volume view: isosurfaces of the field’s log interpolated between cell centers (off, the default: of the value smoothed by a cubic B-spline, which places a surface within the cells it crosses and takes away their terraces) |
| `light` | The view | A white background instead of the dark one |
| `box` | The view | Volume view: outline the box (on by default) |
| `flines` | The view | A run with a magnetic field: draw field lines (on by default with a field). In 2D they are contours of the flux function |
| `nflines` | The view | How many field lines |
| `flfloor` | The view | Field lines end where \|B\| falls below this fraction of the largest \|B\|, and start only where it is above twice that (default 0.01): lower it to follow a weak field farther out, such as to the surface of a star |
| `mirror` | The view | An octant (or 2D quadrant) of a mirror-symmetric run: show the whole configuration, the computed part with its mirror images (on by default) |
| `turn` | The view | Volume view of a run in a rotating frame: turn the box at the frame’s rate, so the picture is the one a camera at rest sees (a rotor turning, its wake left behind) |
| `cmap` | The view | Colormap |
| `tf` | The view | Volume view: the opacity curve, as x:a points separated by commas |
<!-- /gen:link-keys -->

Keys left out take the preset's defaults. A value that is unknown or out of range is ignored
or clamped, never trusted. `state().query` gives back the setup actually in force, as a
minimal query string. Store it next to your results, and `load` will reproduce the run.

### Setup documents

A preset is itself a file: a setup document, described in [setups.md](setups.md). A run can
be built from any such document, so a script can assemble one from parts (point masses, stars,
disks, gas regions, obstacles, a magnetic field) rather than choose among the presets:

```js
const doc = await shaderous.preset('binary');        // a preset's document, to start from
doc.parts[2].ecc = 0.4;                              // the orbit's eccentricity
doc.parts.push({ kind: 'mass', id: 'c', gm: 0.01, pos: [3, 0, 0], vel: [0, 0.58, 0], soft: 0.06 });
await shaderous.load('dim=2&n=512&amr=3', doc);      // the grid, then the document
await shaderous.load(doc);                           // or the document alone, on its own grid
```

The document is checked as a file from a stranger is: unknown keys are dropped, values out of
range are held to it, and a wrong type rejects the call with the reason. `describe().setup`
lists what a document may hold. A run built from a document that is not a preset has the
problem id `custom`; `state().setup` is its document, with every setting changed since.

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
- `variables`: what `slice` can read. `stats` reads the first six.
- `resolutions`: the grid sizes on offer for 2 and 3 dimensions, and `uniformMax`. A larger
  grid must be adaptive, with `amr` of 1 or more.
- `eos`, `cooling`, `riemann`, `limiter`, `gravity`, `colormaps`: the allowed values.
- `methods`: this API's own methods, each with `name`, `summary`, `args` (as JSON Schema) and
  `changesRun`.
- `setup`: what a setup document may hold. `format` is the value of a document's `format`
  key (`shaderous-setup`). `options` are the general settings, each with its
  `path` in the document, `id` (its key in a link, where it has one), `label`, `title`, `kind`,
  range or `options`, `default`, and `apply`: what changing it costs a running simulation
  (`live`, `shaders` or `restart`). `parts` are the kinds of part with their fields, and
  `flows` the built-in initial conditions a `flow` part can name, with their parameters.

### `preset(id) → document`

A preset's whole setup document: the content of `src/presets/<id>.json`, with a variant's
parent resolved. Change it and hand it to `load`.

### `check(document) → { ok, error?, notes, issues, document? }`

Checks a setup document without running it. `error` is set, and `ok` is false, if the
document cannot be read at all; the message names the key at fault. Otherwise `notes` lists
what reading it changed (keys dropped, values held to their range), `document` is the whole
document as read, and `issues` is what the Setup dialog would list under it, each as
`{ level, text, path? }`. An `error` is a rule broken (a star without isolated self-gravity,
two flows that both fill the box) or a value that cannot work, and makes `ok` false; a
`warning` is a choice to look at again: a softening under two cells of the grid, bodies that
overlap or start outside the box, a disk heavier than a tenth of what it orbits, a grid that
needs more GPU memory than the page may take. `load` does not refuse a document over its
issues, so call `check` first.

### `load(setup, document?) → state`

Builds a new run from a setup (see [Setups](#setups)) and returns `state()` with the run
paused at t = 0. With a `document`, the run is built from that setup document, and `setup`
says the grid and what differs from it; a document may also be given alone. If the build
fails, the call rejects with the page's error. A document whose `mesh` shapes name files the
page does not have is refused, naming them: give each with `meshFile` first (a URL is fetched;
a name is also looked up among the files this browser kept from the Setup dialog).

### `meshFile(name, base64) → { name, bytes }`

Gives the page a mesh file a setup document's `mesh` shape names (STL, OBJ, PLY, glTF, GLB; an
airfoil `.dat` or an SVG outline), by that name, as base64. A `.gltf` whose buffers are separate
files needs each of them too, by the URI the `.gltf` gives it. The runner and the MCP server do
this for you, from the folder of the setup document ([headless.md](headless.md)).

### `loadAthdf(base64?, opts?) → state`

Starts a run from an `.athdf` file, paused at the file's time. `opts.switchBurn`: the file is of a
run that watched for burning (he4, c12 and o16 alone, stopped `'burn'`); it carries on with all
the network's species, the other ten at zero. With no argument, from the file
`athdfPut` has put together (for a file too large for one string; see `athdf`). A file Shaderous wrote carries
its own setup, so the run continues exactly where the file left off. A file from another code
(Athena++, or anything else that writes the same layout) is loaded into the current preset at
the nearest resolution on offer.

The result is `state()` plus `warnings`, a list that is empty unless one of our files came
from another build: it then says so, and names any setting this build could not use as
written (a key it no longer reads, a value it rejects or holds to its limits). Those settings
take this build's values. A file of ours whose preset this build does not have is refused.

### `viewAthdf(url?, { oneField?, faces? }?) → state`

Opens an `.athdf` checkpoint for viewing only: for a run too large to continue on this device.
The file is read in a worker without being copied whole (files larger than WebAssembly's 4 GB
open). A uniform grid's file is read as blocks of 16 cells. The run never steps: `run`, `step`,
`record` and the page's Run, Step and Reset refuse. Gravity is off, so the potential is not
shown; nor are the shock flag, time bins, burning or radiation.

It is shown in one of two ways, whichever keeps more of the file's levels in the GPU memory
budget (`viewBudget()`):

- **With its state**: the state at one parity, the primitives, the display's atlas and a
  field's faces, some 0.6 MB a block with a field (against 2.7 MB to run it). Every field,
  field lines, `slice`, `stats` and `probe` work, as for a run.
- **Holding one field**: only the field shown and the density at each cell, some 0.1 MB a block
  in 3D, worked out on the GPU from the file's primitives (`AmrSim.bake`). Changing the field
  reads it from the file again (seconds to tens of seconds for a file of gigabytes; `view` and
  `control` return once it is done). Fields that need neighboring cells (vorticity, divergence,
  density gradient, the divergence of B) are not offered. `probe` reads the file's own cells,
  at its full resolution, with the magnetic field (`B`); `slice` and `stats` refuse. Field lines
  need the face field, 0.13 MB a block more: turning them on opens the file again with it, at
  the finest level where it fits (`faces: true` asks for it from the start).

If even that does not fit, the finest levels are merged into their ancestors, one level at a
time, until it does: density, pressure and the magnetic field averaged by volume, velocities,
the passive scalar and species by mass, and the face field by area, so it stays
divergence-free. A uniform grid is downsampled the same way, by twos.

`url` is a file the page can fetch (the dev server serves the project's own files, e.g.
`/runs/x/0010.athdf`); with none, the file `athdfPut` put together. `oneField` picks the mode
instead (true: one field; false: the state). The result is `state()` plus `view`: `level`, the
finest level kept; `fileLevel`, the file's finest; `blocks` and `fileBlocks`, the leaf blocks
after and before merging; `oneField`; and `faces`, whether the face field is held.

### `viewBudget() → bytes`

The GPU memory the adaptive mesh may use (the `gpumem` control), which a view-only load plans
against.

### `state() → object`

Returns `problem`, `label`, `ndim`, `n`, `amr`, `t`, `dt`, `steps`, `tEnd` (null if none),
`running`, `finished` and `preparing`, plus `query`, the setup as a query string, `setup`,
the setup document the run stands for now (its preset or file with every setting changed
since: `load` it to get the same run), and `adapter`, the GPU adapter the run is on. On an adaptive mesh, `mesh` is the status bar's
line for it (`"8123 blocks, 4% of uniform"`, and whether the block budget is full); `null` on a
uniform grid.
`preparing` is true while a star in the initial conditions is still being settled into
hydrostatic balance. No steps are taken during that time, and `run` waits for it to finish.

### `run({ steps?, until?, timeoutMs? }) → result`

Advances the run, stopping at whichever limit comes first:

- `steps`: take at most this many steps;
- `until`: stop when the clock reaches this time. The last step is shortened so that the run
  lands on `until` exactly;
- `timeoutMs`: give up after this much wall time and return where the run got to.

The result is `{ t, dt, steps, taken, seconds, stopped, lastDt }`, where `taken` is the steps
this call took and `lastDt` the timestep of the last step taken (`dt` reads 0 while the clock
is held at a stop). `stopped` is one of:

- `'steps'`: the step count was reached;
- `'time'`: the clock reached `until`;
- `'end'`: the run reached its end time (`tEnd`), and the state is now `finished`;
- `'timeout'`: `timeoutMs` ran out;
- `'burn'`: a run that watches for burning (he4, c12 and o16 with the network's burning, docs/setups.md) found it:
  it is to be carried on with all the network's species (`loadSnapshot({ switchBurn: true })`, the runner's `--switch-burn`).

To run past `tEnd`, load the setup with a later `tend`, or `control('tend', '')` for no end
time.

With checkpoints on (the panel's Save checkpoints), the clock stops on each checkpoint time
exactly, as it does when the page plays: `run()` writes the checkpoint there and goes on.

Steps go to the GPU in batches, with only the clock read back after each, not the full
diagnostics. On a uniform grid two batches of about 50 ms of GPU time are kept in flight, so
the GPU steps on while the last one's clock is read. An adaptive mesh takes one batch at a
time (about 100 ms), because its refinement checks are applied at the start of the batch after
the one that read them: a batch encoded earlier would move the re-mesh and make the run depend
on timing. Batch sizes double from one step and carry over from one call to the next, so a
movie, which calls `run()` once per frame, doesn't start again from one step each time. A batch
goes no further than the next stop (the `until` time, a checkpoint, the end, or a movie's frame
time) as far as the last timestep tells, since steps past a stop still cost a full pass.

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
- `nuclearHeating`: for a run that burns, the energy burning released per unit time over the
  last step, summed over the gas (0 otherwise; the panel's L nuc, and the view "Nuclear heating"
  per unit volume);
- `com`: the gas's center of mass relative to the box center;
- `perturber`: the first point mass's position relative to the box center, or null;
- `masses`: the point masses, with position, velocity, mass and accreted mass;
- `bodies`, for a setup with obstacles: per obstacle (the first four apart, in the setup's
  order, any after them summed with the fourth) its `id`, the `force` the gas exerts on it, the
  `pressure` and `viscous` parts of that, the `torque` about its center, and with a reference
  (a wind tunnel's) `coefficients`: `drag`, `lift`, `side` and `moment` (x, y, z). The force
  is the momentum the gas gave the body's walls over the last step
  ([numerics.md](numerics.md#26-bodies-in-a-stream));
- `frame`, for a rotating frame that follows a binary (an orbit's `track`): `omega`, its angular
  velocity about z now, `omegaDot`, its rate of change, and `omega0`, where it started. For one
  released after a merger (the orbit's `release`), also `release`: `binaryLike`, how much the
  mass still looks like a binary (smoothed: 1 for two points, 0 for a body symmetric about z;
  the frame hands over between 0.7 and 0.3; null before the first step), and `atRest`, true
  once the frame has stopped and the run is inertial;
- `ranges`: the least and greatest value of each field, keyed by field name (the panel's
  δρ max for a liquid, |ρ / ρ₀ − 1|, comes from the density's);
- `symmetry`, for a mirror-symmetric run: the mirrored axes. The totals are then those of the
  whole configuration;
- `scale` and `redshift`, in comoving coordinates: the scale factor and 1 / a − 1;
- `radiation`, for a run with radiation transport: `eRad` (the radiation energy in the box,
  which `eTot` does not include), `eRadReduced` (the sum of E / r, r being each cell's reduced
  speed of light over the true one: `eTot + eRadReduced` is what the scheme conserves),
  `reducedShare` (the share of `eRad` in cells with r < 1), `luminosity` (the outward flux
  through the box's open faces now),
  `escaped` (the radiation energy that has left through the open faces since the run
  started, summed step by step) and `escapedAt` (the time `escaped` is the total at),
  `streaming` (cells at the streaming limit), `tooFast` (cells where the gas's speed plus its
  sound speed is over a third of the reduced speed of light), `maxReducedFlux` (the largest
  |F| / (c E)) and `maxTrad` (the largest radiation temperature, (E / a)^¼). The panel's L rad, the light curve, is the change of
  `escaped` between two readouts over the change of `escapedAt`;
- `magnetic`, for a run with a magnetic field: `eMag` (the magnetic energy, which `eTot`
  includes and `eInt` then does not), `divB` (the largest |div B| dx of any cell over
  the largest |B|), `bMax` and `betaMin` (the least plasma beta, 2p / B^2, of any cell).
  The panel's Diagnostics show the same four as E mag, |B| max, β min and ∇·B. For a disk
  preset (the magnetized accretion disk), `magnetic.disk` adds `alphaMaxwell` and
  `alphaReynolds` (the stresses over the pressure, summed over the disk's body) and `qz`,
  `qphi` (the mass-weighted MRI quality factors), shown as α M, α R, Q z and Q φ; see
  numerics.md section 21, Disk diagnostics.

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

### `impact() → object`

What a giant impact has left, for 3D runs with gravity (the `moon-forming-impact` preset). The
gas is sorted as the moon-forming impact calculations sort it (Canup 2004; numerics.md section
25): about the densest body, every cell is on the Kepler orbit its position and velocity give
it around that body's mass, and it is escaping if the orbit is unbound, the planet's if the
orbit's pericenter is inside the planet, and the disk's otherwise. The planet's mass and radius
are taken again from what fell to it until they settle. Gas thinner than 1e-7 of the densest is
left out.

| Field | |
|---|---|
| `planet` | `mass`, `radius` (that of a body of its mass at `meanDensity`), `center`, `velocity`, and `shares` |
| `disk` | `mass`, `angularMomentum` (about the planet, along the whole system's), `overRoche` (that over the angular momentum of the disk's mass on a circular orbit at the Roche radius, 2.9 planet radii), and `shares` |
| `escaping` | `mass` and `shares` |
| `moon` | The mass of the moon the disk would gather into: 1.9 L / √(G M a_R) − 1.1 M_D − 1.9 M_esc, with M_esc = 0.05 M_D (Ida et al. 1997; Kokubo et al. 2000); 0 if the disk is too compact |
| `shares` | Of that part's mass, the fraction that is each species (`fe85si15`: how much of it is iron) and its mean `dye` (the passive scalar, which tells the two bodies apart: 1 in the first and 0.5 in the second in `moon-forming-impact`) |
| `meanDensity` | The planet's mean density as used: the Earth's, 5513 kg/m³, in code units where the run has units, and half the densest gas's density where it has none |
| `units` | `time`, `mass`, `length` of a code unit in SI, or `null` |

Masses are in code units: Earth masses in `moon-forming-impact`, where the Moon is 0.0123.
While the bodies are still colliding the orbits are not Kepler's and the sorting means little;
it settles some hours after the last large clump has fallen back.

### `stats() → { t, steps, variables }`

The least, greatest and volume-weighted mean value of each variable over the whole box:
`rho`, `press`, `vel1`, `vel2`, `vel3`, `r0` (the passive scalar), and each species' mass
fraction. The whole state is read back from the GPU, which takes a moment on a large grid.

### `slice({ variable = 'rho', axis = 2, pos = 0.5, res, encoding = 'json' }) → object`

A variable sampled on a `res` × `res` grid, on the plane perpendicular to `axis` at the
fraction `pos` of the box. `res` defaults to the grid's size, at most 256, and can go up to
2048.

The result is `{ t, variable, axis, pos, res, axes, min, max, values }`. `values` is a flat
array: `values[j * res + i]`, where `i` runs along `axes[0]` and `j` along `axes[1]`.

With `encoding: 'base64'`, `values` is left out and the values come as `array` instead:

```js
array: { dtype: 'float32', shape: [res, res], order: 'C', endian: 'little', base64 }
```

`base64` holds the values' bytes as single-precision floats, in the same order as `values`
(row `j`, column `i`: numpy's C order). They are the same numbers: every variable is single
precision already. Through `page.evaluate` this is ten to twenty times faster than a list of
numbers at `res` 1024 or more, where a JSON slice takes seconds. In Node:

```js
const { array, ...rest } = await S('slice', { variable: 'rho', res: 1024, encoding: 'base64' });
const b = Buffer.from(array.base64, 'base64');
const values = new Float32Array(b.buffer, b.byteOffset, b.length / 4); // values[j * res + i]
```

and in Python, `np.frombuffer(base64.b64decode(a['base64']), '<f4').reshape(a['shape'])`.
An unknown `encoding` is an error.

A variable can be any of those listed for `stats`, or:

- `level`, the refinement level of the block at each point;
- `temp`, the temperature in code units, from the run's equation of state (with each cell's
  composition where the species set it), the density and the pressure. The view computes it
  from the energy instead; the two agree to within the single-precision rounding of the
  pressure;
- `tempK`, the same in kelvin, for a setup with physical units (an error otherwise).

On an adaptive mesh each sample is taken from the finest block covering it. In 2D the axis is
always 2.

### `probe(points) → { t, points }`

The state at a list of points `[x, y, z]`. Each point returns `rho`, `press`, `vel` (a 3-vector),
`r0`, `temp` (and `tempK` with physical units, as for `slice`), `level`, and `X`, the species'
mass fractions by name, when the run has species.

### `screenshot({ width?, height?, native?, overlays?, layout? }) → string`

The view as a PNG data URL. With no size, it is drawn at the canvas's size on the page. Give
`width` alone for a square image. The overlays the page draws on top of the canvas (color bar,
markers, labels) are not included, but `overlays: { colorbar: true, triad: true, status: true }`
paints a color bar, an axis triad and the status line onto the image, as a movie's frames have
them. With the
range set to automatic, it is fitted to the current data before the image is drawn.

`native` takes a plane (a 2D run, or the slice or projection view of a 3D run) at the grid's
own resolution: one pixel per cell of the finest level, drawn as the page draws it (contours,
block outlines, flat field lines) but without interpolation. `true` or `'box'` takes the whole
box; `'view'` the part of it in view (zoom and pan, left of the panel on the right), rounded out
to whole cells. A coarse cell of an adaptive mesh is a square of equal pixels. The map is capped
at the GPU's largest texture (`maxTextureDimension2D`, typically 8192 or 16384 a side), so a
larger grid comes out coarser than its cells. Row 0 of the image is the top of the box, so row
`j` holds `slice()`'s row `res - 1 - j`. `native` is not given with `width` or `height`, and is
an error in the volume view, which has no pixel per cell.

`layout` says where the overlays go: `'over'` paints them over the picture's edges, as in a
movie; `'beside'` puts them in margins around it, so no cell is covered (the status line in a
strip above, the color bar and the axes in a column to the right; their sizes follow the
picture's height, or 600 pixels for a smaller one). The default is `'beside'` with `native`,
`'over'` otherwise. Without overlays a native picture is the map alone. A picture larger than a
canvas can be (16384 a side, or 2^28 pixels) is an error that names its size: `'view'` takes
less.

### `record(options) → result`

Records the run from where it is as an MP4 movie (H.264, encoded by the browser's WebCodecs as
each frame is drawn; only the encoded stream is held in memory). The run is stepped as it goes,
and paused when the recording ends. Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `dt` | a 600th of the problem's dynamical time | A frame near every multiple of this much code time, so that by default one dynamical time (`tDyn` of each preset: a star's sqrt(R³/GM), a binary's period, a shear layer's roll-up) plays in 10 s at 60 fps. A frame is taken at the step boundary nearest its time: no step is shortened for it, so the recorded run is bit for bit the unrecorded one. A frame is then up to half a step early or late, and when a step is longer than `dt` the frame times it jumps over are skipped |
| `frames` | | Stop after this many frames, the first (at the start) included |
| `until` | | Stop when the clock reaches this time, landed on exactly as `run()` does |
| `timeoutMs` | | Stop after this much wall time |
| `every` | | Or a frame every this many steps |
| `fps` | 60 | Playback rate. It only sets the frames' timestamps: frame i is shown at i / fps seconds |
| `width`, `height` | the view's | The movie's size in pixels, rounded down to even numbers. Without either, the part of the view area the box takes when the recording starts, at the page's pixels, with a small margin and, when the axes or the color bar are on, room for them at the left and the right: the empty sides of a wide window are left out, and where the box runs off the view area the frame ends with it (1920 x 1080 when the page has no layout). `width` alone gives a 16:9 frame |
| `bitrate` | 0.2 bits per pixel per frame | Bits per second (25 Mbit/s at 1080p60) |
| `overlays` | all on | `{ colorbar, triad, status }`: the field's color bar with ticks in the displayed units; the x, y, z axes, which turn with the camera in the volume view; and the page's status line (grid, simulation time, point masses, mesh) in the upper left |
| `download` | true | Hand the file to the browser as a download |
| `base64` | false | Return the file in the result as `base64` |

It also ends at the run's end time, at ten minutes of movie, when the run is replaced, or on
`stopRecording()`; the file is finished with the frames recorded so far. The result is
`{ name, frames, seconds, width, height, t, steps, bytes, stopped, base64? }`, where `stopped` is
`frames`, `time`, `end`, `stopped`, `replaced` or `timeout`.

While it records, the page's canvas shows the movie's frames, letterboxed, and the camera does
not follow the pointer; view changes through the panel or `view()` still apply, and are
recorded. Checkpoints that are on are written as the run reaches them, as `run()` writes them;
a frame time never moves a checkpoint, and a checkpoint's shortened step is the one the run
takes unrecorded as well. A browser
without an H.264 encoder fails with a message saying so.

A run steps 2 to 3 times slower while it records: each frame is drawn, read back and encoded,
and the clock holds at each frame time. On the 128³ MRI disk (`mri-disk`), from t = 1.5 to
2.5 at 1920 x 1080 with 101 frames, on one desktop GPU, recording took 4.6 s against 1.6 s for the same run
unrecorded on a uniform grid, and 5.5 s against 2.2 s with one level of refinement.

### `stopRecording() → boolean` and `recordingStatus() → { frames, seconds, t } | null`

End the recording under way (it resolves as usual, with `stopped: 'stopped'`), and report its
progress. Neither waits behind other calls.

### `view(changes) → view`

Changes what is drawn, leaving the run alone. These keys are accepted (the table is written
from [src/render/view_options.ts](../src/render/view_options.ts) by `npm run gen`); a link
carries each under the key in the second column:

<!-- gen:view -->
| Key | In a link | Meaning |
| --- | --- | --- |
| `mode` | `mode` | `volume`, `slice` or `projection` |
| `field` | `field` | Which field is drawn: an index into `describe().fields` |
| `species` | `species` | Which species, for the mass-fraction field |
| `log` | `log` | Logarithmic color scale |
| `lo` | `lo` | Lower end of the color range; setting either end turns the automatic range off |
| `hi` | `hi` | Upper end of the color range |
| `sliceAxis` | `axis` | Slice and projection axis (0 x, 1 y, 2 z) |
| `slicePos` | `slice` | Slice position along its axis (0 to 1) |
| `minRho` | `rhocut` | Volume view: least density drawn, as a fraction of the way from the smallest to the largest on a log scale |
| `opacity` | `opacity` | Volume view: opacity of the fog |
| `yaw` | `yaw` | Volume view: camera angle about the vertical, in radians |
| `pitch` | `pitch` | Volume view: camera angle above the horizontal, in radians |
| `dist` | `dist` | Volume view: camera distance, in box lengths |
| `tx` | `tx` | Volume view: the point the camera orbits, in fractions of the box (0.5: the center) |
| `ty` | `ty` | The same, in y |
| `tz` | `tz` | The same, in z |
| `zoom` | `zoom` | Slice view: magnification (1 fits the box) |
| `cx` | `cx` | Slice view: the point at the center of the view, in fractions of the box |
| `cy` | `cy` | The same, along the other axis of the slice |
| `showBlocks` | `blocks` | Outline the blocks of an adaptive mesh |
| `contours` | `contours` | Contour lines (isosurfaces in the volume view) |
| `nContours` | `ncont` | How many contours |
| `fog` | `fog` | Volume view: draw the gas as fog (off leaves the isosurfaces alone) |
| `surfaceOpacity` | `isoa` | Volume view: how opaque the isosurfaces are |
| `exact` | `exact` | Volume view: isosurfaces of the field’s log interpolated between cell centers (off, the default: of the value smoothed by a cubic B-spline, which places a surface within the cells it crosses and takes away their terraces) |
| `light` | `light` | A white background instead of the dark one |
| `box` | `box` | Volume view: outline the box (on by default) |
| `fieldLines` | `flines` | A run with a magnetic field: draw field lines (on by default with a field). In 2D they are contours of the flux function |
| `fieldLineCount` | `nflines` | How many field lines |
| `fieldLineFloor` | `flfloor` | Field lines end where \|B\| falls below this fraction of the largest \|B\|, and start only where it is above twice that (default 0.01): lower it to follow a weak field farther out, such as to the surface of a star |
| `mirrorImages` | `mirror` | An octant (or 2D quadrant) of a mirror-symmetric run: show the whole configuration, the computed part with its mirror images (on by default) |
| `turnWithFrame` | `turn` | Volume view of a run in a rotating frame: turn the box at the frame’s rate, so the picture is the one a camera at rest sees (a rotor turning, its wake left behind) |
| `colormap` | `cmap` | A name from `describe().colormaps` |
| `tf` | `tf` | Volume view: the opacity curve, as `x:a` points separated by commas (2 to 16 of them; x a fraction of the color range, a the opacity, each 0 to 1), as in a link. The default is `0:0,0.25:0.02,0.6:0.25,1:0.9` |
| `autoRange` |  | Fit the color range to the data |
<!-- /gen:view -->

It returns the view as it then stands: every key above, and `colormap`, the color map in force.
`view({})` reads it without changing anything.

### `controls() → array`

Everything that can be set on a running page, in two parts.

First the controls of the side bar, each as `{ id, kind, label, title, value }`: the preset
menus, the preset's featured fields, run and time control, the view, checkpoints, the movie
settings, GPU memory. A select also has `options`, and a number field has `min` and `max`
where it sets them. Controls inside collapsed sections are included.

Then the settings of the run's setup document, which a person changes in the Setup dialog:
the grid, the gas, gravity, the boundaries, the mesh, the solver, and the fields of each part.
Each has `id` (its key in a link, such as `gamma` or `riemann`, or its path in the document,
such as `parts.0.soft`), `path`, `label`, `title`, `page`, `kind`, `value`, its range or
`options`, `apply` (what changing it costs: `live`, `shaders` or `restart`), and `unavailable`,
the reasons it cannot be used in this setup, where there are any. A field of a part also has
`part`, that part's `id` (or its kind).

### `control(id, value?)`

Sets one control, or presses a button when it is given no value, exactly as a person would.
It then waits for whatever the change starts, such as a rebuild of the shaders or a restart.

An `id` that is not a control of the bar is a setting of the setup document, by its link key,
its path, or the name it had when it was a control of the bar (`res`, `levels`, `visc`,
`bc-lo-0`, ...: old scripts keep working). The page then does the least that gets the run
there, as the dialog's Apply does: a uniform write, new shaders, or a fresh start.

This lets you change a run partway through, the way a user can: switch the Riemann solver,
the equation of state, gravity, viscosity, or a preset parameter, without starting from t = 0.
A few settings start the run again whenever they change, as their `apply` of `restart` in
`controls()` says: among them radiation transport (`rad`), implicit pressure (`psplit`,
`pcycles`) and the ANEOS `material`.

The GPU memory the simulation may take is the control `gpumem`, in GB: `control('gpumem', 4)`
sets it (it decides how many blocks the adaptive mesh can hold, and starts the run again), and
`control('gpumem', '')` returns to the guessed default. The page keeps it for later visits.
Changes that a person's edit would restart (initial conditions, resolution) restart here too.

Number fields of the bar hold values in the units the panel shows. For a problem with physical
units, call `control('units', 'code')` first if you mean code units. Settings of the setup
document are always in code units. The run stays paused afterwards, unless the control was
`run`.

### `tool(name, options?) → { tool, notice?, ... }`

Uses one of the page's editing tools, as a person would with the pointer. The tool is applied
right away, between steps, and logged as the page's own tools are (`actions()`). Positions are
fractions of the box, `[x, y]` or `[x, y, z]` (z defaults to the middle, and is ignored in 2D).
Velocities are in code units. The other settings mean what the tool's sliders mean, and
default to the tool's defaults. An option the tool does not take is an error.

| Tool | Options | What it does |
| --- | --- | --- |
| `mass` | `pos`, `radius` (0.06), `softness` (0.5), `amount` (0.25), `atRest` (true) | Adds `amount` times the densest gas's density at the center (a negative amount removes gas). With `atRest`, the new gas shares the momentum already there. |
| `velocity` | `pos`, `radius` (0.08), `softness` (0.5), `vel` | Sets the velocity, blending into the flow toward the edge. |
| `temperature` | `pos`, `radius` (0.06), `softness` (0.5), `factor` (1.5) | Multiplies the temperature by `factor` at the center. |
| `obstacle` | `shape`, then `center` and `half` (rect, ellipse) or `points` (polygon); `axis` (2); `erase` | A solid body. The shape is drawn in the plane normal to `axis` and cut through the box along it, as the pointer's outline is cut along the line of sight. The plane's coordinates are x and y for axis 2, y and z for axis 0, x and z for axis 1. |
| `field` | `pos`, `radius` (0.08), `softness` (0.5), `factor` (1.5) | Makes the magnetic field `factor` times stronger in the brush's hard core (below 1, weaker). |
| `pull` | `from`, `to`, `radius` (0.1), `softness` (0.7), `grip` (0.5) | Drags the field lines from `from` toward `to`, in the steps the tool would take frame by frame. Returns `frames`, how many. |
| `source` | `pos`, `vel`, `radius` (0.04), `softness` (0.5), `rate` (0.25), `fixed` (false) | Feeds gas in at the tool's Flux `rate`. Not fixed, it rides a massless particle set moving at `vel`. |
| `sink` | `pos`, `vel`, `radius` (0.04), `rate` (0.25), `mass` (0), `soften` (0.04), `fixed` (false) | Takes gas away. With a `mass` (a multiple of the gas in the box) it is a point mass. Returns `index`, its place among the run's masses. |
| `star` | `pos`, `vel`, `n`, `rhoc`, `radius`, `field` (none, dipole, random), `beta` | A polytropic star, settled into balance on the grid (3D only), with the tool's defaults for what is not given. Resolves once it has settled. |
| `clear` | `what` (sources, sinks) | Removes every source, or every sink, placed with the tools. |
| `query` | `pos` | The gas at a point, as `probe()` gives it. Changes nothing. |

The result may carry a `notice`, the same message the page shows for the tool. Examples are
"self-gravity switched on" for a star, and a full set of point masses for a sink.

```js
await shaderous.load({ problem: 'mhd-rotor', dim: 2, n: 256 });
await shaderous.tool('field', { pos: [0.3, 0.5], factor: 2 });
await shaderous.tool('pull', { from: [0.3, 0.5], to: [0.45, 0.6], grip: 0.4 });
await shaderous.run({ until: 0.2 });
```

### `actions() → log`

The session log of the run: what was done to it since it started, so it can be run again
(`replay`). The page's Save session log button saves the same, as a `.json` file. The log
holds:

- `query`: the setup the run started from, as the page's address has it (the view included);
- `replayable`: false for a run that started from an imported file, with `why`;
- `entries`: each edit, with `steps` (steps taken when it was applied, as `diagnostics()`
  counts them), `t`, and `action`. The kinds of action are:
  - `brush`: a brush's operation (mass, velocity, temperature, obstacle, field, pull);
  - `mass`: a point mass placed (a sink, a moving source's carrier, a pinned query's tracer);
  - `remove`: point masses taken away;
  - `sources`: the sources and sinks as set;
  - `massChanged`: the gas's mass told to self-gravity after a tool added or took gas;
  - `star`: a star placed;
  - `set`: settings changed mid-run, with `changes` (the link keys that changed) and the setup
    document after them;
  - `regrid`: a re-mesh of the adaptive mesh, as the steps the mesh took (`steps`: each block
    it refined or merged, in order, as `[what, level, coordinates...]`), a few dozen bytes;
  - `clock`: the clock's stop changed (`stop`, the time the step reaching it is shortened to
    land on: the end time, a checkpoint, a scripted run's `until`), or a movie's frame time
    (`frame`);
- `trace`: the run's diagnostics as the page read them back, about five times a second
  (steps, t, dt, mass, eTot, maxMach, floored, and for a magnetized run bMax, betaMin, divB).

What is logged is what reached the simulation. For example, a dab of mass is logged as the
density it added, worked out against the densest gas at the time. The pointer's path is not
logged. A log starts over when the run does (a load, Reset).

### `replay(log, { untilSteps? }) → { applied, actions, stopped?, ...state }`

Runs a session log again. It loads the setup the log started from, steps to each action, and
applies it. Then, with `untilSteps`, it runs on to that many steps. The adaptive mesh makes
none of its own refinement checks meanwhile. Instead it re-meshes where the log says, taking
the steps the log kept. That is what makes the replay exact: the step a live re-mesh lands
on depends on when its check was read back from the GPU. (Logs of the first version kept every
leaf's refinement indicator for each re-mesh instead, about 30 kB each; they still replay.) GPU runs repeat to the bit on the same
machine and browser, so the replay reaches the state the logged run reached. `stopped` says why
it ended early, if the run went another way, for example by reaching its end time before an
action logged later. The run is left paused, its log the replay's own.

This turns "it broke when I did this" into something that can be looked at. Save the session
log, replay it up to just before the trouble, and inspect the state with `slice`, `probe` or
`screenshot`.

Two parts of a live run are not replayed exactly. Per-block timesteps (`tbins`) pause
themselves for a while after a read-back of their saving, and that pause depends on timing.
A change to the equation of state mid-run is replayed through the setting, not the conversion
the page made.

### `athdf(opts?) → { name, time, bytes, base64 }`

The state of the gas as an `.athdf` file, base64-encoded: HDF5 in Athena++'s layout, which yt
and Athena++'s own tools read. The file also carries the setup, so `loadAthdf` can continue
the run exactly.

A file of more than a few hundred megabytes is past the longest string the browser can hand
over. With `{ chunked: true }` the page keeps the file and returns no `base64`; read it out
with `athdfChunk(offset, length) → { offset, end, base64 }`, bytes `offset` to `end`, until
`end` reaches `bytes`. The page lets the file go once its last byte is read. The runner and the
MCP server's `save_athdf` read a file this way, 64 MB at a time, when the page is not served
from the source (the built page of the npm package).

A file of several gigabytes cannot be made in the page at all: a Chromium page holds some 8 GB
of arrays in all, and the state, the file and the copy made while the file grows are each a few
gigabytes on a large adaptive mesh. So the runner and the MCP server, when they serve the page
from the source, take the state instead and write the file themselves:
`snapshotOut() → { name, header, arrays }` holds the state in the page and gives its fields but
the arrays, and each array's `key` (`levels`, `locations`, `faces`, `data.<variable>`), `type`
(`int32` or `float32`) and `length`; `snapshotChunk(key, offset, length) → { offset, end, base64 }`
reads an array's bytes out; `snapshotDone()` lets the state go. Loading goes the other way:
`snapshotIn(header, arrays)` sets out the arrays, `snapshotPut(key, offset, base64)` fills them,
and `loadSnapshot(opts?)` starts the run from them, as `loadAthdf()` does from a file (`opts` as there).

### `athdfPut(offset, base64, total) → { got, total }`

One piece of an `.athdf` file for `loadAthdf()`: its bytes from `offset`, base64-encoded, of a
file of `total` bytes. Send the pieces in order from offset 0, then call `loadAthdf()` with no
argument to start the run from the whole file. The runner's `--from` and the MCP server's
`load_athdf` send files this way, 48 MB at a time.

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
- **Large results are slow to bring out of the page.** `page.evaluate` copies a list of
  numbers one by one: a JSON `slice` at `res` 2048 takes about 13 s to cross, against under a
  second with `encoding: 'base64'`.
- **A rotating frame's velocities are the frame's.** With `physics.rotation` set, every
  velocity the API returns (diagnostics, momenta, slices, probes) and every velocity in an
  `.athdf` file is relative to the turning frame, and `eTot` is the Jacobi integral.
- **`play()` uses the page's own loop.** In headless mode that loop is off, so step with `run`.
- **It is one page and one run.** For runs in parallel, open several pages. Each one takes its
  own GPU memory.
