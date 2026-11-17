# Plan: the setup dialog

Status: done, 2026-11-17 (built 2026-11-16 in commits 7596e14, bf702a3 and the one after).
What differs from the text below:

- Section 7 was already done when this was written: a run holds 32 point masses (5fe9aed).
- A setup that is no preset runs under the problem id `custom` (`src/presets/custom.ts`), so
  everything that finds a problem by id finds it as it finds a preset. `SimConfig` did not
  gain a `setup` field: `Problem.setup` is the document it was compiled from.
- Preset parameters are the featured options whose paths are in the parts (`featured` entries
  carry the old id, label and range); the run still holds their values in `SimConfig.params`.
- The disk part has two profiles, `thin` and `balanced`, one per preset it came from: one
  formula could not start both bit for bit.
- A downloaded preset is the resolved, whole document, not the repository's file with its
  `extends`.
- No rule ties a magnetic field to a frozen mesh: the field is re-meshed with it.
- Added since, on request: a point mass held on its orbit (`onOrbit`, a live toggle); disks
  about stars; and a disk's density in radius and height as a formula (`radialFn`,
  `verticalFn`; `src/setup/expr.ts`) or a table of points. Formulas lift the non-goal below in
  one place: they are parsed into arithmetic over a fixed list of functions and never run.
- Finished 2026-11-17: the dialog has controls for `physics.speciesGamma`,
  `physics.toolScale`, both forms of burning and units by density and velocity. A group of
  the option table may come in several shapes (`GroupForms`), told apart by its keys; the
  dialog offers them as a menu. A unit's `per` is shown beside its value, not edited. Fuel to
  ash burning is refused where the species are nuclides that set the equation of state.
- Tests: `masses-many` is the existing `point-mass-many`. `setup-restart` and `setup-triple`
  are page tests in `tests/gpu/setup.spec.ts`: a restarted custom run matches the
  uninterrupted one (gas bit for bit, masses to 1e-8), and three stars at 21 cells across hold
  their peak density to 5% and binding energy to 2% for a dynamical time. Writing the restart
  test found that a saved file's time and step count were one step behind its gas, and that
  `run()` took one step too many after a load or a restart; both are fixed.
- Setup time (section 12), 512³ with 3 levels of refinement, from `load()` to a run that can
  step, two runs each, in seconds: the hand-written problems (cdc4a29, the parent of 7596e14)
  against the compiled documents (this commit). The compiled setup is no slower; the runs
  differ by up to half from one to the next on either build.

  | Preset | Hand-written | Compiled |
  |---|---|---|
  | `sedov` | 4.9, 8.7 | 5.5, 8.3 |
  | `mri-disk` | 16.1, 14.2 | 15.9, 13.4 |
  | `tidal-disruption-full` (settling 1.3 to 1.7 s of it) | 11.9, 10.1 | 10.2, 10.7 |

## 1. Goal

One dialog holds every setting of the simulation: the grid, the physics, the boundaries, the
initial conditions and the objects in the box. The left-hand bar keeps what concerns
choosing and watching a run: the preset and variant menus, the preset's few featured
settings, time control, the view, diagnostics and performance.

In the dialog a user assembles a run from parts: any number of point masses, stars and
disks, gas regions, obstacles, a magnetic field, in every combination that makes sense
together. Every problem the menu offers today becomes a **preset**: a JSON file in the
repository that sets the dialog's options to the values that problem needs. A setup a user
makes is the same kind of file. Either can be saved, loaded on another machine, shared as a
link, and passed to the headless runner and the MCP tools.

Four rules shape the design:

1. **One document describes a setup.** Everything the dialog can set is a field of a
   `SetupSpec`: plain JSON, versioned, no code in it. Presets and custom setups differ only
   in where the file lives.
2. **A setup compiles to the solver's existing input.** `compileSetup(spec)` returns a
   `Problem` and the overrides of a `SimConfig`, so the solvers, the renderer, the mesh,
   settling, the diagnostics and the tools need no second path.
3. **Every option declares what it cannot be combined with.** The declarations are data,
   one list per option. The dialog greys out an option whose conditions are not met and
   names, in its tooltip, the option that disables it.
4. **Presets are not special.** The dialog opens a preset as it opens any file; nothing a
   preset does is out of a user's reach.

### Non-goals

- Free-form expressions or user code in a setup (a density given as a formula, custom
  WGSL). A file from a stranger must be safe to open.
- New physics. The dialog combines what the code has. Parts that need new machinery (the
  galaxy potential, particles) join it when their own plans land.
- Recording what the editing tools do during a run. "Export state" already keeps that.

## 2. What exists and what is in the way

Used as is:

- `Problem` (`src/problems/types.ts`) and `SimConfig` (`src/sim.ts`) already carry every
  setting. They stay, as what a setup compiles to.
- Building blocks inside the problem files: `laneEmden` and `polytropeScales`, `whiteDwarf`
  and `dwarfsSettling`, `binaryMasses`, `collisionOrbit`, the disk of `mri_disk.ts`
  (`diskState`, `loopPotential`), `dipoleField` and `randomField` (`mhd/star_fields.ts`).
- `configFor`, `resolveGrid`, `amrFits`: a problem and a grid into a run, within the memory
  budget.
- Links (`ui/link.ts`): decoding that clamps, rejects and reports, never trusts.
- `.athdf` files carry the settings of the run that wrote them (`SnapshotMeta.query`).
- The three ways a change reaches a run: `applyParams()` (a uniform write),
  `rebuildPipelines()` (shaders reassembled, the run carries on), and `load()` /
  `restartRun()` (new initial conditions).

In the way:

| Obstacle | Where | What it needs |
|---|---|---|
| 45 presets are TypeScript objects holding functions | `src/problems/*.ts` | Each as JSON over a library of named, parameterized parts (section 5) |
| Each preset has private parameters (`Problem.params`: q, beta, ecc…) | `ProblemParam`, the `p.*` link keys | They become ordinary fields of parts; old link keys map onto them |
| Simulation settings are built into the side bar | `src/ui/app.ts`, 3800 lines: Grid, Hydro, Cooling, Diffusion, Boundaries, Gravity, the problem block | They move to the dialog, generated from the option table |
| Which settings conflict is spread over `syncControls` and the handlers as hidden rows and ad hoc checks | `app.ts` | The option table (section 4) |
| A problem is found by its id everywhere | `problemById` in `ui/link.ts`, `ui/app.ts`, `amr/init_worker.ts`, `amr/init_pool.ts`, restart, `ui/api.ts` | By its spec instead; ids remain as names of preset files |
| Workers rebuild a problem from its id; any other is set up on the main thread | `init_pool.ts` `shareable()` | Send the spec; the worker compiles it |
| At most 4 point masses, written out one by one | `MAX_MASSES` in `sim.ts`, `Masses.list` in `gravity/bh.wgsl`, `bh_cell.wgsl`, the reduction record, the sink panel | Section 7 |
| One damping zone, as WGSL written by hand per preset | `Problem.damping`, `binary.ts`, `mri_disk.ts` | WGSL generated from a disk's numbers |
| GPU settling handles a limited number of stars | `SETTLE_MAX_STARS` in `amr/amr_sim.ts` | A limit in the option table, or a larger GPU limit |
| A magnetic field is either a potential or face values, not both | `MhdSpec` | A sum of potentials plus a uniform part on the faces |

## 3. The setup document

New `src/setup/`: `spec.ts` (types and defaults), `options.ts` (the option table),
`codec.ts`, `compile.ts`, and one file per kind of part under `src/setup/parts/`.

```jsonc
{
  "format": "shaderous-setup",
  "version": 1,
  "id": "binary",                    // presets only: the name links and tests use
  "name": "Binary accretion",
  "family": "binary", "variant": "Standard", "group": "Point masses",   // its place in the preset list
  "blurb": "Two point masses orbiting inside a gas disk: …",
  "domain": { "ndim": 2, "box": 12, "n": 512, "amr": 3, "supports2d": true,
              "bc": { "lo": [1, 1, 1], "hi": [1, 1, 1] },
              "units": { "length": { "pc": 0.01 }, "mass": { "Msun": 2e5 } } },   // absent: scale free
  "physics": { "eos": "barotropic", "gamma": 1, "K": 0.01,
               "gravity": "off", "G": 1, "gext": [0, 0, 0],
               "cooling": "off", "viscosity": 1e-3, "conductivity": 0,
               "riemann": "hllc", "limiter": "koren", "cfl": 0.8,
               "floors": { "rho": 1e-9, "p": 1e-14 }, "fluff": { "rho": 0, "u": 0, "v": 0 },
               "frame": "masses" },
  "mesh": { "refineHi": 0.8, "refineLo": 0.2, "refineMasses": true, "refineDense": 0, "refineRho": 0, "timeBins": 0 },
  "background": { "rho": 1e-8, "p": 1e-10, "v": [0, 0, 0] },
  "parts": [
    { "kind": "mass", "id": "a", "gm": 0.5, "soft": 0.06, "sink": 0.06 },
    { "kind": "mass", "id": "b", "gm": 0.5, "soft": 0.06, "sink": 0.06 },
    { "kind": "orbit", "of": ["a", "b"], "separation": 1, "ecc": 0, "incl": 0 },
    { "kind": "disk", "about": ["a", "b"], "rIn": 2.5, "rOut": 5, "sigma0": 1e-4, "slope": 0,
      "cs": 0.1, "damp": { "from": 4.5, "rate": 3 } }
  ],
  "run": { "tEnd": null, "checkpointDt": 1, "tDyn": 6.2832, "startPaused": false },
  "view": { "field": 0, "log": true, "mode": "slice", "axis": 2 }
}
```

Every key is optional but `format` and `version`; what is absent takes the default of the
option table. A preset file therefore lists only what makes it that preset.

### Parts

| Kind | Fields | Reuses |
|---|---|---|
| `mass` | G M, position, velocity, softening, sink radius and rate | `PointMass` |
| `orbit` | A helper, not an object: puts two bodies (masses or stars, or a body and a group's center of mass) on a Kepler orbit from separation or pericenter (or the tidal radius over it), eccentricity, inclination and phase | `binaryMasses`, `collisionOrbit`, the tidal disruption's orbit |
| `star` | Polytrope (index, mass, radius) or cold white dwarf (mass); position, velocity, dye value, composition; an optional field inside (dipole or random, with beta and seed) | `laneEmden`, `whiteDwarf`, `Settling`, `star_fields.ts` |
| `disk` | The bodies it orbits, radii and tapers, surface density and slope, sound speed (or H/R at a radius), inclination, a damping zone, field loops | The disk module (6.3) |
| `region` | Sphere, box, slab or half-space with a state (density, pressure or temperature or energy, velocity, dye, composition) that replaces or adds to what is under it; a smooth edge; seeded noise | New, small |
| `obstacle` / `inflow` | The same shapes, marking cells solid or held fixed | `Prim.solid`, `Prim.fixed` |
| `field` | A uniform field | `MhdSpec` |
| `flow` | A named initial condition with its parameters, for what shapes cannot say: the Kelvin-Helmholtz shear layer with its seeded modes, the Rayleigh-Taylor interface, a Jeans or Alfvén wave, Orszag-Tang, the rotor, the field loop, the detonation's ZND profile, the Evrard sphere | The `init` functions of today's problem files, moved to `src/setup/parts/flows/` |

A `flow` is what lets every preset be a JSON file without formulas in it: the file names a
built-in function and gives its numbers. A flow declares, like any option, what it cannot
be combined with (most fill the whole box, so they exclude one another and a background).
Presets that are only shapes (Sod, Sedov, the shock bubble, the forward step, the sphere)
use `region` and `obstacle` and no flow, and the conversion (section 9) prefers that
wherever the result is bit for bit the same.

Positions and sizes are in code units of the setup's own box. With `units` given, the
dialog shows and accepts physical values beside them, as the binary's fields do today.

## 4. The option table

`src/setup/options.ts` lists every option once. The dialog's forms, the conflict checks, the
defaults, the codec's ranges, the link keys and the scripting interface's list of controls
are all read from it.

```ts
interface Option {
  path: string;                 // where it sits in the spec: 'physics.gravity', 'parts.star'
  label: string; tip: string;
  page: 'domain' | 'gas' | 'gravity' | 'boundaries' | 'mesh' | 'solver' | 'parts' | 'run';
  control: NumberControl | SelectControl | CheckControl | PartControl;   // range, step, unit, choices
  apply: 'live' | 'shaders' | 'restart';     // what changing it costs a running simulation
  excludes?: Rule[];            // cannot be on (or have this value) together with these
  requires?: Rule[];            // can only be on together with these
  linkKey?: string;             // its key in a link ('gamma', 'bc', …), today's names kept
}
interface Rule { option: string; is?: Value | Value[]; reason?: string }
```

- A rule may be attached to one choice of a select (`physics.gravity = periodic`), to a
  checkbox, or to a kind of part (`parts.star`: whether one may be added).
- `excludes` is read both ways: declared on either option, it disables whichever of the two
  the user reaches second.
- The tooltip of a greyed-out option is made from the rule: "Not available with Dimensions:
  2D", or the rule's own `reason` where the bare names would not explain it ("a star's
  gravity is not periodic").
- A unit test checks the table itself: every rule names an option that exists, and no
  preset file breaks a rule.

Rules known today, to be entered as declarations:

| Option | Excludes / requires |
|---|---|
| Self-gravity (periodic or isolated) | requires 3D |
| Self-gravity, periodic | requires every face periodic |
| Self-gravity, isolated | excludes periodic faces |
| A periodic face | requires the opposite face periodic |
| Star | requires 3D, isolated self-gravity |
| White dwarf star | requires the degenerate or Helmholtz equation of state; excludes polytropic stars |
| Polytropic star | requires an equation of state with an energy; excludes white dwarfs |
| Point mass, star, disk | exclude periodic faces |
| Disk | requires a mass or star to orbit |
| Damping zone | requires a disk; excludes a second damping zone |
| Frame follows the masses | requires a point mass |
| Cooling, conduction, burning | require an equation of state with an energy |
| Network burning | requires its 13 species and physical units |
| Local timesteps, refinement thresholds, refine around masses | require an adaptive mesh |
| Jeans floor, V-cycles, energy-conserving work | require self-gravity |
| Magnetic field | excludes mesh refinement that changes during the run (the mesh stays as first refined) |
| A whole-box `flow` | excludes other whole-box flows and a background |

What a list of options cannot express stays as checks on values, run by the same code path
and shown in the dialog's issues list: bodies that overlap or leave the box, a softening
under two cells, a disk heavier than a tenth of what it orbits, a polytrope whose gamma is
below 1 + 1/n, the grid against the GPU memory budget.

## 5. Presets

- Each of today's problems is a file `src/presets/<id>.json`, in the format of section 3.
  `src/presets/index.ts` imports them (Vite bundles JSON), compiles each, and builds what
  `MENU`, `FAMILIES`, `PROBLEMS` and `problemById` provide today, from the files' own
  `group`, `family` and `variant`. Preset ids do not change, so links, state files, the
  validation suite and the tests keep working.
- Picking a preset in the dialog loads its file into the editor: every option shows the
  value the preset sets. Changing anything makes it "Binary accretion (modified)"; Save
  writes the user's own file.
- **Download.** The dialog's preset list has a download button per preset (the file as it
  is in the repository) and "Download all" (a zip, or the folder's link on GitHub).
- **Preset parameters.** `Problem.params` goes away. The binary's mass ratio and
  eccentricity are fields of its `orbit` part; the tidal disruption's impact parameter is
  the orbit's pericenter in tidal radii; a field's beta and seed belong to the star. Old
  links' `p.q`, `p.ecc`… are mapped to those paths by an alias list in the preset file.
- **Featured fields.** A setup file may list the options that matter most to it
  (`"featured": ["parts.2.ecc", "parts.0.soft", "physics.viscosity"]`): any path of the
  spec, a part's field or a general setting. They are shown in two places, from the same
  list and the same option table: at the top of the dialog, above its pages, and in the
  side bar under the preset menus, where `Problem.params` are shown today. Each is the
  same control as on its own page, with the same greying and the same mark of what
  changing it costs. A user's own setup can feature fields too (a star button beside any
  field in the dialog), and the list is saved in the file.
- **Variants** are separate files that share most of their content. A file may name a
  parent (`"extends": "tidal-disruption"`) and give only the differences; the codec
  resolves it on load, and a downloaded file is always the resolved, whole document.
- `src/problems/*.ts` shrinks to the libraries the parts use (`lane_emden.ts`,
  `white_dwarf.ts`, `settle.ts`). The functions that are today's `init` bodies move to
  `src/setup/parts/`.

## 6. Compiling a setup

`compileSetup(spec)`, deterministic and free of the DOM, so the page, a worker and Node get
the same run from the same document.

### 6.1 Wiring

- `SimConfig` gains `setup: SetupSpec`. Wherever a problem is found by id today, the spec
  is compiled instead (a preset id is looked up to its spec first).
- Workers: `InitRequest` carries the spec, `init_worker.ts` compiles it, `shareable()`
  accepts every setup that has no restart.
- The compiled `init` paints in layers: a flow or the background, then disks, then
  regions; stars are left to settling, as the collision presets leave theirs. Each part
  precomputes what it needs (a Lane-Emden table, a disk's constants), so a cell costs a few
  lookups.
- An option whose `apply` is `live` on a part's field (a softening) is taken up by the run
  where the masses are now, as `ProblemParam.live` does today.

### 6.2 Stars

`settle` returns one `SettleStar` per star part. Dye values are spread over (0, 1] so the
passive scalar tells the stars apart, as the collision's 1 and 0.5 do. To check: how many
stars `SETTLE_MAX_STARS` allows on the GPU and what raising it costs; the option table
holds that limit on an adaptive mesh.

### 6.3 Disks

New `src/setup/parts/disk.ts`, from the two presets' disks: a midplane density with tapers,
a vertical profile in balance with the softened potential of what it orbits, rotation from
the enclosed mass and the pressure gradient, tilted by a rotation of the coordinates. About
several bodies it uses their total mass at their center of mass, flattened inside their
orbit as `binary.ts` does. It writes the damping zone's WGSL from the same numbers
(formatted as float literals by the compiler, never taken as text from a file), so the
target and the initial state cannot drift apart. The binary and the magnetized disk are the
test: both must start bit for bit as before.

Several disks add their densities and share momentum by mass. The disk diagnostics
(`Problem.disk`) stay tied to the first mass.

### 6.4 Magnetic fields

Potentials of the star and disk parts and of flows are summed (their curl stays
divergence-free). A uniform field is set on the faces, since its potential is not periodic.
`MhdSpec` gains a form that takes both.

## 7. More than four point masses

"Any number" in practice means a limit set by cost, not by four hand-written copies:

- The mass count becomes a number the shaders are assembled with. `bh_cell.wgsl`'s unrolled
  blocks, and the others written out mass by mass, are generated for the run's count, as
  `reduceDefs()` generates the reduction record. No run-time-indexed function arrays (the
  Firefox device loss of c14460e).
- `Masses.list` in the uniform grows to the build's maximum; `MASS_FLOATS`, `MASS_RED`,
  `packMasses`, `MassReader`, the sink readouts and the diagnostics CSV follow the count.
- The reduction record is 1 + 10 per mass: 161 floats at 16 masses. To measure first: the
  point-mass reduction's time and the gas update's time against the count, in Chrome and
  Firefox. If the record is the cost, sinks become optional per mass (a mass without a sink
  needs 3 channels, not 10).
- First limit: 16, raised if the measurements allow.

Runs with four or fewer masses must give the same results bit for bit, and compile no
slower.

## 8. The dialog and the side bar

### 8.1 What stays in the side bar

| Stays | Moves to the dialog |
|---|---|
| The preset and variant menus, the blurb, and a **Setup…** button | |
| The setup's featured fields (section 5), under the menus | Every other field of the parts |
| Run, Step, Reset, Record, Share, Autoplay, Autoload | |
| Time: end time, checkpoints | CFL, local timesteps |
| Movie | Grid: dimensions, resolution, refinement and its thresholds |
| View, including units of the readouts | Hydro: equation of state, solver, limiter, tenuous-gas limits |
| Diagnostics, the sinks' readouts | Cooling, Diffusion |
| Performance: GPU memory, pass timings | Boundaries and the damping zone |
| Export and Import state | Gravity |

The editing tools on the rail stay where they are: they act on the running gas.

The preset menu works as the problem menu does now: picking a preset or a variant loads
its file and starts it (or, with Autoload off, waits for Run). It lists the presets in
their groups and then the user's library. A setup that is neither (a preset since changed,
or a file just opened) shows as "Binary accretion (modified)" or by its own name, with the
variant menu hidden when it has no variants.

A featured field changed in the bar goes the way its option's `apply` says, as a preset
parameter does today: a `live` one takes effect at once, a `restart` one starts the run
again or waits for Run with Autoload off. The bar and the dialog edit the same spec, so a
value changed in one shows in the other.

### 8.2 Layout

A `<dialog>` in new files `src/ui/setup/` (`dialog.ts`, `forms.ts`, `preview.ts`,
`library.ts`, `setup.css`). It edits a copy of the spec; the run is untouched until Apply.

- **Left: presets and files.** The presets in their groups, each with its download button;
  the user's library; New, Open file.
- **Middle: the featured fields, then pages.** The setup's featured fields sit in a strip
  above the pages, whichever page is open. The pages: Domain and grid, Gas (equation of state, cooling, diffusion), Gravity,
  Boundaries, Mesh, Solver, Objects (the parts list with its Add menu: point mass, star,
  disk, gas region, obstacle, inflow, field, orbit), Run. Every field is generated from the
  option table. A greyed-out field keeps its place, with its reason as the tooltip and, on
  touch screens where there is no hover, as a line under the field when tapped.
- **Right: a preview.** Top and side views drawn on a 2D canvas from the spec alone: the
  box and its boundary kinds, each body with its softening and sink radius, stars to scale,
  disk extents, regions, and the paths the bodies would take as point masses for a chosen
  time (a small float64 N-body integration). Bodies can be dragged; the form follows. No
  GPU work, so it cannot disturb the run behind the dialog.
- **Bottom:** the issues list, the memory the grid needs against the budget, and Save file,
  Copy link, Save to library, Cancel, Apply.

On a phone the three columns become tabs.

### 8.3 Applying changes

Apply compares the edited spec with the running one and does the least that is needed, as
the handlers do today: only `live` options changed, a uniform write and the run carries on;
any `shaders` option, the pipelines are rebuilt (the equation of state through `switchEos`,
which carries the pressure across); any `restart` option, the run starts again. Each field
carries a small mark of its kind, and the Apply button says what it will do ("Apply",
"Apply and recompile", "Apply and restart") before it is pressed.

The run goes on behind the dialog, which can stay open after Apply, so a live setting
(viscosity, a cooling rate) can still be tuned while watching, though less directly than in
the bar today. See the open decisions.

## 9. Saving, loading and sharing

- **File.** `name.shaderous.json`: the whole, resolved spec, pretty-printed. Save and Open
  in the dialog; dropping the file on the page opens the dialog with it.
- **Library.** Named setups in `localStorage` (`shaderous.setups`), listed under the
  presets, with rename, delete and export. Files are the durable copy; the dialog says so.
- **Link.** As today a link names a preset and what differs from it, option by option,
  under each option's `linkKey`: every link made so far still opens. A setup that differs
  from its preset in its parts, or has no preset, puts the spec in `?setup=`, minified,
  deflated (`CompressionStream`) and base64url-encoded. If the link would pass about 8000
  characters, Copy link says so and offers the file.
- **State files.** `SnapshotMeta` gains `setup`, the whole spec, beside `query`. A `.athdf`
  then restarts as itself whatever the preset files have become since. Files without it are
  read through `query` as now. `SNAPSHOT_FORMAT` goes to 2.
- **Scripting.** `window.shaderous.load()`, the headless runner and the MCP `load` tool
  take a preset id with overrides (as now) or a spec (`--setup file`). `controls` and
  `control` are served from the option table; `describe` returns a preset's spec and the
  table's schema, so an agent can write a setup; `state` returns the running spec.
- **Versions.** `version` counts changes to the spec. `codec.ts` keeps one migration per
  step and applies them in order; a file from a newer version loads with a warning and its
  unknown parts listed. Unknown keys are dropped and listed, values out of range are held
  to it and listed, a wrong type is an error: the discipline of `decodeLink`. The rule of
  `PROBLEM_ALIASES` applies: add, never remove.

## 10. Tests

Unit (vitest):

- Option table: every rule names an existing option; rules read the same both ways; every
  preset file satisfies every rule; every option has a default the codec accepts.
- Codec: save then load returns the same spec; `extends` resolves; every migration;
  malformed and hostile input (wrong types, huge arrays, deep nesting) ends in a clean
  report.
- **Presets against today's code**, the gate of section 11's step 2: for every preset, the
  compiled `SimConfig` equals `configFor(oldProblem)` field by field, and `init`, the
  masses, settling and the magnetic potential agree at sampled points, in 2D and 3D. The
  old objects are kept in the test tree until the last preset has passed, then deleted.
- Compiler: the same spec compiled twice, and in a worker, gives identical leaf states
  (extend `init_cpu.test.ts`); generated damping WGSL against the float64 disk.
- Links: every key of today's `link.test.ts` still decodes to the same run; `p.*` aliases;
  `?setup=` round trip.

GPU:

- The validation suite and the existing GPU tests, unchanged in their numbers.
- `masses-many`: 8 and 16 masses with gas; momentum of gas and masses conserved as well as
  with 2; the four-mass presets bit for bit.
- `setup-restart`: export a custom run, import it, continue; identical to the
  uninterrupted run.
- `setup-triple`: three stars settle and hold for a dynamical time.

Page (Playwright): the side bar holds no simulation setting but the preset menus and the
featured fields; pick a preset and a variant in the bar and see it load; change a featured
field in the bar and see the same value in the dialog; open the dialog, pick a preset, see a conflicting option greyed out with its reason, change a live option and see
the run carry on, change a restart option and see it start again, save the file, reload,
open the file, and get the same link; download a preset and get the repository's file.

## 11. Order of work

1. **The option table, the spec, the codec and the compiler** for every setting `SimConfig`
   and `Problem` hold today, with the parts `mass`, `region`, `obstacle`, `field` and
   `flow`. No dialog. A handful of presets as JSON (the empty box, Sod, Kelvin-Helmholtz,
   the polytrope, the binary) beside their old objects, passing the comparison test.
2. **Every preset as JSON**, family by family, each through the comparison test; the disk
   module (6.3) and the star and orbit parts come in here, since presets need them.
   `PROBLEMS` is then built from the files, the spec goes to the workers, links and state
   files, and the old objects are deleted. **Gate:** validation suite and GPU tests
   unchanged.
3. **The dialog**, with every page and Apply, the presets list with downloads, file save
   and open, the library. In the same change the side bar loses its simulation sections,
   its preset menus are built from the preset files, and its parameter block becomes the
   featured fields, so no setting is ever lost between the two.
4. **The preview** and dragging.
5. **More than four masses** (section 7), after its measurements.
6. **Composition beyond the presets**: several disks, stars with disks, regions with
   species, and the rules they turn up.
7. Documentation: `docs/api.md`, `docs/headless.md`, `docs/architecture.md`, and
   `docs/teaching.md`, whose activities name controls in the side bar today.

Steps 1 and 2 change nothing a user sees and can ship as they pass. Step 5 is independent
of 3 and 4.

## 12. Risks

- **Converting 45 presets without changing a result.** This is most of the work and all of
  the danger: a default that differs in the last digit changes a validation number. The
  comparison test, preset by preset, with the old objects alive until the end, is the
  defense. Presets that resist being shapes become flows, which is always possible since a
  flow is the old function under a name.
- **Live tuning gets a step further away.** Viscosity, a cooling rate or the CFL number
  are one click in the bar today and will be behind a button, unless the setup features
  them. The dialog stays open over a running simulation and applies live options at once;
  if that is not enough in use, see the open decisions.
- **Featured lists that grow.** If every preset features a dozen fields the bar is back
  where it started. Keep presets to about six, the number the largest parameter block has
  today.
- **Teaching.** `docs/teaching.md` and anyone's habits point at the side bar. The preset
  menus stay there, and the featured fields keep a preset's main numbers beside them.
- **The compiled problem is slower to set up than a hand-written one.** Precompute per
  part, and measure a 512³ adaptive setup against today's time for the same preset.
- **Combinations nobody has run.** Presets are each tuned (floors, tenuous-gas limits,
  damping, refinement densities); free combinations will find untuned corners. Defaults
  derived from the parts (ambient density, `refineRho`, softening from the cell size) and
  the issues list carry most of it. Expect the rule table to grow from use.
- **Spec churn.** Migrations from version 1 on, even before release, keep saved files and
  the 45 preset files in step.

## 13. Open decisions

| Decision | Options | Leaning |
|---|---|---|
| The dialog over a running simulation | Blocking (`showModal`); or a panel that floats and lets the view be used behind it | Blocking, kept open after Apply; revisit if live tuning suffers |
| Live options: apply at once, or on Apply | As typed, as the bar does now; or with everything else on Apply | As typed for `live` options, Apply for the rest |
| Featured fields a user adds | Saved in the setup file only; or also remembered per preset in the browser | In the file only |
| Where GPU memory lives | Performance, in the bar (it describes the machine, and is not saved in a setup); or the dialog's grid page | The bar |
| End time and checkpoint interval | Bar only (time control); or also in the spec's `run` so a preset can set them | Both: the preset sets the starting value, the bar changes it |
| Variants | `extends` a parent file; or whole files with repeated content | `extends`, with downloads always resolved |
| Limit on point masses | 8, 16, or sinks optional and more | 16, decided by the measurements of section 7 |
| Disks about a star | Only about point masses; or about stars too | Masses first |
| File extension | `.shaderous.json`, `.json` | `.shaderous.json` |
