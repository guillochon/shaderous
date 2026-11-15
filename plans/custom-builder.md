# Plan: a custom setup builder

Status: draft, 2026-11-15. Nothing here is built yet.

## 1. Goal

A dialog in which a user assembles a run from parts instead of picking a preset: any number
of point masses, stars and disks, gas regions, obstacles, a magnetic field, the boundaries
and the physics, in every combination that makes sense together. The setup is one plain
document that can be saved to a file, loaded on another machine, shared as a link, kept in a
library in the browser, and passed to the headless runner and the MCP tools.

Three rules shape the design:

1. **One document describes a setup.** Everything the builder can do is a field of a
   `SetupSpec`: plain JSON, versioned, no code in it. The dialog is one editor of that
   document; a text editor is another.
2. **A setup compiles to an ordinary `Problem`.** `compileSetup(spec)` returns the same
   object a preset file exports, so the solvers, the renderer, the mesh, settling, the
   diagnostics and the tools need no second path.
3. **Conflicts are prevented, not reported after the fact.** A pure function judges a spec
   and names what clashes; the dialog greys out a choice that would clash and says why in
   its tooltip.

### Non-goals

- Free-form expressions or user code in a setup (a density given as a formula, custom WGSL).
  A file from a stranger must be safe to open.
- New physics. The builder combines what the code has. Parts that need new machinery (the
  galaxy potential, particles) join the builder when their own plans land.
- Recording what the editing tools do during a run. "Export state" already keeps that.
- Replacing the presets. They stay as they are; some also become templates (section 9).

## 2. What exists and what is in the way

Used as is:

- `Problem` (`src/problems/types.ts`) already carries every part: `masses`, `settle` (any
  number of stars, each with its own barotrope, position and velocity), `init`, `mhd`,
  `damping`, `frame`, `refineMasses`, `refineDense`, `species`, `burn`, `units`.
- Building blocks inside the presets: `laneEmden` and `polytropeScales`, `whiteDwarf` and
  `dwarfsSettling`, `binaryMasses`, `collisionOrbit`, the disk of `mri_disk.ts`
  (`diskState`, `loopPotential`), `dipoleField` and `randomField` (`mhd/star_fields.ts`).
- `configFor`, `resolveGrid`, `amrFits`: turning a problem and a grid into a run, within the
  memory budget.
- Links (`ui/link.ts`): careful decoding that clamps, rejects and reports, never trusts.
- `.athdf` files carry the settings of the run that wrote them (`SnapshotMeta.query`).

In the way:

| Obstacle | Where | What it needs |
|---|---|---|
| A problem is found by its id everywhere | `problemById` in `ui/link.ts`, `ui/app.ts`, `amr/init_worker.ts`, `amr/init_pool.ts`, `io` restart, `ui/api.ts` | A second way to name a problem: by its spec |
| Workers rebuild a problem from its id; any other problem is set up on the main thread | `init_pool.ts` `shareable()` | Send the spec; the worker compiles it |
| At most 4 point masses, written out one by one | `MAX_MASSES` in `sim.ts`, `Masses.list` in `gravity/bh.wgsl`, `bh_cell.wgsl`, the reduction record (1 + 10 per mass), the sink panel in `app.ts` | Section 6 |
| One damping zone, as WGSL written by hand per preset | `Problem.damping`, `binary.ts`, `mri_disk.ts` | WGSL generated from a disk's numbers |
| Disk code is private to two presets, with constants baked in | `binary.ts`, `mri_disk.ts` | One disk module both presets and the builder use |
| GPU settling handles a limited number of stars | `SETTLE_MAX_STARS` in `amr/amr_sim.ts`; more fall back to the CPU, which reads the whole mesh | A cap in the validator, or a larger GPU limit |
| A magnetic field is either a potential or face values, not both | `MhdSpec` | Sum of several potentials plus a uniform part on the faces |
| No dialog exists; `app.ts` is 3800 lines | `ui/` | The builder lives in its own files |
| Which controls restart a run is spread over the handlers | `load()`, `restartOrWait()`, `rebuildPipelines(true)` against `applyParams()` in `app.ts` | One table (section 7.2) |

## 3. The setup document

New `src/setup/`: `spec.ts` (types and defaults), `validate.ts`, `compile.ts`, `codec.ts`,
and one file per kind of part under `src/setup/parts/`.

```jsonc
{
  "format": "shaderous-setup",
  "version": 1,
  "build": "…",                      // informational
  "name": "Triple with a circumbinary disk",
  "notes": "…",
  "domain": { "ndim": 3, "box": 12, "n": 256, "amr": 2,
              "bc": { "lo": [1, 1, 1], "hi": [1, 1, 1] },
              "units": { "length_m": 3.1e14, "mass_kg": 4e35 } },   // absent: scale free
  "physics": { "eos": "barotropic", "eosParams": { "gamma": 1, "K": 0.01 },
               "gravity": "off", "G": 1, "gext": [0, 0, 0],
               "cooling": "off", "viscosity": 1e-3, "conductivity": 0,
               "floors": { "rho": 1e-9, "p": 1e-14 }, "fluff": { "rho": 0, "u": 0, "v": 0 },
               "frame": "masses", "tEnd": null },
  "mesh": { "refineMasses": true, "refineDense": 0, "refineRho": 0 },
  "background": { "rho": 1e-8, "p": 1e-10, "v": [0, 0, 0] },
  "parts": [
    { "kind": "mass", "id": "a", "gm": 0.5, "pos": [-0.5, 0, 0], "vel": [0, -0.5, 0], "soft": 0.06, "sink": 0.06 },
    { "kind": "orbit", "of": ["a", "b"], "ecc": 0, "incl": 0 },      // sets pos and vel of a pair
    { "kind": "star", "id": "s", "model": "polytrope", "n": 1.5, "mass": 1, "radius": 0.3, "pos": [...], "vel": [...] },
    { "kind": "disk", "about": ["a", "b"], "rIn": 2.5, "rOut": 5, "sigma0": 1e-4, "slope": 0,
      "cs": 0.1, "incl": 0, "damp": { "from": 4.5, "rate": 3 }, "loops": null },
    { "kind": "region", "shape": "sphere", "center": [...], "radius": 0.1, "state": { "rho": 1, "p": 10 } },
    { "kind": "obstacle", "shape": "box", ... },
    { "kind": "field", "model": "uniform", "b": [0, 0, 0.01] }
  ],
  "settings": "tend=…&riemann=…",     // the live controls, as a link's query (section 7.2)
  "view": "mode=slice&axis=2"
}
```

Parts, in the order they would be built:

| Kind | Fields | Reuses |
|---|---|---|
| `mass` | G M, position, velocity, softening, sink radius and rate | `PointMass` |
| `orbit` | A helper, not an object: puts two bodies (masses or stars, or a body and the center of mass of a group) on a Kepler orbit from separation or pericenter, eccentricity, inclination and phase | `binaryMasses`, `collisionOrbit` |
| `star` | Polytrope (index, mass, radius) or cold white dwarf (mass); position, velocity, dye value; an optional field inside (dipole or random, with beta) | `laneEmden`, `whiteDwarf`, `Settling`, `star_fields.ts` |
| `disk` | The bodies it orbits, inner and outer radius and tapers, surface density and slope, sound speed (or H/R at a radius), inclination, an optional damping zone, optional field loops | The new disk module (5.3) |
| `region` | Sphere, box, slab or half-space with a state (density, pressure or temperature, velocity, dye, composition) that replaces or adds to what is under it; an optional smooth edge and seeded noise | New, small |
| `obstacle` / `inflow` | The same shapes, marking cells solid or held fixed | `Prim.solid`, `Prim.fixed` |
| `field` | Uniform field, or a field tied to a star or disk (those live on their part) | `MhdSpec` |

Regions cover the test-problem shapes too (a shock tube is two half-spaces, a blast a small
hot sphere, a shear layer three slabs with noise), so the builder is not limited to
astrophysical setups.

Positions and sizes are in code units of the setup's own box. With `units` given, the
dialog shows and accepts physical values beside them, as the binary preset's fields do.

## 4. Validation

`validateSetup(spec, limits) → Issue[]`, pure and shared by the dialog, the file loader, the
API and the tests. An issue is an error (the setup cannot be built) or a warning (it can, and
here is what to expect), names the fields involved, and has a one-line reason the dialog shows
in place.

Rules known today (the table grows with the parts):

| Rule | Kind |
|---|---|
| Stars need 3D and isolated self-gravity | error |
| Self-gravity needs 3D; periodic self-gravity needs every face periodic | error |
| Periodic faces come in pairs | error |
| Point masses, stars and disks with periodic faces (their pull is not periodic) | error |
| White dwarfs need the degenerate (or Helmholtz) equation of state; polytropes an ideal or radiation one: the two kinds of star do not mix | error |
| A polytrope of index n with gamma below 1 + 1/n is convectively unstable | warning |
| Cooling, conduction, hot regions and burning need an equation of state with an energy | error |
| The network burner needs its 13 species and physical units | error |
| A disk needs at least one body to orbit; `frame: masses` needs a mass | error |
| More than one damping zone | error |
| A disk heavier than a tenth of what it orbits (its own pull is left out of its rotation) | warning |
| Bodies that overlap, or reach outside the box | error / warning |
| Softening or sink under two cells of the level the body sits on (it is raised, as the presets do) | warning |
| More masses or stars than the build allows (sections 6, 5.2) | error |
| A magnetic field on an adaptive mesh: the mesh stays as first refined | warning |
| The grid does not fit the GPU memory budget | error, from `amrFits` |

Before the rules run, `codec.ts` normalizes: unknown keys are dropped and listed, numbers
outside a field's range are held to it and listed, a wrong type is an error. This is the
discipline of `decodeLink`, and its report is shown the same way.

## 5. Compiling a setup

`compileSetup(spec): Problem`, deterministic and free of the DOM, so the page, a worker and
Node all get the same problem from the same document.

### 5.1 Wiring

- The problem's id is `custom`. `SimConfig` gains `setup?: SetupSpec`. Wherever a problem is
  found by id, a helper `problemFor(id, setup)` compiles the spec when the id is `custom`.
- Workers: `InitRequest` carries the spec, `init_worker.ts` compiles it, and `shareable()`
  accepts a compiled problem. Without this every custom setup on an adaptive mesh would be
  set up on the main thread.
- The compiled `init` paints in layers: background, then disks, then regions, with stars
  left to settling as the collision presets leave theirs. Layers are precomputed per part
  (a Lane-Emden table, a disk's constants), so a cell's cost stays a few lookups.
- The setup's own controls (`Problem.params`) are empty in the first version: every value
  is edited in the dialog. Later a spec may mark fields as `live` panel controls (a
  softening, say), which is how the presets' live parameters would carry over.

### 5.2 Stars

`settle` returns one `SettleStar` per star part. Each star's dye value is spread over (0, 1]
so the passive scalar tells them apart, as the collision's 1 and 0.5 do. To check: how many
stars `SETTLE_MAX_STARS` allows on the GPU and what raising it costs; until then the
validator caps stars at that number on an adaptive mesh.

### 5.3 Disks

New `src/setup/parts/disk.ts`, from the two presets' disks: a midplane density with tapers,
a vertical profile in balance with the softened potential of what it orbits, rotation from
the enclosed mass and the pressure gradient, tilted by a rotation of the coordinates. About
several bodies it uses their total mass at their center of mass, flattened inside their
orbit as `binary.ts` does. It also writes the damping zone's WGSL from the same numbers
(formatted as float literals by the compiler, never taken as text from the file), so the
target and the initial state cannot drift apart. `binary.ts` and `mri_disk.ts` then call
this module, which is the test that it is right: both presets must start bit for bit as
before.

Several disks add their densities and share momentum by mass. Only one may have a damping
zone, and the disk diagnostics (`Problem.disk`) stay tied to the first mass.

### 5.4 Magnetic fields

Potentials of the star and disk parts are summed (their curl stays divergence-free). A
uniform field is set on the faces, since its potential is not periodic. `MhdSpec` gains a
form that takes both.

## 6. More than four point masses

"Any number" in practice means a limit set by cost, not by four hand-written copies:

- The mass count becomes a number the shaders are assembled with. `bh_cell.wgsl`'s unrolled
  blocks, and the others written out mass by mass, are generated for the run's count, as
  `reduceDefs()` generates the reduction record. No run-time-indexed function arrays
  (the Firefox device loss of c14460e).
- `Masses.list` in the uniform grows to the build's maximum; `MASS_FLOATS`, `MASS_RED`,
  `packMasses`, `MassReader`, the sink panel and the diagnostics CSV follow the count.
- The reduction record is 1 + 10 per mass: 161 floats at 16 masses. To measure first: the
  point-mass reduction's time and the gas update's time against the count, in Chrome and
  Firefox. If the record is the cost, sinks become optional per mass (a mass without a sink
  needs 3 channels, not 10).
- First limit: 16, raised if the measurements allow. The masses' own mutual forces are
  summed on the GPU already; their cost is nothing next to the gas.

Runs with four or fewer masses must produce the same results bit for bit, and compile no
slower.

## 7. The dialog

### 7.1 Layout

A `<dialog>` opened with `showModal()`, in new files `src/ui/builder/` (`dialog.ts`,
`forms.ts`, `preview.ts`, `library.ts`, `builder.css`). It edits a copy of the spec; nothing
touches the run until Build.

- **Left: parts.** The list of parts with an Add menu (point mass, star, disk, gas region,
  obstacle, inflow, field, orbit). Entries that would conflict are greyed out with the
  reason. Duplicate, delete, reorder (order matters for regions).
- **Middle: the form** for what is selected, or one of the fixed pages: Domain and grid,
  Gas and physics, Boundaries, Mesh. Fields are generated from a description of each part
  (label, range, step, tooltip, unit), the pattern `ProblemParam` already sets.
- **Right: a preview.** Two schematic views (top and side) drawn on a 2D canvas from the
  spec alone: the box, each body with its softening and sink radius, stars to scale, disk
  extents, regions, and the paths the bodies would take as point masses for a chosen time
  (a small float64 N-body integration). Bodies can be dragged; the form follows. No GPU
  work, so it is instant and cannot disturb a run in progress.
- **Bottom:** the issues list, the memory the grid needs against the budget, and the
  buttons: Open file, Save file, Copy link, Save to library, Cancel, Build.

Entry points: a "Custom" group at the top of the problem menu ("New custom setup…" and the
library's entries), an "Edit setup…" button beside Reset while a custom run is loaded, and
"Customize…" on presets that have a template (section 9). Build goes through `load()`, so
Autoload and Autoplay behave as for a preset.

On a phone the three columns become three tabs.

### 7.2 Which controls go where

The rule: a control whose change starts the run again belongs in the dialog; one that takes
effect in a running simulation stays in the side panel. Today that split is implied by which
handler calls `load()` or `restartOrWait()` and which calls `applyParams()` or
`rebuildPipelines()`. The first step of the UI work is one table of every control with its
kind (`restart`, `shaders`, `live`), which the panel's handlers and the dialog both read, so
they cannot disagree.

In the dialog (restart): dimensions, resolution, refinement levels, box size, units, floors,
the frame, the mesh's standing rules, the background, every part, the species and burning.
It also holds a few controls that are live in the panel but that the parts are judged
against: the equation of state and its constants, the gravity mode, G, and the boundary
kinds. Their values in the dialog are where the run starts; the panel can still change them
afterwards.

Everything else that is live (solver choices, CFL, refinement thresholds, cooling numbers,
diffusion, the view) is saved with the setup as the `settings` and `view` strings, in the
link's own keys and read by the link's own decoder, and is taken from the panel when the
user saves. A custom run with the dialog closed shows the side panel as any preset does,
with an empty parameter block.

## 8. Saving, loading and sharing

- **File.** `name.shaderous.json`: the spec, pretty-printed. Save and Open in the dialog;
  dropping the file on the page opens the dialog with it; "Import state" recognizes it.
- **Library.** Named setups in `localStorage` (`shaderous.setups`), listed in the problem
  menu's Custom group, with rename, delete and export. Files are the durable copy; the
  dialog says so.
- **Link.** `?setup=` holds the spec, minified, deflated (`CompressionStream`) and
  base64url-encoded, with the usual link keys beside it. If the link would pass about 8000
  characters, Copy link says so and offers the file instead.
- **State files.** `SnapshotMeta` gains `setup`. A `.athdf` from a custom run restarts as
  itself anywhere. An older build that meets `problem=custom` refuses the file with its
  existing "this build does not have" message. `SNAPSHOT_FORMAT` goes to 2.
- **Scripting.** `window.shaderous.load({ setup })`, the headless runner's `--setup file`,
  and the MCP `load` tool take a spec; `describe` returns the spec's schema with ranges, so
  an agent can write one. `state` returns the current spec.
- **Versions.** `version` counts changes to the spec. `codec.ts` keeps one migration per
  step and applies them in order; a file from a newer version loads with a warning and its
  unknown parts listed. The rule of `PROBLEM_ALIASES` applies: add, never remove.

## 9. Presets as templates

Presets that are a combination of parts get a spec that compiles to the same problem:
binary accretion, the magnetized disk, the polytrope and its field variants, the white dwarf
collision, tidal disruption, the empty box. "Customize…" opens the dialog on that spec. Each
template is also a regression test (same initial state as its preset), and together they are
the dialog's starting points under New. Presets built on special functions (Kelvin-Helmholtz,
the MHD tests) stay presets only.

## 10. Tests

Unit (vitest):

- Codec: save then load returns the same spec; every migration; malformed and hostile input
  (wrong types, huge arrays, deep nesting, numbers out of range) ends in a clean report.
- Validator: each rule of section 4, both ways.
- Compiler: each template's `init`, masses and settling against its preset's at sampled
  points; the same spec compiled twice, and in a worker, gives identical leaf states (extend
  `init_cpu.test.ts`); generated damping WGSL against the float64 disk at sampled points.
- Links: `?setup=` round trip beside ordinary keys (extend `link.test.ts`).

GPU:

- `custom-binary`: the binary template for a few orbits against the preset, diagnostics
  equal.
- `masses-many`: 8 and 16 masses with gas; total momentum of gas and masses conserved as
  well as with 2; the four-mass presets unchanged bit for bit.
- `custom-restart`: export a custom run, import it, continue; identical to the uninterrupted
  run.
- `custom-triple`: three stars settle and hold for a dynamical time.

Page (Playwright): open the dialog, add parts, see a conflicting choice disabled with its
reason, build, save the file, reload the page, open the file, and get the same link.

## 11. Order of work

1. **Spec, codec, validator, compiler for masses, background, regions, boundaries and
   physics; the `custom` problem through links, workers, state files and the API.** No
   dialog: a setup is a JSON file loaded through the API or headless. This proves rule 2.
2. **Dialog shell**: the control table (7.2), Domain, Physics, Boundaries, point masses,
   the preview, file save and open, the library.
3. **Stars and orbits.**
4. **The disk module**, the two presets moved onto it, disks in the builder, generated
   damping.
5. **More than four masses** (section 6), after its measurements.
6. **Obstacles, inflows, magnetic fields, species and burning.**
7. **Templates and "Customize…"**, links, `docs/api.md`, `docs/headless.md`, a builder
   activity in `docs/teaching.md`.

Steps 3, 4 and 5 do not depend on one another. Step 1 is useful alone to anyone scripting
runs.

## 12. Risks

- **The compiled problem is slower to set up than a hand-written one.** Layered `init` with
  many parts costs more per cell. Precompute per part, and measure a 512³ adaptive setup
  against the binary preset's time.
- **Combinations nobody has run.** The presets are each tuned (floors, fluff, damping,
  refinement densities); a free combination will find untuned corners. The validator's
  warnings and sensible defaults derived from the parts (ambient density from the lightest
  part, `refineRho` from it, softening from the cell size) carry most of this. Expect the
  rule table to grow from use.
- **Scope of the dialog.** Every part is a form, a preview glyph, rules and tests. The
  order of work lands them one kind at a time, each usable when it lands.
- **Spec churn.** Early versions will change. Migrations from version 1 on, even while
  unreleased, keep saved files alive.

## 13. Open decisions

| Decision | Options | Leaning |
|---|---|---|
| Limit on point masses | 8, 16, or sinks optional and more | 16, decided by the measurements of section 6 |
| Live panel controls for a custom setup | None; or fields of the spec marked live | None at first |
| Units in the dialog | Code units only; or physical beside them when `units` is set | Physical beside them, as the binary preset |
| Where the live settings are edited before Build | Only in the panel; or mirrored in the dialog | Panel, with the few the rules depend on mirrored |
| Disks about a star | Only about point masses; or about stars too | Masses first; stars once a star's settling with a disk around it is understood |
| File extension | `.shaderous.json`, `.json` | `.shaderous.json` |
| Link encoding | Deflate + base64url; or a short key per field | Deflate: no second schema to maintain |
