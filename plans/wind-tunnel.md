# Plan: wind tunnel runs of arbitrary objects

Status (2026-11-20): stages 0 to 2 built (first release: static bodies from mesh files in a
tunnel, with forces), stage 3 built (a rotor's frame), stage 4 begun (moving spheres and boxes
on the uniform grid; its Galilean test fails, which settles decision 13 against the staircase
for moving bodies). Each section has an "As built" note with the measurements. The values and
citations below were from memory when written; 2.2 says which were checked and how (all eight
citations of section 15 were confirmed against Crossref). Cost figures in section 9 are
estimates, not measurements.

Goal: load a 3D model file, put it in a stream of air, and get the flow, the drag, the lift
and the torque. Bodies may have parts that move: a turbine rotor, a propeller, a flap, a
spinning wheel.

| Stage | What it adds | Grids |
|---|---|---|
| 0 | Baseline: what today's obstacles give for a sphere and a cylinder; three checks of the code | uniform and blocks |
| 1 | Mesh files to a signed distance field; a `mesh` shape for obstacles and regions | none (CPU) |
| 2 | The tunnel: inlet, outlet, floor, sponge; forces on each body; drag and lift coefficients | uniform and blocks |
| 3 | Rotors without moving cells: the run made in the rotor's frame | uniform and blocks |
| 4 | Moving bodies: prescribed motion, cells that change between gas and solid | uniform, then blocks |
| 5 | Bodies the flow moves: a rotor that spins up against a load | both |
| 6 | Accuracy: partly filled cells in place of the staircase; a wall model; eddy viscosity | both |
| 7 | Display: the body's surface colored by pressure, streamlines, smoke | both |

Stages 1 and 2 together are the first useful thing (a static object in a tunnel). Stage 3 gives
an isolated rotor early, cheaply. Stage 6 is decided by what stages 0, 2 and 4 measure.

**Contents**

1. [What the code has, and what is missing](#1-what-the-code-has-and-what-is-missing)
2. [Stage 0: baseline](#2-stage-0-baseline)
3. [Stage 1: mesh files](#3-stage-1-mesh-files)
4. [Stage 2: the tunnel and the forces](#4-stage-2-the-tunnel-and-the-forces)
5. [Stage 3: rotors in their own frame](#5-stage-3-rotors-in-their-own-frame)
6. [Stage 4: moving bodies](#6-stage-4-moving-bodies)
7. [Stage 5: bodies the flow moves](#7-stage-5-bodies-the-flow-moves)
8. [Stage 6: accuracy](#8-stage-6-accuracy)
9. [What a wind turbine costs](#9-what-a-wind-turbine-costs)
10. [Stage 7: display](#10-stage-7-display)
11. [Validation tests](#11-validation-tests)
12. [Not covered](#12-not-covered)
13. [Open decisions](#13-open-decisions)
14. [Order of work](#14-order-of-work)
15. [References](#15-references)

---

## 1. What the code has, and what is missing

**Has.**

- Solid cells. The fourth component of the second state texture marks a cell: 1 solid, -1
  held at its state (an inlet), 0 gas (`hydro/common.wgsl`). A face beside a solid cell is a
  reflecting wall, made by a mirror image (`hydro/flux.wgsl`, `q_mirror`); with viscosity it
  is a no-slip, adiabatic wall at rest (`hydro/diffusion_core.wgsl`).
- Obstacles and inlets in the setup format (`ObstaclePart` in `setup/spec.ts`), painted cell by
  cell on the CPU (`setup/parts/region.ts`), and the obstacle brush in the page.
- Slow flow: the liquid equation of state and the implicit pressure step, on the uniform grid
  and on blocks, with obstacles and inlets (`plans/incompressible.md`, 4.12 and 5.11). A wall
  is a face coefficient $`\beta = 0`$ in the pressure solve.
- Fast flow: the explicit scheme with an ideal gas, shocks included.
- Box faces: outflow, reflecting, no-slip and moving walls (kinds 4 and 5, `domain.bc`,
  `domain.wallVelocity`).
- A mesh that refines along an obstacle's outline (`AmrSim.refineOutline`), and per-block
  timesteps.
- The `cylinder-wake` preset: Strouhal number 0.174 at Reynolds number 100, 16 cells across
  the cylinder.
- Rotating frames in the setup format.

**Missing.**

| Missing | Stage |
|---|---|
| Shapes other than a sphere, a box and a half-space | 1 |
| Any measure of the force on a body | 2 |
| A sponge at the outlet (damping zones belong to disks in the setup format) | 2 |
| Solid cells that move: `load_vel` returns zero for a solid cell, and the mark is set once | 4 |
| Walls better than a staircase | 6 |
| A wall model or a subgrid model: the viscosity is one constant | 6 |
| The body drawn as a surface | 7 |

**Two limits to say plainly, in the docs and in the page.**

1. *Reynolds number.* A car at 30 m/s is at a Reynolds number of several million, a turbine
   blade the same. No grid here resolves those boundary layers. What the code will compute is
   the large eddies, with the scheme's own dissipation as the subgrid model. That gives the
   pressure drag of bluff bodies, where the shape fixes the separation (boxes, buildings,
   cars' rear ends, plates, cylinders below the drag crisis), to perhaps 10 to 20%. It does
   not give skin friction, the drag of a streamlined body, stall, or the drag crisis. Lift of
   an airfoil in attached flow is mostly an inviscid pressure effect and should be usable;
   its drag is not.
2. *Thin parts.* A wall thinner than two cells is not a wall. A blade 18% as thick as its
   chord, with 16 cells along the chord, is 3 cells thick, and its trailing edge is less than
   one. Stage 1 reports the thinnest part of each body in cells, and warns.

## 2. Stage 0: baseline

No new physics. Three measurements and three checks, through the headless runner.

**Measurements** (the force comes from stage 2's diagnostic, so this stage and the first half
of stage 2 are done together):

| Run | Vary | Measure |
|---|---|---|
| `cylinder-wake`, 2D, Reynolds number 100 | 8, 16, 32, 64 cells across; explicit and implicit pressure; uniform and blocks | mean drag coefficient, lift amplitude, Strouhal number |
| Sphere, 3D, Reynolds number 100 | 8, 16, 32 cells across | drag coefficient, length of the wake's recirculation |
| Sphere, 3D, no viscosity, slip walls | 16, 32 cells across | drag coefficient (the staircase's own drag: it should be zero before separation sets in) |

The convergence rate of the drag with the cell size is the number stage 6 is judged against. A
staircase should give first order.

**Checks.**

1. Do obstacles, the liquid and implicit pressure run in a rotating frame? Stage 3 depends on
   it. If one is refused, find out why.
2. Does the page have a vorticity field, and can the field-line renderer
   (`render/fieldlines.ts`) take a vector field other than the magnetic one? Stage 7.
3. How long does a re-mesh take with 2 000 to 5 000 blocks? Stage 4 on blocks, and the choice
   in 9.2.

### 2.1 Answers to the checks (2026-11-20)

1. **Rotating frame.** The explicit scheme takes all three together: `cylinder-wake` with
   `physics.rotation.omega = [0, 0, 0.5]` (liquid, obstacle, inlet, viscosity) runs at 256 on
   the uniform grid and with 2 levels, no errors, to t = 1. Implicit pressure is refused in
   both grids (`implicitUnsupported` in `sim.ts`: "rotating frames and shearing boxes are not
   available with it"). The reason is in the step's order, not the physics: the centrifugal
   force is part of `ext_accel` (`gravity/ext.wgsl`), which the pressure passes already kick
   with (`pressure/kick.wgsl`, `amr/pressure/faces_gas.wgsl`), and the update applies the
   first half-turn of the Coriolis force (`frame_open`); the second half-turn, `frame_close`,
   is in `gravity/correct.wgsl` and `amr/correct.wgsl`, which an implicit step does not run.
   The fix is to give the implicit step its second half-turn (in the update before the solve,
   so that the projection sees it, or after the correction), lift the refusal, and test that
   a liquid at rest in the inertial frame stays so. Days, not weeks: stage 3 goes ahead.
2. **Vorticity and field lines.** The display has vorticity (field 11, `render/field_value.wgsl`:
   the signed z component in 2D, the magnitude in 3D), but no Q criterion. The field-line
   renderer is attached only to runs with a magnetic field (`FieldLines.attach` returns when
   `!sim.cfg.problem.mhd`), and traces the field volume `BV`. Its sampling pass
   (`render/bsample.wgsl`, `amr/bsample.wgsl`) already resamples the velocity into a second
   volume, `VV`, for moving the seeds. Streamlines are therefore: a sampling variant that
   writes the velocity into `BV` (or a tracer that reads `VV`), seeds on a rake, and the
   attach gate opened for them. A small change.
3. **Re-mesh time.** Measured on the RTX 3080 (Chrome, D3D11) with a 3D blast on blocks of
   16³, the mesh filled by hand to 2 300 to 5 100 leaves and then one leaf in 20 refined per
   re-mesh (`tree.adapt`, then `applyNow`, timed to `onSubmittedWorkDone`):

   | Leaves | Tree (CPU) | Encode and upload | GPU | Total | One step |
   |---|---|---|---|---|---|
   | 2 276 → 3 459 | 2.3 ms | 2.3 ms | 4.0 ms | 8.6 ms | 28 ms at 2 276 |
   | 3 459 → 4 733 | 4.0 ms | 2.5 ms | 2.9 ms | 9.4 ms | |
   | 5 100 (budget of 5 832 full) | 21 to 29 ms | 3 to 4 ms | 3.5 to 5.5 ms | 29 to 36 ms | |

   A re-mesh costs a third of a step at 3 000 leaves and about one step with the budget full,
   where the CPU's tree search dominates. A mesh that follows the blades, re-meshing every 20
   to 40 steps, would spend 1 to 5% of its time re-meshing: affordable. 9.2's default (refine
   the swept disk once) stands for its other reason, no refinement boundary crossing a blade.
   (Probe: a Playwright script over `tools/headless/session.mjs`; not kept in the repository.)

### 2.2 Measurements (2026-11-20)

With the force diagnostic of 4.2, on the RTX 3080, through the runner. The published values
were checked against comparison tables in open papers (the primary papers' texts were closed;
notes in the session's shared folder): cylinder at Reynolds number 100, unbounded, mean drag
coefficient 1.32 to 1.35 (Park, Kwon & Choi 1998: 1.33; Posdziech & Grundmann 2007: 1.325;
Liu et al. 1998: 1.35), lift amplitude about 0.33 (rms 0.23), Strouhal number 0.164 to 0.165;
sphere at 100, drag coefficient 1.08 to 1.09 (Johnson & Patel 1999: 1.08; Fornberg 1988:
1.085; Clift, Grace & Weber's correlation 1.087) and a recirculation 0.88 diameters long. The
Allen and Vincenti correction puts a 6% blockage at about +4.5% on the drag and +2.3% on the
Strouhal number (an estimate, not a published result).

**The `cylinder-wake` preset as it is** (box 4 = 16 diameters, periodic sides, the cylinder 4
diameters from the inlet), statistics over t = 30 to 60:

| Cells across | Scheme, grid | Mean Cd | Cd swing | Lift amplitude | St |
|---|---|---|---|---|---|
| 8 | explicit, uniform / blocks | 1.681 / 1.627 | 0.025 / 0.047 | 0.34 / 0.35 | 0.167 / 0.163 |
| 8 | implicit, uniform / blocks | 1.693 / 1.727 | 0.008 / 0.012 | 0.35 / 0.42 | 0.167 / 0.169 |
| 16 | explicit, uniform / blocks | 1.624 / 1.566 | 0.027 / 0.042 | 0.43 / 0.45 | 0.175 / 0.170 |
| 16 | implicit, uniform / blocks | 1.656 / 1.652 | 0.008 / 0.009 | 0.43 / 0.43 | 0.175 / 0.175 |
| 32 | explicit, uniform / blocks | 1.541 / 1.544 | 0.023 / 0.034 | 0.40 / 0.40 | 0.179 / 0.179 |
| 32 | implicit, uniform / blocks | 1.563 / 1.563 | 0.005 / 0.005 | 0.40 / 0.40 | 0.180 / 0.179 |
| 64 | explicit, uniform / blocks | 1.520 / 1.534 | 0.021 / 0.029 | 0.40 / 0.41 | 0.179 / 0.180 |
| 64 | implicit, uniform / blocks | 1.551 / 1.544 | 0.007 / 0.011 | 0.39 / 0.38 | 0.181 / 0.179 |

The four ways agree to 2% from 32 cells across. The drag is 15% above the published value and
the Strouhal number 9%: the box is small and periodic across (a 6% blockage), and the cylinder
4 diameters from a held inlet. The explicit scheme's drag swings 3 to 5 times more than the
implicit one's: sound at Mach 0.1 that the implicit step does not carry. Runs took 1 to 12
minutes each to t = 60.

**In the tunnel** (box 8 = 32 diameters, slip sides 32 diameters apart: 3% blockage; the
cylinder 8 diameters from the inlet; implicit, uniform grid), statistics over t = 40 to 80:

| Cells across | Mean Cd | Lift amplitude (rms) | St |
|---|---|---|---|
| 16 | 1.548 | 0.48 (0.34) | 0.168 |
| 32 | 1.463 | 0.43 (0.31) | 0.171 |
| 64 (2 levels of blocks) | 1.435 | 0.40 (0.28) | 0.170 |

The drag falls by 0.085 and then 0.029: faster than first order at first, then slower. A
first-order extrapolation from 16 and 32 gives 1.378 (what test `cylinder-forces` checks,
within 1.30 to 1.45); from 32 and 64, 1.406. The drag is settling near 1.41, 4 to 6% above the
published 1.32 to 1.35, of which the 3% blockage accounts for part. The Strouhal number is 2 to
4% high; the lift amplitude 20 to 30% high (not understood; the published amplitude is itself
only quoted, and lift is more sensitive than drag to the wall's shape). 

**The sphere in the tunnel** (box 4 = 16 diameters, slip sides: 1.2% blockage; the sphere 4
diameters from the inlet; implicit; to t = 20, statistics over t = 15 to 20, where it is
steady):

| Cells across | Grid | Cd | Without viscosity, t = 4 to 6 |
|---|---|---|---|
| 8 | uniform 128³ | 1.210 | |
| 16 | 256, 2 levels | 1.158 | 0.44 |
| 32 | 512, 3 levels | 1.163 | 0.50 |

6 to 7% above the published 1.08 to 1.09, and not falling from 16 to 32. Without viscosity the
sphere's drag should be nil until the flow separates; the staircase gives 0.44 and 0.50, and
does not fall with the cell size (the scheme's own dissipation separates the flow from a
stepped surface). This is the number section 8.1 is judged against. (`sphere-drag`, the GPU
test, checks 0.95 to 1.35 at 16 cells across over t = 10 to 12: 1.253.) The recirculation's
length was not measured.

## 3. Stage 1: mesh files

### 3.1 Formats

| Format | Why | Notes |
|---|---|---|
| STL, binary and ASCII | what every CAD program and every 3D printing site writes | no units, no hierarchy; told apart by the size the header implies, not by the word `solid` |
| OBJ | common, text | polygons are fanned into triangles; `o` and `g` names become parts |
| PLY, binary and ASCII | scans | positions and faces only |
| glTF 2.0 and GLB | has units (meters), named nodes and a transform tree | the natural format for bodies with moving parts: each node is a part. No Draco or meshopt compression at first: the loader says so |
| 3MF | what newer CAD exports instead of STL | a zip of XML; after the others |
| Airfoil coordinates (`.dat`, Selig and Lednicer order), SVG paths | 2D runs | an outline, extruded along z in 3D |

Not read: STEP and IGES. They hold exact surfaces, not triangles, and turning them into
triangles needs a CAD kernel. The docs say how to export STL or glTF from FreeCAD, Blender,
Fusion and Onshape.

The parsers are written here, in `src/geometry/` (`stl.ts`, `obj.ts`, `ply.ts`, `gltf.ts`,
`outline.ts`), each a function from bytes to
`{ positions: Float32Array, indices: Uint32Array, parts: { name, first, count, transform, parent }[] }`.
The project has no runtime dependency beyond Playwright and these formats are small. Each is
unit-tested on fixture files of a few hundred bytes.

### 3.2 From triangles to a signed distance field

Each rigid part becomes one 3D texture of signed distance, in the part's own frame. A part
that moves keeps its texture and changes only its transform (stage 4), so the triangles are
never looked at again after loading.

1. **Bounding-volume tree** over the triangles, on the CPU, in a worker.
2. **Distance** on a grid in the part's frame: exact distance to the nearest triangle within a
   band of a few cells of the surface, and a clamped value beyond. The grid's spacing is the
   finest cell of the run (or coarser, with a cap on the texture's size: 256 cubed is 64 MB as
   `r32float`, 32 MB as `r16float`).
3. **Sign** by the generalized winding number ([Jacobson et al. 2013][Jacobson2013]), with the
   tree's far-field expansion for speed ([Barill et al. 2018][Barill2018]). Files from the
   internet have holes, flipped triangles and parts that pass through each other. Ray parity
   fails on all three; the winding number degrades smoothly, and 0.5 is the threshold.
4. **Report**: volume, surface area, bounding box, frontal area along each axis (the reference
   area of the drag coefficient), the thinnest part in cells, and how far the winding number
   strays from 0 and 1 (how broken the mesh is).

The same code serves the 2D outlines, with segments for triangles.

On the GPU the texture is sampled with trilinear filtering. On the CPU the same array is
sampled by the painters, so the adaptive mesh's CPU initialization (`amr/init_cpu.ts`) gets
mesh shapes with no new path.

### 3.3 In the setup format

`Shape` gains one member, so both `obstacle` and `region` can use it (a region in the shape of
a mesh: gas in the shape of a rabbit, for nothing):

```json
{ "kind": "obstacle", "id": "rotor",
  "shape": { "type": "mesh", "file": "turbine.glb", "node": "Rotor",
             "fit": 0.3, "center": [0, 0, 0], "rotate": [0, 0, 0], "up": "y" } }
```

| Key | Meaning |
|---|---|
| `file` | a path relative to the setup document (runner, MCP), a URL, or the name of a file dropped on the page |
| `node` | glTF node, OBJ group; all of the file if absent |
| `scale` or `fit` | a factor on the file's lengths, or the longest side as a fraction of the box. STL has no units, so one of the two is required for it |
| `center`, `rotate`, `up` | where the bounding box's center goes, degrees about x, y, z as for a box, and which of the file's axes is up (glTF is y up; the box is z up in 3D) |

Where the file lives:

- **Runner and MCP server**: read from disk beside the setup document.
- **Page**: a file picker and drag-and-drop on the Setup dialog; the bytes are kept in the
  origin's private file system under their SHA-256, so a reload finds them.
- **Links**: a query string cannot hold a mesh. A link carries `file` as a URL if it has one,
  and otherwise the hash, and the page asks for the file if it does not have it.
- **`.athdf` checkpoints**: the solid mark is in the state, so a static body restarts as it is.
  A moving body needs its distance field: the checkpoint stores each part's hash and pose, and
  the runner writes the distance fields once beside the outputs.

### 3.4 Done when

A Stanford bunny, a car body and a glTF with three nodes each load in the page and the runner;
the unit tests of 11.1 pass; `setups.md` documents the shape; the schema
(`public/shaderous-setup.schema.json`, `tools/headless/schema.json`) has it; random testing
(`tools/random-test`) draws mesh obstacles from two fixture files.

### 3.5 As built (2026-11-20)

`src/geometry/`: `stl.ts`, `obj.ts`, `ply.ts`, `gltf.ts`, `outline.ts` (airfoil `.dat`, SVG
paths), `mesh.ts` (parts, transforms, `flatten`, `orient`), `sdf.ts`, `store.ts` (files by name,
placement, the page's private file system). Unit tests in `tests/unit/geometry.test.ts` on
fixtures in `tests/unit/fixtures/meshes/` (written by `make.mjs` there). Departures and
measurements:

- **The winding number does not survive flipped triangles**, against 3.2's claim: a flipped
  triangle subtracts its solid angle, and a sphere with half of its triangles flipped had a
  winding number near 0 inside (`sdf-broken` failed). The triangles are now made consistent
  first (`orient`: weld, propagate across edges of exactly two triangles, turn each closed piece
  outward, leave an open one as most of it was given). Edges of three or more triangles, where
  pieces touch, join nothing: welded across them, two touching cubes came out as one "closed"
  piece and the shared face's side was wrong (seen in a run; test `sdf-touching`).
- **Sign**: one closed piece takes the side of the nearest face where the nearest point is
  inside a face; anything else the winding number (the shortcut is wrong where pieces touch
  or pass through each other).
- **Decision (13): the distance field on the CPU**, default kept. On one core: 82 000 triangles
  on 129³ points 2.7 s, 330 000 on 249³ 19 s, 1.3 million on 249³ 26 s (after measuring the
  stray with the far field beyond 8 radii; with the exact sum it was 32, 156 and 478 s). It is
  made on the thread that compiles the setup, not yet in a worker of its own; on blocks each
  initialization worker makes its own copy. Distances are kept as float32 on the CPU; no GPU
  texture yet (stage 4 and 7 need one).
- `sdf-volume`: relative volume error 1.2e-2, 3.5e-3, 5.3e-4, 7.1e-5 at 10, 20, 40, 80 points
  across a sphere (order 1.7 to 2.9). `sdf-sphere`: within the band, 0.1 of a cell.
- **In the format**: a `mesh` member of `Shape` (decision 13 default), with `file`, `node`,
  `scale` or `fit`, `center`, `rotate`, `up`. In 2D the body is the cross-section at z = 0; an
  outline file (`.dat`, `.svg`) is a 2D body directly (in 3D it is extruded through the box).
- **Where the file lives**: the runner and the MCP server send the files beside the document
  (`sendMeshFiles` in `tools/headless/session.mjs`; MCP `load` and `check_setup` take `dir`);
  the page API has `meshFile(name, base64)`; the Setup dialog takes mesh files by "Open file…"
  and by drop, keeps them in the origin's private file system under their SHA-256 (an index
  by name), and adds one the setup does not name as an obstacle. The page's build looks a
  missing file up there, or fetches a URL. Links carry the name, not the hash: a link to a
  setup with a mesh works only in a browser that has the file, or with a URL.
- **Not done**: `.athdf` checkpoints do not store the hash or pose (needed only for moving
  bodies), random testing does not draw mesh obstacles yet, and the preview draws a mesh as
  the hull of its vertices (no surface rendering: stage 7). **The drop on the Setup dialog,
  checked in the page (2026-11-20)**, on the static build in the app's browser: a binary STL
  cube (a real File, dropped by a script's DataTransfer on the dialog: the drag from the
  desktop itself could not be driven) was added as the obstacle "cube" (fit 0.25), drawn in the
  preview, applied, and ran beside the preset's cylinder with its own force in the panel; the
  browser kept it under its SHA-256. A fresh page, without the file in memory, loaded a document
  naming `cube.stl` and found it there. Not tried: the "Open file..." picker, glTF with a
  separate .bin, a real drag from the desktop.

## 4. Stage 2: the tunnel and the forces

### 4.1 A tunnel part

A new part, `tunnel`, that writes what a person would otherwise assemble by hand:

```json
{ "kind": "tunnel", "speed": 1, "axis": "x", "floor": "moving", "sides": "slip",
  "sponge": 0.1, "mach": 0.1 }
```

| Piece | Built from |
|---|---|
| Inlet | a slab of `inflow` cells at the upstream face, at `speed` |
| Outlet | outflow face, and a sponge: a damping zone over the last `sponge` of the length, pulling the gas toward the inlet's state (`hydro/damping.wgsl`, given a box form beside the disks' radial one) |
| Sides and roof | `slip` (reflecting), `open` (outflow) or `periodic` |
| Floor | `slip`, `none` (as the sides), `fixed` (no-slip, kind 4), or `moving` (kind 5 at the wind's speed: a car on a road) |
| Gas | the liquid with implicit pressure when `mach` is below 0.3; an ideal gas, explicit, above |
| Start | the whole box at `speed`. The first pressure solve turns that into flow around the body |

It reports the blockage (frontal area over the tunnel's cross-section) and warns above 5%.
Yaw and angle of attack turn the body, not the wind.

Units: the setup format already has units. A tunnel in SI takes the speed in m/s, the body in
meters and air's $`\nu = 1.5\times10^{-5}`$ m²/s, and reports the Reynolds number asked for
and the one the grid can resolve (below).

### 4.2 Forces

The force on a body is the momentum its wall faces take from the gas. The flux pass already
computes that: the flux of a wall face is the pressure on it, plus the viscous stress. So the
force is a sum over wall faces of fluxes the update forms anyway, and it is exact in the sense
that it is the momentum the gas lost.

- **Which body.** The mark becomes `1 + index` for body `index` (every test of it is
  `> 0.5`, which still holds; `hydro/q.wgsl`'s flag stays 1).
- **Reduction.** New channels of `hydro/diagnostics.wgsl`, summed: force (3), torque about the
  body's reference point (3), and the pressure and viscous parts apart. For up to 4 bodies at
  first; more bodies share a channel and are reported together.
- **With implicit pressure**, the wall's pressure force comes from the correction pass
  (`pressure/correct.wgsl`), not the flux: the sum is taken there.
- **On blocks**, a block's wall faces are its own cells' faces, so the sum over leaves counts
  each face once. Per-block timesteps: weight by the block's step.
- **Reported**: $`C_d = F_\parallel / (\tfrac12 \rho U^2 A)`$, lift and side coefficients, the
  moment coefficients, as time series in the diagnostics panel, the runner's
  `diagnostics.json` and the MCP `diagnostics` tool, with running means over a window.
- **A second measure, for tests only**: the momentum balance of a box around the body (flux
  through its faces plus the rate of change inside). The two must agree to round-off; that is
  test `force-balance`.

### 4.3 Refinement

A body asks for the finest level within a band of its surface (from the distance field, in
`amr/flag.wgsl`'s terms a floor on the level), and a `wake` option keeps a cylinder of
intermediate level behind it. `refineOutline` is the model.

### 4.4 Done when

`tunnel` plus a mesh obstacle runs in the page from a dropped STL; the stage 0 table is filled
in; tests `force-balance`, `sphere-drag` and `cylinder-forces` pass; a preset `wind-tunnel`
ships with a small public-domain body.

### 4.5 As built (2026-11-20)

**Forces (4.2).** As planned, with these choices and departures:

- The mark is 1 + the body's index (`cellMark`, `problems/types.ts`); the obstacles are the
  bodies, in order, `MAX_BODIES` = 4 apart and the rest summed with the fourth. A restricted
  block averages its children's marks, as it did; a body's surface is never merged, so the
  averages are of equal marks.
- Each gas cell writes its wall-face fluxes (pressure part, viscous part, the body's mark)
  into two rgba32float textures laid out as the state (`hydro/walls_on.wgsl`,
  `amr/walls_on.wgsl`); with implicit pressure the correction writes the pressure part
  (`pressure/pwalls_on.wgsl`, `amr/pressure/pwalls_on.wgsl`), less a gas's reference pressure.
  A separate reduction of 36 channels (`hydro/forces.wgsl`, `amr/forces.wgsl`,
  `src/hydro/forces.ts`) runs with each diagnostics readback. Memory: two more atlas textures
  (+32 bytes a cell) for runs with obstacles only; without obstacles the shaders are as before.
- The torque takes each cell's force at its center: exact for the pressure, half a cell's
  lever off for viscosity's. A gas cell touching two bodies gives its whole wall force to the
  one of the larger mark.
- With per-block timesteps each block's texture holds its own last step's force; after a
  re-mesh a reused slot may hold another block's force until that block next steps.
- Reported as `bodies` in the diagnostics, the page's panel (a block per body beside the
  sinks', with C_d, C_l, the force and the torque, and CSV columns `body<i>_*`), the runner's
  `diag.jsonl` and the MCP `diagnostics` tool. Running means over a window are not built: the
  runner's series serve for now.
- `force-balance` (GPU): the change of a periodic box's momentum against the sum of force
  times dt over 30 steps, liquid and ideal gas, explicit and implicit, uniform and blocks:
  3e-7 to 5e-6 of the impulse (threshold 1e-4).

**The tunnel (4.1).** A part `tunnel` (`setup/parts/tunnel.ts`), compiled as a transform of the
document (box faces, gas, how the pressure acts, background) plus an inlet painter and a
damping zone with a target written for it. Departures:

- The sponge's rate rises as the square of the depth to 4 U / (sponge x box) at the outlet;
  damping zones are now allowed with implicit pressure for the liquid (still refused for a
  gas).
- Units (SI speed, meters, air's viscosity) and the report of the Reynolds number the grid
  resolves are not built; the tunnel is in code units.
- The reference area is the first obstacle's shadow along the stream (`shadowOf`: exact for a
  sphere, from the triangles for a 3D mesh, sampled otherwise), or the part's `area`; lift is
  along the box's up axis. The blockage warning is in the check.
- Refinement along the body's surface is the existing outline refinement; the `wake` option of
  4.3 (a cylinder of intermediate level behind the body) is not built.
- **The inlet on blocks** (found by the measurements): held cells were those within `inlet`
  finest cells of the upstream face, but that face is at the coarsest level, whose cell centers
  lie further in: on blocks no cell was held, and the stream died away (the cylinder's drag fell
  from 1.0 to 0.33 over t = 80 at 64 cells across). The painters now get the block grid's
  levels (`Problem.init`'s `levels`, `PaintContext.levels`), and the inlet is `inlet` cells of
  the coarsest level deep. The block runs of 2.2 and 4.5 were redone after the fix.
- The preset `wind-tunnel` uses a body made in code, `builtin:ahmed` (the Ahmed body's nominal
  proportions: `geometry/builtin.ts`), rather than a shipped file: nothing to fetch, public by
  construction. At 256 finest cells with 2 levels it gives C_d near 0.9 and C_l 0.34 early on (a
  showcase at Reynolds number 12 000, not the body's measured 0.29).
- `mesh-vs-shape` (GPU): the icosphere through the mesh path and the built-in sphere mark the
  same cells and give the same drag, to the bit, at 8 and 16 cells across.
- Dropping an STL on the Setup dialog: checked in the page (3.5).

## 5. Stage 3: rotors in their own frame

An isolated rotor in a wind along its axis is steady in the frame that turns with it. In that
frame the blades are static obstacles (stages 1 and 2), and the gas feels the Coriolis and
centrifugal forces the code already has.

- Inlet state: $`\mathbf{v} = U\hat{\mathbf{z}} - \boldsymbol\Omega\times\mathbf{r}`$. The
  inlet painter and the sponge's target take a velocity that depends on position.
- Sides: open, far enough out (the gas there moves at $`\Omega r`$ along the face, which sets
  the timestep, so the box should be a cylinder's worth and no more: a `tunnel` with
  `"frame": "rotor"` masks the corners with inlet cells).
- No tower, no ground, no yaw, no shear in the wind: anything not symmetric about the axis is
  unsteady in this frame and needs stage 4.
- Torque and thrust come from 4.2; power is $`\Omega`$ times the torque; the power coefficient
  is $`P / (\tfrac12 \rho U^3 \pi R^2)`$.
- The display turns the picture back to the tunnel's frame, as an option.

This stage is small if check 1 of stage 0 passes. It is also the reference stage 4 is tested
against: the same rotor run both ways must give the same torque.

### 5.1 As built (2026-11-20)

- **Implicit pressure in a rotating frame**, for the liquid (check 1's fix): the update gives
  the cell the centrifugal force's whole kick and the Coriolis force's second half-turn before
  the solve (`hydro/update.wgsl`, `amr/update.wgsl`). For a liquid of uniform density the
  centrifugal force is a gradient, which the solve takes into the pressure. A gas with implicit
  pressure is still refused in a rotating frame (its solve also carries the energy; not tried).
- **The tunnel's `rotor`** (an angular velocity about the stream's axis through the box's
  center; 3D only), rather than `"frame": "rotor"`: it sets the run's rotation, and the inlet,
  every cell's starting state and the sponge's target take U - Omega x r; the cells outside the
  cylinder the box holds (less the inlet's cells) are held at that state.
- **Test `rotor-frame`** (GPU): two flat blades pitched at +-45 degrees, Omega = 2, U = 1, 64³:
  the torque about the axis from the explicit scheme and from implicit pressure, averaged over
  t = 1.5 to 3, -3.99 and -4.06 (1.7%; threshold 5%). This checks the implicit rotating frame
  against the explicit one, which other tests (`rot-*`) check against theory. The plan's own
  `rotor-frame` (a helical paddle whose torque vanishes where its blades are edge-on to the
  relative wind) is not built.
- The display turned back to the tunnel's frame came later: the view's `turnWithFrame` (View
  section, Turn with frame; link `turn`) turns the volume view's camera back about the frame's
  axis by Omega t, t carried on between readouts by the steps submitted. Any 3D rotating frame
  but a shearing box; volume view only. (The power coefficient came with 5.2.) An empty rotor-frame tunnel holds U - Omega x r (kinetic energy to 1e-6 over 423
  steps; not discriminating, as that state is steady with or without the frame). Measured
  again with 5.2's tunnel at Omega = 2.5, 128³: after t = 1.6 the axial velocity a plane
  upstream is 1 to 6 digits and the swirl is the frame's.

### 5.2 As built (2026-11-20): the `wind-turbine` preset

- **The rotor**, `builtin:turbine` (`geometry/builtin.ts` `turbineRotor`): three blades about
  +x and a spherical spinner of radius 0.2 R. Chord 0.4 R at the root to 0.25 R at the tip
  (linear), NACA 00xx thickness of 20% thickened to 0.06 R where thinner (a blunt trailing
  edge), the section set at 10 degrees to the relative wind at a tip-speed ratio of 1 with the
  Betz axial induction (inflow (2U/3, -Omega r)), pivoted at 30% chord. The blade starts at
  0.08 R, inside the spinner, narrowing and keeping the spinner's pitch there, so no flat root
  cap is exposed. The mesh's `anchor: "origin"` (new) puts the hub on the axis.
- **The preset**: box 3, uniform 128³ (cells R/34; chords 14 to 8.5 cells, the thinnest
  section 2 cells by construction), liquid at Mach 0.1, nu = 1e-4, sponge 0.15, Omega = 1.25.
  Featured: the rotor's speed and the wind's. Default view: the volume of the density gradient
  (in the liquid, the pressure's), on which the blades and the vortex trail they shed stand out
  on a clear background. Vorticity was tried first: in the turning frame the whole box has the
  frame's 2 Omega, a haze over everything (adding 2 Omega back in the field would change the
  vorticity of every rotating-frame run, the binaries' included: not done).
- **Measured** (2026-11-20). In the page (static build, adapter "nvidia ampere"): torque
  +0.035 about the axis and thrust 0.29 at t = 15 and t = 30 alike, C_p 0.044; 7.4 ms a step
  and 24 ms a screenshot (render and read-back), so at most about 30 frames a second with a step
  a frame. The pane's own frame counter could not be read: the app's browser pane was throttled
  to a frame or two a second even when paused. Headless runner: C_p 0.033 at t = 1.5 and 3
  (still settling); over the rotor's speed at t = 2, C_p 0.038 at Omega = 0.6, 0.048 at
  0.9, about 0 at 1.6, -0.12 at 2.2. The featured rotor and wind speeds did not reach the
  coefficients' reference (C_p was over the preset's Omega whatever the control said): the
  reference is now made from the setup with the featured values in force. No errors in either.
- **Thinnest part**: the mesh's own measure (shortest way through along a triangle's inward
  normal) says 0.51 cells, at the trailing edge of a blade's root just outside the spinner
  (r = 0.23 R), a ray from the blunt edge's corner. Sections of the grid's solid cells there
  and at 0.75 R are continuous and 2 cells thick at the trailing edge.
- **What did not work, and why the rotor is slow.** The first rotor was pitched for a
  tip-speed ratio of 4 (chord 0.3 to 0.15 R, 6 degrees): the wind braked it at every speed
  (C_p -0.3 at Omega = 0.5 to -2.9 at 5; with the frame turned the other way the sign turned
  too). An off-axis sphere in the same frame gives the torque of its drag (c_y F_z = -0.04, the
  sign of drag against the turning), so the forces are right; the blades are not: a fast
  rotor's outer blade lies 3 to 10 degrees from the rotor's plane, and a blade two cells thick
  turns into a flat staircase that keeps none of that pitch, so it only feels drag. At a
  tip-speed ratio of 2 (still negative) and then 1 the pitch is 20 to 40 degrees; with the
  narrow blades C_p was 0.015 at best, with the wide ones 0.04.
- **The adaptive mesh fails for this body**: on `n = 256, amr = 2` the coarsest level sees a
  blade thinner than its cell in pieces, the surface that asks for refinement is only where
  those pieces are, and refinement stalled at 260 blocks with two blades left on level 1 (and
  a third broken into separate cells at mid-span). That run gave negative thrust and torque:
  a rotor that leaks. Refining on the mesh's own distance field (not on the coarse grid's
  solid cells) fixes it: built in 5.3. The plan's 9.2 (refining what the rotor sweeps) is the
  same need.
- **Not shown**: the blades are holes in the gas (stage 7's fractional cells would give them
  their shape below a cell); the tip vortices are a few cells across and fade within a rotor's
  diameter; no tower, nacelle or ground.

### 5.3 As built (2026-11-21): refinement from the body's shape, and what resolution gives a fast rotor

- **Refinement from the shape** (4.3 and 9.2 built; `setup/parts/surfaces.ts`,
  `amr/bodies.ts`). Each obstacle keeps every block within `refineBand` finest cells (4) of its
  surface at `refineLevel` (the finest), from the start and through every re-mesh, decided
  from the shape: exact distances for spheres, boxes and half-spaces, points on the triangles
  for a mesh (a 2D section from its inside on a grid). Below that level the block's indicator
  is infinite; at it, held between the merge and refine thresholds; such blocks are never
  evicted to make room. A block a re-mesh makes there by refinement takes its solid cells from
  the shape at its own level (`amr/regrid.wgsl`, a mark per cell from the CPU), and a cell of
  the body's that the shape does not cover at that level becomes gas. The band is wider than
  the ghost cells reach (2), so no coarser block's cells, nor ghosts filled from them, lie on
  the body: the coarse levels' idea of a thin body never meets the flow beside it. `refine:
  "swept"` takes the union of the surface over the run (or one period of its motions), poses
  sampled close enough that a point moves at most half the band between them: in place for
  stage 4 on blocks; for a body at rest it is the band. `refine: "off"` is the old behavior.
- **Tests.** `thin-plate-blocks`: a plate 0.9 coarsest cells thick, centered on a coarsest
  cell face (no coarsest center inside it), across a closed box of liquid at 128 with 2
  levels: the whole plate (65 536 finest cells) solid at the start, after 150 steps, and after
  the mesh is merged to the coarsest level (the plate gone: 0 solid cells) and refined back
  (65 536 again, painted), no cell within 6 finest cells of it coarser than the finest, and
  the mass on either side constant to 1e-8. With `refine: "off"` the same merge loses the
  plate for good and 3.4% of the liquid crosses where it was. `turbine-blocks`: the preset on
  256 with 2 levels, each blade's solid cells those of the shape on the finest grid (5146, 5153,
  5155), at the start and after 100 steps, nothing within 2 finest cells of the surface
  coarser. `force-balance` (all eight runs, blocks included), `mesh-vs-shape`, `rotor-frame`,
  `psplit-obstacle`, `liquid-amr` pass; `sphere-drag` gives Cd 1.2534 as before (its sphere
  was already refined right); the `wind-tunnel` preset is identical to the bit with the band
  on or off (260 blocks, Cd 0.894 at t = 1).
- **The preset on blocks** (256, 2 levels, 2.5 GB, 1331 slots): 526 blocks at the start, the
  three blades whole at the finest level (a slice through the rotor's plane in the page shows
  them inside the smallest blocks); the wake then fills the budget (1170 blocks, 29% of
  uniform) and a step takes 115 ms; with the flow's own refinement held off (`refhi=1`) 526
  blocks and 73 ms. The uniform 256^3 grid takes 208 ms. **The preset stays on uniform 128^3**
  (7.4 ms a step in the page): blocks are not interactive at this size, and the finer grid
  does not make the slow rotor honest in a way that pays for it.
- **The fast rotor's power against resolution** (`builtin:turbine-fast`: tip-speed ratio 4
  design, chord 0.30 R to 0.15 R, 6 degrees to the relative wind, thinnest 0.03 R). Same tunnel
  and frame as the preset, the band only (`refhi=1`), C_p averaged over t = 1.5 to 3 (2.5 for
  56 cells; each settled within a few percent by t = 1.5). Cells along the chord at 75% span:

  | cells along the chord | finest cell | blocks | GPU memory | ms / step | TSR 2 | TSR 3 | TSR 4 | TSR 5 |
  |---|---|---|---|---|---|---|---|---|
  | 14 | R/68 (256, 2 levels) | 526 | 2.5 GB | 61 to 71 | -0.155 | -0.227 | -0.469 | -0.937 |
  | 28 | R/137 (512, 3 levels) | 1065 | 3 GB | 82 to 90 | -0.110 | -0.141 | -0.314 | -0.790 |
  | 56 | R/273 (1024, 4 levels) | 2745 | 6 GB | 292 | | | -0.193 | |

  The torque is against the turning in every run (at TSR 4: -0.094, -0.063, -0.039 about the
  axis; thrust 0.97 to 1.05). Halving the cell takes about 0.15 off the loss at TSR 4 each
  time, first order; extrapolated to no cell at all (2 x 56 minus 28) C_p is still about -0.07.
  Resolution alone does not get this rotor to drive itself within any grid this machine
  holds: a staircase blade feels its drag and not enough of its lift, and the cure is the cut
  cells of 8.1, not more cells. TSR 6 (Omega = 7.5) did not run: the runner stopped advancing
  at t = 0.3 to 0.4 on both grids, no error reported, twice (not investigated).
- **Not done.** The view turned back to the tunnel frame. The TSR 6 hang. The slow (TSR 1)
  rotor's C_p on blocks against its uniform 0.044.

## 6. Stage 4: moving bodies

### 6.1 Motion in the setup format

```json
{ "kind": "obstacle", "id": "rotor", "shape": { "type": "mesh", "file": "turbine.glb", "node": "Rotor" },
  "motion": { "spin": { "axis": [1, 0, 0], "about": [0, 0, 0.3], "rate": "7 * U / R" } },
  "parent": "nacelle" }
```

| Motion | Parameters |
|---|---|
| `spin` | axis, a point on it, an angular rate |
| `slide` | a velocity |
| `swing` | an axis, an amplitude and a period (a flap, a pitching airfoil) |
| `path` | position and angles as formulas of `t` (`setup/expr.ts`) |

Rates may be formulas of `t`, so a rotor can be spun up over a turn. `parent` chains
transforms: tower, nacelle (yaw), hub (spin), blade (pitch). A glTF's node tree gives the
chain and each node's pivot; `motion` is then given per node name.

### 6.2 The stamp pass

Each step, before the hydro passes, one compute pass per moving part: for every cell near the
part's swept bounding box, take the cell's center into the part's frame, sample the distance
field, and write

- the mark (solid or gas), and
- for a solid cell, the body's velocity there,
  $`\mathbf{v}_s = \mathbf{V} + \boldsymbol\Omega\times(\mathbf{x} - \mathbf{c})`$, in the
  momentum slot, which a solid cell does not otherwise use.

Then the walls move:

- `load_vel` returns the stored velocity for a solid cell, not zero (`hydro/update.wgsl`, the
  block grid's own in `amr/update.wgsl`, and the diffusion loaders).
- `q_mirror` reflects about the wall's normal velocity: the image's normal velocity is
  $`2 v_{s,n} - v_n`$. `Mapped.vadd` already carries "twice a moving wall's velocity" for the
  box's moving faces; the solid-neighbor branch of `hydro/flux.wgsl` fills it from the solid
  cell.
- With viscosity the wall drags the gas along it (no slip at $`\mathbf{v}_s`$).
- **Implicit pressure.** A wall face is closed ($`\beta = 0`$) but it now carries a known
  normal velocity, which enters the divergence in the source pass. This is how the liquid
  learns that a body is pushing it, and it is the main reason moving bodies are easier in the
  implicit scheme than the explicit one: the push is felt everywhere within the step.
- **Timestep.** The body's fastest point enters the rates reduction: the body may cross at
  most the Courant fraction of a cell per step.
- The body does work on the gas, so the energy is not conserved, as with a moving wall of the
  box. The work is the force of 4.2 dotted with $`\mathbf{v}_s`$, tallied.

### 6.3 Cells that change kind

With a staircase the wall jumps a cell at a time.

- **Covered** (gas becomes solid): the cell's mass, momentum and energy go to its gas
  neighbors on the side the wall is moving toward, shared by their mass. Conservative.
- **Uncovered** (solid becomes gas): the cell takes the mean primitive state of its gas
  neighbors, with the wall's velocity, and the mass comes out of those neighbors.
  Conservative.

Both are gathers: a cell looks at the old and new marks of its neighbors and computes what it
gives or takes with one piece of code, so the two sides of an exchange agree. The stamp pass
writes the new mark to the other parity; a second pass does the exchange.

This is known to make the force noisy: each jump is a small piston stroke, and the pressure
spikes ([Seo & Mittal 2011][Seo2011]). The noise falls with the cell size and with the
implicit pressure step. Stage 4 measures it (test `galilean-drag`). If it is too large for the
torque of a rotor to mean anything after a running mean over one blade passage, section 8.1 is
the cure, and comes before stage 5.

### 6.4 Blocks

- The stamp and exchange passes run per block, as the update does.
- A block the body is about to enter must already be fine. Either re-mesh ahead of the body
  with a buffer of blocks, or refine everything the body will ever sweep, once (9.2).
- Ghost cells take the mark and the solid velocity as they take the state.
- Per-block timesteps: a body is stamped at each block's own time.

### 6.5 Done when

Tests `piston`, `galilean-drag`, `oscillating-cylinder` and `rotor-two-frames` pass on the
uniform grid, then on blocks; a spinning-rotor preset runs in the page.

### 6.6 As built so far (2026-11-20): the uniform grid, spheres and boxes

- **Format**: `motion` (`spin`, `slide`, `swing`, `path`; rates and paths numbers or formulas of
  t) and `parent` on obstacles (`setup/spec.ts`), poses and chains in `geometry/motion.ts`
  (unit tests `body-pose motion`: a hub on a turning nacelle against a worked example, position
  to 1e-9 and velocity to 1e-5).
- **On the GPU** the poses are WGSL generated from the setup (`hydro/moving.ts`), evaluated
  once a step at the step's end time by one invocation (`hydro/moving_pose.wgsl`): the step's
  dt is decided on the GPU, so the CPU cannot give the poses. Velocities by central differences
  of the poses (h = 1e-3). A formula's rate is integrated by Simpson's rule (64 pieces) there.
- **Stamp and exchange** run after the step (`hydro/moving_stamp.wgsl`,
  `hydro/moving_exchange.wgsl`), as 6.2 and 6.3 say, with the one departure that an uncovered
  cell takes 1/(n+1) of each of its n donors' states (a donor giving at most half), rather than
  the donors' mean: it starts lighter than its neighbors, a thin band of lower density behind a
  body. Mass is conserved to the bit; momentum is not (the body's push), nor energy (its work).
- **Moving walls in the fluxes** (`hydro/flux.wgsl`, under `MOVING`): the image's normal
  velocity is 2 v_s - v, and a wall face carries no mass, only the Riemann pressure and its
  work p v_s (a staircase face does not move until its cell changes kind, so mass across it
  would be lost into the body). `load_vel` gives a body's cells their velocity, so viscosity
  drags the gas along. The body's cells count in the timestep's rates.
- **Test `piston`** (GPU, 2D 64², a slab at Mach 0.085 through a closed box of ideal gas for
  t = 2): mass to the bit, the compressed gas's mean pressure 1.443 against the adiabat's
  1.456, the energy gained 0.18745 against the work done 0.18735.
- **Test `galilean-drag`** (GPU, built, **fails**): a cylinder 16 cells across in a periodic box
  of liquid (Mach 0.1, Reynolds number 100), at rest in a stream at U = 1 against sliding at -1
  through liquid at rest, drag averaged over t = 1 to 3. Explicit: C_d 1.32 at rest, 2.08
  moving (58% high), and the moving body's force history has an rms difference from the other
  of 11 times the mean: each cell the body crosses is a piston stroke. Implicit pressure: C_d
  1.35 at rest, 16 moving without the wall's velocity in the solve's source, and -4.0 with it
  (`pressure/source.wgsl`: a face onto a moving body's cell takes the cell's density times the
  wall's velocity; added, kept). **Decision 13, "staircase noise acceptable for rotors":
  no.**

### 6.7 Making the forces on moving bodies right (2026-11-20, second pass)

1. **The diagnostic.** The exchange pass now writes the momentum it gives the gas where a cell
   changes kind (an uncovered cell's m v_s, less what its donors gave), over dt, as a force on
   the body (WX, `hydro/forces.ts`; a covered cell's momentum stays in the gas). Test
   `force-balance-moving` (GPU): a cylinder sliding through a periodic box of liquid, the
   box's momentum change against the summed impulse over 200 steps, 6.3e-8 explicit and 5.4e-8
   implicit. The diagnostic is now the momentum the body gave the gas, exactly. It did not
   change `galilean-drag` (to the last digit at its sampled times: the exchange's share is
   small); sampled every step, the explicit moving body's force swings from -2.3 to +5.0 with
   the period of one cell crossed, where its mean should be about 0.17. **So the error is the
   staircase's jumps, not the diagnostic.**
2. **The wall's velocity in the implicit source: sign and weighting are right.** A liquid piston
   in a tube closed behind it: the pocket the piston leaves goes to -15 and lower (suction, as a
   stiff liquid must); with the wrong sign it would be compressed. With both ends open (the liquid
   should simply move at the piston's 0.1, the force near 0): explicit, 0.16 ahead of it and a
   force of -3 to -9; implicit with the source, 0.14 and +1 to +16. A covered cell hands its
   whole mass to one or two neighbors, doubling their density: in a liquid of sound speed 10 a
   pulse of c^2 rho ~ 100. The sign change in `galilean-drag` (16 without the source, -4 with
   it) was the mean of such pulses, not a sign error.
3. **A real error found on the way: the viscous wall assumed a wall at rest**
   (`hydro/diffusion_core.wgsl`: `dn = 2 v_gas`, face velocity 0), so a moving body's viscous
   force had the wrong sign (C_d's viscous part -2.2 where it should be about +0.3). Now
   `dn = 2 (v_r - v_l)` with the solid cell's velocity and the wall's velocity on the face: the
   same numbers for a wall at rest (regression: viscosity, force-balance, psplit-obstacle,
   psplit-cavity, liquid-cavity, rotor-frame, piston all pass).
4. **8.1's cheap form, built** (the liquid with implicit pressure, the uniform grid; on whenever
   such a run has moving bodies): each cell near a moving body gets a gas fraction phi and its
   upper faces open fractions a_f, from the bodies' signed distances at the step's end
   (`hydro/moving_stamp.wgsl`, `moving_dist`; the poses and fractions are now stamped before the
   solve). The solve takes the cell's true density rho / phi in the pressure, phi times its
   compressibility term, and beta = a_f on its faces, and the correction the same a_f on the
   pressure's push in the face mass flux (`pressure/source.wgsl`, `pressure/correct.wgsl`): as a
   body sweeps through a cell, its pressure rises and the solve pushes the liquid out. A cell
   becomes the body's only once phi < 0.05, nearly empty; an uncovered one starts with phi of the
   shares. The wall-velocity source is off with fractions (the swept volume carries the push).
   Results: the open-ended liquid piston moves the liquid at 0.087 to 0.095 (0.1 wanted), the
   force on it within +-0.9 (was +-16); the pressure dips to -80 in some cells the piston
   uncovers, which start nearly empty. **`galilean-drag`: still fails.** C_d (viscous fix in):

   | | At rest | Moving | Mean difference | rms difference / mean |
   |---|---|---|---|---|
   | explicit, staircase | 1.317 | 2.459 | 87% | 11.2 |
   | implicit, staircase (wall velocity in the source) | 1.348 | -4.03 | | 10.1 |
   | implicit, cheap 8.1 (fractions) | 1.348 | 3.565 | 164% | 4.4 |

   The noise fell by half; the mean did not come near. Most of the excess is the pressure part.
   Why the cheap form cannot fix it: the momentum update of a partly filled cell still takes
   the pressures of its full faces (the wall's share of a cut face pushes as if it were gas),
   the advection still crosses full faces into cells the body is sweeping, and the explicit
   fluxes still see the wall jump a whole cell when a cell flips, however little it holds.
   **Stopped here, as decided: the full 8.1 comes next, by its float64 prototype (8.1.1).**
5. **Not verified in the cheap form**: 3D, blocks, a gas, mesh bodies (none moves yet).
- **Not built**: blocks (6.4; refused there), moving mesh bodies (they need the distance field
  on the GPU), the work tally in the diagnostics (the tests sum it from the force), tests
  `oscillating-cylinder`, `spinning-cylinder` and `rotor-two-frames`, and a spinning-rotor
  preset. Not started, as the plan said, until moving drag is right.

## 7. Stage 5: bodies the flow moves

One degree of freedom first: a rotor's angle.

```math
I\,\dot\Omega = \tau_\text{flow} - \tau_\text{load}(\Omega), \qquad \dot\theta = \Omega ,
```

with $`\tau_\text{flow}`$ from 4.2 and the load a formula (a generator: $`k\Omega^2`$; a
brake; nothing). The moment of inertia is given, or computed from the mesh's volume and a
density. Integrated on the CPU from the last step's torque, read back with the diagnostics
(a step late, which is stable when the body is much denser than the gas: true of anything
solid in air, false of a light body in water, which this stage refuses by the density ratio).

Then free translation and all six degrees of freedom, with gravity: a falling, tumbling body.
Collisions between bodies and with the walls are not covered.

Result: start a turbine at rest in a wind and watch it reach its tip-speed ratio.

## 8. Stage 6: accuracy

Three independent pieces, each built only if a measurement asks for it.

### 8.1 Partly filled cells

The staircase is first order, rough on a smooth body (stage 0's inviscid sphere), and noisy
when it moves (6.3). The distance field knows where the surface is within a cell, and the
cure is to use it.

- Each face gets an **open fraction** $`a_f`$ from the distance at the face's corners, formed
  on load by one function of the face (both cells of a face then form the same bits). Each
  cell gets a gas fraction $`\phi`$.
- The update becomes
  $`\phi\,\partial_t U = -\sum_f a_f F_f / \Delta x + (\text{wall pressure on the cut}) `$,
  and for a moving body $`\partial_t\phi`$ matches the volume the wall sweeps, which replaces
  the jumps of 6.3 with a continuous change.
- The pressure solve takes $`\beta_f = a_f/\rho_f`$ in place of 0 or 1: its coefficient
  textures already exist, so this part is nearly free ([Ng, Min & Gibou 2009][Ng2009]).
- **Small cells.** A cell with $`\phi \to 0`$ cannot take an explicit advective step. The
  remedy that fits a gather is state redistribution ([Berger & Giuliani 2021][Berger2021]): a
  small cell is averaged with its neighbors after the update, conservatively.
- Reconstruction near the wall, the viscous wall stress at the true distance, the refinement
  faces of the adaptive mesh and the force sum all need their cut-face forms.

This is the largest piece of the plan, comparable to the implicit pressure work. A float64
prototype in `src/reference/` comes first, on the cylinder and the piston, as
`reference/psplit.ts` did for the pressure split.

A cheaper alternative to measure first: keep the staircase for the fluxes and use only
$`\beta_f = a_f/\rho_f`$ in the pressure solve, with the swept volume as a smooth source. For
the liquid that may remove most of the noise by itself. **Measured (6.7): it halves the noise
and does not fix the mean; the full form is needed.**

#### 8.1.1 What the float64 prototype needs (2026-11-20)

A reference solver in `src/reference/` (as `reference/psplit.ts`), 2D, float64, on the CPU, of
the liquid with implicit pressure and a moving rigid body, before any GPU work:

- **Geometry, exact.** For a circle and a turned rectangle: each cell's gas fraction phi and
  each face's aperture a_f by clipping the cell's square against the shape (polygon clipping,
  or the circle's segment areas), at the step's start and end; the cut face's length and normal
  in each cut cell; the swept volume of each cell over the step, phi(t + dt) - phi(t), to
  round-off.
- **The conservative update** phi dU/dt = -(1/dx) sum_f a_f F_f + (the cut face's flux), the
  cut face's flux being the moving-wall one (no mass through it, the pressure p* on it and its
  work p* v_s n, viscosity at the true distance to it): no cell flips. A cell born from the body
  starts at phi = 0 and fills by the fluxes; one being swallowed empties by them.
- **Small cells: state redistribution** (Berger & Giuliani 2021): after each update a cell with
  phi below about 1/2 is merged with a neighborhood, its own and their states replaced by
  volume-weighted averages, conservatively. This replaces the exchange of 6.3 entirely.
- **The pressure solve with beta_f = a_f / rho_f and the swept volume in its source** (Ng, Min &
  Gibou 2009): div(a_f u_f) = -(1/dt) (swept volume) in each cut cell, the same a_f on the
  correction's mass flux, so that solve and correction agree to round-off.
- **Forces from the cut faces**: the integral of p n and the viscous stress over each cut face,
  which must equal the momentum balance of a box around the body to round-off (as
  `force-balance-moving` checks now).
- **Tests the prototype must pass before the GPU port**: Galilean drag (moving and at rest within
  2 to 3% in the mean, the moving body's force noise under 10% after a running mean over one
  cell crossed); the inviscid cylinder's drag (should be nil) falling at better than first
  order; the open-ended piston (the liquid at the piston's speed to 1%, the force near nil);
  the oscillating cylinder of Dütsch et al. (1998), its drag history.
- **Then the GPU port**: phi and a_f textures from the distance field (analytic shapes first,
  then mesh bodies, whose distance field must go to the GPU for this), the cut-cell update in
  `hydro/update.wgsl` (the uniform grid first), a redistribution pass, apertures in the source
  and correction, and the force sum over cut faces; then blocks. Comparable in size to the
  implicit pressure work (plans/incompressible.md).

#### 8.1.2 The float64 prototype, as built (2026-11-21)

`src/reference/cutcell.ts` (unit tests `tests/unit/cutcell.test.ts`, scripts in
`tools/cutcell/`, run with `node --experimental-strip-types`): 2D, periodic, the liquid
(p = c^2 (rho - rho0)) with implicit pressure, a rigid convex polygon (a circle of 256 sides)
moving and turning. What it does:

- **Geometry**, exact for the polygon: phi by clipping each cell's square, each face's open
  share and the first moment of its open part by clipping the face, the wall vector by closure
  (W = -sum_f a_f n_f), the wall's distance and nearest point from the gas centroid. Over a
  step the faces' shares are averaged by the trapezoidal rule on `subSteps` parts (4): with
  only midpoints, a face that opens late in a step was shut through it, leaving a newly
  uncovered cell with no open face, and the piston blew up at some sample counts.
- **Update**, per whole cell (Q = phi rho, M = phi rho v): MUSCL advection through the open
  faces (one-sided beside the wall), viscosity through them, nothing through the wall; the
  pressure solve (phi'/c^2) p + (dt^2/dx^2) sum a_f (p - p_nb) = Q* - phi' rho0, which empties
  a cell being covered and fills one being uncovered; the mass it moves carrying the mean
  velocity of its two cells (the upwind one beside a small cell); the face pressures at their
  open parts' centroids, the wall pressure carried to the wall along the gradient; the viscous
  wall stress at the true distance, point-implicit. Then state redistribution (Berger &
  Giuliani 2021, unweighted, first order) for cells under half full. A cell covered with no
  open face left hands what it holds to its neighbors (round-off: 5e-13 of the mass in the
  Galilean run).
- **Measured**:
  - mass to 1e-15 and the box's momentum balance against the force on the body to 1e-15,
    at rest, moving and turning;
  - liquid at rest around a body at rest stays at rest, force 0 to round-off;
  - **Galilean drag** (box 4, 256^2, 16 cells across, Re 100, Mach 0.1, t = 1 to 2.2): C_d
    1.405 at rest, 1.412 moving: 0.5% apart (criterion 2 to 3%); rms difference of the two
    histories 0.75% of the mean, the moving body's raw noise 1.2% (criterion 10% after a running
    mean). The GPU's staircase gave 87% and 164% apart (6.7);
  - **the open-ended piston**, started over 0.5: the liquid at 0.5 to 0.2% (criterion 1%) at
    32, 64 and 128; the force's mean near nil (|mean| <= 0.05 against a dynamic pressure times
    height of 0.25); what is left is sound sloshing between the plate's faces (period 0.36),
    excited by the start;
  - **the inviscid cylinder**, started from the potential flow, C_d over t = 0.1 to 0.3 at 8,
    16, 32, 64 cells across: 0.96, 0.49, 0.23, 0.12, **first order** (criterion: better than
    first order, not met); a staircase in the same prototype: 1.15, 0.89, 0.69, 0.52. The
    error is the rear's pressure recovery (Cp 0.33 against 0.62 at 32 across), not the force's
    sum: the face centroids, the wall's pressure gradient, one-sided slopes at the wall and a
    central carry changed it by under 3% each; halving the timestep changed nothing. Likely
    first-order terms left: the unweighted redistribution (threshold 0.25 instead of 0.5: 7%
    less drag), and the pressure's compact face gradient between cut cells' centers, not their
    centroids.
- **Not done**: the oscillating cylinder of Dütsch et al.; second order at the wall; walls of
  the box other than periodic; a body crossing the periodic edge.


### 8.2 A wall model

At high Reynolds number a no-slip wall on this grid gives the wrong stress (the first cell is
hundreds of wall units thick), and a slip wall gives none. A wall-function wall takes the
tangential velocity at a point one or two cells out along the distance field's gradient, and
sets the wall stress from the log law. An option of the body: `"wall": "slip" | "noslip" |
"model"`. It needs `tunnel` to know the true viscosity apart from the one the grid resolves.

### 8.3 Eddy viscosity

The viscosity is one constant (`P.diff`). A Smagorinsky or WALE eddy viscosity makes it a
function of the local strain, computed in the diffusion pass from the velocity gradients it
already forms. Whether it beats the scheme's own dissipation on these grids is a measurement
(the sphere's drag at Reynolds numbers of $`10^4`$ to $`10^5`$, against 0.4 to 0.5), not an
assumption: many Godunov codes do without.

## 9. What a wind turbine costs

Rough numbers for a three-bladed rotor of radius $`R`$ at a tip-speed ratio of 7, blades with
a chord of $`0.05R`$ over most of the span.

### 9.1 Cells and steps

- 16 cells along the chord: $`\Delta x = R/320`$. The blade is 3 cells thick. That is the least
  that deserves the name, and it needs section 8.1 to be more than a rough paddle.
- A box of $`6R \times 4R \times 4R`$ at that cell size would be $`3\times10^9`$ cells. So the
  adaptive mesh: 5 levels above a base of $`96\times64\times64`$.
- The tip moves at 7 times the wind, and sets the step: at a Courant number of 0.4, about
  5 000 steps of the finest level per turn. Ten turns for the wake to form: 50 000 steps.

### 9.2 Refining what the rotor sweeps

The blades sweep a disk about $`0.1R`$ thick. Refined once to the finest level, that is
$`\pi R^2 \times 0.1R / \Delta x^3 \approx 10^7`$ cells: 640 MB of state. A mesh that follows
the blades would hold perhaps a fifth of that and re-mesh every 20 to 40 steps. With 2.5 GB
the swept disk fits, costs no re-meshing, and has no refinement boundary crossing a blade.
Start there (`"refine": "swept"`); following the blades is an optimization for later.

### 9.3 The alternative that does not resolve blades

Wake studies of turbines mostly do not resolve the blades. They replace each blade by a line
of forces taken from tables of its airfoils' lift and drag ([Sørensen & Shen 2002][Sorensen2002]),
or the rotor by a disk of force. That needs no distance field and a grid of
$`\Delta x = R/30`$. It answers "what is the wake, what does the next turbine see", not "what
does this blade shape do". It is a small feature (a body force, as the driving is) and is
listed in section 12 as a possible stage of its own.

## 10. Stage 7: display

- **The body as a surface**: the ray marcher stops at the zero of the distance field and
  shades it, lit, with the distance field's gradient as the normal. Colored plain, or by the
  pressure coefficient $`C_p = (p - p_\infty)/(\tfrac12\rho U^2)`$ of the gas beside it.
- **Streamlines** from a rake of seed points at the inlet, reusing the field-line tubes
  (`render/fieldlines.ts`) with the velocity for the field (check 2 of stage 0).
- **Smoke**: the passive scalar released from points or a rake in the inlet, drawn by the
  volume renderer.
- **Fields**: vorticity magnitude and the Q criterion, if stage 0 finds them absent.
- **Runner**: all of the above in `--views`, and the force series as a plot.

## 11. Validation tests

Thresholds are set from stage 0 and from the papers, after the papers are checked.

### 11.1 Unit tests (vitest, CPU)

| id | Criterion |
|---|---|
| `mesh-parsers` | each format's fixture gives the same triangles; binary and ASCII STL agree; an OBJ with quads; a GLB with a rotated child node |
| `sdf-sphere` | an icosphere's distance field against $`r - R`$ within the band, to a fraction of a cell |
| `sdf-volume` | the volume of cells with negative distance against the mesh's own volume: converges at second order |
| `sdf-broken` | the same sphere with a tenth of its triangles removed, and with half of them flipped: the same inside to within a cell |
| `sdf-thin` | a plate thinner than a cell is reported as such |
| `body-pose` | a chain of three transforms gives the positions and velocities of a worked example |

### 11.2 GPU tests (`src/validation/`)

| id | Stage | Criterion |
|---|---|---|
| `force-balance` | 2 | wall-face force and control-volume force agree to round-off, explicit and implicit, uniform and blocks |
| `cylinder-forces` | 2 | Reynolds number 100: mean drag coefficient near 1.33 to 1.38, lift amplitude near 0.33, Strouhal number near 0.165 (values to check) |
| `sphere-drag` | 2 | Reynolds number 100: drag coefficient near 1.09 (to check) |
| `mesh-vs-shape` | 2 | a sphere from an STL and the built-in sphere mark the same cells and give the same drag |
| `rotor-frame` | 3 | a flat-bladed paddle in its own frame: torque steady, and zero at the tip-speed ratio where the blades are edge-on to the relative wind |
| `piston` | 4 | a wall moving into a closed tube of gas: mass conserved to round-off, pressure on the adiabat, the work done equals the energy gained |
| `galilean-drag` | 4 | a body moving at $`-U`$ through gas at rest and gas at $`U`$ past the body at rest: the same mean drag to a few percent; the noise of the first reported |
| `oscillating-cylinder` | 4 | a cylinder oscillating in line in fluid at rest, Reynolds number 100, Keulegan-Carpenter number 5 ([Dütsch et al. 1998][Dutsch1998]): the drag history |
| `spinning-cylinder` | 4 | the sign and size of the Magnus lift |
| `rotor-two-frames` | 4 | the paddle of `rotor-frame`, run with moving cells in the tunnel's frame: the same mean torque within the noise |
| `free-rotor` | 5 | from rest, the paddle reaches the tip-speed ratio where `rotor-frame` gives the load's torque |
| `cut-convergence` | 6 | the inviscid sphere's spurious drag and the cylinder's drag converge at better than first order |

Showcases, not pass or fail, because they are at Reynolds numbers the grid does not resolve:
the Ahmed body, the DrivAer car (public meshes), a published turbine rotor. Each is reported
beside its measured value with the difference stated.

## 12. Not covered

- **Bodies that bend** (aeroelastic blades, flags, sails). Each part is rigid.
- **Contact** between bodies, and bodies leaving the box.
- **STEP and IGES** files.
- **Actuator lines and disks** (9.3): a small stage of its own if wakes are wanted more than
  blades.
- **Boundary layers resolved** at flight or road Reynolds numbers, transition, the drag
  crisis.
- **Heat**: the walls are adiabatic. A heated body is a small extension of the wall's
  conduction flux.
- **MHD with obstacles**, which the code refuses today.
- **Curved grids** (axisymmetric, spherical): the tunnel is Cartesian.
- **Sound** radiated by the body: the sponge and the liquid's implicit step both damp it.

## 13. Open decisions

| Decision | Where it is settled | Default if nothing argues otherwise |
|---|---|---|
| Distance field on the CPU in a worker, or on the GPU | 3.2, by timing a million triangles | CPU worker |
| Distance texture `r16float` or `r32float` | 3.2 | `r16float`, distances in cells |
| Mesh shapes as a `Shape` member or a new part kind | 3.3 | a `Shape` member |
| Force from wall faces or from a control volume | 4.2 | wall faces; the other as a test |
| Body index in the mark, or a second texture | 4.2 | in the mark |
| Rotor's frame before moving cells | check 1 of stage 0 | yes if it is a week or less |
| Staircase noise acceptable for rotors | `galilean-drag`, `rotor-two-frames` | no: 8.1's cheap form first, then decide on the full one |
| Follow the blades or refine the swept disk | check 3 of stage 0, 9.2 | the swept disk |
| Eddy viscosity | 8.3's measurement | none |
| Explicit ideal gas with moving bodies (transonic propellers) | after stage 4 for the liquid | later |

## 14. Order of work

1. Stage 0's checks (a day).
2. Force diagnostics (4.2) and stage 0's measurements with the built-in shapes.
3. Parsers and the distance field (stage 1), with their unit tests.
4. The `mesh` shape, file handling in the page, the runner and the MCP server.
5. The `tunnel` part, the sponge, the preset. **First release: static bodies.**
6. Rotor's frame (stage 3), if the check allows.
7. Motion in the format; stamp pass; moving walls in the fluxes and the pressure source;
   uniform grid, liquid, implicit pressure. Tests `piston`, `galilean-drag`.
8. The same on blocks, with the swept refinement. **Second release: prescribed motion.**
9. Decide on 8.1 from the noise. Prototype in float64 if it is wanted.
10. Stage 5.
11. Display (stage 7) can go in at any point after step 4; the surface rendering is worth
    doing early, since without it a loaded mesh is seen only as a hole in the gas.
12. `numerics.md` and `setups.md` sections with the measured numbers at each release.

## 15. References

- [Jacobson, Kavan & Sorkine-Hornung (2013)][Jacobson2013], "Robust inside-outside
  segmentation using generalized winding numbers", ACM Trans. Graph. 32, 33.
- [Barill et al. (2018)][Barill2018], "Fast winding numbers for soups and clouds", ACM Trans.
  Graph. 37, 43.
- [Seo & Mittal (2011)][Seo2011], "A sharp-interface immersed boundary method with improved
  mass conservation and reduced spurious pressure oscillations", J. Comput. Phys. 230, 7347.
- [Ng, Min & Gibou (2009)][Ng2009], "An efficient fluid-solid coupling algorithm for
  single-phase flows", J. Comput. Phys. 228, 8807.
- [Berger & Giuliani (2021)][Berger2021], "A state redistribution algorithm for finite volume
  schemes on cut cell meshes", J. Comput. Phys. 428, 109820.
- [Dütsch, Durst, Becker & Lienhart (1998)][Dutsch1998], "Low-Reynolds-number flow around an
  oscillating circular cylinder at low Keulegan-Carpenter numbers", J. Fluid Mech. 360, 249.
- [Sørensen & Shen (2002)][Sorensen2002], "Numerical modeling of wind turbine wakes",
  J. Fluids Eng. 124, 393.
- Mittal & Iaccarino (2005), "Immersed boundary methods", Annu. Rev. Fluid Mech. 37, 239: the
  review of the methods of sections 6 and 8.
- Kwatra et al. (2009): as cited in `plans/incompressible.md`.

The citations above were written from memory and then checked against Crossref and the papers'
own pages (2026-11-20): all correct. Ng, Min & Gibou's title has an en dash ("fluid–solid");
Mittal & Iaccarino's DOI is 10.1146/annurev.fluid.37.061903.175743.

[Jacobson2013]: https://doi.org/10.1145/2461912.2461916
[Barill2018]: https://doi.org/10.1145/3197517.3201337
[Seo2011]: https://doi.org/10.1016/j.jcp.2011.06.003
[Ng2009]: https://doi.org/10.1016/j.jcp.2009.08.032
[Berger2021]: https://doi.org/10.1016/j.jcp.2020.109820
[Dutsch1998]: https://doi.org/10.1017/S002211209800860X
[Sorensen2002]: https://doi.org/10.1115/1.1471361
