# Plan: wind tunnel runs of arbitrary objects

Status (2026-11-20): proposed, nothing built. Published values and citations below are from
memory; each is checked against its paper before it becomes a test threshold. Cost figures in
section 9 are estimates, not measurements.

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
the liquid that may remove most of the noise by itself.

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

The citations above are from memory and have not been checked against the publishers' records;
check each before any of this reaches `numerics.md` or the paper.

[Jacobson2013]: https://doi.org/10.1145/2461912.2461916
[Barill2018]: https://doi.org/10.1145/3197517.3201337
[Seo2011]: https://doi.org/10.1016/j.jcp.2011.06.003
[Ng2009]: https://doi.org/10.1016/j.jcp.2009.08.032
[Berger2021]: https://doi.org/10.1016/j.jcp.2020.109820
[Dutsch1998]: https://doi.org/10.1017/S002211209800860X
[Sorensen2002]: https://doi.org/10.1115/1.1471361
