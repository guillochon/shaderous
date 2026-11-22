# Architecture

This page explains how Shaderous maps a finite-volume hydrodynamics code onto WebGPU: what
the interface allows and forbids, the design choices that follow, and how a step moves through
the GPU. The physics and equations are in [numerics.md](numerics.md).

**Contents**

1. [Overview](#overview)
2. [What WebGPU constrains](#what-webgpu-constrains)
3. [Source layout](#source-layout)
4. [Shader assembly](#shader-assembly)
5. [State storage](#state-storage)
6. [Anatomy of a uniform-grid step](#anatomy-of-a-uniform-grid-step)
7. [Gathered fluxes and bit-identical faces](#gathered-fluxes-and-bit-identical-faces)
8. [Boundaries without ghost cells](#boundaries-without-ghost-cells)
9. [Reductions and the GPU-resident timestep](#reductions-and-the-gpu-resident-timestep)
10. [Multigrid on textures](#multigrid-on-textures)
11. [Adaptive mesh](#adaptive-mesh)
12. [Per-block timesteps on the GPU](#per-block-timesteps-on-the-gpu)
13. [Rendering](#rendering)
14. [Frame loop and pacing](#frame-loop-and-pacing)
15. [GPU memory budget](#gpu-memory-budget)
16. [Input, output and links](#input-output-and-links)
17. [Testing](#testing)
18. [Performance notes](#performance-notes)
19. [Limitations and lessons that carry over](#limitations-and-lessons-that-carry-over)

---

## Overview

Shaderous is a TypeScript application built with Vite. It has no runtime dependencies apart
from [h5wasm](https://github.com/usnistgov/h5wasm), which it loads only when a user imports or
exports a file. Every fluid, gravity and rendering computation runs in WGSL shaders on the
user's GPU through [WebGPU](https://www.w3.org/TR/webgpu/). The CPU:

- holds the settings and the adaptive-mesh tree (a few kilobytes);
- encodes command buffers;
- reads back, without ever blocking, diagnostics and the per-block refinement flags.

Stepping never waits for the CPU, and the CPU never waits for a step.

```
          CPU (TypeScript)                              GPU (WGSL)
 ┌───────────────────────────────┐        ┌────────────────────────────────────────────┐
 │ UI, presets, link <-> config  │ encode │ rates -> clock -> predictor -> update      │
 │ shader assembly (preprocess)  │──────▶ │ -> [species] -> [Poisson V-cycles]         │
 │ AMR tree (blocks, neighbors)  │        │ -> [second half-kick, point masses]        │
 │ async readbacks (diag, flags) │◀────── │ -> render (ray-march or slice)             │
 └───────────────────────────────┘ mapAsync└────────────────────────────────────────────┘
```

## What WebGPU constrains

Four properties of the interface (not of the hardware) shaped the design.

| Constraint | What follows from it |
|---|---|
| A compute pass cannot read and write the same `rgba32float` texture. A four-channel float texture is either sampled (read) or a write-only storage texture. | Every update reads one copy of the state and writes the other (ping-pong). Ghost cells cannot be filled in place, so they are **computed on load** instead ([below](#ghost-cells-computed-on-load)). Single-channel `r32float` textures *can* be read and written (needs the `readonly_and_readwrite_storage_textures` WGSL feature), and the red-black Gauss-Seidel smoother relies on that. |
| Few storage buffers per stage, and 128 MB per binding by default. | The state lives in **3D textures**, not buffers. On the same flux stencil, storage buffers measured about 1.4 times slower than textures. |
| No 64-bit floats in WGSL. | Everything is `f32`. The scheme has to hold hydrostatic balance and the internal energy of fast flows at that precision: hence dual energy, well-balanced reconstruction, settled initial states, and series forms of the degenerate equation of state ([numerics.md](numerics.md#7-robustness-in-single-precision)). |
| Readback is asynchronous: data reaches the CPU through a mapped buffer, frames later. | The **timestep is computed and used on the GPU**. The CPU reads diagnostics and refinement indicators whenever they arrive and never waits for them. |

WGSL also has no dynamic dispatch or function pointers. Physics modules are therefore chosen
when the shaders are compiled ([below](#shader-assembly)).

At startup `src/gpu/device.ts` asks for a high-performance adapter. It checks a short list of
minimum limits and refuses to run, with a clear message, if any is missing. It then requests
the adapter's full buffer sizes, plus `timestamp-query` if available.

## Source layout

```
src/
  gpu/         device setup and limit checks, shader preprocessor, hierarchical reductions,
               GPU memory budget, timestamp timing
  hydro/       rates, clock, predictor and update passes; riemann/ and limiters/ modules;
               dual energy, diffusion, cooling hooks, species passes
  eos/         WGSL equation-of-state modules and their float64 mirrors (eos.ts); the tables
               of white dwarf matter, hydrogen and helium and the ANEOS materials (aneos/),
               loaded only when a run uses them (tables.ts). table.ts loads and binds all three
               and has the Hermite basis (table_tex.wgsl on the GPU); the free-energy lookup and
               its solve for T (table.wgsl) are the first two's, ANEOS's its own (aneos.wgsl)
  gravity/     multigrid passes, multipole boundary, half-kick corrector, point masses, sinks
  pressure/    the implicit pressure step: its multigrid for alpha p - div(beta grad p) = b,
               the face velocities it corrects, and their passes for the liquid and the gas
  radiation/   gray M1 radiation transport: advection, the implicit exchange with the gas,
               escape and diagnostics; opacity/ has the opacity modules it slots in
  mhd/         constrained transport, HLLD, the field's passes on blocks
  burn/        the nuclear network surrogate and its reference network
  debris/      bound and unbound mass, and the giant-impact census (impact.ts)
  cooling/     WGSL cooling functions and their float64 mirrors
  species/     packing and bookkeeping of mass fractions
  amr/         block tree (tree.ts), atlas passes and the adaptive driver (amr_sim.ts);
               gravity/ for the Poisson solve on blocks
  render/      field extraction, ray-marcher, slice view, picking, transfer functions
  edit/        the brushes' passes on the uniform grid (amr/ has the blocks')
  setup/       the setup document: its types, the option table, reader and writer, and the
               compiler that turns one into a Problem; parts/ has the stars, orbits, disks,
               regions and the built-in flows
  presets/     one JSON setup document per problem of the menu
  problems/    the Problem type and the libraries the parts use: Lane-Emden polytropes, cold
               white dwarfs, settle.ts (discrete hydrostatic balance)
  reference/   float64 CPU version of the hydro scheme, exact Riemann solver
  units/       physical constants, unit systems, composition
  validation/  the in-page validation suite (validate.html)
  io/          snapshots, Athena++ HDF5 reader and writer, checkpoints, capture
  ui/          the simulator front end, shareable links
  schema/      the generated schema and doc tables (`npm run gen`, below)
  sim.ts       owns the textures and encodes a uniform-grid step
tests/         unit/ (Vitest, Node) and gpu/ (Playwright, real GPU)
tools/burn/    offline Python tools for the nuclear-burning surrogate
tools/eos/     offline tools that build the equation-of-state tables, and their notes (NOTES.md)
tools/headless/ the command-line runner and the MCP server
paper/         the method paper (AASTeX)
```

## Shader assembly

`src/gpu/preprocess.ts` is a small preprocessor (about 60 lines). Vite's `import.meta.glob`
bundles every `.wgsl` file under `src/` into a registry keyed by path, and `assemble(entry, slots)`
expands two kinds of include:

```wgsl
#include "hydro/common.wgsl"   // a file from the registry, pulled in once per assembly
#include <eos>                 // a slot: whatever the caller maps it to
```

A slot maps to either a registry path or literal WGSL text. This is how the equation of state,
Riemann solver, limiter, cooling function, diffusion terms and species hooks are swapped:

```ts
assemble('hydro/update.wgsl', {
  eos: 'eos/degenerate.wgsl',
  riemann: 'hydro/riemann/hllc.wgsl',
  limiter: 'hydro/limiters/koren.wgsl',
  species: speciesSlot(groups),          // 'hydro/species_on.wgsl' or 'hydro/species_off.wgsl'
});
```

Features that are off are compiled out entirely. A run without diffusion, cooling or species
pays nothing for them, because their slots map to stubs (`*_off.wgsl`) whose constants let the
compiler drop the code. Changing a module in the interface rebuilds only the affected pipelines.
Each equation of state also declares compile-time flags such as `EOS_HAS_ENERGY` and
`EOS_KEEP_TOTAL`, which the shared passes branch on statically.

## State storage

On the uniform grid the state is two pairs of `rgba32float` 3D textures:

| Texture | Channels |
|---|---|
| `U1` | $`\rho`$, $`\rho v_x`$, $`\rho v_y`$, $`\rho v_z`$ |
| `U2` | $`E`$, $`\rho e`$, $`\rho s`$ (passive scalar), spare |

Each pair has a ping and a pong copy; a step reads one and writes the other. Other storage:

- **Gravity:** one `r32float` texture per multigrid level for each of $`\Phi`$, the right-hand side
  and the residual, plus an `rgba32float` field texture holding $`-\nabla\Phi`$ (with the
  point-mass pull added in).
- **Species:** an extra `rgba32float` texture with four mass fractions per texel, the groups of
  four stacked along $`z`$. Group $`g`$ of cell $`(x, y, z)`$ is texel $`(x, y, z + g\,n_z)`$.
  When the species set the equation of state, one more `rgba32float` texture holds each cell's
  four composition moments (`species/mixture.ts`); hydro passes read it at binding 27, and the
  passes that write it use binding 26. The species passes also read the predictor's half-step
  state (binding 24), for how far toward first order it went where the gammas jump.
- **Equation-of-state tables:** a tabulated equation of state reads one `rgba32float` 2D
  texture, uploaded once per device and handed to every bind group whose shader reaches it
  (`setSharedBinding` in `gpu/device.ts`). White dwarf matter, hydrogen and helium, and the
  ANEOS materials all use binding 22 (`EOS_TAB`, declared in `eos/table_tex.wgsl`), since a run has only one;
  for a pair of ANEOS materials, the mixture's levels are stacked along the texture's rows. The files are fetched only when a
  run picks that equation of state: 2.2 MB per ANEOS material, 5.2 MB per hydrogen–helium
  composition.
- **Radiation:** $`(E, \mathbf{F})`$ and the gas's energy and momentum gained from it, each in
  two parities that the advection, the implicit sweeps and the commit to the gas alternate
  between (`radiation/passes.ts`).
- **Implicit pressure:** per multigrid level, the pressure, right-hand side, residual and
  coefficient textures of `pressure/multigrid.ts`, beside gravity's.

By default the uniform grid has **no primitive-variable textures**: the predictor and the update
convert the conserved state as they load it, about twenty times a cell in a 3D step. Where memory
bandwidth is scarce this is the cheaper choice: on the GPU where it was measured, the flux update
with all of its arithmetic removed still took 85% as long, so a pass that only rewrites the state
in another form cost more than repeating the conversion. On data-center GPUs (an A100) the update
is limited by arithmetic instead: doubling its texture loads costs 4%, a cheaper limiter saves
17%. And an equation of state that solves for its temperature or reads a table makes each
conversion expensive. **Stored primitives** (`primStored` in `sim.ts`, the setup key
`physics.storedPrims`) therefore convert each cell once a step, in the rates pass, into two
textures that `load_q` reads (`hydro/prim_load_on.wgsl`, `hydro/prim_store_on.wgsl`). The values
are the conversion's own, so results change only at round-off: bit for bit on some drivers,
while others compile the conversion a little differently in the rates pass than in the predictor
(1e-8 relative after 30 steps with the degenerate gas on an A100, 1e-6 with Helmholtz). They cost 32 bytes per cell. On by default for every equation of state but the
ideal gas, the barotropic gas and the liquid: on an A100 a 3D white dwarf's step is 3.0 to 3.1
times faster with Helmholtz and 1.35 to 1.46 times with the degenerate gas, an ANEOS planet's 2.7
to 3.9 times, while the ideal gas gains 1 to 3% there and loses 3 to 5% on an RTX 3080. (The adaptive mesh stores its primitives anyway,
ghosts included: `amr/assemble.wgsl`.)

## Anatomy of a uniform-grid step

`sim.ts` encodes a step as a sequence of compute passes in one command buffer:

| Pass | Shader | Work |
|---|---|---|
| 1. Rates | `hydro/rates.wgsl` | Reads the state once and writes the first stage of the timestep reduction: max signal rate and max $`\lvert g\rvert`$ per $`4^3`$ block, folded in workgroup-shared atomics. With stored primitives it also writes each cell's primitives for the predictor and the update. |
| 2. Clock | `hydro/clock.wgsl` + `gpu/reduce.wgsl` | Finishes the reduction, applies the point-mass and output limits, and writes $`\Delta t`$ and the time to a small buffer that later passes read. |
| 3. Predictor | `hydro/predictor.wgsl` | Limited slopes, the half-step primitive state, shock and first-order flags. |
| 4. Update | `hydro/update.wgsl` | Face states, Riemann solves on all $`2d`$ faces, diffusion fluxes, the conservative update, the first half kick, cooling, dual-energy sync, floors. |
| 5. Species *(optional)* | `hydro/species*.wgsl` | Move each species with the face mass fluxes the update recorded. |
| 6. Implicit pressure *(optional)* | `pressure/source*.wgsl`, `pressure/multigrid.ts`, `pressure/correct*.wgsl` | The pressure equation from the old and the advected state, its V-cycles, and the correction of the face fluxes (with self-gravity, the potential is solved first, for the advected density). Replaces the second half kick. |
| 7. Radiation advection *(optional)* | `radiation/advect.wgsl` | $`E`$ and $`\mathbf F`$ moved with the update's face mass fluxes. |
| 8. Gravity *(optional)* | `gravity/advance.wgsl`, `gravity/rhs.wgsl`, V-cycles, `gravity/field.wgsl` | The guess extrapolated in time, the new source, V-cycles, the field texture. |
| 9. Correct *(optional)* | `gravity/correct.wgsl`, `bh_*.wgsl` | Second half kick and the flux-form work moved to the mean of the two potentials; point masses kick, drift and accrete. |
| 10. Radiation sweeps *(optional)* | `radiation/sweep.wgsl`, `commit.wgsl`, `escape.wgsl` | $`N_r`$ sweeps of transport and exchange with the gas, then their commit to the state. |
| 11. Burning *(optional)* | `hydro/burn*.wgsl` | The network surrogate on the step's end state. |

With a magnetic field, passes 3 and 4 are the field's own two-stage step (`mhd/`).

`sim.profile()`, or **Time GPU passes** in the Performance panel, reports the GPU time of each
pipeline from timestamp queries (`gpu/timing.ts`).

### Interior and shell variants

The predictor and update are each compiled **twice from one source** (`hydro/region_pass.ts`):

- an **interior** variant for cells at least two cells from every face, with the boundary
  mapping compiled out (`INTERIOR = true`);
- a **shell** variant in general form, dispatched as slabs whose workgroups are two cells thick,
  so the shell launches no more invocations than it has cells.

Fluxes on the seam between the two variants agree bit for bit; the flux-determinism test covers
it.

### Workgroup shapes

3D passes use $`8\times8\times4`$ workgroups (256 invocations, the default limit). A 2D run uses
$`16\times16\times1`$, which more than halves the cost of its flux update compared with
$`8\times8\times4`$ and three idle layers.

## Gathered fluxes and bit-identical faces

Most GPU finite-volume codes compute each face flux once, store it, and difference the stored
values in a second pass. Shaderous **gathers**: each invocation owns one cell and computes the
fluxes on all of its faces itself. Every interior face is therefore evaluated twice, once by
each of its cells.

Conservation then requires the two evaluations to be **bit-identical**. They are when both
cells read identical inputs and do identical arithmetic in the same order, so the code
ensures that:

- face states are reconstructed by a function of the two cells' stencils that does not depend
  on which side asks;
- every quantity that enters a face flux is formed from the same values whichever cell asks,
  including the potential differences of the well-balanced reconstruction and the diffusion
  gradients;
- the Riemann solver is called with the same (left, right) order from both sides, in the
  frame where the face normal is $`+x`$.

The trade: two flux evaluations per face, in exchange for **no flux storage, no second pass,
and no write races**. The idea extends to refinement and time-bin boundaries, where a cell
*recomputes* its neighbor's fluxes rather than reading stored ones (below). The validation
suite checks that the two cells of every face produce bit-identical mass fluxes for three
flows, and that a uniform state advected ten times across the box conserves mass and momentum
to round-off.

## Boundaries without ghost cells

The uniform grid stores no ghost cells. Every texture load goes through `bc_map`
(`hydro/common.wgsl`), which maps an out-of-range cell index to the texel that holds its value,
with the signs of its velocity and magnetic field components and, at a face that lets nothing
in, the clamp on its normal velocity. Cells inside the grid return at once, and in the interior
variant `bc_map` is compiled out. Obstacles are solid cells flagged in the state and treated as
reflecting (or no-slip, with viscosity) walls by the same functions.

The kinds of box face are chosen per face: periodic, outflow, reflecting, outflow only, the
line-tied wall, and the moving wall (whose velocities are in the parameter block, `P.wallv`). What each does to a ghost is written once, in `hydro/boundary.wgsl`
(`bc_ghost_cell`, `bc_vsign`, `bc_bsign`, `bc_side`, `bc_ghost_face`, `bc_wall_velocity`), and
every grid's mapping
calls it: `bc_map` on the uniform grid, `ghost_src` and `row_src` for the gas on blocks (where
the block table holds a domain face as `-1 - kind`), and `map_cell` and `face_map`
(`mhd/amr_lookup.wgsl`) for MHD on blocks. The interface, the setup format and links read one
table, `problems/boundaries.ts`. A new kind is an entry there, its rules in
`hydro/boundary.wgsl`, and whatever physics is its own: the line-tied wall also zeroes the
electric field along it (`bc_tied_keep`, in both edge passes) and the resistive Poynting flux
through it, and the moving wall sets it to $`-\mathbf{v}_\text{wall}\times\mathbf{B}_n`$
(`moving_wall_e` in `mhd/corner.wgsl`, from the wall faces: `bface` on the uniform grid, the
assembled faces on blocks). The CPU reference solvers (`src/reference`) keep their own mappings of the first
four kinds, as independent checks.

One trap when touching these: `map_cell` must be called inside the per-axis loop of
`face_map`, as written. Hoisted out of the loop, it is inlined at every call site, and the MHD
block passes took twice as long per step.

## Reductions and the GPU-resident timestep

`gpu/reduce.ts` and `gpu/reduce.wgsl` implement multi-stage hierarchical reductions (sum, max,
min) for any number of values per cell. Each stage folds $`4^3`$ blocks of the previous stage's
records ($`4`$ measured faster than $`8`$: shorter serial loops, more threads). Sums are formed
hierarchically to limit single-precision round-off in diagnostics. Maxima of non-negative
floats are accumulated with atomic max on their bit patterns, since those order the same way as
the floats.

The timestep reduction ends in a GPU buffer, never on the CPU. Diagnostics (mass, momentum,
energies, max Mach number, floored-cell count, under-resolved Jeans cells) use the same
machinery and are copied to a mapped buffer that the CPU reads whenever it is ready.

## Multigrid on textures

`gravity/multigrid.ts`. Each level is an `r32float` texture set ($`\Phi`$, right-hand side,
residual), from $`n^3`$ down to $`2^3`$. Red-black Gauss-Seidel smoothing reads and writes $`\Phi`$ in
place, one color per dispatch. Restriction and trilinear prolongation are separate small passes.
For isolated boundaries a reduction gathers the mass moments, and `multipole_eval.wgsl` gives
the boundary potential through the same index-mapping trick the hydro uses. The potential of
the previous step is kept, both as a warm start and to extrapolate $`\Phi^{n+1/2}`$ for the
energy source.

## Adaptive mesh

`src/amr/`.

### Tree on the CPU, data on the GPU

The **tree** (`amr/tree.ts`) is small and lives on the CPU: blocks, levels, parents, children
and face neighbors, with at most one level of difference across a face. Whenever the mesh
changes it is uploaded as flat tables: per-block records, neighbor slots, and the lists of
blocks each pass is dispatched over.

The **data** lives in one 3D **texture atlas** per field. Each block of $`16^3`$ cells (with 2
ghost layers, so $`20^3`$ texels, padded to $`24`$ where a pass needs it) occupies a slot in the
atlas. Parents keep their slots, because gravity and restriction use them. In 3D the atlas
holds at most 2197 ($`13^3`$) blocks, about 3 GB with gravity, and shrinks to fit the
[memory budget](#gpu-memory-budget). When a re-mesh wants more blocks than the budget has,
the most strongly flagged blocks are refined first, and room for them is made by merging the
blocks least flagged (those closest to being merged anyway), as long as these are flagged under
0.8 of the block that needs the room (`EVICT_MARGIN`). If the initial mesh of a
self-gravitating run cannot fit a whole level, that level is left out (the status line says so)
rather than letting a refinement boundary cut through a star at an arbitrary place.

### A step on blocks

```
assemble   primitives for every block, ghosts included, straight from the conserved state
timestep   reduce the signal rate over leaf cells; set dt
predict    half-step states for leaf blocks and their first ghost layer
update     leaf cells only, with fluxes matched at refinement boundaries
[radiation] E and F moved with the update's mass fluxes
[pressure] implicit pressure: the source, V-cycles over the tree's levels, the correction
[gravity]  restrict density, relax every level in the same dispatches, field, second half kick
[radiation] N_r sweeps, each after assembling (E, F) with ghosts, then the commit
```

The two are not taken together. With either every block takes one timestep (per-block
timesteps are refused with implicit pressure and switched off with radiation).

### Ghost cells computed on load

Because a pass cannot write the `rgba32float` texture it reads, ghost layers cannot be copied
within the atlas. Shaderous does not fill them at all. The **assemble** pass
(`amr/assemble.wgsl`), which converts conserved to primitive variables, computes every texel of
a block, ghost or interior, from wherever its value lives:

| The texel lies… | Its value comes from |
|---|---|
| inside the block | the block itself |
| in a same-level neighbor | that neighbor's slot |
| beyond a physical boundary | the boundary image of an interior cell |
| over a finer neighbor | the average of the $`2^d`$ finer cells covering it |
| over a coarser neighbor | limited linear interpolation from the coarse block |

There is no separate exchange step, and no ghost cell is ever stale.

### Conservation at refinement boundaries without flux registers

Berger-Colella AMR, and the GPU codes that follow it, store the fluxes the fine side computes on
a coarse face and correct the coarse cell afterwards. In Shaderous the coarse cell instead
**recomputes the $`2^{d-1}`$ fine fluxes** that tile its face, from the same fine states and with
the same arithmetic as the fine cells, and sums them. Nothing is stored. The fluxes agree bit
for bit between blocks on the same level and to $`10^{-6}`$ across levels, and mass, momentum and
energy are conserved to round-off.

### Thread layout on blocks

Passes over a block and its ghost layers (18 or 20 texels across, not a multiple of 8) run as
flat rows of 64 threads per block (`inv_flat` in `amr/list.wgsl`). $`8\times8\times4`$ tiles had
left about half their threads idle there. In 2D the four $`z`$-layers of a workgroup take four
consecutive blocks. Together with predicting only leaves, this made a 2D Kelvin-Helmholtz step
(2048², 3 levels) 2.8 times faster and a 3D Sedov step 19% faster.

### Regridding

Every 8 steps (`REGRID_EVERY`) a pass reduces the refinement indicator per block into a
buffer with atomic max (`amr/flag.wgsl`). The CPU reads it back without blocking. The flow may
run at most 24 steps past an unanswered check: a signal crosses at most about a third of a cell
per step, and flagged blocks keep a buffer block around them, so no feature can leave its
refinement in that time. The CPU then adapts the tree, and one pass (`amr/regrid.wgsl`) fills
the new leaves into the other state texture by copying, prolonging or restricting. When a
preset needs every block above a density at the finest level (the white dwarfs), that is
applied during the first construction too.

### Gravity on blocks

`amr/gravity.ts`, `amr/gravity/*.wgsl`. Every block, parents included, holds $`\Phi`$. A cycle
restricts the density upward, fills each level's ghosts from its neighbors or by interpolation
from the level above, and runs a multigrid V-cycle *inside* each block (16, 8, 4, 2, 1 cells).
All levels relax in the same dispatches, so the cost does not grow with tree depth. A gate pass
(`gate.wgsl`) skips later cycles once the residual stops falling. Prolongation reads the patch
of coarse cells a workgroup needs into workgroup memory once, and ghost fills launch threads
only for face ghosts. These two changes cut a 3D Evrard step from 3.88 to 3.55 ms of GPU time,
with identical results.

## Per-block timesteps on the GPU

Per-block timesteps ([numerics.md](numerics.md#18-local-per-block-timesteps)) are scheduled
entirely on the GPU:

1. `bins_init.wgsl` gives each block a bin from its own signal speed.
2. `bins_relax.wgsl` runs relaxation sweeps that enforce the neighbor rules.
3. `bins_lists.wgsl` / `bins_args.wgsl` build, for each substep, the list of blocks that start
   or finish a step and write **indirect dispatch arguments**. The CPU encodes the same command
   sequence every cycle without knowing how many blocks each pass covers.
4. `bins_on.wgsl` gives the update the bin logic: a slower block takes a faster neighbor's
   fluxes, recomputed, at the start and midpoint of its own step. `mid_on.wgsl` lets a faster
   block read a neighbor that is part way through a longer step from that neighbor's stored
   half-step prediction. `*_off.wgsl` stubs compile all of this out when the feature is off.

Every few cycles (`BIN_POLL_CYCLES = 4`) the share of block updates actually made is read
back. Above 80% (65% with a gravity field) the run pauses local timesteps for 192 steps, and
longer each time the answer is the same, up to 1536.

## Rendering

`src/render/`. The renderer runs on the same `GPUDevice` and reads the simulation textures
directly. Nothing is copied to the CPU.

- **Field extraction** (`field.wgsl`, `field_value.wgsl`) computes the displayed quantity per
  cell into a texture. The quantities are density, temperature, pressure, scalar, speed and
  velocity components, Mach number, entropy $`p/\rho^\gamma`$, net heating, vorticity,
  divergence, $`|\nabla\rho|/\rho`$, potential, kinetic energy density, Jeans length in cells,
  timestep limit, mass fraction of each species, and refinement level. The color range follows
  the field, ignoring the most extreme 0.5% of cells for derived fields.
- **Ray-marcher** (`raymarch.wgsl`): front-to-back emission-absorption through the unit box with
  a 1D transfer function and a dithered start to hide banding. What it marches through comes
  from a slot:
  - On a uniform grid (and a 2D mesh), a display volume of at most $`256^3`$, sampled at a
    fixed step (`volume_uniform.wgsl`).
  - On a 3D adaptive mesh, the leaf blocks at their own resolution (`volume_atlas.wgsl`). The
    display pass's `atlas` entry (`amr/field.wgsl`) writes each leaf block's display values into
    an atlas laid out like the state's. Each block gets one ring of border cells holding the leaf
    there, at any level, so hardware trilinear filtering is continuous across blocks.
  - A sample finds its leaf through the leaf index and the block table. The ray then moves on by
    two thirds of that cell, so fine regions are sampled finely and the coarse rest cheaply.
  - The atlas costs 8 bytes per atlas texel: about 140 MB at the default block budget, the same
    as the $`256^3`$ volume it replaces. A volume at the finest resolution would be 1 GiB at
    $`512^3`$.
  - The color range comes from the same pass, each cell counted as the number of finest cells
    it covers.
- **Slice** (`slice.wgsl`): an axis-aligned slice with zoom up to 64× and a color bar.
- **Lines** (`lines.wgsl`): block outlines of the adaptive mesh and tool guides over the volume
  view.
- **Picking** (`pick.wgsl`): the query tool reads one cell's state (and its block, level and
  indicator) through a tiny readback. Its depth pick marches a ray through the same volume slot
  as the ray-marcher.
- **Brushes** (`edit/brush*.wgsl`, `amr/brush.wgsl`) write mass, velocity, heat, obstacles,
  sources and stars straight into the state textures. The field brushes (`edit/field_brush*.wgsl`,
  `amr/field_brush.wgsl`) change the face field through a vector potential (strength) or the
  induction equation (pull), then the total energy by the field's change (numerics.md section
  21).

## Frame loop and pacing

`ui/app.ts`. Each frame encodes $`N`$ steps and a render pass into command buffers. $`N`$ adapts to
how long the GPU takes. The loop measures completed work with `queue.onSubmittedWorkDone()`.
Firefox reports finished GPU work only about every 100 ms, so there the loop instead submits a
frame every display refresh up to a small backlog and steers by the backlog. Results agree
across browsers to round-off. Axis-permuted runs are bit-identical in Chrome and differ in the
last bit in Firefox.

## GPU memory budget

`gpu/memory.ts`. No browser reports how much video memory there is. A desktop GPU raises an
out-of-memory error, which the adaptive mesh answers with a smaller atlas. iOS does not: memory
is shared, and the system kills the page before WebGPU reports anything. Shaderous therefore
guesses a budget (2.5 GB for a discrete desktop GPU, 2 GB for shared memory, 1 GB on phones),
lets the user override it in the Performance panel, and writes a note to `localStorage` before
each large allocation. If a note is still there when the page loads, the page died holding that
much, and the automatic budget drops below it for good. A run that is replaced is torn down
before the next one allocates, so memory never spikes during a rebuild.

The adaptive mesh sizes its atlas from the budget and its bytes per block
(`AmrSim.blockBytesFor`): the state, primitive, predictor and gravity textures, a magnetic
field's textures and flux register, and species' four atlas textures per group of four, their
composition moments and face mass fluxes. For the 13 network species those are more than the
rest of a block together; left out, a white dwarf with burning asked Firefox for twice its
budget, which Firefox refuses where Chrome lets it spill into shared memory. Resource errors are
checked once everything is made, before the initial conditions are set up, so an atlas the GPU
cannot make is retried smaller within seconds.

## Setups: one document, compiled

Every setting of a simulation is a field of one JSON document, a `SetupSpec`
(`setup/spec.ts`; [setups.md](setups.md) describes the format). The presets are 79 such files
in `src/presets/`, and a setup made in the Setup dialog is the same kind of file. A document
holds numbers and names and no code, so one from a stranger is safe to open; `setup/codec.ts`
reads it as untrusted input.

`compileSetup` (`setup/compile.ts`) turns a document into what the solvers have always taken:
a `Problem` (its `init`, `masses`, `settle` and `mhd` functions close over the document) and
the overrides of a `SimConfig`. Nothing downstream knows about documents: the solvers, the
mesh, settling, the renderer and the diagnostics see a `Problem`. The compiler is free of the
DOM and deterministic, so the page, the workers that build an adaptive run's initial state
(`amr/init_worker.ts`, which gets the document and compiles it again) and Node get the same run
from the same document. The initial state is painted in layers: a whole-box flow or the
background, then disks, then stars, then regions, obstacles and inflows in the order of the
parts. What shapes cannot say is a `flow`: a named function of this build with parameters
(`setup/parts/flows/`), so a file never carries a formula.

The option table (`setup/options.ts`) lists every option once: its place in the document, its
control and range, what changing it costs a running simulation (`live`, `shaders` or
`restart`), its key in a link, and the rules that say what it requires and excludes. The
dialog's forms, the greying of options that cannot be combined (with the reason as the
tooltip), the reader's ranges, the link codec and the scripting interface's `controls()` are
all read from it.

A run stands for a document: `runningSpec` (`setup/running.ts`) is the one its problem was
compiled from, with its grid, its parameter values and every setting changed since.
`changeBetween` compares two documents and says what getting from one to the other takes. The
dialog edits a copy and hands it back; the page (`applySpec` in `ui/app.ts`) then does the
least that is needed: a uniform write, a rebuild of the shaders, or a fresh start. The side
bar keeps what concerns choosing and watching a run, and shows a setup's featured options from
the same table.

### One table, read everywhere

What a setup, the view or a call of the scripting API accepts is written once, and every place
that repeats it is made from that place:

- A list of choices (`RIEMANN_KINDS`, `GRAVITY_MODES`, `STAR_TYPES`, ...) is a constant array
  from which its type is made. The option table's menus must name exactly its list, and the
  document codec takes its choices from the option table.
- The view's settings are a table, `src/render/view_options.ts`: key, link key, type, limits
  and a line of documentation. The link codec loops over it and `shaderous.view()` reads its
  keys from it. The renderer's `ViewState` must match it or the build stops.
- The scripting API's methods are a table, `src/ui/api_schema.ts`: what each does and its
  arguments as JSON Schema.
- The MCP server's tools are a table, `src/ui/mcp_schema.ts`: the API method each wraps, the
  arguments it passes on (described as the API describes them) and what it adds of its own.
- The command-line runner's flags are a table, `tools/headless/flags.mjs`, which makes its
  parser's options and its help.

The headless tools cannot import the page's TypeScript, so `npm run gen` writes what they need
to files that are committed:

- `tools/headless/schema.json`: link keys, view settings, variables, choices, API methods and
  the finished definitions of the MCP tools. The server serves those definitions as they are
  and holds only what each tool does when called.
- The tables of the docs that restate one of these tables, between `<!-- gen:NAME -->`
  markers: the link keys and view settings of [api.md](api.md); every key of a setup document,
  every part's fields and every flow's parameters in [setups.md](setups.md); the runner's
  options and the MCP tools in [headless.md](headless.md); and the lists of keys, tools and
  flags in the agent skill (`.claude/skills/shaderous-sim/SKILL.md`).
- `public/shaderous-setup.schema.json`, the format of a setup document as JSON Schema.

The generator is the unit test `tests/unit/generated.test.ts` run with a flag: run plainly, as
`npm test` does, it fails when a generated file is out of date. After changing one of the
tables above, run `npm run gen` and commit what it changes. A new generated block is a
function in `src/schema/docs.ts` and a pair of markers where it should appear.

The presets were hand-written TypeScript objects until November 2026. Every preset file was
held bit for bit to the object it replaced before those objects were retired.

## Input, output and links

- **Snapshots** (`io/athdf.ts`) are HDF5 files in the Athena++ `.athdf` layout: primitive
  variables block by block, with levels and logical locations. [yt](https://yt-project.org/)
  opens them directly (`yt.load("file.athdf")`). Files written by Shaderous carry a JSON
  `Shaderous` attribute with the run's settings, time, step count and point masses, so importing
  one continues the same run on the same mesh. Species are written as variables named after
  them (`he4`, `c12`). A file from another code is sampled onto the nearest available
  resolution.
- **Viewing a checkpoint too large to run** (`io/view.ts`, `io/view_worker.ts`,
  plans/view-only.md). The file is opened in a worker through Emscripten's WORKERFS, which
  reads the `File` in place, and read 1024 blocks at a time (`openAthdf`'s ranged reads). Blocks
  finer than the level that fits the budget are merged into their ancestors as they are read
  (`readForView`), and the merged snapshot is loaded like any restart by an `AmrSim` with
  `viewOnly`: every atlas only the step uses (the second parity, the predictor, gravity, a
  field's compensation, half-step, assembly and impulse textures) is one texel, so the bind
  groups stay valid while it never steps. The atlas is sized to the merged tree exactly
  (`amrBudget`). Merged blocks are handed on as soon as the last of the file's blocks inside
  them is read (`streamForView`), so only the blocks still being filled are held. A uniform
  grid's file is read as blocks of 16 at one level below a coarser root (`blocksOfUniform`), so
  merging downsamples it.
- **Holding one field** (`viewBaked`): not even the state is kept. A second atlas of slots of
  16^3 cells without ghosts (rg32float: the field shown and the density) is filled on the GPU by
  `amr/field.wgsl`'s `bake` entry from batches of the file's primitives, and the display passes,
  compiled with the override `BAKED`, read it in `leaf_value` in place of the state, so the
  volume, slice and projection views and the automatic color range work unchanged. Another field
  is baked by reading the file again; the reader's worker keeps it open. Field lines need the
  face field (`viewFaces`), loaded with the bake, and sample the baked density
  (`amr/bsample.wgsl`).
- **Checkpoints** (`io/checkpoints.ts`) write the same file every $`\Delta t`$. In Chrome and Edge
  they go to a folder chosen once through the File System Access API; other browsers download
  them. The cell data is gzip-compressed at level 1. Uncompressed, a state is 24 bytes per cell (about 400 MB at
  $`256^3`$); compressed files measured 8% (an early Sedov blast) to 86% (tidal debris) of that.
- **Links** (`ui/link.ts`): the address bar holds the preset, dimension, resolution, refinement
  and *only what differs from the preset's defaults*, for example
  `?problem=tidal-disruption&dim=3&n=256&amr=3&p.beta=2&mode=slice`. Anything unrecognized or out
  of range in a link is ignored rather than trusted.
- **Snapshots in memory** (`io/capture.ts`) save a running simulation and start runs from it; Reset returns to the loaded state.
- **Session logs** (`ui/session.ts`) record what was done to a run as it ran: what the tools
  and settings changed (as it reached the simulation, not the pointer's path), the changes of
  the clock's stop, and the adaptive mesh's re-meshes as the steps the tree took (each block
  refined or merged, in order: `BlockTree.lastSteps`, about a hundred bytes a re-mesh). Each entry is stamped with the step it was applied at, from a copy of the GPU clock
  submitted at the moment it is applied. `replay()` (ui/api.ts) loads the setup, steps to each
  entry and applies it, with the mesh's own refinement checks off. A re-mesh is the one thing a
  live run decides by timing: its check is read back while the run goes on, so the step it
  lands on varies. The replay takes the same steps on the same tree instead
  (`BlockTree.replayAdapt`), which gives the same mesh slot for slot and makes it repeat to the
  bit. The page saves the log (Save session log), and the API and the MCP server read and
  replay it.

## Testing

- **Unit tests** (`npm test`, Vitest in Node): the preprocessor, the equations of state against
  their float64 mirrors and analytic limits, white dwarf structure and the collision orbit, the
  exact Riemann solver, the float64 reference scheme, the AMR tree, links, units and checkpoints.
- **GPU tests** (`npm run test:gpu`): Playwright drives an installed Chrome (or Edge with
  `PW_CHANNEL=msedge`) through the 180-test validation suite on the real GPU. Each test closes
  its page, bounded to 30 s, before its result is judged, and a global timeout (180 minutes,
  `PW_GLOBAL_TIMEOUT`) runs in the runner process, so a failed test cannot hold the suite. The
  specs take `test` from `tests/gpu/fixtures.ts`, whose browser fixture closes Chrome through
  `tools/headless/close.mjs` (as the headless session does): on macOS Chrome's crash handler can
  outlive Chrome while holding the pipe that Playwright's `Browser.close` waits on, and every
  full run then passed its tests and never exited.
- **In the browser**: [validate.html](https://guillochon.github.io/shaderous/validate.html) runs
  the same suite on the visitor's GPU, with each test's pass criterion next to its result.
- **Float64 mirrors**: every WGSL equation of state and cooling function has a TypeScript
  twin, and `src/reference/hydro.ts` implements the whole hydro scheme in double precision.
  This separates shader bugs from limits of the scheme.

## Performance notes

- Where memory bandwidth is scarce (the GPU of the measurement above) the flux update is bound
  by texture fetches, not arithmetic. This is why the uniform grid stores no primitives by
  default and why a slope-storage pass was never added. On an A100 it is bound by arithmetic,
  which is what stored primitives and the Koren limiter's early return in smooth cells address.
- Textures beat storage buffers for the stencil by about 1.4×.
- Nearly all of the update's cost is face work (reconstruction and Riemann solves), which the
  gather form does twice per face. On the uniform grid the update therefore computes each face
  of its workgroup's box once, into workgroup memory, and each cell reads its six
  (`hydro/tile.wgsl`); faces on the box's edge are computed again by the neighboring workgroup,
  so no flux leaves the pass. On an Apple M4 Pro in Chrome this made the largest uniform grids
  1.3 to 1.5 times faster (256^3 and 2048^2; adaptive meshes use their own update). Caching the
  stencil's cell states in workgroup memory instead was slower than the texture loads it
  replaced, and the shape of the workgroup made no difference within a few percent.
- Bit-identical faces cannot rest on two copies of the same arithmetic. WGSL allows a compiler
  to fuse and reassociate floating-point operations (WGSL 15.7.5), and Dawn compiles Metal
  shaders in relaxed math mode: on an Apple M4 Pro the gather's two call sites for a face (a
  cell's lower face and its neighbor's upper face) disagreed in the last bit on thousands of
  faces, and the flux checks of the validation suite failed. The tiled update instead computes
  every shared face with the same code in one compiled pipeline (it is not split into interior
  and shell pipelines like the other stencil passes), and every face agrees. This costs about
  10% against splitting it, for the boundary mapping that runs at run time.
- Viscosity makes the flux update 1.6× (2D) to 1.8× (3D) as long, because of the extra diagonal
  loads. Conduction alone is not measurable.
- Systematic throughput measurements (cell updates per second on named GPUs and browsers) have
  not been made yet.

## Limitations and lessons that carry over

Shaderous runs on one GPU, inside the memory the browser grants. Its radiation transport (gray,
M1: [numerics.md](numerics.md#24-radiation-transport)) and its implicit pressure take one
timestep for all blocks, and its diffusion is explicit. Single precision is enough for the shipped
problems. Round-off is biased, though, when a uniform background sits exactly on a power of two
(see `problems/advection.ts`), and long integrations accumulate errors that double precision
would not.

The main choices came from WebGPU's restrictions, but they do not depend on it.
Gathering fluxes and recomputing them across refinement and time-bin boundaries trades extra
arithmetic for less storage and memory traffic, which suits any processor limited by memory
bandwidth. Computing ghost cells on load removes a synchronization step from every AMR cycle.
Keeping the timestep and scheduling on the GPU, with indirect dispatch, removes every
CPU round trip from the step.
