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

Four properties of the interface (not of the hardware) shaped the design. The
[original plan](../plan.md) lists the limits that were checked when development began.

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
  eos/         WGSL equation-of-state modules and their float64 mirrors (eos.ts)
  gravity/     multigrid passes, multipole boundary, half-kick corrector, point masses, sinks
  cooling/     WGSL cooling functions and their float64 mirrors
  species/     packing and bookkeeping of mass fractions
  amr/         block tree (tree.ts), atlas passes and the adaptive driver (amr_sim.ts);
               gravity/ for the Poisson solve on blocks
  render/      field extraction, ray-marcher, slice view, picking, transfer functions
  problems/    one file per preset, plus settle.ts (discrete hydrostatic balance)
  reference/   float64 CPU version of the hydro scheme, exact Riemann solver
  units/       physical constants, unit systems, composition
  validation/  the in-page validation suite (validate.html)
  io/          snapshots, Athena++ HDF5 reader and writer, checkpoints, capture
  ui/          the simulator front end, shareable links
  sim.ts       owns the textures and encodes a uniform-grid step
tests/         unit/ (Vitest, Node) and gpu/ (Playwright, real GPU)
tools/burn/    offline Python tools for the nuclear-burning surrogate
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

On the uniform grid there are **no primitive-variable textures**. The predictor and the update
convert the conserved state as they load it. The flux update is bound by texture fetches: with
all of its arithmetic removed it still takes 85% as long. A pass that only rewrites the state in
another form therefore costs more than repeating the conversion.

## Anatomy of a uniform-grid step

`sim.ts` encodes a step as a sequence of compute passes in one command buffer:

| Pass | Shader | Work |
|---|---|---|
| 1. Rates | `hydro/rates.wgsl` | Reads the state once and writes only the first stage of the timestep reduction: max signal rate and max $`\lvert g\rvert`$ per $`4^3`$ block, folded in workgroup-shared atomics. |
| 2. Clock | `hydro/clock.wgsl` + `gpu/reduce.wgsl` | Finishes the reduction, applies the point-mass and output limits, and writes $`\Delta t`$ and the time to a small buffer that later passes read. |
| 3. Predictor | `hydro/predictor.wgsl` | Limited slopes, the half-step primitive state, shock and first-order flags. |
| 4. Update | `hydro/update.wgsl` | Face states, Riemann solves on all $`2d`$ faces, diffusion fluxes, the conservative update, the first half kick, cooling, dual-energy sync, floors. |
| 5. Species *(optional)* | `hydro/species*.wgsl` | Move each species with the face mass fluxes the update recorded. |
| 6. Gravity *(optional)* | `gravity/rhs.wgsl`, V-cycles, `gravity/field.wgsl` | New source, warm-started V-cycles, the field texture. |
| 7. Correct *(optional)* | `gravity/correct.wgsl`, `bh_*.wgsl` | Second half kick; point masses kick, drift and accrete. |

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
(`hydro/region.wgsl`), which maps an out-of-range cell index to the texel that holds its value,
and a velocity transform (a sign flip for reflecting walls, clamping for no-inflow outflow).
Periodic, outflow, outflow-without-inflow and reflecting boundaries are chosen per face.
Obstacles are solid cells flagged in the state and treated as reflecting (or no-slip, with
viscosity) walls by the same functions. In the interior variant `bc_map` is compiled out.

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
[memory budget](#gpu-memory-budget). If the initial mesh of a self-gravitating run cannot fit a
whole level, that level is left out (the status line says so) rather than letting a refinement
boundary cut through a star at an arbitrary place.

### A step on blocks

```
assemble   primitives for every block, ghosts included, straight from the conserved state
timestep   reduce the signal rate over leaf cells; set dt
predict    half-step states for leaf blocks and their first ghost layer
update     leaf cells only, with fluxes matched at refinement boundaries
[gravity]  restrict density, relax every level in the same dispatches, field, second half kick
```

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
  sources and stars straight into the state textures.

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

## Input, output and links

- **Snapshots** (`io/athdf.ts`) are HDF5 files in the Athena++ `.athdf` layout: primitive
  variables block by block, with levels and logical locations. [yt](https://yt-project.org/)
  opens them directly (`yt.load("file.athdf")`). Files written by Shaderous carry a JSON
  `Shaderous` attribute with the run's settings, time, step count and point masses, so importing
  one continues the same run on the same mesh. Species are written as variables named after
  them (`he4`, `c12`). A file from another code is sampled onto the nearest available
  resolution.
- **Checkpoints** (`io/checkpoints.ts`) write the same file every $`\Delta t`$. In Chrome and Edge
  they go to a folder chosen once through the File System Access API; other browsers download
  them. The cell data is gzip-compressed at level 1. Uncompressed, a state is 24 bytes per cell (about 400 MB at
  $`256^3`$); compressed files measured 8% (an early Sedov blast) to 86% (tidal debris) of that.
- **Links** (`ui/link.ts`): the address bar holds the preset, dimension, resolution, refinement
  and *only what differs from the preset's defaults*, for example
  `?problem=tidal-disruption&dim=3&n=256&amr=3&p.beta=2&mode=slice`. Anything unrecognized or out
  of range in a link is ignored rather than trusted.
- **Snapshots in memory** (`io/capture.ts`) save a running simulation and start runs from it; Reset returns to the loaded state.

## Testing

- **Unit tests** (`npm test`, Vitest in Node): the preprocessor, the equations of state against
  their float64 mirrors and analytic limits, white dwarf structure and the collision orbit, the
  exact Riemann solver, the float64 reference scheme, the AMR tree, links, units and checkpoints.
- **GPU tests** (`npm run test:gpu`): Playwright drives an installed Chrome (or Edge with
  `PW_CHANNEL=msedge`) through the 36-test validation suite on the real GPU.
- **In the browser**: [validate.html](https://guillochon.github.io/shaderous/validate.html) runs
  the same suite on the visitor's GPU, with each test's pass criterion next to its result.
- **Float64 mirrors**: every WGSL equation of state and cooling function has a TypeScript
  twin, and `src/reference/hydro.ts` implements the whole hydro scheme in double precision.
  This separates shader bugs from limits of the scheme.

## Performance notes

- The flux update is bound by texture fetches, not arithmetic. This is why the uniform grid
  stores no primitives and why a slope-storage pass was never added.
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

Shaderous runs on one GPU, inside the memory the browser grants. It has no magnetic fields or
radiation transport, and its diffusion is explicit. Single precision is enough for the shipped
problems. Round-off is biased, though, when a uniform background sits exactly on a power of two
(see `problems/advection.ts`), and long integrations accumulate errors that double precision
would not.

The main choices came from WebGPU's restrictions, but they do not depend on it.
Gathering fluxes and recomputing them across refinement and time-bin boundaries trades extra
arithmetic for less storage and memory traffic, which suits any processor limited by memory
bandwidth. Computing ghost cells on load removes a synchronization step from every AMR cycle.
Keeping the timestep and scheduling on the GPU, with indirect dispatch, removes every
CPU round trip from the step.
