# Plan: a view-only mode for checkpoints too large to run here

Status (2026-11-21): not begun. The memory figures below are worked out from
`AmrSim.blockBytes` and the code as it stands, not measured; stage 0 measures them.

Goal: open an `.athdf` checkpoint of a run that does not fit in this device's GPU memory (made
on a Runpod pod or a cluster, say) and look at it: slices, projections, the volume view,
isosurfaces, probes, the colour bar, the field menu. Nothing steps. The GPU holds only what the
display reads, and if even that does not fit, a coarser version of it.

| Stage | What it adds |
|---|---|
| 0 | Measurements: the largest checkpoint the page opens today, and where it fails |
| 1 | Reading a file in pieces, in the page, without holding it in memory |
| 2 | `ViewSim`: blocks holding only the state the display passes read |
| 3 | One field at a time, baked on the GPU in chunks from the full state |
| 4 | Level of detail: coarser levels, or a region at full resolution, when even that does not fit |
| 5 | The page and the scripting API: opening for viewing, what is greyed out, going on to run |

Stages 1 and 2 are the first useful thing (several times more blocks). Stage 3 is the large gain
(tens of times). Stage 4 makes any file open, at some resolution.

---

## 1. What limits a checkpoint today

**The file read.** The page's load (`ui/app.ts`, `readAthdf` in `io/athdf.ts`) copies the whole
file into h5wasm's in-memory file system, inside the WebAssembly heap, and then reads every
variable into one `Snapshot` in JavaScript. A 32-bit WebAssembly heap stops at 4 GB (less in
practice), and the `Snapshot` doubles it. A file of a few GB fails before the GPU is involved.
The runner's path (`readAthdfFile`, h5wasm/node) reads from disk but still builds the whole
`Snapshot`.

**The GPU.** A block is PAD^3 = 20^3 texels in 3D (16^3 cells and two ghost cells a side,
`amr/tree.ts`). `AmrSim.blockBytes` charges each texel 9 rgba32float textures (two state pairs,
primitives, predictor, gravity) and two r32float ones: 152 bytes, so 1.2 MB a block before a
magnetic field, species, radiation or implicit pressure add theirs. In the default 2.5 GB
budget (`gpu/memory.ts`) that is some 2,000 blocks, 8 million cells (a 200^3 uniform grid's
worth). The display adds its own volume atlas (rgba16float per leaf cell) and display volume.

**Restarts need all of it.** `restartFrom` (`io/capture.ts`) builds a full `SimConfig` and an
`AmrSim` with every atlas, because a restart steps.

What the display actually reads (`amr/field.wgsl`): the conserved state `U1`, `U2`; the
primitives `W1` (for the gradients of vorticity, divergence, density gradient and shocks); the
gravity field `GF` (the potential field only); the species `S` (mass fractions); the leaf index
`IDX`; and with a magnetic field, its faces (`render/bsample.wgsl`). It writes the volume atlas
the ray-marcher reads (`render/volume_atlas.wgsl`).

## 2. Stage 0: measurements

Before building anything, with a few checkpoints of rising size (a 3D blocks run at 1, 2, 4 and
8 GB, one with a magnetic field, one with the 13 species):

- where the page's load fails (heap, `Snapshot`, atlas allocation) and at what size;
- the actual GPU bytes per block (`noteAllocation`) against `blockBytes`;
- the time to read each, and the time spent in h5wasm against in conversion.

These set the targets and say which of stages 1 to 3 matters first.

## 3. Stage 1: reading a file in pieces

- **In the page:** run h5wasm in a worker with the file mounted lazily (Emscripten's WORKERFS
  over a `File`, which reads with `File.slice` as HDF5 asks, so the file is never copied into
  the heap). The page's file picker and drop already hand over a `File`; the File System
  Access handle the checkpoint folder uses (`io/checkpoints.ts`) works too. Check that the
  h5wasm build we ship includes WORKERFS; if not, a small custom VFS over synchronous
  `FileReaderSync` in the worker does the same.
- **Reading by blocks:** read a variable for a range of blocks at a time (an HDF5 hyperslab of
  `prim` and of the species and `Bcc` datasets), as the face field already is in `FACE_SLAB`
  pieces. Expose it as an iterator over chunks of blocks: `{ blocks, levels, locations,
  vars: Record<name, Float32Array> }`, a chunk being some 64 MB. No `Snapshot` of the whole
  file ever exists.
- **gzip:** compressed files (`writeAthdf`'s `gzip`) read chunk by chunk too; HDF5 decompresses
  per chunk, so the dataset's chunk shape matters. Check what we write (one variable's chunk
  should be a few blocks, not the whole variable) and fix the writer if needed, since old files
  with a huge chunk would need decompressing whole.
- **The same iterator on the runner** (h5wasm/node on a path), so the MCP server and the
  command-line runner can view files on a pod without loading them.
- **Other codes' files** (plain Athena++): the same layout, so the same reader, as long as the
  variables are there; they open with the current problem as `loadAthdf` does today.

Test: a unit test (vitest, node build) that reads a written file chunk by chunk and gets the
same values as `readAthdf`; a GPU test opening a file larger than the heap could take, made by
the runner.

## 4. Stage 2: `ViewSim`, the state the display reads and nothing else

A new class beside `Sim` and `AmrSim`, implementing what the renderer, the probes, the slice and
the diagnostics need of `Simulation` (`display`, `displayAtlas`, `readLeaves`-like reads for
probes, `diagnostics` of what is loaded), and refusing `step`, edits and brushes.

- The tree comes from `Levels` and `LogicalLocations`, without the refinement machinery.
- Atlases: `U1`, `U2` (rgba32float) filled from the file's primitives through the equation of
  state, as `restartFrom` converts them (the tabulated ones' pressure to energy inversion
  included), chunk by chunk; `W1` derived on the GPU from them; ghost cells by the existing
  assembly pass (it only needs the state atlases and the tree). With a magnetic field, the
  file's faces into the face atlas when a magnetic quantity is shown, cell-centred `Bcc` for the
  rest. Species: the groups when a mass fraction is shown.
- Not allocated: the second state pair, the predictor, gravity's field and multigrid, the flux
  registers, the face mass fluxes, radiation's and implicit pressure's atlases, time bins.
  Fields needing them (potential, time bin, timestep limits) are greyed out in the menu with the
  reason ("not in the checkpoint").
- About 3 rgba32float textures a texel (48 bytes) against 152: some three times the blocks.
  Modest; this stage is mostly the structure stage 3 builds on, and the shape of the class.

The run's setup comes from the file's `Shaderous` metadata (`meta`), compiled only for what
display needs: the equation of state, units, cosmology (proper densities at the file's a),
species names, the box and its symmetry.

## 5. Stage 3: one field at a time, baked on the GPU

The display shows one quantity. Hold only that, and work it out from the full state a chunk at a
time, with the existing passes, so no physics is written twice:

1. A small **work atlas**, a full `ViewSim` state (stage 2) for a chunk of a few thousand blocks
   plus their face neighbours (for the ghost cells the gradient fields need), allocated once.
2. For each chunk: stream its blocks and neighbours in (stage 1), assemble ghost cells, run the
   display pass's field evaluation (`field_value` in `render/field_value.wgsl`, as
   `amr/field.wgsl`'s atlas mode calls it) into the **view atlas**, one value per leaf cell.
3. The view atlas has no ghost cells (16^3 per block, not 20^3) and one channel: r16float (2
   bytes) or r32float (4). The ray-marcher, slice and projection read it in place of the volume
   atlas's channels; normalization and the log move into those passes, so changing the colour
   range does not re-bake.

Memory: a block is 16^3 x 2 bytes = 8 kB at half precision, against 1.2 MB: some 150 times the
blocks, so the default budget holds about 300,000 blocks, 1.2 billion cells (a 1,000^3 grid's
worth), less the work atlas. The atlas's side limit (2048 texels, 128 blocks a side without
ghosts) allows 2 million blocks, so memory binds first.

Costs and decisions:

- **Changing the field re-bakes**: one pass over the file. At the speeds of stage 0 this may
  take tens of seconds for a large file. Show progress; keep the last two fields baked if
  memory allows; let the user pick several fields to bake at load.
- **Isosurfaces** use the log channel interpolated between leaf centres (`smooth_log`); with
  one channel holding the value, interpolate it in the ray-marcher instead (needs neighbours:
  read across block edges through `IDX`, as the slice already does at the finest level).
- **Half precision**: 3 significant digits and a range of 6e-5 to 65504. Store log10 of
  positive quantities (density, temperature, pressure) in half precision; signed quantities
  (velocity components, B components, vorticity) in r32float, or half with a stored scale.
  Probes and the colour bar's numbers read from the file, not the view atlas, so they keep full
  precision.
- **Range and histogram**: computed during baking (`RANGE` buffer), accumulated over chunks.

Tests: the baked field against a full `AmrSim` load of a checkpoint that fits, cell by cell
(exact in r32float; to half precision's rounding in r16float), for a gradient field and a plain
one; screenshots of the slice and volume view against the full load, compared by
`test-results`' image diff.

## 6. Stage 4: level of detail

When even one field does not fit, or to open quicker:

- **Level cap**: replace leaves finer than level L by their ancestors at L, the value
  volume-averaged (mass-weighted for velocities) while baking; L chosen as the finest that fits
  the budget, from the count of blocks per level read from `Levels` at open. Shown as "viewing
  to level L of M".
- **Region at full resolution**: a box (the slice view's visible area, or drawn) baked at full
  depth inside, capped outside; re-baked from the open file when the box moves. The file stays
  open in the worker for this.
- Uniform-grid files (one block) are split into 16^3 blocks on read, so the same machinery
  applies (and a level cap is a downsampling).

## 7. Stage 5: the page and the scripting API

- **Opening**: beside today's load, "Open for viewing" (and automatically offered when a file's
  block count times `blockBytesFor` exceeds the budget: "This run needs 14 GB to continue here;
  open it for viewing?").
- **Mode**: a "View only" badge; the run controls, setup editing and brushes disabled with the
  reason; the field menu greying described above.
- **Continuing**: "Run from here" loads it fully (today's path) when it fits.
- **Several files**: stepping through a folder of checkpoints with the time slider re-bakes the
  current field from the next file, so a movie of a large run can be made here (`io/movie.ts`).
- **API and MCP**: `loadAthdf` gets a `view: true` option; `load_athdf` a `view` flag; `slice`,
  `probe`, `screenshot`, `record` work; `run` and `control` refuse with the reason. This is what
  makes looking at a pod's 512^3 output from the MCP server practical.

## 8. Not covered

- Remote files (viewing a file on a pod without downloading it): HTTP range reads would fit the
  stage 1 VFS, but need a server on the pod; later, if wanted.
- Particles, sinks and point masses' trails: drawn from the metadata as now, no change.
- Editing a checkpoint in view mode.

## 9. Open decisions

1. Half or single precision for the view atlas by default (stage 3): measure the visual
   difference and the memory; probably log-half for positive fields, single for signed.
2. Whether `ViewSim` is a new class or `AmrSim` with a view flag that skips allocations. A new
   class keeps `AmrSim` free of conditionals; reusing the assembly and display passes is the
   same either way.
3. Whether stage 2 ships on its own or only as the inside of stage 3.

## 10. Order of work

1. Stage 0 measurements.
2. Stage 1 reader, with its unit test; the writer's chunk shape fixed if needed.
3. Stage 2 `ViewSim` with the field menu greying, against a full load.
4. Stage 3 baking, half precision, ray-marcher changes.
5. Stage 5's page and API pieces (enough to use it), then stage 4.
6. `docs/` and the CHANGELOG: what view-only shows, how much it holds.
