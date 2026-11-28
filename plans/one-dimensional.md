# Plan: one-dimensional runs

Status (2026-11-27): planned, nothing built.

A run with `dim: 1`: the gas varies along one axis only. Cartesian runs vary along x; spherical
runs along r (spherical symmetry); cylindrical runs along R (cylindrical symmetry, no z). Such
runs support the adaptive mesh, and are drawn in the same view as 2D runs, the one row of cells
repeated across the axis the run does not resolve: y for a cartesian run, the polar angle for a
spherical or cylindrical one. The main window says plainly that the run is 1D, so the stripes
or rings on screen are not mistaken for a 2D flow that happens to be uniform. In a cartesian 1D
run the profile tool draws horizontal lines only.

Why: the shock tubes, blast waves and collapse problems in `src/presets` (`sod-*`,
`riemann-*`, `noh-*`, `uniform-collapse-*`, `sedov`, `blast-spherical`, `radiative-shock*`,
`brio-wu`) are 1D problems run today on 2D grids of n x n cells, n times the work they need. In
1D a 65536-cell AMR run of a spherical blast costs less than a 256^2 grid, can be checked
against the exact solutions in `src/reference` directly, and fits on a phone.

## What is there to build on

- **The degenerate axis already exists.** A 2D run is a 3D grid with one layer along z:
  `gridDims()` in `src/domain.ts` sets `d[2] = 1`, and the shaders test `P.dims.z == 1u`
  (`flat`) to skip the z fluxes, predictor terms and species faces (`hydro/update.wgsl`,
  `hydro/predict.wgsl`, `hydro/species*.wgsl`, `hydro/diffusion_core.wgsl`, `amr/grid.wgsl`).
  1D generalizes this to "axes 1 and 2 are degenerate": `P.dims.y == 1u` as well.
- **Spherical runs are already (r, theta) with theta from 0 to pi** (`src/spherical.ts`,
  `hydro/common.wgsl` `sph_cell` / `sph_volume`). A 1D spherical run is that grid with one theta
  cell spanning [0, pi]: its volume is the whole shell (`sphGrid().shell()`), its theta faces lie
  on the poles where the flux is zero, and the theta source terms of one cell over the whole
  sphere cancel. No new geometry, only a check that one cell gives exactly the shell.
- **Cylindrical runs are (R, z)** with the axis at R = 0 (`domainLo`). Dropping z leaves R with
  its geometric terms, which are already there.
- **The slice view reads a 2D map texture** (`render/slice.wgsl` `fs`, `MAP` built by the
  projection pass in `render/project.wgsl` / `render/field.wgsl` / `amr/field.wgsl`). A map one
  texel tall repeats along y on its own, since `floor(st.y * 1) = 0`; and in a spherical run
  `sph_code()` maps the view to (r, theta), so the same one-row map repeats over theta and draws
  concentric half-rings, unchanged.
- **Block outlines** come through the map's green channel (`m.g`): in 1D they become vertical
  lines (cartesian) or arcs (spherical) at block edges, which show the refinement well.
- **The profile tool** (`plans/profile-tool.md`, `src/ui/profile.ts`, the profile section of
  `src/ui/app.ts` from `profilePress`) already snaps to 15° steps with Shift (`snapAngle`), and
  is withheld on spherical grids (`noProfile`).
- **The exact Riemann solver** (`src/reference/riemann_exact.ts`) and the float64 1D references
  (`src/reference/hydro.ts`, `radiation.ts`) give exact or near-exact answers to compare with.
- **Problems say which dimensions they support** (`supports2d` in `src/problems/types.ts`, the
  preset's `domain.supports2d`); `resolveGrid()` in `app.ts` forces 2D for cylindrical and
  spherical problems and 3D where 2D is unsupported.

## Decisions

1. **A real `ndim = 1`, not a thin 2D grid.** The alternative, a 2D grid one block (16 cells)
   tall with y periodic, needs almost no code but costs 16 times the work on the uniform grid,
   and worse on blocks: every refinement splits along y too, so a level L region is 2^L blocks
   tall. Instead `ndim` becomes `1 | 2 | 3` everywhere (`GridShape`, `AmrSim`, `Snapshot`,
   `create.ts`, `geometry/store.ts`, the drive and moving-body helpers, ...: about a thousand
   uses in 68 files, which the type checker lists once the type changes). Each site is either
   already written over `axes = [0, 1, 2].slice(0, ndim)` and works, or has a `ndim === 2 ? ... :
   ...` that needs a third branch. The grid is `[n, 1, 1]`.
2. **Blocks of BS x 1 x 1 cells, a binary tree.** `BS` stays 16 along x so the block code,
   ghost layers (`NG = 2`), the 2:1 balance and the time bins are shared; along y and z a 1D
   block has one cell and no ghosts. The tree splits a block into 2 children (`2^ndim`), has
   neighbours only at -x and +x (the other four face slots hold the "periodic self" marker so
   the existing neighbour walk reads the block's own cells), and the regrid table's restriction
   takes 2 children. The atlas packs 1D blocks PAD x 1 into rows of the 2D atlas texture
   (`atlasShape` gets a 1D branch: many slots per row, one layer), so the texture formats and
   bindings stay as in 2D. 16-cell blocks are small, but AMR passes dispatch over all blocks at
   once, so the per-dispatch cost (`dispatch-overhead` memory: ~1.7 us) does not grow with the
   number of blocks; if block bookkeeping shows up in timings, a later option is BS = 64 in 1D.
3. **The shaders test the degenerate axes at run time, as they do for z today.** Every `flat`
   (`dims.z == 1`) gains a sibling `line` (`dims.y == 1`), and loops over axes stop at
   `select(3, select(2, 1, line), flat)`. No new shader variants: the uniform-grid and block
   shaders read `P.dims`. Workgroup shapes that are 8 x 8 in 2D become 64 x 1 in 1D
   (`dispatch.ts`, `flatGroups`).
4. **Drawn as 2D, the values repeated across the axis not resolved.** The view is the slice
   view (the volume view and projection are not offered: `flatView()` is true for 1D). The map
   the projection pass writes is `n_view x 1` texels; the slice shader repeats it.
   - **Cartesian:** the square view as for a 2D run; every column one value. The y axis keeps
     its extent (the box) so zoom, pan, the ruler and the brushes behave as in 2D.
   - **Spherical:** the meridional half-disk the 2D spherical view already draws, each ring
     one value: no change to `slice.wgsl`.
   - **Cylindrical:** a full disk in the plane perpendicular to the axis (R, phi), each ring one
     value, R from 0 at the center to the box at the rim. This is a new branch in `fs`, beside
     `sph_code()`, keyed on a camera flag (`C.mir.w` is taken: a new uniform word), with its CPU
     twin beside `sphFromView()` / `sphToView()` (`cylFromView()` / `cylToView()`), used by
     `locate()`, the query readout and the profile tool. (The 2D cylindrical view is the (R, z)
     half-plane; repeating that along z would look like a cartesian run, which is why the
     polar angle is the axis repeated.)
5. **The window says it is 1D in three places,** so the repetition cannot be missed:
   - a chip in the top-left corner of the view, over the canvas, reading **1D** with what is
     repeated: "1D · repeated along y", "1D · spherically symmetric", "1D · cylindrically
     symmetric" (a `title` explaining that only one axis is simulated). It is part of the view,
     so it is in screenshots and recordings (drawn into the frame, not only an HTML overlay; the
     profile line's SVG is not in screenshots, see `plans/profile-tool.md`, and the chip must be).
   - the repeated axis is drawn dimmed, with its tick labels dropped and the label "y (not
     simulated)" in a cartesian run; in spherical and cylindrical runs there is no y axis to
     dim, and the chip carries the message.
   - the query readout leaves out the repeated coordinate ("x = 0.31" alone, or "r = ...") and
     the HUD's grid line reads `1D, 4096 cells` (or `1D, 256 + 4 levels`) where it now reads
     `512^2`.
   The grid menu offers 1D beside 2D and 3D, only for problems that support it.
6. **The profile tool in a cartesian 1D run draws horizontal lines only.** The second end
   follows the pointer along x and keeps the first end's y (Shift has nothing to snap); dragging
   either end of a placed line moves it along x only; a line placed by the link (`prof`), the API
   (`profile({ line })`) or the runner (`--profile`) has its ends' y set to the first end's.
   The y it is drawn at is free, since every y gives the same values; the plot is the same.
   `encodeProfileLines` writes two numbers an end in 1D (x0;x1) and `decodeProfileLines` reads
   either form. In spherical and cylindrical 1D runs the profile tool keeps the rules the 2D
   grid has today (withheld on spherical grids, free on cylindrical ones); see open question 1.
7. **What 1D runs offer at first:** hydrodynamics (every EOS, Riemann solver and limiter),
   AMR with time bins, cooling, species and burning, point-mass gravity, the spherical monopole
   (`amr/sph_monopole.wgsl`) for spherical self-gravity, and cartesian self-gravity by direct
   integration (the field in 1D is `g(x) = -2 pi G (M(<x) - M(>x))`, a prefix sum, cheaper than
   the multigrid and exact). **Not at first**, each refused by the setup checker with the
   reason (`src/setup/check.ts`) until built: MHD (1D MHD needs Bx constant and only the face
   fields of y and z, a stage of its own; `brio-wu` would be its first preset), radiation,
   moving bodies, meshes and cut cells, driving, particles, mirror symmetry other than x (the
   only axis there is).
8. **One setup key.** `dim: 1` in setups, links (`dim=1`, `src/ui/link.ts`), the API
   (`load({ dim: 1 })`), MCP `load`, the runner (`--dim 1`); presets say `supports1d: true`.
   Resolutions for 1D: 256 to 65536 uniform (`RES_1D`), and up to `MAX_AMR_LEVELS` above a root
   grid of at least two blocks. `uniformMax(1)` and `amrFits(n, 1, levels)` follow from bytes
   per cell, which are the 2D figures with one row.
9. **Files.** athdf output and input in 1D (Athena++ writes 1D files natively: `nx2 = nx3 = 1`,
   `src/io`); snapshots and replays carry `ndim: 1`; the runner's slice and probe outputs are
   one row.

## Stages

1. **Uniform cartesian 1D.** `ndim: 1 | 2 | 3` through the types; `gridDims` gives `[n, 1, 1]`;
   `line` beside `flat` in the hydro shaders; 1D workgroup shapes; the map one texel tall; the
   1D chip, the dimmed axis, the HUD and readout; `dim: 1` in the link and API; `supports1d` on
   `sod-x`, the `riemann-*` presets and `noh-planar`. Done when Sod and LeBlanc at n = 1024
   match `riemann_exact.ts` to the error the 2D `sod-x` strip has today, and the view and
   readout show the run as 1D.
2. **Profile tool lock (cartesian).** Decision 6, with its link, API and runner forms.
3. **AMR in 1D.** The binary tree (`amr/tree.ts`: children, neighbours, balance, `adapt`,
   `replayAdapt`), the 1D atlas (`AmrSim.atlasShape`, `blockBytes`), `assemble`, `flag`,
   `regrid` (2-child restriction and prolongation), `correct` (flux correction at one face per
   side), time bins, diagnostics; block outlines in the view. Done when a refined Sod and the
   `sedov`-style blast agree with the uniform run at the finest resolution, and replay is exact.
4. **Spherical and cylindrical 1D.** One theta cell over [0, pi] (check `sph_volume` gives the
   shell to float precision); cylindrical R alone; the cylindrical disk view (decision 4) and its
   CPU twins; the chip's wording. Presets: `noh-spherical`, `noh-cylindrical`,
   `uniform-collapse-spherical`, `blast-spherical`, `sedov`. Done when Noh in each geometry
   matches its exact solution (`src/validation/exact.ts`) and the views show rings.
5. **Gravity, cooling, species, burning in 1D.** Cartesian prefix-sum gravity; the monopole for
   spherical runs; `uniform-collapse-planar`, `radiative-shock`, the detonation presets.
6. **Later, each if wanted:** MHD in 1D (`brio-wu`), radiation in 1D, BS = 64 blocks.

## Tests

- Unit (vitest, so they run in the Pages deploy's `npm test`): `gridDims`, `domainHi`,
  `boxFrac` for `ndim: 1`; the binary tree's adapt and balance; `cylFromView` / `cylToView`
  round trips; `encodeProfileLines` / `decodeProfileLines` in 1D; the horizontal lock on placed
  and dragged lines; the setup checker's refusals; link round trip of `dim=1`.
- GPU (Playwright, on a static build: `gpu-tests-static-build` memory): Sod and LeBlanc against
  the exact solution, uniform and AMR; Noh in three geometries; conservation of mass and energy
  to float precision on a periodic 1D box with AMR; a screenshot test that the 1D chip and the
  repeated rows are drawn.
- Random setup testing (`random-test` skill): add `dim: 1` to the option table so the fuzzer
  draws it with the rules of decision 7.

## Open questions

1. In a spherical or cylindrical 1D run, should the profile tool be locked to radial lines from
   the center (the only line that is not redundant), as the cartesian one is to horizontal
   lines? A radial line in a spherical 1D run is easy to sample (r alone), so it would also lift
   the "not on a spherical grid" restriction for 1D.
2. Cylindrical view: a full disk (decision 4) or a half-disk on the left like the spherical
   view, so the two geometries frame alike?
3. Is the chip enough, or should the repeated direction also be shaded (a faint gradient or
   hatching across y) so a still screenshot without the chip still reads as 1D?
