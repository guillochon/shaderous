# Plan: one-dimensional runs

Status (2026-11-27): phases 1 to 5 built; 6 to 14 not started. Open questions of the first
draft answered: the profile tool is locked to radial lines in spherical and cylindrical 1D runs,
the cylindrical view is a half-disk like the spherical one, and the 1D chip alone marks the view.

**As built, phases 1 to 5.**

- **Uniform grid** (decisions 1, 3): `ndim` is `1 | 2 | 3`; a 1D grid is `n x 1 x 1`
  (`gridDims`), y and z periodic (a spherical run's poles kept), each cell its own neighbor across
  them, so the y fluxes cancel and the shaders were not changed to skip them: every cell of a 1D
  run goes through the boundary-mapped pass (`region_pass.ts`), and `grid_n()` has one cell
  along y. Workgroups along the row (64 x 1). Uniform 1D grids go up to 2048 cells, a 3D
  texture's side; finer ones are adaptive, up to 32768 (the leaf index is a row of the same cap).
- **Blocks: not decision 2 as written.** A 1D block is laid out on the GPU as a 2D one, sixteen
  like rows, each block its own neighbor along y (the block table's y faces hold its own slot)
  and each child's slot written for both octants along y (`tree.ts writeRow`, `regridTable`).
  Every row computes the same bits, so the 2D block shaders run unchanged but for: positions on
  y = 0 (`blk_pos`), a spherical run's one theta cell on every row (`blk_sph`,
  `sph_children`, `sph_fine_face`), the leaf index a row (`indexGrid`, `leaf_at`), and one row
  of each block in the reductions, the readback and the monopole. The tree is truly binary
  (`nchild` 2, two faces). The cost is sixteen times the cells a block of one row would compute;
  true `BS x 1` blocks remain a later option (phase 14), and only matter for speed.
- **View** (decisions 4 to 6): the slice map is one texel tall and repeated by the slice shader;
  spherical and 1D axisymmetric runs are the half-disk (`curvedView`; the axisymmetric one with
  no inner radius, `sphViewParam` 1e-30). The chip is over the view (`#oned-chip`) and painted on
  screenshots and movies whatever overlays are asked for (`paintChip`). The HUD and the Setup
  dialog say `1D, 4096 cells`; the query readout and the profile caption give x (or r) alone.
- **Profile tool** (decision 7): `lockLine` in `ui/profile.ts`; a click places a radial line; the
  link's `prof` is `x0,x1` in 1D (`x0,y0,x1,y1` read too).
- **Gravity** (phase 5): constant gravity along -x; self-gravity by the monopole passes with a
  `LINE` mode (`gravity/sph_monopole.wgsl`, `amr/sph_monopole.wgsl`, `sim.ts lineGravity`): the
  spherical monopole, Cartesian slabs (the 1D Green's function, isolated or periodic with the
  mean taken out) and axisymmetric rings. The option table allows self-gravity in 1D
  (`gravity.grid`). Point masses only at a spherical run's center; stars not in 1D.
- **Refused** (decision 8, `oneDimUnsupported` and the option table's `oneD.blocked`): everything
  of phases 6 to 13, with the phase named; and the sink and obstacle tools.
- **Checked in the page**: Sod at 1024 cells uniform and 4096 with 3 levels against the exact
  plateau (p* 0.3031, u* 0.9274); Noh in all three geometries uniform and on blocks (post-shock
  densities 4, 15.9, 63 against 4, 16, 64, pre-shock profiles as (1 + t/r)^(k-1)); the Jeans
  wave's growth to t = 0.5 (amplitude 1.17% against 1.16% from linear theory, uniform and on
  blocks); a uniform cylinder's gravitational energy, -pi^2 / 4 to five figures; 2D and 3D runs
  on blocks unchanged. Unit tests: `tests/unit/one_d.test.ts`.
- **Open**: refined cylindrical Noh at 4096 cells with 3 levels blew up once at t = 0.12 when it
  was run in many short `run()` calls with diagnostics read in between, and not when run in long
  calls (twice, identically); not found yet. Presets on in 1D: Sod (x), the Riemann problems,
  strong shock, planar, cylindrical and spherical Noh, advection, Sedov, the spherical blast,
  the uniform collapses, Coggeshall, Jeans and the second collapse. The atmosphere is not: its
  faces along x are periodic, and a 1D atmosphere needs walls there.

A run with `dim: 1`: the gas varies along one axis only. Cartesian runs vary along x; spherical
runs along r (spherical symmetry); cylindrical runs along R (cylindrical symmetry, no z). Such
runs support the adaptive mesh, and are drawn in the same view as 2D runs, the one row of cells
repeated across the axis the run does not resolve: y for a cartesian run, the polar angle for a
spherical or cylindrical one. The main window says plainly that the run is 1D, so the stripes
or rings on screen are not mistaken for a 2D flow that happens to be uniform. The profile tool
draws horizontal lines only in a cartesian 1D run, and radial lines only in a spherical or
cylindrical one.

Why: the shock tubes, blast waves and collapse problems in `src/presets` (`sod-*`,
`riemann-*`, `noh-*`, `uniform-collapse-*`, `sedov`, `blast-spherical`, `radiative-shock*`,
`brio-wu`, `zeldovich-pancake`, `polytrope`, `white-dwarf`) are 1D problems run today on 2D
grids of n x n cells, n times the work they need. In 1D a 65536-cell AMR run of a spherical
blast costs less than a 256^2 grid, can be checked against the exact solutions in
`src/reference` directly, and fits on a phone.

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
- **Cylindrical runs are (R, z)** with the axis at R = 0 (`domainLo`), and carry the swirl
  v_phi in the third velocity component with its centrifugal term (`hydro/common.wgsl`).
  Dropping z leaves R with its geometric terms and the swirl, which are already there.
- **The slice view reads a 2D map texture** (`render/slice.wgsl` `fs`, `MAP` built by the
  projection pass in `render/project.wgsl` / `render/field.wgsl` / `amr/field.wgsl`). A map one
  texel tall repeats along y on its own, since `floor(st.y * 1) = 0`; and in a spherical run
  `sph_code()` maps the view to (r, theta), so the same one-row map repeats over theta and draws
  concentric half-rings, unchanged.
- **Block outlines** come through the map's green channel (`m.g`): in 1D they become vertical
  lines (cartesian) or arcs (spherical, cylindrical) at block edges, which show the refinement.
- **The profile tool** (`plans/profile-tool.md`, `src/ui/profile.ts`, the profile section of
  `src/ui/app.ts` from `profilePress`) already snaps to 15° steps with Shift (`snapAngle`), and
  is withheld on spherical grids (`noProfile`).
- **The exact Riemann solver** (`src/reference/riemann_exact.ts`), the float64 1D references
  (`src/reference/hydro.ts`, `radiation.ts`) and the exact solutions of `src/validation/exact.ts`
  give answers to compare with.
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
   once, so the per-dispatch cost (~1.7 us) does not grow with the number of blocks; if block
   bookkeeping shows up in timings, a later option is BS = 64 in 1D.
3. **The shaders test the degenerate axes at run time, as they do for z today.** Every `flat`
   (`dims.z == 1`) gains a sibling `line` (`dims.y == 1`), and loops over axes stop at
   `select(3, select(2, 1, line), flat)`. No new shader variants: the uniform-grid and block
   shaders read `P.dims`. Workgroup shapes that are 8 x 8 in 2D become 64 x 1 in 1D
   (`dispatch.ts`, `flatGroups`).
4. **Three velocity components, one flux direction ("1.5D").** The state keeps vx, vy, vz; only
   x faces carry fluxes, and vy and vz are carried along by them. This is what 1D MHD needs
   (Brio-Wu's v_y and B_y), and what a cylindrical run's swirl needs (v_phi, with its
   centrifugal term). In a spherical run, whose one theta cell is the whole sphere, v_theta and
   v_phi must be zero: the setup checker refuses them.
5. **Drawn as 2D, the values repeated across the axis not resolved.** The view is the slice
   view (the volume view and projection are not offered: `flatView()` is true for 1D). The map
   the projection pass writes is `n_view x 1` texels; the slice shader repeats it.
   - **Cartesian:** the square view as for a 2D run; every column one value. The y axis keeps
     its extent (the box) so zoom, pan, the ruler and the brushes behave as in 2D.
   - **Spherical:** the meridional half-disk the 2D spherical view already draws, each ring
     one value: no change to `slice.wgsl`.
   - **Cylindrical:** the same half-disk in the left half of the square, R from 0 at the middle
     of the left edge to the box at the rim, the polar angle around it, each ring one value; so
     the two curved geometries frame alike. It is the spherical mapping without r_min and with
     R in place of r: `sph_code()` gains a flag (a new camera uniform word, `C.mir.w` being
     taken) and `sphFromView()` / `sphToView()` gain the cylindrical case for `locate()`, the
     query readout and the profile tool. (The 2D cylindrical view is the (R, z) half-plane;
     repeating that along z would look like a cartesian run, which is why the polar angle is
     the axis repeated.)
6. **The window says it is 1D with a chip,** in the top-left corner of the view: **1D** with
   what is repeated, "1D · repeated along y", "1D · spherically symmetric", "1D · cylindrically
   symmetric", and a `title` explaining that only one axis is simulated. It is drawn into the
   frame, not only as an HTML overlay, so it is in screenshots and recordings (the profile
   line's SVG is not in screenshots, see `plans/profile-tool.md`; the chip must be). Besides the
   chip, what would say something false about a 1D run follows it: the query readout leaves out
   the repeated coordinate ("x = 0.31" alone, or "r = ..."), the HUD's grid line reads
   `1D, 4096 cells` (or `1D, 256 + 4 levels`) where it now reads `512^2`, and the grid menu
   offers 1D beside 2D and 3D, only for problems that support it.
7. **The profile tool is locked to the one direction that is not redundant.**
   - **Cartesian: horizontal lines.** The second end follows the pointer along x and keeps the
     first end's y (Shift has nothing to snap); dragging either end moves it along x only. The y
     it is drawn at is free, since every y gives the same values.
   - **Spherical and cylindrical: radial lines.** The line runs from the center (r = 0, or
     r_min) outward; a press places the outer end anywhere in the half-disk and the line is
     drawn from the center to it, along that polar angle; dragging the outer end changes its
     radius and angle; the inner end is fixed at the center. The plot's x axis is r (or R), not
     distance along the line, and its samples are taken at r alone, so a spherical 1D run gets
     the profile tool, which the 2D spherical grid still withholds.
   - A line placed by the link (`prof`), the API (`profile({ line })`) or the runner
     (`--profile`) is made to obey the same rule (y set to the first end's; or the inner end
     moved to the center), with a note that it was. `encodeProfileLines` writes two numbers a
     line in 1D (the x or r of each end) and `decodeProfileLines` reads either form.
8. **Every physics module is planned for 1D,** in the phases below. Until its phase is built,
   the setup checker (`src/setup/check.ts`) refuses a module in a 1D run with the reason and
   the phase, as it does for options that do not fit a 2D run. A few modules have no meaning in
   1D and stay refused for good (decision 9).
9. **Not applicable in 1D, refused for good:** cut cells and meshes (`hydro/cut*.wgsl`,
   `src/geometry`: a 1D obstacle is an interval, decision 8's phase 12 covers it), disks
   (`disk`: a disk needs two axes), the shearing box (`shear`), mirror symmetry along y or z,
   constant gravity along an axis other than x, and point masses off the center (a point mass
   in 1D is a sheet in a cartesian run and a line in a cylindrical one; phase 5 allows the
   central mass of a spherical run, the line mass on a cylindrical run's axis, and a sheet at a
   point of a cartesian run, each with its exact 1D field).
10. **One setup key.** `dim: 1` in setups, links (`dim=1`, `src/ui/link.ts`), the API
    (`load({ dim: 1 })`), MCP `load`, the runner (`--dim 1`); presets say `supports1d: true`.
    Resolutions for 1D: 256 to 65536 uniform (`RES_1D`), and up to `MAX_AMR_LEVELS` above a
    root grid of at least two blocks. `uniformMax(1)` and `amrFits(n, 1, levels)` follow from
    bytes per cell, which are the 2D figures with one row.
11. **Files.** athdf output and input in 1D (Athena++ writes 1D files natively: `nx2 = nx3 = 1`,
    `src/io`); snapshots and replays carry `ndim: 1`; the runner's slice and probe outputs are
    one row.

## Phases

Each phase lists its modules, what 1D asks of them, the presets it turns on (`supports1d`) and
what says it is done.

1. **Uniform cartesian hydrodynamics.** `ndim: 1 | 2 | 3` through the types; `gridDims` gives
   `[n, 1, 1]`; `line` beside `flat` in the hydro shaders; 1D workgroup shapes; the map one
   texel tall; the 1D chip, the HUD and readout; `dim: 1` in the link, API, MCP and runner.
   Modules that act cell by cell come with it: every equation of state (`src/eos`: ideal,
   liquid, barotropic, degenerate, Helmholtz, H-He, ANEOS, tables, ionizing, gas plus
   radiation, quartic), every Riemann solver and limiter, floors, fluff, dual energy, the
   pressure floor, heat sources (`heat`), damping and start damping, reference states and
   region passes. Presets: `sod-x`, the `riemann-*` presets, `strong-shock`, `noh-planar`,
   `advection`. Done when Sod and LeBlanc at n = 1024 match `riemann_exact.ts` to the error the
   2D `sod-x` strip has today, and the view and readout show the run as 1D.
2. **Profile tool, cartesian.** Decision 7's horizontal lock, with its link, API and runner
   forms.
3. **AMR.** The binary tree (`amr/tree.ts`: children, neighbours, balance, `adapt`,
   `replayAdapt`), the 1D atlas (`AmrSim.atlasShape`, `blockBytes`), `assemble`, `flag`,
   `regrid` (2-child restriction and prolongation), `correct` (flux correction at one face a
   side), time bins, diagnostics and their reductions, refinement criteria (`refineRho`,
   `refineDense`, `refineZones`, `refineMasses`); block outlines in the view. Done when a
   refined Sod and a refined blast agree with the uniform run at the finest resolution, mass
   and energy are conserved to float precision on a periodic box, and replay is exact.
4. **Spherical and cylindrical geometry.** One theta cell over [0, pi] (check `sph_volume` gives
   the shell to float precision), uniform and logarithmic r; cylindrical R alone with its swirl;
   the half-disk views (decision 5) and their CPU twins; the radial profile lock (decision 7);
   the chip's wording. Presets: `noh-spherical`, `noh-cylindrical`, `blast-spherical`,
   `sedov`, `uniform-collapse-spherical`, `uniform-collapse-cylindrical`. Done when Noh in each
   geometry matches its exact solution (`src/validation/exact.ts`) and the views show rings.
5. **Gravity.** Constant gravity along -x. Self-gravity, which in 1D is a prefix sum rather than
   a Poisson solve (`src/gravity`, `amr/gravity`): cartesian isolated `g(x) = -2 pi G (M(<x) -
   M(>x))` per unit area, cartesian periodic the same with the mean density taken out;
   spherical `g = -G M(<r) / r^2` through the monopole (`amr/sph_monopole.wgsl`);
   cylindrical `g = -2 G M(<R) / R` per unit length. On blocks the sum runs over leaves in x
   order, at each leaf's own resolution. The potential (for the view's potential field and the
   energy diagnostics) by a second prefix sum. Point masses as decision 9 allows them, with
   sink accretion for the central mass. Hydrostatic settling (`settle`), stars and planets built
   by setup parts (`src/problems/lane_emden.ts`, `white_dwarf.ts`, `cored_polytrope.ts`,
   `planet.ts`) evaluated at r. Presets: `jeans`, `uniform-collapse-planar`, `polytrope`,
   `white-dwarf`, `isothermal-collapse`, `evrard`, `second-collapse`, `atmosphere`. Done when
   a polytrope holds still to the 2D run's error, and Jeans growth rates match linear theory.
6. **Species, burning and cooling.** Species advection along x (`hydro/species*.wgsl`,
   `amr/species*.wgsl`, the mass-flux faces one per side), composition-dependent equations of
   state, the burn networks and their watch (`src/burn`: aprox13 and the rest), screening,
   cooling (`src/cooling`: ISM, power law). Presets: `detonation-stable`,
   `detonation-pulsating`, `detonation-co`, `thermal-instability`, `thermal-instability-hhe`.
   Done when a stable detonation's
   speed matches the Chapman-Jouguet speed as the 2D run does, and the pulsating one pulsates
   with its known period.
7. **Diffusion and driving.** Viscosity and conduction along x only (`hydro/diffusion*.wgsl`,
   `diff_rate.wgsl`: the cross terms vanish, the 1D viscous stress is (4/3) mu dv/dx for
   vx and mu dv/dx for vy, vz); turbulent driving with modes along x (`hydro/drive*.wgsl`,
   `driveModes(1, ...)`, `drive_ou.wgsl` with the 1D case beside the `flat` one), which in 1D is
   driven Burgers turbulence. Presets: `driven-turbulence` (1D version), a conduction front
   test against its similarity solution. Done when a viscous shock's width and a conduction
   front's profile match their analytic forms.
8. **Frames and cosmology.** Comoving coordinates (`src/cosmo.ts`) with periodic gravity from
   phase 5; the accelerating frame (`frame`) along x; the rotating frame in a cylindrical run,
   about its axis (the swirl carries the frame's rotation; refused in cartesian and spherical
   runs, where it would need a second axis). Presets: `zeldovich-pancake`, the classic 1D
   cosmology test. Done when the pancake matches the Zel'dovich solution before shell crossing.
9. **Magnetic fields.** 1D MHD in the 1.5D form (decision 4): Bx constant in space and time,
   By and Bz evolved by the x fluxes; constrained transport reduces to this in 1D
   (docs/numerics.md, the corner EMF section), with the y and z face fields one value a cell;
   the face registers and their AMR flux correction (`registerBytes(1)`); resistivity. Presets:
   `brio-wu`, `riemann-slow-shock` and the other MHD Riemann problems, `alfven-wave` along x.
   Done when Brio-Wu matches the published 1D solution at the resolution of the reference, and
   Bx stays exactly constant.
10. **Radiation.** The radiation module along x (`src/radiation`, `amr/rad_*.wgsl`: M1 in 1D,
    the sweep, the escape and the regrid of its fields) with every opacity (`src/opacity`:
    constant, electron scattering, Kramers), and the opaque-edge handling of
    `plans/radiation-opaque-edges.md` where built. `src/reference/radiation.ts` is already a
    float64 1D version of the scheme, so the GPU 1D run can be compared to it cell by cell.
    Presets: `radiative-shock`, `radiative-shock-supercritical`, `radiating-sphere`,
    `sedov-rad`, `sedov-rad-transport`, `radiation-star`. Done when the 1D run agrees with the
    float64 reference and the radiative shocks match their semi-analytic profiles.
11. **Implicit pressure and incompressible flow.** The implicit pressure solve
    (`src/pressure`, `amr/pressure`, `src/reference/psplit.ts`) in 1D, where its multigrid
    becomes a tridiagonal system (solved directly on the uniform grid; the block multigrid kept
    for AMR); the liquid equation of state's incompressible limit, in which 1D flow is a uniform
    velocity set by the boundaries. New presets: a low-Mach acoustic pulse and a 1D liquid
    piston. Done when the low-Mach run matches the
    explicit one at Mach 0.3 and stays stable at Mach 0.01.
12. **Walls, obstacles, moving bodies and inflow.** A 1D obstacle is a solid interval
    (`solid` cells, `hydro/walls*.wgsl`), a moving body a piston moving along x
    (`hydro/moving*.wgsl`: the body's distance function is |x - x_b| - half width), forces on it
    the pressure difference across it (`forceRef`, drag; no lift); moving walls at the
    boundaries (`wallVelocity`); inflow boundaries as the wind tunnel's (`plans/wind-tunnel.md`)
    along x. The obstacle, source, sink, temperature and star brushes act on an interval (the
    brush's circle cut by the line), drawn as a band across the repeated axis. New preset: a piston
    driving a shock into gas at rest. Done
    when the piston's shock speed matches the exact solution.
13. **Particles and debris.** Tracer and debris particles along x (`src/debris`, the census
    passes, `plans/hybrid-particles.md` if built), placed at their x only and drawn at random y
    (or polar angle) across the repeated axis so they can be seen. Done when tracers follow the
    flow of an advected pulse.
14. **Later, if wanted:** BS = 64 blocks in 1D; special relativity in 1D
    (`plans/special-relativity.md`) once it is built in 2D, whose shock tubes are 1D problems.

## Tests

- Unit (vitest, so they run in the Pages deploy's `npm test`): `gridDims`, `domainHi`,
  `boxFrac` for `ndim: 1`; the binary tree's adapt and balance; the cylindrical half-disk
  mapping's round trip; `encodeProfileLines` / `decodeProfileLines` in 1D; the horizontal and
  radial locks on placed and dragged lines; the setup checker's refusals for every module not
  yet built and for those that do not apply; link round trip of `dim=1`; the 1D prefix-sum
  gravity against analytic fields.
- GPU (Playwright, on a static build): for each phase, the test named in its "done when";
  conservation of mass and energy on a periodic 1D box with AMR; a screenshot test that the 1D
  chip and the repeated rows (or rings) are drawn.
- Random setup testing (`random-test` skill): add `dim: 1` to the option table so the fuzzer
  draws it with the rules of decisions 8 and 9, widened as each phase lands.
