# Shaderous

A browser-based 3D compressible hydrodynamics code for astrophysical test problems. All fluid
and gravity computation runs in WGSL compute shaders on WebGPU. The design is in
[plan.md](plan.md).

**Live:** https://guillochon.github.io/shaderous/ (needs a WebGPU browser). Validation suite:
https://guillochon.github.io/shaderous/validate.html

- **Hydro:** unsplit MUSCL-Hancock, third-order reconstruction limited only where the flow is
  not smooth (or a classic slope limiter), HLLC (optionally with a low-Mach fix, or HLL) fluxes,
  positivity fallback, shock-flagged HLL blend, dual-energy formalism, floors.
- **Self-gravity:** multigrid Poisson solve every step, periodic or isolated (multipole) boundaries.
- **Self-gravity or a uniform field:** a constant external acceleration drives the
  Rayleigh-Taylor preset.
- **Point masses:** up to four, with Plummer-softened gravity on the gas and a sink around each
  that takes gas in. They pull on one another, feel the gas, and grow by what they accrete.
  They work with or without self-gravity, in 2D and 3D, on the uniform grid and the adaptive mesh.
  Presets bring their own, and the sink tool places more: with its mass slider at zero it
  places a plain sink that stays put, and with a mass (a multiple of the gas in the box) a
  point mass, set moving by a drag. A preset without a gravitational constant gets G = 1.
- **Tidal disruption:** a polytropic star passed by a point mass on a parabolic orbit, run in the
  star's rest frame. The frame accelerates with the gas's center of mass, the point mass's orbit
  is integrated alongside the gas, and the boundaries let gas out but not in.
- **Binary accretion:** two point masses on a Kepler orbit inside an isothermal disk with a
  constant kinematic viscosity (1e-3 in units of the binary's separation squared times its
  orbital frequency: alpha = 0.1 at the binary's separation). They clear a cavity, draw
  streams across it and accrete from the disks that form around them.
- **Optically thin cooling and heating:** a power-law cooling function with uniform heating, or
  the interstellar curve of Koyama & Inutsuka (2002), chosen when the shaders are assembled
  and available to every preset whose equation of state has an energy. The thermal instability
  preset starts interstellar gas in unstable balance and lets it separate into cold clouds and
  a warm medium, with thermal conduction setting the size of the clouds.
- **Viscosity and thermal conduction:** a constant kinematic viscosity with the full stress
  tensor, and a conductivity per unit density (the heat flux is `-kappa rho grad T`), on the
  uniform grid and the adaptive mesh. The binary's disk is viscous and the thermal
  instability conducts; elsewhere both are off, and every preset can set them under
  Diffusion (`nu` and `kappa` in a link). Conduction needs an equation of state with an
  energy. Obstacles hold a viscous gas at their walls.
- **2D or 3D:** presets that make sense in the plane (shock tubes, blasts, Kelvin-Helmholtz,
  Rayleigh-Taylor, advection) can run on an n x n grid up to 2048 squared. A 2D run is one periodic
  layer of the same 3D scheme, without self-gravity.
- **Adaptive mesh refinement:** a tree of 16-cell blocks packed into texture atlases, refined
  where density or pressure jump and merged where they do not. Fluxes are matched across
  refinement boundaries, so mass, momentum and energy are conserved to round-off. Every
  preset can use it, including the self-gravitating ones: the potential is solved level by
  level on the same blocks, and gas that approaches its Jeans length is refined.
- **Equations of state:** ideal gas, barotropic (isothermal / polytropic), gas plus radiation
  pressure, chosen when the shaders are assembled.

## Running

Needs Node 20+ and a browser with WebGPU (current Chrome, Edge or Firefox).

```bash
npm install
```

```bash
npm run dev
```

Then open http://localhost:5183. The validation suite is at http://localhost:5183/validate.html
and runs on your own GPU.

The address bar follows the settings: preset, grid, refinement, and whatever differs from the
preset's defaults (physics, the preset's own parameters, view and camera), for example
`?problem=tidal-disruption&dim=3&n=128&amr=2&p.beta=2&mode=slice`. Copy it, or use the Share
button, to hand someone the same setup. A bare visit opens Kelvin-Helmholtz at 1024 squared in 2D.

## Saving and loading the state

**Export state** writes the gas as an HDF5 file in the layout Athena++ uses (`.athdf`), and
**Import state** starts a run from one.

- The file holds the primitive variables `rho`, `press`, `vel1`-`vel3` and the passive scalar
  `r0` in a `prim` dataset, block by block, with each block's refinement level and position.
  A uniform grid is one block; an adaptive run writes its leaf blocks. yt opens these files
  as Athena++ data (`yt.load("file.athdf")`), in 2D and 3D, uniform and refined.
- Files written here also carry the run's settings, time, step count and point masses, so
  importing one restores the preset, grid, physics and view, and continues from the same
  state on the same mesh. The loaded run starts paused, and Reset returns to the loaded state.
- A file from another code needs `rho`, `press` and `vel1`-`vel3` on a Cartesian, uniformly
  spaced, square or cubic domain. It is sampled onto the nearest resolution on offer, cell
  center by cell center (not conservatively), keeping the current preset's physics.
- HDF5 comes from h5wasm, loaded the first time either button is used (about 1 MB compressed).
- The cell data is gzip-compressed at level 1 (HDF5's standard filter; higher levels gain
  about a percent more for up to twice the time). Uncompressed, a state is 24 bytes per cell,
  about 400 MB at 256^3. Compressed files measured from 8% of that (an early Sedov blast) to
  86% (tidal debris), and compressing a hard-to-compress 128^3 state takes about 2 s.

## Tests

```bash
npm test
```

runs the unit tests (preprocessor, equations of state, exact Riemann solver, float64
reference scheme) in Node.

```bash
npm run test:gpu
```

drives the installed Chrome through Playwright and runs the validation suite on the real GPU.
Set `PW_CHANNEL=msedge` to use Edge, or `PW_HEADED=1` if headless Chrome finds no adapter.

## Layout

```
src/
  gpu/         device setup and limit checks, shader preprocessor, hierarchical reductions
  hydro/       timestep rates, clock, predictor and update passes; riemann/ and limiters/ modules
  eos/         WGSL equation-of-state modules and their float64 mirrors
  gravity/     multigrid passes, multipole boundary, half-kick corrector, point masses and sinks
  cooling/     WGSL cooling functions and their float64 mirrors
  amr/         block tree, block-atlas passes and the adaptive driver; gravity/ for its Poisson solve
  render/      field extraction, ray-marcher, slice view, transfer function
  problems/    one file per preset
  reference/   float64 CPU implementation of the hydro scheme, exact Riemann solver
  validation/  the in-page validation suite
  io/          snapshots of the state, and the Athena++ HDF5 reader and writer
  ui/          the simulator front end
  sim.ts       owns the textures and encodes a step
tests/         unit/ (Vitest) and gpu/ (Playwright)
```

## Notes on the numerics

- **Gravity coupling** is kick-drift-kick: half a kick with the old potential inside the update
  pass, the other half after the new potential is solved. Applying the whole kick after the
  flux update leaks `rho g^2 dt^2 / 2` of internal energy per step from gas in hydrostatic
  balance, which slowly cools a star.
- **Gravitational energy source** defaults to the flux form (face mass flux times the potential
  difference), which conserves total energy but can overheat low-density cells next to a sharp
  density contrast. The alternative adds exactly the kinetic energy of each kick: gravity never
  touches the internal energy, but total energy is not conserved where the flow is
  under-resolved. Both are selectable in the UI (`gravEnergyFlux` in `SimConfig`).
- **Float32.** Conservation is at round-off level, but round-off is biased when a uniform
  background sits exactly on a power of two (see `problems/advection.ts`).
- **Reconstruction.** The default gives each face of a cell the value of the parabola through
  the cell and its two neighbors, which is third-order accurate on the stencil a slope
  limiter already uses. A three-cell limiter cannot tell a smooth maximum from a jump and
  flattens both, so the limiter (Koren's) is applied only where a quantity changes by more
  than 1% of its natural scale across a cell: the density, the pressure, or the sound speed
  for velocities. A smooth peak then keeps its height (the largest error in the advection
  test is 2.5 to 5 times smaller than with the MC limiter) and hydrostatic states are
  quieter. The price is that kinks and jumps below 1% are not limited: shock-tube errors are
  a few percent higher on axis-aligned tubes. MC, van Leer and minmod remain selectable.
- **A step is four hydro passes:** timestep rates, clock, half-step predictor, update. On the
  uniform grid there are no primitive-variable textures: the predictor and the update convert
  the conserved state as they load it. The flux update is bound by texture fetches (with all
  of its arithmetic removed it still takes 85% as long), so a pass that only rewrites the
  state in another form costs more than repeating the conversion. The rates pass writes
  nothing but the first stage of the timestep reduction, folded in workgroup-shared atomics.
  Storage buffers in place of 3D textures were measured and are slower (about 1.4 times for
  the same stencil). The predictor and the update are each compiled twice from one source
  (`hydro/region_pass.ts`): with the boundary mapping compiled out for cells at least two
  from every face, and in general form for the shell, whose slabs get workgroups two cells
  thick so they launch no more invocations than they have cells. Fluxes on the seam between
  the two agree bit for bit (the flux-determinism test covers it). A 2D run uses 16 x 16 x 1
  workgroups, which more than halves its flux update against 8 x 8 x 4 with three idle
  layers. With gravity it adds the Poisson source, V-cycles, a pass that writes the
  acceleration field, and the second half kick. `sim.profile()` times each pass.
- **Frame pacing** adapts the number of steps per frame to the GPU. Firefox reports finished
  GPU work only every ~100 ms, so there the loop steers by backlog rather than by timing.
  Results agree across browsers to round-off; axis-permuted runs are bit-identical in Chrome
  and differ in the last bit in Firefox.
- **Adaptive mesh.** Ghost cells are never copied within a texture (WebGPU cannot read and
  write one rgba32float texture in a pass). Instead the primitives pass computes every texel,
  ghost or not, from wherever its value lives: a neighboring block, a boundary image, the
  average of finer cells, or an interpolation from a coarser block. At a refinement boundary
  the coarse cell recomputes the fine fluxes rather than using its own, which conserves
  without storing fluxes. The mesh is
  rebuilt from the CPU every 8 steps, from a per-block indicator read back without blocking.
- **Local timesteps** (adaptive mesh, off by default; "Local timesteps" under Hydro, `tbins`
  in a link). Time advances in cycles of up to 8 substeps of the smallest step. At the start
  of a cycle every block gets a bin from its own signal speed, on the GPU: a block in bin k
  steps 2^k substeps at a time. Face neighbors are at most one bin apart, a block next to a
  faster one steps as fast, and a block is never slower than a coarser neighbor. Across a
  face whose other side steps twice as often, a block takes that side's fluxes, recomputed
  from its stored states: half when its own step starts, half at its midpoint. So nothing is
  stored and conservation stays at round-off. A faster block reads a neighbor that is part
  way through a step from that neighbor's half-step prediction. With self-gravity the
  potential is solved on every block when a cycle ends and, in between, only on the blocks
  finishing a step; elsewhere it is carried forward at the rate it last changed there, which
  is also what extrapolates it half a step ahead for the energy source. Point masses move
  every substep. The extra passes are not free (measured in 3D: each block update costs about
  10% more without gravity and 50 to 70% more with it, where every substep also relaxes the
  potential), so the share of updates actually made is read back every few cycles and the
  run falls back to one timestep while that share is above 80% (65% with a gravity field);
  the status line says which. Measured against one timestep: a 3D Sedov blast at 256 with 3
  levels makes 2.2 times fewer block updates and runs 1.9 times faster; an isothermal
  collapse at 256 with 3 levels 2.3 times fewer and 1.6 times faster. Smooth flows, and stars
  that fill their refined region, gain nothing and stay on one timestep; with the option on
  they still pay for the occasional trial cycles and the bin checks in the passes, measured
  at 0 to 3% in 3D and 15 to 25% in 2D, where a block is cheap and its ghost cells are a
  large share of it. A block that steps
  less often has its half-step potential extrapolated further, so energy is conserved a
  little less well with gravity: the Evrard collapse loses 2.1 times as much energy at 128
  cells with 2 levels, 1.1 times at 256 with 3.
- **Gravity on the adaptive mesh.** The potential lives on every block, parents included.
  The density is averaged into the parents, ghost cells of a level come from its neighbors
  or are interpolated from the level above, and a multigrid cycle runs inside the blocks (16,
  8, 4, 2 and 1 cells across). The coupling is one-way, so a fine level never corrects the
  coarse potential beneath it; on a uniform sphere the error stays below 0.3% with three
  levels. A fresh mesh is solved level by level from the coarsest down. During a run every
  level relaxes in the same dispatches, each against the boundary values its coarser level
  had at the start of the cycle, so the cost does not grow with the depth of the tree; the
  solve then takes about as long as on a uniform grid of the finest resolution when half the
  volume is refined, and less when the refined region is smaller.
  Refinement also triggers where the Jeans length falls below `jeansRefine` cells (16 by
  default). The atlas holds at most 2197 blocks in 3D (about 3 GB of GPU memory with
  gravity; it shrinks automatically if the GPU cannot allocate that); if the initial mesh of a
  self-gravitating run cannot fit a whole level within that, the level is left out (the
  status line says so) rather than letting a refinement boundary cut through a star at an
  arbitrary place, which disturbs hydrostatic balance. Periodic gravity assumes periodic hydro boundaries on the adaptive mesh.
- **Barotropic gas** uses HLL, since there is no contact wave for HLLC to resolve.
- **HLLC, low Mach** (a third choice of Riemann solver). A Godunov flux damps a velocity jump
  at a rate set by the sound speed, so slow flow is damped far faster than it moves. Following
  Thornber et al. (2008) the velocity jump handed to HLLC is scaled by the local Mach number,
  with a floor of 0.3: this scheme's time integration (one predictor, one update) with the
  unlimited third-order reconstruction relies on the solver's damping, and a Gresho vortex at
  Mach 0.01 gains kinetic energy without bound for a floor of 0.1 or less. So the fix removes
  up to 70% of the damping of slow flow, not all of it: that vortex on 64 x 64 cells loses 6%
  of its kinetic energy in two time units where plain HLLC loses 15%. Above Mach 1, shocks
  included, it is plain HLLC. (Scaling the acoustic wave speeds instead, as in HLLC-LM, takes
  the damping of pressure jumps away too and is unstable here even at Mach 0.03.)
- **Well-balanced reconstruction** (always on, for equations of state with an energy; `wb=0`
  in a link turns it off for comparison). In hydrostatic balance it is not the
  pressure that is smooth across a stratified cell but its balance with gravity. So what is
  limited and reconstructed is each neighbor's deviation from that balance,
  `(p_r - p_c) + (rho_c + rho_r) / 2 * (phi_r - phi_c)`, and the balanced part of the step to
  the face, `-rho_c * dphi`, is added back unlimited (after Kappeli & Mishra 2014). Where the
  cells obey that trapezoid rule the two cells sharing a face agree on its pressure, and a
  cell's two face pressures differ by exactly what the gravity kick adds to its momentum, so
  the state holds to round-off however steep it is. Walls and obstacles are exact too: the
  image beyond a wall has the cell's own potential, while the wall face itself is reached
  along the true one. The shock detector measures pressure jumps against the same balance,
  or a steep atmosphere at rest would be flagged as a shock. An isothermal column falling by
  e every 4 cells keeps its largest Mach number at 4e-5 after a sound crossing (0.35
  without); the background of the Rayleigh-Taylor preset is 240 times quieter and the error
  of the Jeans growth rate 12 times smaller. On the adaptive mesh the potential is
  kept on both ghost layers for this.
- **Stars start in the grid's own balance.** A star set up from its analytic profile is in
  balance with the analytic potential, not with the one the Poisson solver finds on the grid,
  and answers the mismatch by ringing. So the polytrope and the star of the tidal disruption
  are settled before the first step, in the manner of Zingale et al. (2002) for mapped stellar
  models: the density is set to satisfy `h(rho) + phi = C` with the grid's potential and the
  star's mass, the potential is solved again, and the two are repeated until the density
  changes by less than 1e-5 (6 to 8 rounds; the first moves it by about 1%). For n = 1 that
  relation is exactly the trapezoid rule above. On the adaptive mesh (128 cells, 2 levels)
  the settled n = 1 polytrope's central density then strays by 0.14% over three dynamical
  times against 0.26% for the plain scheme, and its energy by 0.33% against 0.77%. On a
  uniform grid of 64 cells over ten dynamical times the energy also drifts less (2.0%
  against 3.5%), but the central density strays further (2.5% against 0.7%). The ambient gas
  around the star is uniform, not in balance, and falls onto it at up to Mach 0.8; with less
  damping in the scheme, the star shows more of that.
- **Point masses** share the gravity field texture with self-gravity, so the hydro passes do
  not know they exist. Each step they drift, then one reduction over the updated gas gives
  every mass its pull on the gas (whose reaction moves the mass) and what its sink takes; the
  second half-kick pass removes exactly that from the cells. The removal rate falls from
  `sqrt(G M / r^3)` at the center to zero at the sink radius, and the gas that goes takes its
  momentum to the mass, so mass and momentum are conserved between the two (to 1e-7 in the
  validation test). The pull between two masses is softened by the larger of their two
  lengths. The frame is inertial unless the preset asks for the gas's rest frame, as the
  tidal disruption does. Each pair of masses also limits the timestep, to a sixteenth of the
  CFL number times the lesser of `r / v` and `sqrt(r^3 / G M)` for their softened separation
  and relative speed (a twentieth at the default CFL number: about 125 steps for a circular
  orbit). The gas alone would step straight through a close encounter: two masses on an
  orbit of eccentricity 0.9 in near-empty gas keep the orbit's energy to 0.7% over two
  periods with the limit, and lose 94% of it without. The binary preset's gas already steps
  more finely than its orbit asks for.
- **Cooling** is applied to the internal energy at the end of the update, at fixed density,
  in up to 12 second-order substeps that each change the energy by at most a tenth. A substep
  that would cross thermal balance goes to the balance point instead, so a cooling time far
  below the timestep is stable and does not overshoot. The timestep is never shortened for
  cooling: a cell that cools faster than the substeps can follow loses at most 70% of its
  energy in a step. Temperature is `p / rho` in code units (the interstellar curve takes a
  conversion to kelvin, and reads the density as atoms per cm^3). Without conduction the
  smallest clouds of the thermal instability are set by the grid and shrink as it is
  refined: the scale of the structure at t = 0.6 falls by 20% from 512^2 to 1024^2 cells.
  With the preset's conduction (`kappa` = 3e-4) it changes by 8%, for a quarter more steps
  at 1024^2.
- **Viscosity and conduction** are explicit fluxes added to each face's Riemann flux, from
  the state at the start of the step: `-tau . n` and `-(tau . n) . v` for the stress
  `tau = rho nu (grad v + its transpose - 2/3 div v)`, and `-kappa rho grad T` for the heat,
  with `nu` and `kappa` constant (as Athena++ defines `nu_iso` and `kappa_iso`; an ideal
  gas's temperature diffuses at `(gamma - 1) kappa`). Derivatives along a face's normal are
  differences of its two cells, those across it the mean of the two cells' centered
  differences, which reach the cells diagonally across from the one being updated. Being
  part of the face flux, they inherit what holds for it: the two cells of a face compute the
  same bits, and across refinement boundaries and between blocks on different timesteps both
  sides use one flux, so momentum and energy stay conserved to round-off. On the adaptive
  mesh the edge ghost cells such a difference would need on the rim of a block's face are
  never filled; there both blocks take one-sided differences along the rim instead. What the
  stress dissipates goes into the separately advected internal energy as the energy the
  faces carry in, less what the cell's kinetic energy gains. An obstacle's face is a wall at
  rest half a cell away that the gas sticks to and that passes no heat; the faces of the box
  stay as they are for the gas (a reflecting one is a plane of symmetry, without shear
  stress). The timestep counts diffusion as a signal speed of `2.5 d D / dx` beside the
  gas's own, for `d` dimensions and the larger diffusivity `D` of `4/3 nu` and the thermal
  one: it falls as `dx^2` where diffusion dominates, and per-block timesteps take that up
  like any other signal. There is no implicit or super-stepped solve, so strong diffusion on
  a fine mesh is slow. Which of the two are in use is compiled into the passes: switched off
  they cost nothing, conduction alone adds no measurable time, and viscosity makes the flux
  update 1.6 times as long in 2D and 1.8 times in 3D (its cross derivatives are 4 and 8 more
  loads per face). Measured: the kinetic energy of a shear wave falls as
  `exp(-2 nu k^2 t)` to 0.1% on 64 cells per wavelength (along an axis or the diagonal, on
  the uniform grid and across block faces) and to 0.3% in 3D on 32; between two no-slip walls
  to 0.05%; a temperature wave at constant pressure decays at `kappa k^2 (gamma - 1) / gamma`
  to 0.3%.

## License

MIT; see [LICENSE](LICENSE).
