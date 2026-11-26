# Changelog

What changed in each release of Shaderous. A release's section here is the body of its
[GitHub release](https://github.com/guillochon/shaderous/releases) and ships in the
[npm package](https://www.npmjs.com/package/shaderous). To cut a release, see
[docs/headless.md](docs/headless.md#releases).

## Unreleased

### Added

- **Up to seven levels of refinement** (plans/seven-levels.md), from five: 128 times finer than
  the coarsest grid, for problems whose interesting part is small against the box. A level needs
  a whole 16-cell root block per side (7 levels from 2048 cells across). 3D grids go to 4096³ at
  the finest, adaptive only; the leaf index (64 MB there) is counted in a grid's memory. A 2D
  Sedov blast on one root block, 7 levels deep, matches the similarity solution to 0.3% and the
  same blast on 4 levels exactly (validation `amr-seven-levels`).
- **Longer local timesteps:** `tbins` 4 and 5, blocks stepping up to 16 and 32 times the
  smallest step (validation `amr-time-bins-32`). With self-gravity they stop at 16: over 32
  substeps an Evrard collapse lost four times the energy it does with one timestep.
- **The Setup dialog's grid menus adjust each other.** Every resolution and number of levels can
  be picked; one the other cannot hold changes the other (more levels than the resolution has
  room for raise it; a grid too large for the GPU memory lowers the resolution, or raises the
  levels), and such choices are marked with an asterisk whose tip says what will change.
- **A warning when a crash has cut the GPU memory budget.** Under the status line, in amber:
  the budget in force against the automatic one, and a click opens Performance. Auto there now
  also forgets the crash and restores the full automatic budget (before, a lowered budget stayed
  until the user typed a value).

## 0.4.0 (2026-11-23)

### Added

- **Releasing a followed frame after a merger** (an orbit's `release`). A frame that follows a
  binary keeps the axis of its mass; once one star is torn apart over the other there is no axis
  left, and the frame turned at random (in a 1024³ double white dwarf its rate fell from 0.36 to
  0.07 as the donor came apart, and was back at 0.13 two time units later). Released, it watches
  how much the mass still looks like a binary (e, its quadrupole over its moment of inertia in
  the orbital plane: 0.90 to 0.94 for that pair before it merged, 0.2 to 0.4 after), and as e
  falls from 0.7 to 0.3 it hands over from keeping the axis to slowing down by its starting rate
  per orbit; once stopped, the run carries on in an inertial frame. In a merging pair of
  polytropes the released frame stopped an orbit after the merger and the remnant stayed compact
  and centered, keeping 68% of the mass in the box, where the frame left following swung between
  -0.22 and 0.47 and 52% stayed. A checkpoint saved before the release is released when loaded
  with it set. The diagnostics' `frame.release` says how far it has got.
- **Opening a checkpoint for viewing only** (plans/view-only.md), for runs too large to continue
  on this device: "Open for viewing" in the page, `viewAthdf()` in the API, `load_athdf` with
  `view: true` in the MCP server. The file is read in parts, never copied whole (files past
  WebAssembly's 4 GB open), its finest levels merged into coarser ones until it fits the GPU
  memory budget (mass and momentum kept, the magnetic field divergence-free), and shown by a mesh
  that holds only what the display reads, about a quarter of what running it takes. It does not
  step. Where that keeps more of its levels, it holds only the field shown (about 0.1 MB a block
  in 3D), worked out on the GPU from the file and read again when the field changes; probes then
  read the file's own cells. Field lines load the face field when turned on. Uniform-grid files
  open too, downsampled by twos if they must be. A 3.5 GB magnetized double white dwarf
  checkpoint (19,419 blocks to level 5) opens whole in the default 2.5 GB budget in about 35 s,
  another field in about 20 s, and with field lines at level 4.
- **Wind tunnels and bodies from mesh files** (plans/wind-tunnel.md). Obstacles may be read from
  STL, OBJ, PLY, glTF and GLB files (a `mesh` shape; airfoil `.dat` and SVG outlines for 2D),
  given beside a setup document, through the MCP server's `dir`, by URL, or dropped on the Setup
  dialog, which keeps them in the browser. The diagnostics report the force and torque the gas
  exerts on each obstacle, exactly the momentum its walls took (`bodies`; test `force-balance`
  to 5e-6), with drag and lift coefficients in a `tunnel`: a part that sets an inlet, an outlet
  with a sponge, the sides and floor, and the gas (the liquid with implicit pressure below Mach
  0.3). A rotor can be run in its own frame (`rotor`), now that the liquid's implicit pressure
  runs in a rotating frame. Spheres and boxes can move on the uniform grid (`motion`: spin,
  slide, swing, path), pushing the gas with mass conserved. A preset, `wind-tunnel`, puts a
  car-like body in a tunnel. The cylinder at Reynolds number 100 gives a drag coefficient of
  1.38 extrapolated (published 1.32 to 1.35, with 3% blockage).
- **Stars in binaries, and Roche lobe overflow.** A `binary` orbit may now hold stars as well as
  point masses: a star and a point mass, or two stars. On a circular orbit its stars start
  tidally locked and settled in the Roche potential (the grid's, the point masses' and the
  centrifugal term), stretched toward their companions, each out to its L1 point. `fill` sets
  the separation from how full the first star is of its Roche lobe (Eggleton 1983), and
  `corotating` runs in the frame turning with the orbit, where a binary in balance stands
  still; with `track` the frame follows the orbit as it changes, keeping the axis of the binary's
  quadrupole in place, with the Euler force of its changing turn. Two stars held still in a
  followed frame keep their orbit; moving across the grid, the same pair spiraled in. The setup checks warn of a star that overflows its lobe from the start. A new preset,
  **Roche lobe overflow**: a Sun-like star pouring gas through L1 onto a 3 solar mass black hole,
  stably, in a stream that winds into a disk; and its variant **Two stars**, a Sun-like star
  filling its lobe beside a 0.6 solar mass star, both resolved. A circular binary's orbit is
  found as its stars settle, from the settled stars' own pull: at the point masses' rate two
  stretched stars fell together (a separation 3.1% smaller by t = 10; now 0.1%).
- **Rigidly rotating stars**, settled with the centrifugal potential, and a preset
  **polytrope-rotating**.
- **Partly filled cells** (`physics.cutCells`): obstacles at rest cut the uniform grid's cells
  for the liquid with implicit pressure, with the wall's shear taken off the wall and small cells
  redistributed. Drag and lift converge toward the reference values where the staircase walls
  did not (a NACA 0012 at Re 1000, the cylinder at Re 40). Built-in bodies `naca0012` and blade
  sections; a **wind turbine** preset, a three-bladed rotor in its own frame with its power
  coefficient. Obstacles refine the adaptive mesh from their own shapes (`refine`,
  `refineBand`, `refineLevel`), so parts thinner than a coarse cell stay whole.
- **More combinations of physics.** Radiation transport with implicit pressure; species in
  axisymmetric and spherical runs; species that set the equation of state under implicit
  pressure and with MHD on the adaptive mesh; the Jeans floor and Jeans refinement in comoving
  coordinates (in proper lengths); axisymmetric runs mirrored across z = 0; any number of stars
  under mirror symmetry. Each has a validation test.
- **A magnetized double white dwarf** example setup, and Slurm scripts for chained runs on a
  cluster (resubmitted from checkpoints, requeued jobs resuming, checkpoints kept per stretch of
  simulated time).
- **The command-line runner**: `--from FILE.athdf` carries a run on from a checkpoint; several
  views per output, `--overlays` (color bar, axes, status line), a wall-clock budget,
  checkpoints on their own schedule; the adaptive mesh's status (blocks, a full budget) in
  `diag.jsonl` and `state()`; `SHADEROUS_DIST` serves another built page. Checkpoints of
  gigabytes: the runner and the MCP server write and read the .athdf file themselves, array by
  array, past the browser's memory.
- **Setup checks** refuse many more setups that could not run (found by randomized testing of
  setup documents drawn from the option table): equation-of-state constants and table scales
  that do not fit 32-bit numbers in the units chosen, a gamma of 1 outside the barotropic gas,
  flows without what they need (units, a star, their species), a star with G = 0, uniform
  fields with a divergence of their own in (R, z) and (r, θ), and others.
- The view turns with a rotating frame; the block grid draws derivative fields from the state
  shown; presets are grouped by what they model.

### Changed

- **Faster steps.** Equations of state that iterate or read a table keep each cell's primitives
  from the timestep pass on the uniform grid instead of converting the state at every load
  (`physics.storedPrims`): a 3D white dwarf with the Helmholtz gas about 3 times faster on an
  A100, an ANEOS planet 2.7 to 3.9 times. The Koren reconstruction skips its limiter where the
  flow is smooth (4 to 9% a step). Shaders compile faster where the equation of state is
  costly.

### Fixed

- HLLC and its low-Mach version take the HLL flux where gas too cold for its wave speeds gave
  0 / 0 (every flux NaN), and HLLD takes HLLE where the contact speed rounds onto a fast wave.
- The Helmholtz equation of state starts Newton's iteration for thin hot gas from the ideal gas
  and radiation; below its tables' densities the solve by pressure returned up to 1e5 times the
  energy, and stars' thin surroundings could run away to NaN.
- Radiation pushes the gas by the absorbed share of the flux: in thin gas the difference it took
  rounded to nothing or to noise, depending on the GPU's compiler.
- Implicit pressure takes gas at zero pressure (its sound speed from the pressure floor).
- The helium-shell ignition flow requires the Helmholtz gas it uses.

- Opening a checkpoint for viewing picks a level the GPU's atlas can really hold: it planned
  against the memory budget's bytes alone, but the atlas is a cube of slots (and, with a magnetic
  field, was bounded by a flux register a view does not use), so a file just under the budget
  failed with "the mesh has no leaf at level ..." and was drawn with holes. If the GPU makes a
  smaller atlas than asked, the view is planned again against it.
- A view's field lines no longer flicker "Reading ... and the magnetic field" without end: the
  reader's WebAssembly memory could run short of a 60 MB run for the faces after many reads, and
  the page tried again every frame. The faces are read 128 blocks at a time, a reader that runs
  short starts over in a fresh worker, and a read that still fails says why once.
- Turning on a view's field lines keeps the view as it was (a volume stays a volume): the file,
  opened again with its face field, reset the view as for a file just opened.
- The command-line runner starts `diag.jsonl` empty: a second run into the same folder appended
  its diagnostics to the first run's.

## 0.3.0 (2026-11-19)

### Added

- **Frames and coordinates.** Rotating frames, mirror symmetry (half, quadrant and octant
  boxes), comoving coordinates for cosmology, the shearing box, and axisymmetric (R, z) and
  spherical (r, θ) grids, each on the uniform grid and the adaptive mesh.
- **Radiation transport.** Two-moment (M1) transport with a reduced speed of light chosen cell
  by cell, on the uniform grid and the adaptive mesh, with a light curve in the Diagnostics
  panel and presets for a clump's shadow, radiative shocks and a star held up partly by
  radiation.
- **Nearly incompressible flow.** A weakly compressible liquid, and implicit pressure for the
  liquid and the ideal gas: steps set by the flow speed instead of the sound speed, with
  self-gravity, uniform fields, point masses, species, obstacles and inlets, on both grids.
- **Equations of state.** Hydrogen and helium with molecules, dissociation and ionization from
  a table of one free energy; ANEOS rock, iron and water, with planets built from them and a
  Moon-forming impact preset.
- **Stars.** A point-mass core (a giant), with a common-envelope preset and example setup;
  presets with the hydrogen and helium equation of state (a tidal disruption, a common
  envelope, interstellar gas).
- **The page and the API.** A setup preview where shapes are clicked, dragged and turned;
  field brushes and field-line tubes; session logs with exact replay; screenshots at the grid's
  own resolution; a gallery of the presets in the README.
- **Releases** have a changelog: this file, in the npm package and as the notes of each GitHub
  release.

### Changed

- **Self-gravity conserves energy better and costs less.** The Poisson solve starts from the
  potential extrapolated in time and takes one V-cycle a step on the uniform grid instead of
  two (about 10% faster per step at 128³), and the gravitational work is moved to the mean of
  the old and new potentials after each solve. The Evrard collapse at 64³ now drifts 0.23% in
  total energy (0.82% before).
- **Implicit pressure is time-centered** where the step resolves sound, on both grids, at the
  same one solve a step: a resolved sound wave keeps 98% of its amplitude over a period (86%
  before), and a shock tube's error falls by 40%.
- Per-block timesteps: a run starts with 16 single steps before its first cycle, and the
  half-step potential of each block follows the potential's curvature in time (the Evrard
  collapse loses no more energy with per-block steps than with one timestep).

### Fixed

- Adaptive mesh: cells created by refinement in cold, fast gas got an internal energy many
  orders of magnitude too large (a Zel'dovich pancake on blocks was off by 6.5% in density,
  now 0.7%, as on the uniform grid).
- Adaptive mesh: the potential's boundary values at the edges and corners of refined regions
  were first order, so the error grew with each level added (a uniform sphere on three levels:
  0.26%, now 0.05%).
- Implicit pressure: species in a liquid lost 0.2% of their mass over a run; they are now
  carried on the whole step's mass fluxes and conserved to round-off.
- Diffusion: a face takes the smaller of its two cells' densities, so thin gas beside dense gas
  no longer blows up.
- Species that set the equation of state no longer raise a WebGPU validation error on the
  uniform grid.
- Restarts resume at the step they were saved at.
- Headless runs and the GPU test suite no longer wait forever at their end on macOS, where
  Chrome's crash handler could outlive Chrome and hold the browser's close.

## 0.2.0 (2026-11-17)

### Added

- Resistivity, with tearing-mode and island-coalescence presets.
- Turbulence driving, with driven-turbulence presets with and without a magnetic field.
- Moving walls (the lid-driven cavity preset) and line-tied walls (the X-point collapse
  preset).
- Presets: an Alfvén-wave collision, Elsasser packets and a magnetized Kelvin-Helmholtz
  instability.
- Nuclear heating as a view field and the L nuc diagnostic.
- Temperature in `slice()` and `probe()`.

### Changed

- Faster self-gravity on the uniform grid (the isolated boundary is summed once per solve),
  burning on the adaptive mesh (only the leaves' interior cells), and `slice()` and `probe()`
  (pressure only where they sample).
- A snapshot's pressure is that of each cell's own composition.

### Fixed

- The timestep shown while a run is held at a frame time no longer reads 0.

## 0.1.2 (2026-11-16)

The first release on npm: the built page with the headless runner (`npx shaderous`) and the MCP
server (`npx shaderous mcp`), listed in the MCP Registry.
