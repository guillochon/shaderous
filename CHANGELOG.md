# Changelog

What changed in each release of Shaderous. A release's section here is the body of its
[GitHub release](https://github.com/guillochon/shaderous/releases) and ships in the
[npm package](https://www.npmjs.com/package/shaderous). To cut a release, see
[docs/headless.md](docs/headless.md#releases).

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
