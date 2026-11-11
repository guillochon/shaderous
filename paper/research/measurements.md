# Sources of the numbers in the manuscript

Every number quoted in `paper/main.tex` comes from one of the sources below. All refer to the Shaderous repository at commit `6f397a2` (2026-11-10), except the rows on viscosity, conduction and the point-mass pair limit, which were measured with the change that added them. When the code changes, rerun the measurement and update both this note and the manuscript.

## From the repository README (`README.md`, "Notes on the numerics")

| Quantity | Value | README section |
|---|---|---|
| Flux update time with all arithmetic removed | 85% of the full pass | "A step is four hydro passes" |
| Storage buffers against 3D textures, same stencil | about 1.4 times slower | same |
| 2D workgroups 16 x 16 x 1 against 8 x 8 x 4 | more than halves the flux update | same |
| Largest advection error, third-order reconstruction against MC | 2.5 to 5 times smaller | "Reconstruction" |
| Smoothness threshold for applying the limiter | 1% of the natural scale | "Reconstruction" |
| Isothermal column falling by e every 4 cells: largest Mach number after a sound crossing | 4e-5 well balanced, 0.35 without | "Well-balanced reconstruction" |
| Rayleigh-Taylor background | 240 times quieter | same |
| Jeans growth-rate error | 12 times smaller | same |
| Settling: rounds, tolerance | 6 to 8 rounds, density change < 1e-5 | "Stars start in the grid's own balance" |
| Settled n = 1 polytrope, AMR 128 cells 2 levels, 3 dynamical times: central density, energy | 0.14% vs 0.26%; 0.33% vs 0.77% | same |
| Low-Mach HLLC floor; Gresho vortex at Mach 0.01, 64 x 64, two time units | floor 0.3; loses 6% vs 15% | "HLLC, low Mach" |
| Local timesteps: cycle length; Sedov 3D 256 with 3 levels; isothermal collapse 256 with 3 levels | up to 8 substeps; 2.2x fewer block updates, 1.9x faster; 2.3x fewer, 1.6x faster | "Local timesteps" |
| Local timesteps: fallback thresholds | share of updates above 80% (65% with gravity) | same |
| Local timesteps: cost of extra passes | about 10% per block update without gravity, 50 to 70% with gravity (3D) | same |
| Evrard energy loss with local timesteps | 2.1 times (128, 2 levels), 1.1 times (256, 3 levels) | same |
| AMR gravity: error on a uniform sphere | below 0.3% with three levels | "Gravity on the adaptive mesh" |
| AMR block budget in 3D | 2197 blocks, about 3 GB with gravity | same |
| Viscosity: cost of the flux update; diffusive signal speed | 1.6 times in 2D, 1.8 times in 3D, conduction alone unmeasurable; 2.5 d D / dx | "Viscosity and conduction" |
| Point-mass pair limit: fraction, steps per circular orbit; eccentricity-0.9 orbit over two periods | CFL / 16, about 125 steps; energy to 0.7% with the limit, 94% lost without | "Point masses" |
| Thermal instability at t = 0.6, scale of the structure from 512^2 to 1024^2 cells | changes by 8% with conduction (kappa = 3e-4), falls by 20% without | "Cooling"; `clouds_*` metrics of the conduction test |
| Preset coefficients | binary nu = 1e-3; thermal instability kappa = 3e-4 | `src/problems/binary.ts`, `src/problems/thermal_instability.ts` |
| Jeans refinement default | 16 cells | same |
| Mesh rebuild interval | every 8 steps | "Adaptive mesh" |
| Point-mass mass and momentum conservation | 1e-7 in the validation test | "Point masses" |
| Cooling substeps | up to 12, each changes the energy by at most a tenth; a step loses at most 70% | "Cooling" |
| Compressed snapshot sizes | 8% to 86% of 24 bytes per cell | "Saving and loading the state" |
| 2D grid maximum | 2048 squared | feature list |

## From the code

| Quantity | Value | Source |
|---|---|---|
| Block size | 16 cells per side, two ghost layers | `src/amr/amr_sim.ts` header |
| Mesh refinement check interval | `src/amr/amr_sim.ts` | header comment |
| Lines of WGSL and TypeScript in `src/` | 4,532 and 10,163 | `find src -name "*.wgsl" | xargs cat | wc -l` (and `*.ts`) |
| Number of validation tests | 35 | `src/validation/tests.ts` |
| Validation pass criteria | as quoted in Table 2 | `criterion:` fields in `src/validation/tests.ts` |

## Not yet measured

- Throughput (cell updates per second) on named GPUs and browsers: needs a benchmark run at normal priority. The manuscript leaves these in `\todo{}`.
- The measured values (not the criteria) of each validation test: needs a run of `npm run test:gpu` with the JSON report saved here.
