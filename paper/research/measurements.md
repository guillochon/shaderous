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
| Thermal instability at t = 0.6, scale of the structure from 512^2 to 1024^2 cells | changes by 6% with conduction (kappa = 3e-4; 8% before faces took the smaller density), falls by 20% without | "Cooling"; `clouds_*` metrics of the conduction test |
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
| Number of validation tests | 35 (36 from `db8c5c7`) | `src/validation/tests.ts` |
| Validation pass criteria | as quoted in Table 2 | `criterion:` fields in `src/validation/tests.ts` |
| Lines of WGSL and TypeScript in `src/` at `dd42075` | 6,235 and 14,978 (quoted as about 6,000 and 15,000) | same commands |
| CFL number; dual-energy threshold; shock-detector threshold; density and pressure floors | 0.8; 1e-3; 0.5; 1e-10 and 1e-14 | `configFor` in `src/sim.ts` |
| Smoothness threshold and blend of the reconstruction | 0.01 of the scale; weight `clamp(1.5 - eta, 0, 1)` | `src/hydro/recon.wgsl` |
| Multigrid sweeps and V-cycles | 2 before, 2 after, 8 on the coarsest level; 2 V-cycles uniform, up to 3 on the adaptive mesh | `src/gravity/multigrid.ts`, `vcycles` in `src/sim.ts` |
| Refinement filter | EPS = 0.01 | `src/amr/flag.wgsl` |
| GPU memory budgets | 2.5 GB discrete, 2 GB shared, 1 GB phone | `src/gpu/memory.ts` |
| Reduction fold per stage | 4^3 records | `RED_F` in `src/gpu/reduce.ts` |
| Species: no local timesteps | per-block timesteps are off with species | header of `src/amr/species.wgsl` |
| Burning network | 13-isotope alpha chain built with pynucastro; softmax output; DAgger rounds | `tools/burn/build_net.py`, `train.py`, `dagger.py` |

## Measured in the browser, November 2026

Runs driven through the in-page console on an NVIDIA Ampere GPU in Chrome, with the commits named. GPU times come from the timestamp-query panel (`src/gpu/timing.ts`).

| Quantity | Value | Commit, setup |
|---|---|---|
| AMR step, GPU time: 2D Kelvin-Helmholtz 2048^2 with 3 levels; 3D Sedov 256 with 2 levels | 1.19 to 0.43 ms (2.8 times); 1.02 to 0.82 ms (19%) | `e745009` (flat dispatch, four 2D blocks per workgroup) against `8388aad` |
| Saha equation of state: least Gamma_1 in the ionization zone | between 1 and 1.3 (solar, rho = 1e-9 g/cm^3) | `967ab54`, `tests/unit/eos_physical.test.ts` |
| Degenerate series below x = 1/2 | agree with the closed forms to 1e-7 at 0.3 <= x < 0.5 | same |
| Collision energy with E rebuilt at resets, and with E kept (1.0 + 0.8 M_sun, head-on, 256 with 2 levels, box 12) | +2.3% at the greatest compression; -0.6% through the bounce | `7ef7ccf` and `0fcf59e` |
| Collision without gravity: energy until gas leaves the box | +0.34% with E rebuilt; 0.000% with E kept | same, 128^3 |
| Settling of white dwarfs: relaxation, tolerance, rounds, time | 1.6, 1e-3, 7 to 12 rounds, 1.6 s (collision 256 with 2 levels) to 5.5 s (512 with 3 levels) | `7ef7ccf` |
| 1 M_sun white dwarf: central density, unsettled (256 with 2 levels, box 8) | swung from 17.0 to 12.8 and back to 13.9 (25%) | `7a837cd` |
| 1 M_sun white dwarf, settled (128 with 2 levels, box 3, 2.5 time units = 6 dynamical times) | within 3.5% (14.23 to 14.88 about 14.38) | `db8c5c7` with the change of `0fcf59e` |
| Cold white dwarf test (0.6 M_sun, 64^3): central density, energy | within 3.3%; energy to 1.3e-5 | `0fcf59e`, `white-dwarf` in `src/validation/tests.ts` |
| Number of validation tests | 36 | `src/validation/tests.ts` at `0fcf59e` |

## Added with the update to the code at `1bb702b` (2026-11-15)

Numbers that `edits/code-catchup.py` adds to the manuscript. "numerics" is `docs/numerics.md` in the code's repository; a commit means its message, which records the measurement.

| Quantity | Value | Source |
|---|---|---|
| Helium-air contact at one temperature: pressure error without and with the first-order blend; blend thresholds | 3.7% and 0.2%; 1% to 5% in gamma - 1 | numerics §12 |
| Shock-bubble: mirror-cell differences with an on-off switch; helium at the finest level without and with the composition criterion | 5%; 24% and 99% | numerics §12, §17 |
| Arrhenius burn: substeps, energy per substep, burning-length refinement threshold | 8; a fifth of the internal energy; 80 cells | `src/burn/burn.ts`, `src/hydro/burn.wgsl` |
| Arrhenius detonation: resolution, speed with burning in shocks, shortfall without | 12.8 cells per half-reaction length; within 1% of CJ; 5%, halving with resolution | `detonation` test in `src/validation/tests.ts`; `src/problems/detonation.ts` |
| Network: layers, units, DAgger rounds, half-precision weights | 4 x 128; round 30 | commit `6f68adb`; `src/burn/network.ts` |
| Network: GPU against float64 mirror | within 2% (L1) on 48 held-out states | `burn-network` test |
| Network: least temperature and density that burn; longest network step | 1e8 K, 1e4 g/cm^3; 10 ms | `src/burn/network.ts` |
| Burning limiter: tube detonations 0.5 to 4 km; collision of two 0.64 M_sun dwarfs on 940 km cells | f < 0.2 stops detonations, f >= 0.3 matches; Ni56 0.45 M_sun unlimited, 0.008 and 0.056 at f = 0.1 and 0.4 | commit `4c1667e` |
| Point masses in kick form: entropy of the disrupted n = 3 star | cut a hundredfold in flux form; holds to about 0.1 dex (0.12) | numerics §14; commits `3cd1a52`, `74aec88` |
| n = 3 star of Gamma = 5/3 gas: destroyed between beta | 1.8 and 2 (against 1.85 of Guillochon & Ramirez-Ruiz 2013) | commit `74aec88` |
| Face-flux tile: speed-up of the largest uniform grids on an Apple M4 Pro; cost of not splitting the update | 1.3 to 1.5 times; about 10% | `docs/architecture.md`, "Performance notes" |
| MHD divergence: without compensation; with | 3.5e-6 after 750 steps of Orszag-Tang; 7e-8 after 751, 9e-8 after 3483 | numerics §21 |
| MHD memory | 80 bytes per cell on top of 112 | numerics §21 ("Memory") |
| Edge electric fields, field loop at 128^2 | energy kept 93.3% (GS05) against 94.5% and 94.6%; peak 1.08 against 1.15 to 1.21; stray 0.17 against 0.21 to 0.26 | numerics §21 (Lee's estimates) |
| First-order flux correction threshold | density changed by a factor of 4 in stage 1 | numerics §21 |
| MHD on refined blocks: energy; magnetic energy against the uniform grid | 2e-7; 0.3% (rotor), 0.1% (blast) | numerics §21 ("Block grid") |
| Moving mesh: rebuilds, divergence, magnetic energy kept | 144; 1.5e-7; 96% (uniform 256^2: 96%) | numerics §21 ("Re-meshing") |
| MHD time bins: share of block updates | 85% (2D blast, t = 0.05); 98 to 100% (rotor, loop, 3D blast) | numerics §21 ("Time bins") |
| Magnetized disk: beta, Alfven ceiling; alphas and Q after three orbits at 256^3 with 2 levels | 50, 3; alpha_M 0.026, alpha_R 0.003, Q_z 4.2, Q_phi 15 | numerics §21 ("Magnetized accretion disk", "Disk diagnostics") |
| Largest run: RTX 4090, 512^3 with 3 levels | 5,832 blocks, about 24 million cells, 16.8 GB, 6,359 steps in 14 min per orbit; magnetic energy x5.4, alpha 0.009 | `docs/headless.md`, "Largest run so far" |
| 3D mesh limits; H200 run | 35^3 (field), 33^3 (species); with self-gravity only memory since `a9d3ab9` (was 31^3); 1024^3 with 4 levels at 35^3 = 42,875 blocks of 16^3 = 175,616,000 cells (quoted as about 175 million), 117 GB, 10 steps in 10 s | `docs/headless.md`; commits `dc34d29`, `1bb702b` |
| Throughput on the Apple M4 Pro | from `paper/figures/data/performance/throughput_m4pro.json` via `make_performance.py` | commit `34beb4a` |
| Lines of WGSL and TypeScript in `src/` at `1bb702b` | 12,002 and 26,501 | `find src -name "*.wgsl" | xargs cat | wc -l` (and `*.ts`) |
| Number of validation tests | 64 | `src/validation/tests.ts` at `1bb702b` |

## Figures and validation run again at `6fbf2cb` (2026-11-15)

Every figure's simulations were run again with its script (`--run`) on the RTX 3080 12GB in Chrome 156, and the validation suite was run and its report saved.

| Quantity | Value | Source |
|---|---|---|
| Measured column of the three validation tables | one value per criterion, in the criteria's order | `research/validation-run.json` (Playwright's JSON report of `tests/gpu/validation.spec.ts`; each test's metrics are in its `stdout`), 68 of 68 passed |
| Largest div B over the MHD tests | 2.8e-7 (`loop_blocks_divB_later` of `mhd-restart`) | same |
| Tidal disruption, beta = 1.5: peak return rate, time, bound clump at t = 40 | 1.60 M_sun/yr, 61 d after pericenter, 0.117 M_sun (was 5.3, 37 d, 0.09 before the point masses' pull took kick form) | `make_tde.py` output; `figures/data/tde/full/` |
| Tidal disruption, beta = 0.6: surviving core | 0.947 M_sun | `make_tde.py` output; `figures/data/tde/partial/` |
| White dwarf of 1 M_sun at 256 with 2 levels: settled, mapped | 0.52%, 30.4% | `make_balance.py` output |
| Isothermal column: largest Mach number after a sound crossing | 3.8e-5 balanced, 0.35 plain | `make_balance.py`; `figures/data/balance/column.json` |
| Sedov cells in Figure 1b | 6,455,296 leaf cells, every 33rd plotted | `make_hydro.py` output |
| Throughput on the RTX 3080 12GB in Chrome 156 and Firefox 160 | from `paper/figures/data/performance/throughput.json` and `throughput_firefox.json` via `make_performance.py` (its printed table gives every point, its range over three passes and the time per step); measured with the GPU otherwise idle | `make_performance.py --run [--firefox]` at `6fbf2cb` with the working tree's `throughput` (timed again with more steps while a timing is under a second) |
| Shaderous against AthenaK per unit of bandwidth, 256^3 | 1.52e6 per GB/s against 3.01e5 to 3.95e5: 3.9 to 5.1 times (quoted as 4 to 5) | `make_performance.py` output |
| Firefox against Chrome | largest uniform grids 1.63e9 against 1.85e9 (2048^2) and 1.38e9 against 1.39e9 (256^3); adaptive 1.5 to 7.3 times fewer; ms per step 1.9 to 2.5 (2D) and 4.0 to 4.6 (3D) against 0.33 to 0.77 and 1.9 to 3.7 | same |
| Nickel from the collision of two 0.64 M_sun dwarfs with the tabulated equation of state | 0.61 M_sun | no record of the run was found in the repository on 2026-11-15; with a network still trained on the first equation of state's heating. Measure again after the retraining |
