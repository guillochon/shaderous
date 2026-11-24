# Plan: hyperlocal weather from a parent forecast

Status (2026-11-24): not started. An exploration plan: what to measure, and which design
choices the measurements settle, before any atmosphere-specific code is written. Written from
reading `docs/numerics.md`, `plans/incompressible.md` and `plans/wind-tunnel.md`; nothing
here has been run. The citations in section 12 are from memory and must be checked (Crossref)
before they are quoted anywhere else. Cost figures are estimates, marked as such.

Goal: take the forecast of an operational model (HRRR at 3 km in the US; ICON or ECMWF
through Open-Meteo elsewhere) and compute the flow over a few kilometres of real ground at
10 to 50 m: the wind over a ridge, the cold air that pools in a hollow at night, the thermals
of a sunny afternoon, the sea breeze, the wind between buildings. This is **dynamical
downscaling**, not a better forecast of the large scale. The parent model owns the large
scale; this run adds what the terrain and the surface force, which a 3 km grid cannot
resolve.

What the code cannot promise, said plainly from the start: individual eddies at these scales
lose predictability within minutes. The useful outputs are statistics and terrain-locked
features (mean wind and gust factor at a point, where the cold pool sits, where the lift is),
not "a gust at 14:03".

| Stage | What it adds | Settles |
|---|---|---|
| 0 | Reading checks: which existing pieces combine with implicit pressure | the list of gaps |
| 1 | Dry benchmarks of atmospheric solvers on today's code | D1, D2, D3, the cost |
| 2 | Real terrain, dry, steady wind from a sounding | D4, D5 |
| 3 | Heat at the ground: the daytime mixed layer and the night's stable layer | D6, D7 |
| 4 | Driven by the parent model through a day; compared with stations | D8, D9 |
| 5 | Moisture: fog and shallow cumulus | D10 |
| 6 | The product: fetching inputs, the page, ensembles | D11 |

Each stage ends with a go or no-go (section 9). Stages 1 to 3 use nothing outside the repo
but sounding numbers and a terrain file.

## 1. What the code has

Reading the docs changed the first guess, which was that the time step and hydrostatic
balance would be the obstacles. They are mostly solved already:

- **Implicit pressure for the ideal gas** (`docs/numerics.md` section 9; `plans/incompressible.md`
  4.12, 5.10, 5.11). The step is not limited by the sound speed: a Gresho vortex takes 212
  steps at Mach 0.1, 0.01 and 0.001 alike, where the explicit scheme takes 3 411 and 32 198
  at the first two. It runs on the uniform grid and on blocks, with obstacles, inlets,
  species and a uniform gravity field. Atmospheric winds are at Mach 0.01 to 0.05.
- **Balance under a uniform field with implicit pressure.** Columns whose density falls by e
  every 4 cells hold at Mach 1.2e-6 through a sound crossing (`psplit-gas-balance`). The
  real atmosphere is far gentler: its density falls by e in about 8 to 10 km.
- **The reference pressure** that keeps single precision usable at low Mach number is taken
  where the pressure varies by less than a factor of 4 over the grid. The pressure falls by a
  factor of about 1.25 over the lowest 2 km, the depth of every box past stage 1, so those
  qualify easily. The 10 km columns of the benchmarks (a factor of about 3.8) are close to the
  limit. This must be checked, not assumed (E1).
- **Well-balanced reconstruction** for the explicit scheme (section 8), and low-Mach HLLC
  (section 6), as a fallback and for comparison.
- **Solid cells, inlets, box faces** of several kinds, refinement along an obstacle's outline,
  mesh files read into a signed distance field (`plans/wind-tunnel.md` 3.5), and partly filled
  cells being built (`wind-tunnel.md` 8.1; `src/hydro/cut.ts`).
- **A box-shaped damping zone** (the tunnel's sponge; `hydro/damping.wgsl`) that pulls the gas
  toward a target state.
- **A rotating frame** with the centrifugal force switchable off: an f-plane, which is the
  Coriolis force this problem needs.
- **Species** carried with the gas and conserved to 1e-7 under implicit pressure.
- **Constant viscosity and conduction** (section 11), **optically thin heating and cooling**
  (section 10), **refinement zones** (`mesh.zones`).
- **Units**: the setup format takes SI (the tunnel's m/s and metres).

## 2. What is missing

| Missing | Why it matters here | Stage |
|---|---|---|
| A stratified initial state from a sounding (temperature and wind against height) | every run starts from one | 1 |
| Terrain from an elevation grid (GeoTIFF or similar) into the solid mask | the whole point | 2 |
| Inflow faces that carry a profile against height, not one state | wind shear at the inlet | 2 |
| A surface heat flux and a momentum flux at the ground (a wall model) | thermals, cold pools, the right wind near the ground | 3 |
| A subgrid model (Smagorinsky or TKE) | if the scheme's own dissipation is not enough (D6) | 3 |
| A time-varying target for the damping zone, read from the parent model | driving the box through a day | 4 |
| Turbulent inflow: laminar inflow takes kilometres to become turbulent | the upstream part of the box is wasted otherwise | 4 |
| Water vapour and cloud water with latent heat | fog, cumulus, humidity | 5 |
| Readers for HRRR GRIB2 or Open-Meteo JSON | stages 4 to 6 | 4 |

Implicit pressure today refuses species that set the equation of state, and burning
(`numerics.md` section 9, "Species"). Moisture needs both in a mild form: vapour changes the
gas constant by about 1%, and condensation releases heat. Section 7 (D10) takes this up.

## 3. Stage 0: checks by reading

Answer from the code and the option rules (`src/setup/options.ts`), each in a line, before
running anything:

1. Does implicit pressure accept viscosity and conduction? Heating and cooling? The rotating
   frame without the centrifugal force? The damping zone? Refinement zones?
2. Does the gravity time limit $`C\sqrt{\Delta x/|g|}`$ (kept with implicit pressure,
   section 9) apply to a balanced atmosphere? At 20 m cells it is about 0.7 s, at 5 m about
   0.35 s. The physical limit of a stable atmosphere is the buoyancy period,
   $`2\pi/N \approx 10`$ minutes. If this limit sets the step, it is the first thing to
   question (D3).
3. Can an inlet take a different state in each cell of a face (a profile), or one state per
   part? Can a box face be an inlet?
4. What does the top of the box do? A reflecting lid returns gravity waves to the ground; the
   damping zone at the top is the usual cure (as in the Schär test, E4).
5. Does `physics.gravity` take a value in SI and point along z, and does the SI unit system
   reach the temperature (in kelvin) for diagnostics?

Deliverable: a short table in this plan, "Answers to the checks", in the form of
`wind-tunnel.md` 2.1.

## 4. Stage 1: dry benchmarks

Standard tests of atmospheric dynamical cores, all dry and with no ground physics. Each has a
published answer. Each runs with implicit pressure, and with the explicit scheme as a
comparison where the explicit scheme is affordable. Run on the uniform grid first, then on
blocks.

| Id | Test | What it measures | Reference answer |
|---|---|---|---|
| E1 | An atmosphere at rest: isentropic (θ = 300 K) and isothermal, 10 km deep | the balance, the floor in Mach number, whether the reference pressure is taken | stays at rest; target Mach below 1e-5 for an hour |
| E2 | Rising warm bubble (Robert 1993): a bubble 0.5 K warm in a neutral atmosphere, 2D | buoyancy of small temperature perturbations in single precision | the bubble's height and shape at 10 to 20 min |
| E3 | Density current (Straka et al. 1993): a cold bubble falls and spreads, viscosity 75 m²/s, 2D | the whole scheme against a converged answer; convergence with resolution | front at about 15.5 km at 900 s; θ' profiles at 25 to 200 m |
| E4 | Mountain gravity waves (Schär et al. 2002): flow over a rippled hill, damping layer at the top, 2D | flow over terrain in a stratified atmosphere; the staircase wall against partly filled cells | the linear solution for w |
| E5 | Precision: E2 with a bubble 0.01 K warm | whether float32 resolves the perturbations fog and cold pools need | the bubble rises, symmetric |
| E6 | Cost: E1 and E3 in 3D at 64³, 128³, 256³, implicit pressure | steps per second, cycles per solve | none; the number feeds section 8 |

Notes:

- E3's viscosity is part of its definition: it makes the problem converge. The constant
  viscosity of section 11 is enough.
- E4 is the test that will hurt the staircase. Terrain at 20 m cells is gentle slopes made of
  steps, and steps make spurious vortices and waves. Run it with the staircase, then with
  partly filled cells when they reach the ideal gas.
- Each test becomes a validation test in `src/validation/` once it passes, in the form of the
  `psplit-*` tests, with its tolerance from the measurement.

Done when: E1 to E5 pass at the published tolerances or the reason they do not is known, and
E6 gives a cost per cell-step.

## 5. Stage 2: real terrain, dry, steady wind

- **Terrain to the mask.** An elevation grid (USGS 3DEP at 1 to 10 m in the US, Copernicus
  GLO-30 elsewhere) turned into a height field, and the height field into solid cells or
  partly filled ones. A `terrain` part in the setup format, beside `tunnel`: file, corner
  coordinates, extent, and a smoothing radius for the box edges.
- **The sounding.** A `sounding` part, or a key of a new `atmosphere` part: θ (or T) and wind
  against height, integrated to discrete hydrostatic balance on the grid (numerics section 16
  does this for stars).
- **Inflow and outflow.** Inflow with the sounding's profile on the upwind faces, outflow and a
  sponge downwind, a sponge at the top. A real wind is not along an axis; turning the terrain
  rather than the wind (as the tunnel does for yaw) keeps the faces simple.

Case studies, each with field measurements:

| Id | Site | Measured | Why |
|---|---|---|---|
| E7 | Askervein hill (Taylor & Teunissen 1987), neutral flow | speed-up over the crest along lines of masts | the classic hill test; a smooth, isolated hill |
| E8 | Bolund (Berg et al. 2011), neutral flow | speed-up and turbulence over a steep cliff | a sharp edge, where the staircase may do as well as anything |
| E9 | A ridge-soaring site of our choosing, against a pilot's tracks or a station | lift band position | the first useful output |

Done when: E7's speed-up at the crest is within 10 to 15% of the masts at 10 m above ground,
which is the range published models reach.

## 6. Stage 3: heat at the ground

- **Surface fluxes.** First a prescribed sensible heat flux, uniform or by land cover, varying
  with the time of day, and a wall stress from Monin-Obukhov similarity (the log law with a
  stability correction) at the first cell. This is the wall model of `wind-tunnel.md` 8.2 with
  heat added; build it once for both.
- **Subgrid model.** Run first with the scheme's own dissipation (implicit LES), then with a
  Smagorinsky or TKE eddy viscosity, and compare.

| Id | Test | Reference |
|---|---|---|
| E10 | Convective boundary layer: heat flux into a neutral layer under an inversion, periodic sides | Schmidt & Schumann 1989; profiles of heat flux and velocity variance against z/zᵢ; growth of zᵢ |
| E11 | Stable boundary layer (GABLS1, Beare et al. 2006): the ground cools at 0.25 K/h under a geostrophic wind | the LES intercomparison's profiles; the layer's depth |
| E12 | A cold pool in a real valley at night, against a station on the valley floor | the temperature drop at the floor against the slope |

E11 is the hard one for any LES: the eddies shrink to metres, and coarse runs go laminar or
collapse. Whether 5 to 10 m cells are enough here decides how far down the night side of the
product can go.

## 7. Design decisions

| Id | Decision | Choices | Settled by |
|---|---|---|---|
| D1 | Equations | the existing compressible gas with implicit pressure; a new anelastic or Boussinesq solver | E1 to E5. The existing path is preferred: it is built, tested and keeps shocks out of the way for free. A new solver only if the existing one fails on balance or precision |
| D2 | Prognostic variables | total energy (today); perturbations from a reference sounding | E5. If float32 cannot hold 0.01 K, carry the departure from the initial balanced state, as the well-balanced reconstruction already does for the pressure |
| D3 | The gravity time limit | keep; relax for a balanced atmosphere (limit by N rather than √(g/Δx)) | Stage 0 check 2 and E6. A factor of 2 to 5 in cost |
| D4 | Terrain | staircase; partly filled cells; terrain-following coordinates | E4, E7. Terrain-following coordinates are what weather models use, but they change the grid itself: last resort |
| D5 | Box size and refinement | one box at one resolution; refinement zones around the area of interest; nesting several runs | E7, E9 and the cost |
| D6 | Turbulence | implicit LES; Smagorinsky; TKE (Deardorff) | E10, E11. Shared with `wind-tunnel.md` 8.3 |
| D7 | Ground | prescribed fluxes; a land surface model (soil heat, moisture, radiation) | E12. Prescribed fluxes go a long way; a land model only if the cold pools need it |
| D8 | Lateral boundaries | relaxation zones toward the parent (Davies 1976); open faces with profiles | stage 4 runs |
| D9 | Turbulent inflow | a long fetch; recycling a plane; perturbing cells near the inlet | stage 4. The cell-perturbation method is cheap and fits a gather |
| D10 | Moisture | vapour and cloud water as species with saturation adjustment after the pressure correction; or a full microphysics | stage 5. Start with saturation adjustment: no rain, only fog and cloud. It needs implicit pressure to accept a small source of heat and a gas constant that varies a little (section 2) |
| D11 | Where the inputs are read | in the page (Open-Meteo answers in JSON and allows the browser); a tool in `tools/` that writes a setup document (GRIB2 needs a decoder) | stage 6 |

## 8. Cost (estimates until E6)

A box 5 × 5 km by 2 km deep:

| Cells | Count | Step, limited by advection (u = 10 m/s, C = 0.5) | Steps for 6 h |
|---|---|---|---|
| 50 m | 0.4 M | 2.5 s (the gravity limit, about 1.1 s, may set it: D3) | 9 000 to 20 000 |
| 20 m | 6.3 M | 1.0 s (gravity: 0.7 s) | 22 000 to 31 000 |
| 10 m | 50 M | 0.5 s (gravity: 0.5 s) | 43 000 |

Refinement zones change this a lot: fine cells only near the ground and over the area of
interest. Whether a run keeps up with real time is E6's question. Keeping up is the threshold
for a forecast. Running several times faster than real time is what an ensemble needs.

## 9. Go or no-go

- **After stage 1.** No-go if E1 or E3 cannot be made to pass with implicit pressure on the
  uniform grid. That would mean D1 points to a new solver, which is a different, larger plan.
  Also no-go if E6 puts a 20 m run of 6 h above about a day on one GPU, and neither D3 nor
  refinement offers a clear factor.
- **After stage 2.** No-go for terrain-forced wind if E7 is off by more than 25% at the crest
  with partly filled cells. Stage 3 could still go ahead over flat ground.
- **After stage 3.** If E11 cannot be made to work at the resolutions the cost allows, the
  product is daytime and neutral only. Say so, and drop the night side.

## 10. Stage 4 onward, in outline

- **Stage 4.** Hindcast a real day at one site, the parent model's analysis forcing the
  sponges hour by hour, against personal and official weather stations. Metrics: mean
  absolute error of the 10 m wind and the 2 m temperature, against the parent model
  interpolated to the same point. The question is whether downscaling beats interpolation,
  and where.
- **Stage 5.** Moisture with saturation adjustment. Test: BOMEX shallow cumulus
  (Siebesma et al. 2003), then valley fog.
- **Stage 6.** The product: a site picked on a map, the inputs fetched, a run sized to the
  GPU, the outputs named in the goal drawn on the terrain. An ensemble of a few runs
  perturbed from the parent's members, if the cost allows.

## 11. Order of work

1. Stage 0 checks (reading only).
2. E1, E2, E5 on the uniform grid in 2D: an afternoon. They decide D1 and D2 early.
3. E3 with convergence, then E4 with the staircase.
4. E6, and D3 if the gravity limit sets the step.
5. The sounding part and E1 to E4 on blocks.
6. Stage 2, waiting on partly filled cells for the ideal gas if E4 says so.
7. Stage 3, sharing the wall model and the subgrid model with the wind tunnel.

## 12. References (from memory, unchecked)

- Robert, A. 1993, J. Atmos. Sci. 50, 1865: bubble convection with a semi-implicit formulation.
- Straka, J. M. et al. 1993, Int. J. Numer. Methods Fluids 17, 1: the density current.
- Schär, C. et al. 2002, Mon. Wea. Rev. 130, 2459: terrain-following coordinates; mountain waves.
- Taylor, P. A. & Teunissen, H. W. 1987, Boundary-Layer Meteorol. 39, 15: Askervein.
- Berg, J. et al. 2011, Boundary-Layer Meteorol. 141, 219: Bolund.
- Schmidt, H. & Schumann, U. 1989, J. Fluid Mech. 200, 511: the convective boundary layer.
- Beare, R. J. et al. 2006, Boundary-Layer Meteorol. 118, 247: GABLS1.
- Siebesma, A. P. et al. 2003, J. Atmos. Sci. 60, 1201: BOMEX.
- Davies, H. C. 1976, Q. J. R. Meteorol. Soc. 102, 405: lateral relaxation.
- Maronga, B. et al. 2020, Geosci. Model Dev. 13, 1335: PALM, the closest existing code.
- Dowell, D. C. et al. 2022, Weather and Forecasting 37, 1371: HRRR.
