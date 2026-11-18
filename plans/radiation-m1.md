# Plan: radiation transport with the M1 closure

Status: stages 1 to 4 and most of 7 done (2026-11-18); stages 5 and 6 (adaptive mesh, per-block
timesteps) not started, and refused with a message until then. Written 2026-11-15.

Done, as built (docs/numerics.md section 24): the uniform-grid scheme of sections 3 and 4, with
the per-cell speed of light; opacities; view fields; diagnostics; Er and Fr1-3 in snapshots;
settings, links and scripting; two presets; GPU tests rad-off, -restart, -equilibrium, -f32,
-streaming, -diffusion, -shadow, -advect, -cooling-sphere. Departures from the text below: the
sweep's Courant number is 0.5 (forward Euler with linear reconstruction was unstable at 0.8);
tau_0 is 2.5/3 (the streaming rate counts d, as the hydro's does); the face flux is Rusanov's
with the largest M1 wave speed; the work term takes r and the half-kick velocity, and the
exchange gives its rounding to the larger energy, so E_gas + E / r is kept exactly; N_r is a
setting (default 8), not chosen per step. Not yet: rad-marshak, rad-shock, rad-star,
rad-flux-match; the star's light curve preset (it needs the adaptive mesh).

Goal: gray radiation hydrodynamics on the uniform grid and the adaptive mesh, good enough for
the problems Shaderous already runs: white dwarfs, stars being disrupted, their debris. Those
have interiors of enormous optical depth (10^10 and more) moving at 10^-3 to 10^-1 of the speed
of light, wrapped in gas that is nearly transparent. The scheme has to be right at both ends in
32-bit floats, and it has to fit a code that stores no fluxes.

## 1. The variant, in one paragraph

Radiation energy `E` and flux `F` are evolved **in the gas's frame**, closed with Levermore's
M1 Eddington factor. The **hydro step carries them with the gas**, the way it carries species.
A separate **radiation sweep**, run several times per hydro step, does two things in one stage:
explicit transport between cells (HLL, with the dissipation scaled down in opaque cells), and
a **local implicit exchange** of energy and momentum with the gas. The speed of light is
**reduced only in cells too transparent to afford the true one**; opaque cells use the true
speed, so the inside of a star has the right heat capacity and conserves energy exactly.

### Why this and not the others

| Alternative | Why not |
|---|---|
| Lab-frame (mixed-frame) moments with a reduced speed of light, as [Quokka][He24] | With the reduced speed `ĉ` on the transport term, trapped radiation moves at about `(ĉ/c) v` instead of `v`. Quokka's own papers say the reduction cannot be used where `βτ ≳ 1` ([Wibking & Krumholz 2022][WK22], section 3.2.3), and [Skinner & Ostriker (2013)][SO13] conclude the same. A star's interior has `βτ` of 10^7 |
| One reduced speed everywhere | The exchange then conserves `E_gas + (c/ĉ) E_rad`, so radiation in equilibrium with the gas weighs `c/ĉ` times too much. Quokka measured a 20% error in the equilibrium temperature at `ĉ = 0.1c` |
| Trapped and streaming photons split by cell optical depth, as [RAMSES-RT][RT15] | Good idea, and the nearest relative of this plan. But it leans on the numerical diffusion of one particular first-order flux (GLF), and its trapped energy is still inflated by `c/ĉ` |
| Flux-limited diffusion | An implicit global solve in 32-bit floats, on per-block timesteps. No GPU code we found does it |
| Radiation in the equation of state only (what we have: `eos/gas_radiation.wgsl`) | Exact where gas and radiation are locked together, but nothing leaks out: no cooling, no light curve |

What is borrowed from where:

- Evolving the moments in the gas's frame and advecting them with the hydro fluxes:
  [Fornax][Fornax]. Advection and compression then happen at the gas's speed whatever `ĉ` is.
- A source term in every stage, so the scheme has the right diffusion limit when cells do not
  resolve the mean free path: [He, Wibking & Krumholz (2024)][He24].
- The HLL dissipation scaled by `1/τ_cell`: [Fornax][Fornax], equations 17 and 18.
- Reconstructing `E` and the reduced flux `f`, not `F`, and dropping to first order when
  `|F| > cE`: [Quokka][WK22].

Two parts are **our own construction and untested in the literature**: the reduced speed chosen
per cell (section 3.4), and the combination of gas-frame advection with the source-in-every-stage
update. Each has a switch back to the plain version, and a test that tells them apart
(section 9).

### Non-goals for v1

- More than one frequency group; Compton scattering; lines.
- Point masses or gas sources that shine (the `Mass` record has no room; section 10).
- Radiation with a magnetic field: refused with a message at first, as MHD first refused gravity.
- Terms of order `v/c` in the flux equation, and anything of order `(v/c)^2`.
- Second order in time for the radiation sweep (section 10).

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Fields | One `rgba32float` texture `R = (E, F_x, F_y, F_z)`, gas frame | Four numbers, one texel. Species and MHD each keep their own textures too |
| Advection with the gas | A pass after the hydro update moves `E/ρ` and `F/ρ` with the stored face mass fluxes (`MF`), as species move | Radiation trapped in gas then moves exactly as the gas does. The machinery exists, refinement boundaries included |
| Transport | A separate sweep pass, HLL, limited linear reconstruction, fluxes gathered and bit-identical | The hydro shaders do not grow. Compile time has been the bottleneck |
| Subcycling | `N_r` sweeps per hydro step, at most 16; beyond that the hydro step shortens | The sweep is cheap beside the hydro update; the gas should not pay the light speed |
| Exchange with the gas | In every sweep, implicit and local: the flux in closed form, the temperature from one scalar equation | No global solve. The scalar equation is a sum of positive terms (section 3.3), so 32-bit floats suffice at any optical depth |
| The gas's side during the sweeps | A second texture `G = (ρe, Δ(ρv))`: the gas's internal energy as the sweeps change it, and the momentum it has gained | The sweeps never write `U1`/`U2`. One commit pass folds `G` back, as the burn pass writes the other parity |
| Speed of light | True `c` from the unit system; a run constant `ĉ`; reduction per cell (section 3.4) | Thermodynamics exact where it matters, an affordable step where it does not |
| Equation of state | With radiation on, the EOS's own radiation term is off | `E` carries it now. Otherwise it is counted twice |
| Opacity | A new shader slot `<opacity>`: `kappa_planck(rho, T)`, `kappa_rosseland(rho, T)` | None exists today. Chosen at assembly like cooling |
| When it is off | `rad_off.wgsl` stubs; no texture, no pass | Runs without radiation must not change by a bit |

## 3. The physics

### 3.1 Equations

With `κ_P` the absorption opacity, `κ_R` the total (absorption and scattering), `P = D(f) E`
the pressure tensor and `r` the reduction factor of section 3.4:

```
∂E/∂t + ∇·(v E) + ∇·(v·P)  + r ∇·F      =  r c κ_P ρ (a T⁴ − E)  −  v·g
∂F/∂t + ∇·(v F)            + r c² ∇·P   = −r c κ_R ρ F

gas momentum:        ∂(ρv)/∂t = … + g,        g = κ_R ρ F / c
gas internal energy: ∂(ρe)/∂t = … − c κ_P ρ (a T⁴ − E)
gas total energy:    ∂E_gas/∂t = … − c κ_P ρ (a T⁴ − E) + v·g
```

- The first two terms of each radiation equation are the hydro step's (section 4.2). The rest
  is the sweep's.
- `∇·(v·P) + v·g` is the compression work `P:∇v` written so that what the radiation loses is
  exactly what the gas's kinetic energy gains. Where `r = 1`, `E_gas + E` is conserved to
  round-off, as mass and gas energy are today.
- In steady state `r` drops out of everything: the reduction slows how fast the radiation
  field settles, not what it settles to.
- The gas always sees the true rates. Where `r < 1` the conserved quantity is
  `E_gas + E/r`, as in every reduced-speed scheme.

### 3.2 Closure

Levermore's: with `f = |F|/(cE)` and `n = F/|F|`,

```
χ = (3 + 4f²) / (5 + 2√(4 − 3f²)),     D = (1−χ)/2 · I + (3χ−1)/2 · n n
```

`f` is clamped to 1 for the closure only. A face whose reconstructed state has `f > 1` or
`E ≤ 0` drops to first order, the positivity fallback the hydro has.

### 3.3 The local implicit exchange

Per cell, per sweep, after the explicit transport has made `E*`, `F*`, with
`θ_F = r c κ_R ρ Δt` and `θ = r c κ_P ρ Δt`:

1. **Flux.** `F' = F* / (1 + θ_F)`. In the gas's frame the equilibrium flux is zero, so there
   are no velocity terms and nothing to iterate. The gas gains
   `Δ(ρv) = (F* − F') / (r c²)`.
2. **Temperature.** Backward Euler for the pair `(ρe, E)` eliminates `E'` and leaves

   ```
   ρe(T') + A a T'⁴ = ρe + A E*,        A = θ / (r (1 + θ))
   ```

   For an ideal gas this is the quartic `eos/quartic.wgsl` already solves for the
   gas-and-radiation EOS. For the others it is a Newton iteration on `eos_eint_from_T`, inside
   a bracket. Then `E' = (E* + θ a T'⁴) / (1 + θ)`.
3. **Work.** `E' −= v·Δ(ρv)`, with the velocity of the start of the hydro step.

Why this survives 32-bit floats: nothing subtracts `aT⁴` from `E`. As `θ → ∞` the equation
becomes "gas energy plus radiation energy is what it was", and `E'` a weighted mean of two
positive numbers. At `τ_cell = 10^10` the difference `aT⁴ − E` is far below a rounding of
either, and it is never formed. Quokka's Newton tolerances (10^-10 to 10^-13, in doubles)
are not available to us and not needed in this form.

### 3.4 The speed of light, per cell

Explicit transport with the true `c` needs `Δt < Δx/c` in transparent cells. In an opaque cell
it does not: with the source implicit and the dissipation scaled (section 4.3), what limits the
step is the diffusion `D = c/(3 κ_R ρ)`, the same limit conduction has (`numerics.md` section 3,
`s_diff = 2.5 d D/Δx`). So, with `τ_c = κ_R ρ Δx` and `τ_0 = 2.5 d / 3`:

```
r = min(1, (ĉ/c) · max(1, τ_c/τ_0))
```

- A cell with `τ_c ≥ τ_0 c/ĉ` has `r = 1`: the true speed of light, exact conservation, the
  right heat capacity. With `c/ĉ = 100` that is every cell of optical depth above about 250,
  which is all of a star but its outermost cells.
- In every reduced cell the radiation crosses the cell at about `ĉ`, by streaming or by
  diffusion. So the one condition for validity is `ĉ ≫ max(|v| + c_s)`, not
  [Skinner & Ostriker's][SO13] `ĉ ≫ v τ_max`, which no star can meet.
- `ĉ` is a constant of the run, by default 10 times the problem's largest expected speed. A
  diagnostic counts cells where `|v| + c_s > ĉ/3`.
- **This is ours, and untested.** A switch (`radLocalC = 0`) gives the usual uniform
  `r = ĉ/c`, and `ĉ = c` gives no reduction at all. Section 9's `rad-equilibrium`,
  `rad-diffusion` and `rad-cooling-sphere` tests are run both ways to show the difference.
- `r` changes as a cell's density and temperature do, so `Σ E/r` is not exactly constant in
  reduced cells. Those cells hold little of the energy; the diagnostic in section 7 says how
  much.

### 3.5 Units

`c` in code units is `SI.c` over the code velocity (`src/units/`), and `a` is the `arad` that
`eosConstants()` already makes. A problem without physical units gives `radC` and `arad`
itself. For the white dwarf presets `c` is 30 to 300 code velocities; for a sun-like star near
a 10^6 M_sun black hole, a few hundred.

## 4. The passes

### 4.1 A step on the uniform grid

```
rates, clock        the step, with a radiation term (4.5), and N_r
predict, update     as now; the update stores face mass fluxes (MF) as it does for species
species
rad_advect          R: E/ρ and F/ρ moved with MF, plus the flux v·P            (4.2)
gravity, masses, correct
rad_sweep × N_r     R and G: transport, then the local exchange                (4.3)
rad_commit          U1, U2 into the other parity: add G, synchronize energies  (4.4)
burn
```

### 4.2 `rad_advect`

- Second order, limited, upwind on the sign of the stored mass flux, as
  `hydro/species.wgsl`. To check first: that pass's assumptions about fractions (sums to one,
  clipping to [0, 1]) must not leak in; if they do, `rad_advect` is a copy without them.
- The extra energy flux `v_face · P_face · n` uses arithmetic means of the two cells, so both
  cells compute the same bits.
- A cell marked fixed (`U2.w = −1`) keeps its `R`. That gives the tests an inflow boundary.

### 4.3 `rad_sweep`

- Reads `R`, `G` and `U1` (for `ρ` and `v`, constant through the sweeps); writes `R` and `G`
  of the other parity. Two storage textures; the limit of four is not near.
- Each cell computes its `2d` face fluxes itself, as the hydro update does. Face flux, with
  `S_L`, `S_R` the frozen-Eddington wave speeds and `ε = min(1, 1/τ_face)`:

  ```
  F_E = [S_R F_L − S_L F_R + ε S_R S_L (E_R − E_L)] / (S_R − S_L)
  ```

  and likewise for the flux of `F`, which is `c² P`. `τ_face` uses the arithmetic mean
  opacity of the two cells. A GLF option (`S = ±c`) is kept for comparison:
  [RAMSES-RT][RT15] prefers it for isotropy around sources.
- The cell multiplies its flux divergence by its own `r`. The face flux itself is the same
  bits for both cells.
- Boundaries through `bc_map`: periodic, zero-gradient (radiation leaves), reflecting (the
  normal flux changes sign).
- No arrays indexed at run time: the face loops are written out with constant indices, as the
  rule in `gpu/reduce.ts` requires.
- First sweep of a step reads `ρe` from `U2`; later ones from `G`. One uniform flag.

### 4.4 `rad_commit`

Adds `G`'s momentum to `U1`, sets `ρe` from `G`, adds the same energy and the kinetic energy
gained to `E_gas`, then runs `dual_energy` as the update does. Floors and the EOS's least
internal energy apply. With `EOS_KEEP_TOTAL` (degenerate gas) the rule of `numerics.md`
section 7 holds: `E_gas` is kept and only the pressure comes from `ρe`.

### 4.5 The step

- The sweep's own step is `Δt_r = C Δx / ĉ`, the same in every cell by construction (3.4).
  `N_r = ceil(Δt / Δt_r)`, and the hydro step is shortened so that `N_r ≤ 16`.
- The hydro signal speed gains the radiation's push in opaque gas:
  `c_s² → c_s² + (4/9) min(1, τ_c) E/ρ`. The radiation force is a source term, so the Riemann
  solver does not know of it; the step must.
- Expected cost: with `ĉ` ten times the fastest gas, about 12 sweeps a step. Quokka's ratio
  of hydro to radiation-hydro speed on one GPU is about 6. To measure at the end of stage 2.

### 4.6 Memory

`R` and `G`, two parities each: 64 bytes a cell on top of the uniform grid's 112, and on top
of a block's 152 per cell. If that is too much, `G`'s second parity can go by making the commit
part of the last sweep; decide after measuring.

## 5. Adaptive mesh

Single timestep first.

- `R` and `G` are atlas textures like the others. Ghost cells are computed on load in an
  assembly pass of their own (`amr/rad_assemble.wgsl`): same level, boundary image, the mean
  of finer cells, or limited linear from coarser. `E` and `F` are conserved by restriction and
  prolongation as the gas is.
- **Refinement boundaries.** As the gas does (`amr/update.wgsl`, `match_faces`): the coarse
  cell recomputes the fine fluxes that cover its face, from the fine states, with the fine
  cells' arithmetic. Every block is at the same time in a sweep, so this needs nothing new,
  and no flux register.
- `τ_c`, and so `r`, depend on the cell size. A coarse cell and the fine cells beside it may
  have different `r`; each multiplies the shared face flux by its own. Conservation of `E/r`
  across the face is then approximate where either is reduced, and exact where both are 1.
- Regridding: `amr/rad_regrid.wgsl`, after `species_regrid.wgsl`.
- Refinement criteria: Löhner's indicator on `E` as well as density and pressure, off by
  default; the photosphere usually has a density gradient that refines it anyway.

## 6. Per-block timesteps

Stage 5; refused with radiation until then. The proposal, to be checked against
`encodeCycle()` when that stage starts:

- **The sweeps are not binned.** `ĉ` is the same everywhere, so a block's radiation step
  depends only on its level. All leaves sweep together after each smallest substep of a cycle,
  `N_r` times. Ghost cells and refinement boundaries stay as in section 5, because every
  block's radiation is at the same time. No flux register.
- **The gas of a block in mid-step is not touched.** `G` accumulates what the sweeps did to it
  since its last hydro update; `amr/commit.wgsl` folds `G` in when the block finishes its
  step. The sweeps take the block's temperature from `G`, which is current.
- **Advection is binned**, because it uses the hydro fluxes: `rad_advect` runs for the blocks
  that step, with the half-and-half sharing across a bin boundary that species use. Its result
  is a change to `R` that must be added to an `R` the sweeps have moved on since. To decide:
  apply it at the block's update rather than its commit (first order in the block's step,
  like cooling), or carry the change in a texture of its own.
- A cheaper fallback if this proves awkward: with radiation on, a block's bin is set by its
  level alone.

## 7. Interface, diagnostics, files

- **Settings** (`SimConfig`, `problems/types.ts`): `radiation` (off, m1), `opacity` and
  `opacityParams`, `radChat`, `radLocalC`, `radSolver` (hll, glf). Feature bit in `P.flags`;
  constants in a new `P.rad` vec4 (`c`, `ĉ`, `a`, flags), so `PARAM_BYTES` and `struct Params`
  grow together.
- **Opacity modules** (`src/opacity/`): constant; electron scattering `0.2 (1 + X)` cm²/g from
  the problem's composition; Kramers plus electron scattering. Each with a float64 mirror.
- **Initial states.** `Prim` gains `erad` and `frad`. Default: `E = aT⁴` at the gas
  temperature, and `F` at its diffusion value from the discrete gradient of `E`, so that a
  star built in balance with the gas-and-radiation EOS starts in balance (`numerics.md`
  section 16). Without that the first step has no radiation force.
- **Side panel**: Radiation section with the on/off, opacity law, `ĉ` in units of the fastest
  speed. Links: `rad`, `kap`, `chat`. Anything with a control id is scriptable through
  `control()` and the MCP tools already.
- **View fields** (`FIELDS`, `render/field_value.wgsl`, an extra binding as the magnetic
  fields have): radiation energy, radiation temperature `(E/a)^(1/4)`, `|F|`, reduced flux
  `f`, cell optical depth, `r`. Hidden when radiation is off.
- **Diagnostics**, in a reducer of their own as MHD has (`MAG_OPS`): total `E`, total `E/r`,
  the share of `E` in reduced cells, the energy that left through the box faces (a
  bolometric light curve), cells with `f` clamped, cells with `|v| + c_s > ĉ/3`.
- **Files**: `.athdf` gains `Er`, `Fr1`, `Fr2`, `Fr3`; `primitivesFrom` and the restart path
  read them. A snapshot must restart a run exactly, as it does with a field.
- **Docs**: a section of `numerics.md`; the "no radiation transport" lines in `README.md`,
  `architecture.md`, `teaching.md` and the paper.

## 8. Order of work

1. **Groundwork.** `c` in code units; the opacity slot and modules; the EOS switch that drops
   its radiation term; a float64 reference of the whole scheme in 1D
   (`src/reference/radiation.ts`); unit tests.
2. **Static gas, uniform grid.** `R`, `G`, `rad_sweep` with a uniform `r`, `rad_commit`, the
   view fields and the diagnostics. Tests: equilibrium, streaming, shadow, Marshak, diffusion.
3. **Moving gas.** `rad_advect`, the work terms, the force, the step. Tests: advected pulse,
   radiative shocks, a star at rest and in motion. Timing.
4. **The speed of light per cell** (3.4). The tests of stages 2 and 3 again, with the switch
   both ways. If it fails them, the plan falls back to a uniform `ĉ` with its stated limits,
   and section 10's trapped-pressure item moves up.
5. **Adaptive mesh, single timestep** (section 5).
6. **Per-block timesteps** (section 6).
7. **Files, links, panel, docs**, and presets: a hot sphere cooling into vacuum, the
   radiation-dominated blast with transport, a star's disruption with a light curve.

Stages 1 and 2 touch no existing shader but the EOS switch.

## 9. Tests

Unit (vitest, float64 mirrors):

- The closure: `χ(0) = 1/3`, `χ(1) = 1`, monotonic; wave speeds bounded by `c`.
- The exchange: the scalar equation's root against bisection for each EOS; at
  `θ = 10^20` it returns the equilibrium of the pair to a rounding.
- Opacity laws; `c` and `a` in code units for each preset with units.

GPU (`src/validation/tests.ts`), uniform grid and blocks unless said:

| Test | What it shows |
|---|---|
| `rad-equilibrium` | Gas and radiation in a closed box relax to the analytic temperature. With `r = 1` total energy to round-off; with a uniform `ĉ = 0.1c` the known error; with the per-cell rule and an opaque box, none |
| `rad-f32` | The same at `τ_c = 10^10`, started in equilibrium: no drift over 10^4 sweeps |
| `rad-streaming` | A pulse in vacuum moves at `ĉ` and keeps `f = 1`; a beam across the grid |
| `rad-shadow` | An opaque clump in a beam casts a shadow (2D) |
| `rad-marshak` | The Su & Olson non-equilibrium Marshak wave against its tabulated solution |
| `rad-diffusion` | A Gaussian pulse diffuses at `c/(3κρ)` with `τ_c` from 1 to 10^4: the diffusion limit without resolving the mean free path. Also across a refinement boundary |
| `rad-advect` | A pulse of trapped radiation in gas moving at `β = 10^-2` with `βτ = 10^4` stays with the gas and keeps its width, at `ĉ = c/100`. The float64 reference run in the lab frame with the same `ĉ` shows the failure this design avoids: the pulse lags at `(ĉ/c) v` |
| `rad-shock` | Subcritical and supercritical radiative shocks against the semi-analytic profiles of Lowrie & Edwards (2008) |
| `rad-star` | An `n = 3` polytrope with a tenth of its pressure in radiation stays at rest: largest Mach number, energy drift. The same star crossing the box at ten times its sound speed |
| `rad-cooling-sphere` | The luminosity of a hot opaque sphere is the same at `ĉ` and `2ĉ` |
| `rad-flux-match` | Fine and coarse radiation fluxes agree to 10^-6, as the hydro's test does |
| `rad-off` | With radiation off, a run's bits are what they were |

Timing: step time against `ĉ` and against cell count; the sweep's share.

## 10. Risks and open questions

- **The per-cell speed of light is unproven.** It is the reason this variant fits stars, and
  it is ours. Stage 4 is where it is settled; nothing before it depends on it.
- **Galilean invariance of the work terms.** `∇·(v·P)` and `v·g` are large and nearly cancel
  in a star that moves fast, and they are applied in different passes. `rad-star` in motion
  measures it. If it is poor: use `P:∇v` from velocity differences instead, and give up
  round-off conservation of the total energy for it.
- **Radiation pressure as a source term.** The well-balanced reconstruction (`hydro/q.wgsl`)
  does not see it, so a star held up largely by radiation will be noisier than one held up by
  gas. If `rad-star` shows it: put the pressure of radiation in opaque cells into the Riemann
  flux as a non-thermal pressure, as [RAMSES-RT][RT15] does for trapped photons.
- **First order in time** for the sweep (one stage, forward transport and backward source).
  [He et al.'s][He24] two-stage scheme is the upgrade; it doubles the sweeps.
- **Odd-even decoupling** in dynamic diffusion at low resolution, which [He et al.][He24]
  report. Watch for it in `rad-advect`.
- **The photosphere** is where `τ_c` crosses `τ_0 c/ĉ` and `r` changes fastest, and where the
  light curve is made. `rad-cooling-sphere` is the check.
- **Shining point masses** need room in the `Mass` record, or a luminosity on `SourceSpec`
  (which can already ride a mass). After v1.
- **Compile time and bindings.** The sweep is its own pipeline, but the hydro update gains
  the stored mass fluxes whenever radiation is on. Check the pipeline keys rebuild only what
  changed.
- **The share of a step** the sweeps take grows with `ĉ`. If the debris needs `ĉ` near `c`,
  this scheme is the wrong one for that problem, and implicit transport is the alternative.

[He24]: https://arxiv.org/abs/2404.08247
[WK22]: https://arxiv.org/abs/2110.01792
[RT15]: https://arxiv.org/abs/1411.6440
[SO13]: https://arxiv.org/abs/1306.0010
[Fornax]: https://arxiv.org/abs/1806.07390
