# Plan: special-relativistic hydrodynamics

Status (2026-11-20): proposed, nothing built. Formulas, published test values and citations
below are from memory; step 1 of the order of work checks each against its paper before any
shader is written.

Goal: gas at any speed below light on a flat metric, in the existing MUSCL-Hancock scheme,
tested against the relativistic shock tubes and blast waves that have exact solutions.

| Stage | What it adds | Grids |
|---|---|---|
| 0 | One door from the stored state to the primitives; no new physics | uniform and blocks |
| 1 | Float64 reference: exact Riemann solver, recovery, the whole step | none (CPU) |
| 2 | The scheme on the GPU: recovery, HLLE, timestep, tests | uniform, 2D and 3D |
| 3 | Cold fast flow: the entropy fallback (dual energy's counterpart) | uniform |
| 4 | Adaptive mesh | blocks |
| 5 | HLLC, display fields, presets, `numerics.md` | both |

**Contents**

1. [What was asked, and what this plan changes](#1-what-was-asked-and-what-this-plan-changes)
2. [Equations and variables](#2-equations-and-variables)
3. [Single precision](#3-single-precision)
4. [Stage 0: one door to the primitives](#4-stage-0-one-door-to-the-primitives)
5. [Stage 1: float64 reference](#5-stage-1-float64-reference)
6. [Stage 2: the scheme on the uniform grid](#6-stage-2-the-scheme-on-the-uniform-grid)
7. [Stage 3: cold fast flow](#7-stage-3-cold-fast-flow)
8. [Stage 4: adaptive mesh](#8-stage-4-adaptive-mesh)
9. [Stage 5: HLLC, display, presets, docs](#9-stage-5-hllc-display-presets-docs)
10. [Validation tests](#10-validation-tests)
11. [Not covered](#11-not-covered)
12. [Open decisions](#12-open-decisions)
13. [Order of work](#13-order-of-work)
14. [References](#14-references)

---

## 1. What was asked, and what this plan changes

The request: special-relativistic hydro on a flat metric, with a primitive recovery and HLLE
fluxes, tested against relativistic Sod and blast waves. The plan keeps all of that. Six
things follow from how this code is built, and are not in the request:

1. **The recovery runs once per cell per pass, and its result is stored.** Every pass here
   converts the stored state to primitives on load (`load_q` calls `prim_from_cons`, 13 cells
   per update). An iterative recovery in that position would run 13 times per cell. The code
   already stores a synchronized internal energy beside $`E`$ (`U2.y`, `hydro/dual_energy.wgsl`).
   In a relativistic run that slot holds the recovered proper internal energy density, and with
   it the primitives of an ideal gas follow in closed form (2.3). The iteration runs only where
   a new state is made: the update, the predictor, and the passes that write a state.
2. **The reconstructed velocity is $`\mathbf{u} = W\mathbf{v}`$, not $`\mathbf{v}`$.** Limited
   slopes of $`\mathbf{v}`$ can give a face state faster than light in 2D and 3D, and
   $`1 - v^2`$ has no bits left in `f32` above a Lorentz factor of a few hundred. $`W =
   \sqrt{1 + u^2}`$ is well conditioned at any speed.
3. **The energy stored is $`\tau = E - D`$**, without the rest mass, so that slow gas keeps its
   pressure in `f32`.
4. **Ideal gas only, at first.** The closed form of item 1 needs $`p`$ from the stored energy
   without the density. Other equations of state need the pressure stored instead (12).
5. **A float64 reference comes before any shader**, as for implicit pressure
   (`plans/incompressible.md` 4.2). It answers the two questions the design turns on: which
   predictor (6.3), and where single precision gives out (3).
6. **HLLE first, HLLC after.** HLLE is what was asked and is the simplest flux to get right.
   The code's default is HLLC for a reason (contacts and shear), and the second blast wave's
   thin shell is the standard example of what HLLE smears. Stage 5 adds the HLLC of
   [Mignone & Bodo (2005)][MignoneBodo2005].

"The new primitive recovery" is taken to mean a recovery written for this mode. The one
proposed is the bracketed one-dimensional root find of [Galeazzi et al. (2013)][Galeazzi2013]
(its magnetized successor is [Kastaun et al. (2021)][Kastaun2021]): the root is bracketed
before the first iteration, so a fixed number of iterations always lands on it, which is what
a shader needs.

## 2. Equations and variables

Units with $`c = 1`$ (12, decision 1).

### 2.1 Conserved state and fluxes

With $`W = (1 - v^2)^{-1/2}`$, $`h = 1 + \varepsilon + p/\rho`$ the specific enthalpy and
$`\varepsilon`$ the specific internal energy:

```math
D = \rho W, \qquad \mathbf{S} = \rho h W^2\,\mathbf{v}, \qquad \tau = \rho h W^2 - p - D,
```

```math
\partial_t D + \nabla\cdot(D\mathbf{v}) = 0, \qquad
\partial_t \mathbf{S} + \nabla\cdot(\mathbf{S}\mathbf{v} + p\,\mathsf{I}) = 0, \qquad
\partial_t \tau + \nabla\cdot(\mathbf{S} - D\mathbf{v}) = 0 .
```

These go where the Newtonian state is now: `U1 = (D, S)`, `U2 = (tau, e, D s, solid flag)`,
with $`e = \rho\varepsilon`$ the proper internal energy density. The passive scalar and the
species ride on the flux of $`D`$ exactly as they ride on the mass flux today.

### 2.2 Primitives

`Q.a = (rho, u_x, u_y, u_z)`, `Q.b` as now. The limiters, the positivity check, the shock flag
(the sign of the velocity difference is that of $`\mathbf{u}`$'s) and the mirror images at
walls take $`\mathbf{u}`$ without change.

### 2.3 From the stored state, in closed form

For an ideal gas, $`p = (\Gamma - 1)\,e`$, and then

```math
Q = \tau + D + p, \qquad \mathbf{u} = \frac{\mathbf{S}}{\sqrt{(Q - |\mathbf{S}|)(Q + |\mathbf{S}|)}}, \qquad
W = \sqrt{1 + u^2}, \qquad \rho = D/W .
```

No iteration and no equation-of-state call beyond the one multiplication. $`\Gamma \le 2`$ is
required (the sound speed stays below light); the setup check refuses more.

### 2.4 Recovery

Given $`(D, \mathbf{S}, \tau)`$, with $`q = \tau/D`$, $`r = |\mathbf{S}|/D`$ and
$`k = r/(1 + q)`$, the unknown is $`z = Wv`$:

```math
W = \sqrt{1 + z^2}, \quad \rho = \frac{D}{W}, \quad
\varepsilon = W q - z r + \frac{z^2}{1 + W}, \quad
h = 1 + \varepsilon + \frac{p(\rho, \varepsilon)}{\rho}, \quad f(z) = z - \frac{r}{h} = 0,
```

with the root in $`\left[\tfrac{k/2}{\sqrt{1 - k^2/4}},\ \tfrac{k}{\sqrt{1 - k^2}}\right]`$.
The form of $`\varepsilon`$ has no rest-mass cancellation, so it reduces to
$`E/\rho - \tfrac12 v^2`$ as the speed falls.

- Before the solve: $`D`$ floored, $`\tau`$ floored, and $`\mathbf{S}`$ scaled down where
  $`k`$ exceeds the value of a speed limit (a Lorentz factor of 100 to start; 3). Each counts
  as a floored cell in the existing diagnostic.
- The solve: a fixed number of safeguarded steps (Newton inside the bracket, bisection when it
  leaves). The count is set in stage 1 with the state rounded to single precision.
- After: the pressure floor, as `prim_from_cons` applies it now.

One module, `hydro/sr.wgsl`, holds 2.1 to 2.4 and the wave speeds, with a float64 mirror in
`src/reference/sr.ts`, the way each equation of state has its mirror.

### 2.5 Wave speeds

With $`c_s^2 = \Gamma p/(\rho h)`$ and $`\sigma = c_s^2/[W^2(1 - c_s^2)]`$, along the face
normal:

```math
\lambda_\pm = \frac{v_n \pm \sqrt{\sigma\,(1 + \sigma - v_n^2)}}{1 + \sigma} .
```

This form takes $`W`$ from $`\mathbf{u}`$ and never forms $`1 - v^2`$. The reference checks it
against the textbook form in $`v`$.

## 3. Single precision

Estimates, to be replaced by measurements from stage 1 (`Math.fround` on the state).

**Speed.** $`\mathbf{u}`$ holds any Lorentz factor. The closed form of 2.3 needs
$`Q - |\mathbf{S}| \approx Q/2W^2`$ resolved against the rounding of $`Q`$, $`6\times10^{-8}`$:
1% in $`W`$ at $`W \approx 300`$, nothing left near 3000. The speed limit of 2.4 starts at 100.

**Heat.** The recovery takes $`\varepsilon`$ as the small difference of $`Wq`$ and $`zr`$. An
error of $`\delta`$ in the stored state becomes about $`W^2\delta`$ in $`\varepsilon`$:

| $`W`$ | least $`\varepsilon`$ recovered to 1% |
|---|---|
| slow gas, $`v = 0.01`$ | about $`10^{-9}`$ ($`10^{-5}`$ of the kinetic energy, as in the Newtonian scheme) |
| 2 | about $`10^{-4}`$ |
| 10 | about $`10^{-3}`$ |
| 50 | about $`10^{-2}`$ |

The two test problems sit well inside this (their fast gas is shocked and hot). A cold jet at
$`W = 10`$ does not, which is what stage 3 is for.

## 4. Stage 0: one door to the primitives

Eighteen shader files turn the stored state into a velocity on their own (`u.yzw / rho`):
`hydro/update.wgsl` (`load_vel`), `hydro/diagnostics.wgsl`, `hydro/damping.wgsl`,
`hydro/species_predict.wgsl`, `render/field_value.wgsl`, `render/field.wgsl`,
`render/bsample.wgsl`, `amr/cons.wgsl`, `amr/diagnostics.wgsl`, `amr/bsample.wgsl`,
`debris/debris.wgsl`, `radiation/advect.wgsl`, `pressure/source_gas.wgsl`, and the MHD files.
The other direction is made in `amr/init_cpu.ts` (line 294), `edit/brush_core.wgsl` and
`amr/cons.wgsl`.

1. Add to `hydro/common.wgsl` the pair `state_velocity(u1, u2)` and
   `state_from_prim(rho, v, p, ...)`, and a slot `<relativity>` that defaults to
   `const SR = false;` in `gpu/shaders.ts` (`DEFAULT_SLOTS`), as `geometry` does.
2. Route the hydro, render, diagnostics and block-grid readers through them. The MHD,
   radiation, pressure and debris files are left alone: those modes are refused with
   relativity (6.6).
3. With `SR = false` the compiled shaders must be what they were. Check: the whole GPU
   validation suite, and `equal to the bit` where a test already asks it (`flux-determinism`,
   the uniform-against-blocks comparison).

Done when the suite passes unchanged and a grep for `yzw /` finds only the helpers and the
refused modes.

## 5. Stage 1: float64 reference

`src/reference/` gains three files, each with unit tests in `tests/unit/sr.test.ts`.

**`riemann_exact_sr.ts`.** The exact solution for an ideal gas: [Martí & Müller (1994)][Marti1994]
without tangential velocities, [Pons et al. (2000)][Pons2000] with them. Same interface as
`riemann_exact.ts` (`(xi) => state`), plus the tangential velocities. Tests: the published
intermediate states of the two blast waves (10.1), and the Newtonian solver's answer for a
tube with all speeds below 0.01.

**`sr.ts`.** 2.1 to 2.5 in float64: prim to state, the closed form, the recovery, the wave
speeds, the HLLE flux. Tests: round trips over a grid of $`\rho`$, $`\varepsilon`$ from
$`10^{-8}`$ to $`10^{3}`$, $`W`$ from 1 to 100 and random directions; the same with
`Math.fround` on the state, which fills in the table of section 3 and fixes the iteration
count; the two forms of the wave speeds against each other.

**`RefSr` in `sr_hydro.ts`.** The step of `reference/hydro.ts` with the relativistic pieces,
on small 1D and 2D grids. It decides:

- **The predictor** (6.3): conserved-form half step with a recovery, against the primitive
  Jacobian of [Mignone et al. (2005)][MignonePlewaBodo2005] in $`(\rho, \mathbf{u}, p)`$. Judge
  on the L1 error of both blast waves, the order of the smooth wave (10.2), and whether either
  needs the first-order fallback more often.
- Whether the shock flag's HLL blend has any role (with HLLE alone, none; kept for stage 5).
- The thresholds of section 10.
- Whether the two blast waves pass with the state rounded to `f32` after each step.

Deliverable: the reference passes 10.1 and 10.2 in float64, and this file records the choices
and the measured single-precision limits.

## 6. Stage 2: the scheme on the uniform grid

### 6.1 Configuration

- `SimConfig.relativity: 'off' | 'special'`, default off. Setup key `physics.relativity`, link
  key `sr`, an entry in `setup/options.ts` (page `solver`, `apply: 'shaders'`), the schema and
  `ui/api.ts`.
- `hydroSlots` (`sim.ts`, near line 791) sets `relativity` to `const SR = true;` and forces
  the Riemann slot to `hydro/riemann/sr_hlle.wgsl`, as implicit pressure forces `advect.wgsl`.
- `relativityUnsupported(c)` joins the list at `sim.ts:576` (6.6).
- The setup check refuses a part whose speed is 1 or more, and $`\Gamma > 2`$.

### 6.2 Loads

`prim_from_cons` (`hydro/prim.wgsl`) gets an `SR` branch: the closed form of 2.3, then the
pressure floor. It returns $`\mathbf{u}`$ in `w1.yzw` and the relativistic sound speed in
`w2.y`. `load_q` on both grids and `load_vel` need nothing more.

### 6.3 Predictor

Expected choice, subject to stage 1: **the conserved form.** With the limited slopes the
predictor already makes (`edge_up`, `edge_dn`),

```math
\mathbf{U}^{n+1/2} = \mathbf{U}^n - \frac{\Delta t}{2\Delta x}\sum_d
\left[\mathbf{F}_d\left(\mathbf{W} + \tfrac12\Delta_d\mathbf{W}\right) - \mathbf{F}_d\left(\mathbf{W} - \tfrac12\Delta_d\mathbf{W}\right)\right],
```

then one recovery. It needs no Jacobian (the relativistic one couples every velocity component
to the pressure and is easy to get wrong), and it is the same step in a curved metric later.
The cost is one iterative recovery and six flux evaluations per cell. If stage 1 shows the
primitive Jacobian is as accurate, it saves that cost and is the better choice.

In `hydro/predictor.wgsl`: an `SR` branch after the slopes and the shock flag, in place of
lines 98 to 147. The positivity check after it stays, with one addition: a half-step state
whose recovery had to floor or rescale is first order for the step.

### 6.4 Flux

`hydro/riemann/sr_hlle.wgsl`, the same `riemann_flux(l, r, w_hll) -> Flux`:

- `Side.v` holds $`\mathbf{u}`$ in the face's frame; $`W`$, $`h`$ and the conserved state of
  each side are formed there;
- $`S_L = \min(\lambda_-^L, \lambda_-^R)`$, $`S_R = \max(\lambda_+^L, \lambda_+^R)`$ (2.5);
- `m` is the flux of $`D`$, `mom` of $`\mathbf{S}`$, `e` of $`\tau`$ (as $`S_n - D v_n`$ on
  each side); `s` rides on `m` upwinded, as now;
- `vn` is the face's normal velocity from the HLL state (species and the debris tally read it);
  `ei` is zero until stage 3.

`to_side` (`hydro/flux.wgsl`) is unchanged apart from `eos_face` returning the relativistic
sound speed under `SR`. The barotropic branch and the well-balanced steps are compiled out
(no gravity). Wall images work as they are: the image has $`u_n`$ reversed.

### 6.5 Update and timestep

- `hydro/advance.wgsl`: an `SR` branch. The conservative update of $`(D, \mathbf{S}, \tau)`$
  and the scalar, then the recovery of 2.4, and the recovered $`e`$ stored in `U2.y`. No
  kicks, no `dw`, no `pdv`, no `dual_energy`, no `tame_fluff` (its limits are Newtonian; the
  recovery's floors and speed limit take its place).
- `hydro/rates.wgsl`: the rate is $`\sum_d \max(|\lambda_+^d|, |\lambda_-^d|)`$, never more
  than the number of dimensions. No acceleration term.
- `hydro/diagnostics.wgsl`: mass is $`\sum D`$, momentum $`\sum\mathbf{S}`$, energy
  $`\sum\tau`$; the largest Lorentz factor goes where the largest Mach number goes.
- `DEBUG_FLUX` and the tiled path (`hydro/tile.wgsl`) need no change: they move `Flux` structs.

### 6.6 Initial state, brushes, files, and what is refused

- `amr/init_cpu.ts` and the GPU init path call `state_from_prim` (stage 0), which under `SR`
  takes the setup's $`\mathbf{v}`$ and writes $`(D, \mathbf{S}, \tau, e)`$.
- `edit/brush_core.wgsl`: the brushes rebuild a state from primitives through the same
  helper. A painted velocity is clamped below the speed limit.
- `io/athdf.ts`, checkpoints: the state is written as stored; add the mode to the file's
  attributes so that a restart with the other mode is refused.
- Refused, with a reason each, in `relativityUnsupported`: gravity of any kind (self, point
  masses, uniform field), MHD, implicit pressure, viscosity and conduction, cooling, burning,
  radiation, driving, damping zones, rotating frames, shearing boxes, comoving coordinates,
  axisymmetric and spherical grids, every equation of state but `ideal`, species that set the
  equation of state, and (until stage 4) the adaptive mesh. Passive species, the scalar,
  obstacles, mirror symmetry and every kind of box face are allowed.

### 6.7 Done when

The stage 2 tests of section 10 pass on the uniform grid in 2D and 3D, `npm test` passes, and
every existing GPU test still passes with `SR = false`.

## 7. Stage 3: cold fast flow

The counterpart of dual energy, in the slot the Newtonian scheme already uses for it.

- `Flux.ei` carries the entropy: `m` times $`\sigma = p/\rho^\Gamma`$ of the upwind side.
  The update forms $`(D\sigma)^{n+1}`$ from the cell's own $`\sigma^n`$ (from the stored $`e`$)
  and the net flux, with no source: entropy is advected in smooth flow.
- The recovery of 2.4 runs first. Where its $`\varepsilon`$ is below a fraction $`\eta`$ of
  what single precision resolves there (the measured table of section 3), and the cell is not
  flagged as a shock, the solve is repeated with the closure $`p = \sigma\rho^\Gamma`$ in
  place of $`\varepsilon(z)`$: the same iteration, a different function. $`\tau`$ is then
  rebuilt from the primitives, as $`E`$ is rebuilt today, and the energy is not conserved in
  that cell.
- Tests: a cold beam ($`\varepsilon = 10^{-6}`$, $`W = 10`$) crossing a periodic box keeps its
  pressure to 1%; the two blast waves are unchanged to rounding (no cell takes the fallback).

## 8. Stage 4: adaptive mesh

- `amr/predict.wgsl` and `amr/update.wgsl` share the predictor, the flux and `advance`, so
  they take stages 2 and 3 through their own `load_q`.
- `amr/cons.wgsl` restricts and prolongs the stored state. A mean of admissible states is
  admissible ($`|\mathbf{S}| < \tau + D`$ is a convex condition); limited linear prolongation
  can leave it by rounding. After either, the stored $`e`$ is not the recovery of the new
  state: `amr/regrid.wgsl` and the ghost fill run the recovery on every cell they write.
- Flux correction at refinement faces is in conserved fluxes and needs no change.
- `amr/flag.wgsl`: the indicators read density and pressure, which now mean the proper ones.
- Per-block timesteps: allowed only after the tests pass with them on.
- Tests: the second blast wave with three levels follows its shell; a tube across a refinement
  boundary conserves $`D`$, $`\mathbf{S}`$ and $`\tau`$ to rounding; uniform against blocks
  to $`10^{-5}`$, as the Newtonian test has it.

## 9. Stage 5: HLLC, display, presets, docs

- `hydro/riemann/sr_hllc.wgsl`: [Mignone & Bodo (2005)][MignoneBodo2005], blended toward HLLE
  by the shock flag as the Newtonian HLLC is. The Riemann menu then offers both in a
  relativistic run, HLLC the default.
- Display (`render/field_value.wgsl`, `problems/types.ts` `FIELDS`): density is the proper
  density; velocity is $`\mathbf{v}`$; new field "Lorentz factor"; the Mach number is the
  relativistic one, $`u/(W_s c_s)`$ with $`W_s`$ the sound speed's Lorentz factor. The status
  line shows the largest Lorentz factor. `docs/api.md` (probe, slice, stats) says which
  quantities change meaning.
- Presets, in a group "Relativistic": the two blast waves, the tube with tangential velocity,
  a 2D blast, a relativistic Kelvin-Helmholtz layer, and a jet into uniform gas (an `inlet`
  region at $`W = 7`$).
- `docs/numerics.md`: a section "Special relativity" (after 22) with the equations, the
  recovery, the measured limits of section 3 and the test results; `docs/setups.md` and the
  schema for `physics.relativity`; the `shaderous-sim` skill's list of options.

## 10. Validation tests

In `src/validation/sr.ts`, a group "Special relativity" on the validation page. Thresholds are
set from stage 1; the exact values below come from the exact solver, not from this file.

### 10.1 Shock tubes and blast waves (stage 2)

| id | Setup ($`\Gamma = 5/3`$, at rest, $`t = 0.4`$ unless said) | Criterion |
|---|---|---|
| `sr-sod` | $`(\rho, p) = (1, 1)`$ against $`(0.125, 0.1)`$ | L1 density error against the exact solution, along x, y and z, and equal along the three to rounding |
| `sr-blast-1` | $`(10, 13.33)`$ against $`(1, 10^{-6})`$; post-shock speed about 0.71 | L1 error; shock position within a cell; no floored cells |
| `sr-blast-2` | $`(1, 1000)`$ against $`(1, 0.01)`$; post-shock speed about 0.96, shell about 1% of the box | L1 error; shock position within a cell; peak density of the shell above a fraction set in stage 1 |
| `sr-diagonal` | `sr-blast-1` along (1, 1, 0) | L1 error within a factor of the axis-aligned run's |
| `sr-transverse` | `sr-blast-2` with tangential speeds 0, 0.9 and 0.99 on the right side ([Pons et al. 2000][Pons2000]) | L1 error; the shock's speed within 1% |

The published intermediate states (pressure 1.45 and speed 0.714 for the first blast wave,
18.6 and 0.960 for the second, shell density 10.75) are quoted from
[Martí & Müller (2003)][Marti2003] from memory, and are what the exact solver's unit test
checks first.

### 10.2 Limits, order and conservation (stage 2)

| id | Criterion |
|---|---|
| `sr-recovery` | the WGSL recovery and closed form against the float64 mirror over the grid of section 5 (a test shader like `eos/eos_test.wgsl`) |
| `sr-newtonian` | the Sod tube with pressures scaled by $`10^{-4}`$ (speeds near 0.01): density within $`10^{-3}`$ of the Newtonian run's, cell by cell |
| `sr-advection` | a density pulse in uniform pressure carried at $`v = 0.9`$ and $`0.99`$ once round a periodic box: the Newtonian `advection` test's error bound; pressure uniform to $`10^{-5}`$ |
| `sr-wave` | a smooth isentropic wave ([Zhang & MacFadyen 2006][Zhang2006]) at 64, 128, 256 cells: second order |
| `sr-conservation` | $`D`$, $`\mathbf{S}`$, $`\tau`$ to rounding over 1000 steps of a periodic flow at $`W = 5`$; both cells of every face agree to the bit (`DEBUG_FLUX`) |
| `sr-reference` | the GPU against `RefSr` cell by cell after 20 steps of a 2D blast, to $`10^{-5}`$ |
| `sr-blast-2d` | a 2D blast ($`p = 1000`$ inside radius 0.1, 0.01 outside): quadrants equal to rounding, no floored cells, the same in one layer of a 3D grid |
| `sr-wall` | `sr-blast-1` reflected off a wall and off an obstacle: no $`D`$ through the wall |

Later stages add `sr-cold-beam` (7), `sr-amr-*` (8) and `sr-hllc-contact` (a contact at rest
held to the bit, 9).

## 11. Not covered

- **Curved metrics.** The choices here are the ones a fixed metric needs next: $`\mathbf{u}`$
  as the primitive, $`\tau`$ as the energy, a conserved-form predictor, a recovery that takes
  $`|\mathbf{S}|^2`$ as an input (a 3-metric changes how it is formed, not the solve). What a
  metric adds is volume and area factors on the fluxes, geometric sources, and HLLE in a local
  orthonormal frame.
- Relativistic MHD.
- Gravity. Newtonian gravity with relativistic gas is not consistent; it waits for a metric.
- Equations of state other than the ideal gas (12, decision 2).
- Axisymmetric and spherical grids. Jets are the obvious use; it is its own stage after 5,
  with the geometric sources in $`\rho h u_\phi^2`$.
- Lorentz factors above about 100 (section 3).

## 12. Open decisions

1. **Units.** The plan fixes $`c = 1`$ in code units: a relativistic setup chooses its scales
   so. The alternative is a speed of light as a run parameter carried through `hydro/sr.wgsl`;
   it buys setups in other units and a Newtonian limit by raising $`c`$, at the price of the
   factor in every formula. Recommended: $`c = 1`$.
2. **What the slot stores.** $`e`$ gives the closed form for an ideal gas only. Storing the
   pressure instead gives it for any equation of state, but every reader of `U2.y` as an
   internal energy then needs the equation of state inverted. Recommended: $`e`$ now; revisit
   with the first request for another equation of state (the approximate Synge gas of
   [Ryu et al. 2006][Ryu2006] is the likely one).
3. **Predictor** (6.3): settled by stage 1.
4. **The speed limit** (2.4): 100 unless stage 1's rounding tests say lower.

## 13. Order of work

1. Check the formulas, published values, citations and links of this file against the papers
   (section 14's list); correct the file.
2. Stage 0, pushed alone, with the full GPU suite as its test.
3. Stage 1; record its findings here as a section "What the prototype found".
4. Stage 2: `hydro/sr.wgsl` and `sr-recovery` first, then the flux, the update, the
   predictor, the timestep; tests of 10.1 and 10.2.
5. Stages 3, 4, 5 in that order. Each ends with its tests and an "As built" section here.

`npm test` before each push (the Pages deploy runs it); GPU tests on a static build, at
below-normal priority.

## 14. References

From memory; step 1 of the order of work verifies them.

- [Del Zanna & Bucciantini (2002)][DelZanna2002], A&A 390, 1177: 2D relativistic tests.
- [Galeazzi, Kastaun, Rezzolla & Font (2013)][Galeazzi2013], Phys. Rev. D 88, 064009: the
  recovery (appendix C).
- [Kastaun, Kalinani & Ciolfi (2021)][Kastaun2021], Phys. Rev. D 103, 023018: its magnetized
  form, with error bounds.
- [Martí & Müller (1994)][Marti1994], J. Fluid Mech. 258, 317: exact Riemann solution.
- [Martí & Müller (2003)][Marti2003], Living Rev. Relativ. 6, 7: the two blast waves and
  their solutions.
- [Mignone & Bodo (2005)][MignoneBodo2005], MNRAS 364, 126: relativistic HLLC.
- [Mignone, Plewa & Bodo (2005)][MignonePlewaBodo2005], ApJS 160, 199: primitive form of the
  equations.
- [Pons, Martí & Müller (2000)][Pons2000], J. Fluid Mech. 422, 125: exact solution with
  tangential velocities.
- [Ryu, Chattopadhyay & Choi (2006)][Ryu2006], ApJS 166, 410: an equation of state between
  $`\Gamma = 5/3`$ and $`4/3`$.
- Schneider et al. (1993), J. Comput. Phys. 105, 92: HLLE for relativistic gas.
- White, Stone & Gammie (2016), ApJS 225, 22: the same pieces in a curved metric.
- [Zhang & MacFadyen (2006)][Zhang2006], ApJS 164, 255: the smooth-wave test.

[DelZanna2002]: https://doi.org/10.1051/0004-6361:20020776
[Galeazzi2013]: https://doi.org/10.1103/PhysRevD.88.064009
[Kastaun2021]: https://doi.org/10.1103/PhysRevD.103.023018
[Marti1994]: https://doi.org/10.1017/S0022112094003344
[Marti2003]: https://doi.org/10.12942/lrr-2003-7
[MignoneBodo2005]: https://doi.org/10.1111/j.1365-2966.2005.09546.x
[MignonePlewaBodo2005]: https://doi.org/10.1086/430905
[Pons2000]: https://doi.org/10.1017/S0022112000001439
[Ryu2006]: https://doi.org/10.1086/505937
[Zhang2006]: https://doi.org/10.1086/500792
