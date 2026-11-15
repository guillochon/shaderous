# Plan: nearly incompressible flow

Status: draft, 2026-11-15. Nothing here is implemented. Figures quoted from the current code
come from `docs/numerics.md` and the source comments; no new measurement has been made yet
(stage 0 makes them).

Goal: run flows at low Mach number in Shaderous, with the density nearly constant along each
streamline, in three stages:

| Stage | What it adds | Mach numbers it serves | Grids |
|---|---|---|---|
| 0 | Baseline measurements, no new physics | finds today's floor | uniform |
| 1 | Weakly compressible mode: a stiff liquid equation of state on the explicit scheme | about 0.1 (density varies by about 1%) | uniform and adaptive, 2D and 3D |
| 2 | Implicit pressure: the acoustic terms solved by multigrid, the timestep set by the flow speed | any, down to a pure projection | uniform, 2D and 3D |
| 3 | Implicit pressure on the adaptive mesh | any | block grid, 2D and 3D |

Each stage is useful alone and each ends in validation tests and a section of `numerics.md`.

**Contents**

1. [Where the current scheme stops](#1-where-the-current-scheme-stops)
2. [Stage 0: baseline measurements](#2-stage-0-baseline-measurements)
3. [Stage 1: weakly compressible mode](#3-stage-1-weakly-compressible-mode)
4. [Stage 2: implicit pressure on the uniform grid](#4-stage-2-implicit-pressure-on-the-uniform-grid)
5. [Stage 3: implicit pressure on the adaptive mesh](#5-stage-3-implicit-pressure-on-the-adaptive-mesh)
6. [Not covered](#6-not-covered)
7. [Open decisions](#7-open-decisions)
8. [References](#8-references)

---

## 1. Where the current scheme stops

Four limits, each from a different part of the code.

**Step count.** The timestep follows the sum of the flow and sound speeds
(`hydro/rates.wgsl`: `|v_x| + |v_y| + |v_z| + d c_s`). A flow at Mach number $`M`$ therefore
takes about $`1 + 1/M`$ times the steps the flow speed alone would need: 11 times at
$`M = 0.1`$, 101 times at $`M = 0.01`$.

**Damping.** A Godunov flux damps a velocity jump at a rate set by the sound speed. The
low-Mach HLLC (`hydro/riemann/hllclm.wgsl`) scales the jump by
$`z = \min[1, \max(z_\text{min}, M)]`$ with $`z_\text{min} = 0.3`$. The floor cannot go lower:
one predictor and one update with the unlimited third-order reconstruction gain kinetic
energy without bound for $`z_\text{min} \le 0.1`$. A Gresho vortex at $`M = 0.01`$ on $`64^2`$
cells loses 6% of its kinetic energy in two time units (15% with plain HLLC).

**Single precision.** Pressure fluctuations are of order $`M^2 p`$ and the state is `f32`
(relative spacing $`6\times10^{-8}`$).

| $`M`$ | $`\delta p / p`$ | bits left for the fluctuation |
|---|---|---|
| 0.1 | $`10^{-2}`$ | 17 |
| 0.01 | $`10^{-4}`$ | 10 |
| 0.001 | $`10^{-6}`$ | 4 |

The same table holds for $`\delta\rho/\rho`$ of a barotropic gas. An explicit scheme whose
pressure comes from the stored state cannot go much below $`M = 0.01`$, whatever the flux.

**Barotropic gas takes HLL.** `face_flux` sets the HLL weight to 1 for an equation of state
without an energy (`hydro/flux.wgsl`: "no contact wave to resolve"). With the `hllclm` module
selected that is HLL on velocity jumps already scaled by $`z`$, so the low-Mach fix may already
reach barotropic gas. Stage 0 checks this; nothing in the docs or tests says either way.

## 2. Stage 0: baseline measurements

No code changes except test problems. Purpose: fix the numbers every later stage is judged
against, and decide at what Mach number stage 2 pays for itself.

### 2.1 Accuracy floor

Run through the headless runner (`tools/headless/cli.mjs`, the `shaderous-sim` skill) on a
uniform 2D grid, $`64^2`$, $`128^2`$ and $`256^2`$:

| Problem | Mach numbers | Variants | Measure |
|---|---|---|---|
| Gresho vortex (`gresho()` in `validation/tests.ts`) | 0.1, 0.03, 0.01, 0.003 | ideal gas with `hllc` and `hllclm`; isothermal barotropic with `hll` and `hllclm` | kinetic energy kept after 1 and 4 turns, steps taken |
| Taylor-Green vortex, viscous, 2D | 0.1, 0.03 | same four | decay rate against $`\exp(-2\nu k^2 t)`$ |
| Kelvin-Helmholtz (`problems/kelvin_helmholtz.ts`) | as shipped, and scaled to 0.1 | `hllc`, `hllclm` | growth of the transverse kinetic energy |

The Taylor-Green problem is new; it is needed in stage 1 anyway (3.4).

### 2.2 Cost of a V-cycle against a flux update

`sim.profile()` already times the predictor, the update and one V-cycle
(`sim.ts`, `out.vcycle`). Record all three at $`128^3`$ and $`256^3`$ with self-gravity on, on
the development GPU, at normal process priority (it is a benchmark).

With $`C_h`$ the hydro cost of a step, $`C_v`$ a V-cycle and $`N_v`$ the cycles the pressure
solve needs, the implicit step wins below

```math
M^* \approx \frac{C_h}{N_v C_v + C_\text{extra}},
```

where $`C_\text{extra}`$ is the source and correction passes (about one state read and write).
If $`M^* < 0.05`$, stage 2 only pays for accuracy (the pressure no longer comes from the
state), not for speed, and that should be known before building it.

### 2.3 Deliverable

A table of these numbers in this file, and a short "what stage 1 must beat" list. No commit to
`numerics.md` yet.

## 3. Stage 1: weakly compressible mode

The standard trick of smoothed-particle and lattice codes: keep the explicit compressible
scheme, and give the fluid an artificial sound speed about ten times the largest flow speed.
The density then varies by about $`M^2 = 1\%`$ and the flow is incompressible to that
accuracy. It costs 11 times the steps of an incompressible code and nothing else.

### 3.1 Equation of state

A new barotropic module, `eos/liquid.wgsl`, kind `'liquid'`:

```math
p = c_0^2\,(\rho - \rho_0), \qquad c_s = c_0 .
```

- Parameters: reference density $`\rho_0`$ and sound speed $`c_0`$, in `P.eos.x` and `P.eos.y`
  as the barotropic module uses them.
- `EOS_HAS_ENERGY = false`, so the energy equation, dual energy, cooling and conduction are
  compiled out, as for `barotropic.wgsl`.
- Why not isothermal gas with a large $`c_s`$: its pressure is $`c_s^2\rho \approx 100`$ for a
  flow of speed 1, and the gradient that drives the flow is a difference of order 1 between
  such values. The linear form stores only the fluctuation, which recovers two digits.
- The pressure can be negative. `eos_sound_speed` must return $`c_0`$, never
  $`\sqrt{\gamma p/\rho}`$.
- An optional Tait exponent ($`p = B[(\rho/\rho_0)^\gamma - 1]`$, $`\gamma = 7`$ for water) is
  not needed at 1% compression and is left out.

Files that list the kinds and need an entry: `eos/eos.ts` (`EosKind`, `EOS_MODULES`,
`EOS_LABELS`, `eosConstants1/2`, `makeEos` with its float64 mirror, `mixtureParams`),
`ui/app.ts` (the menu and its parameter fields), `sim.ts` (`conducts`, the well-balanced flag at
line 649, both of which test `c.eos !== 'barotropic'` and should test "has no energy"
instead). `ui/link.ts` takes its list from `EOS_LABELS`.

**Audit for negative or near-zero pressure.** Every place that divides by a pressure or floors
one has to be read with $`p \approx 0`$ in mind:

- the shock flag's relative jump, $`|p_{i+1} - p_{i-1}|/\min(p)`$, in `hydro/predictor.wgsl`
  (use $`\rho c_0^2`$ as the scale, or switch the flag off for this kind);
- `P.p_floor` in `hydro/prim.wgsl`, `hydro/pfloor.wgsl` and `eos_eint_floor`;
- `baro_state` (the barotropic branch of `to_side`);
- the display fields that take logs or ratios of the pressure (`render/field_value.wgsl`:
  temperature, entropy, Mach number);
- the brushes (`edit/brush_core.wgsl`), which rebuild the internal energy from the pressure.

### 3.2 Choosing the sound speed

A preset parameter, **Mach number** (default 0.1): $`c_0 = U/M`$ with $`U`$ the preset's
velocity scale. It is a restart parameter, not a live one. The status line should show the
largest $`|\rho/\rho_0 - 1|`$, which is the error of the approximation: a new diagnostics
channel in `hydro/diagnostics.wgsl`, reduced with `MAX`.

### 3.3 Flux

1. Confirm from stage 0 that `hllclm` reaches barotropic gas as HLL with scaled jumps.
2. Make it the default Riemann solver of the liquid presets.
3. If the Gresho vortex of the liquid still loses more than the ideal gas does with `hllclm`,
   the remaining damping is the density-jump term of HLL, $`S_L S_R(\rho_R - \rho_L)`$. Try
   scaling that jump by the same $`z`$. This is the term HLLC-LM scales, and it was unstable
   for the ideal gas at $`M = 0.03`$, so treat it as an experiment with its own test, not a
   planned change.

### 3.4 Walls that move, and no-slip box faces

Today only obstacle cells are no-slip, and they are at rest (`load_vel` returns zero for a
solid). The faces of the box are planes of symmetry even with viscosity. Two additions:

- **Moving solids.** A solid cell keeps its state, so its stored momentum can serve as the wall
  velocity. `load_vel` returns it instead of zero; `q_mirror` reflects the gas about the wall's
  normal velocity. `Prim.solid` cells then take `vx`, `vy`, `vz` from the preset.
  Only the tangential part matters for a wall that does not move through the grid; a wall with
  a normal velocity is out of scope.
- A lid-driven cavity is then a box lined with solid cells, the top row moving. No new
  boundary kind is needed, and the adaptive mesh gets it through the same loaders
  (`amr/update.wgsl` has its own `load_vel`).

### 3.5 Presets

| Preset | Setup | Why |
|---|---|---|
| `taylor-green` | periodic, viscous; 2D decaying vortex array, 3D breakdown | analytic decay in 2D; the standard 3D turbulence test |
| `gresho` | promoted from the test suite | inviscid, measures damping alone |
| `cylinder-wake` | inflow from `fixed` cells, an obstacle disk, outflow, a damping zone at the outlet (`Problem.damping`) to absorb sound | vortex street; Strouhal number near 0.165 at Reynolds number 100 |
| `cavity` | solid-lined box, moving lid | Ghia et al. (1982) centerline velocities at Reynolds number 1000 |
| `shear-layers` | doubly periodic double shear layer | Bell, Colella & Glaz (1989); shows spurious vortices when under-resolved |

Viscosity is explicit, so the step falls as $`\Delta x^2`$ where diffusion dominates. At
Reynolds numbers of 100 and above on these grids the sound speed still sets the step.

### 3.6 Validation tests (`src/validation/tests.ts`)

| id | Criterion (thresholds to be set from stage 0) |
|---|---|
| `liquid-eos` | unit test: WGSL module against its float64 mirror |
| `liquid-gresho` | kinetic energy kept after two time units at $`M = 0.1`$ no worse than the ideal gas with `hllclm`; largest density deviation below $`2M^2`$ |
| `liquid-taylor-green` | decay rate within 1% of $`2\nu k^2`$ on 64 cells per wavelength |
| `liquid-cavity` | centerline velocity within 2% of Ghia et al. at $`128^2`$ |
| `liquid-amr` | the Gresho vortex across a refinement boundary: mass and momentum to round-off |

`npm test` must pass before each push (the Pages deploy runs it).

### 3.7 Done when

The five tests pass on the uniform grid and the block grid, the presets are in the menu and in
links, and `numerics.md` has a section "Weakly compressible liquid" with the measured
numbers.

## 4. Stage 2: implicit pressure on the uniform grid

The method of [Kwatra et al. (2009)][Kwatra2009]: split the flux into advection and pressure,
take the advection explicitly and the pressure implicitly. The timestep then follows the flow
speed. Shocks stay conservative, the equation of state stays general, and as the sound speed
grows the scheme becomes a projection method.

### 4.1 The scheme

One step from $`t^n`$ to $`t^{n+1}`$, for the conserved state
$`(\rho, \rho\mathbf{v}, E, \rho e)`$:

1. **Advect.** The usual predictor and update, with the pressure taken out of the fluxes:

   ```math
   \mathbf{F}^\text{adv} = \left(\rho v_n,\ \rho\mathbf{v}v_n,\ E v_n\right).
   ```

   This gives $`\rho^{n+1}`$ (final) and $`(\rho\mathbf{v})^*`$, $`E^*`$. Gravity's half kicks
   act on $`(\rho\mathbf{v})^*`$ as now.
2. **Advected pressure.** $`p^a = p^n - \Delta t\,\mathbf{v}^n\cdot\nabla p^n`$, upwinded.
3. **Solve** for the new pressure:

   ```math
   \frac{p^{n+1}}{\rho c^2\,\Delta t^2} - \nabla\cdot\left(\frac{1}{\rho^{n+1}}\nabla p^{n+1}\right)
   = \frac{p^a}{\rho c^2\,\Delta t^2} - \frac{\nabla\cdot\mathbf{v}^*}{\Delta t},
   ```

   with $`\rho c^2`$ from the equation of state at $`t^n`$. The left side is symmetric and
   diagonally dominant. As $`c \to \infty`$ the first terms vanish and this is the pressure
   Poisson equation of a projection.
4. **Correct.** With face pressures $`p_f`$ and face velocities

   ```math
   \hat v_f = v^*_f - \Delta t\,\frac{(p_R - p_L)}{\rho_f\,\Delta x},
   ```

   apply $`-\Delta t\,\nabla p`$ to the momentum from the $`p_f`$, $`-\Delta t\,\nabla\cdot(p\hat{\mathbf{v}})`$
   to $`E`$, and $`-\Delta t\,p\,\nabla\cdot\hat{\mathbf{v}}`$ to $`\rho e`$. All three are
   differences of face values, so momentum and energy stay conserved to round-off.

Face quantities: $`\rho_f = \tfrac12(\rho_L + \rho_R)`$,
$`v^*_f = (\rho_L v^*_L + \rho_R v^*_R)/(\rho_L + \rho_R)`$, and
$`p_f = (\rho_R p_L + \rho_L p_R)/(\rho_L + \rho_R)`$, as in Kwatra et al. Each is a symmetric
function of the two cells, so both cells of a face form the same bits, which the gather form
requires.

### 4.2 Prototype in float64 first

`src/reference/hydro.ts` is the whole hydro scheme in double precision. Add the split step
there before writing a shader, with a dense or conjugate-gradient solve on small grids. It
answers, cheaply, the questions the design depends on:

- Does the MUSCL-Hancock predictor stay stable with the advective Jacobian alone, and is the
  low-Mach floor $`z_\text{min}`$ still needed once the acoustic terms are implicit?
- Which advective flux (4.3)?
- $`p^a`$ from the pressure equation (above), or from the advected internal energy that the
  code already carries (`net.ei`)? The second costs nothing but folds part of the compression
  work into $`p^a`$ for a general equation of state.
- How tightly must the solve converge for the Gresho vortex and for Sod?
- Rerun with `Math.fround` on the state to see what single precision does at
  $`M = 10^{-2}`$ and $`10^{-3}`$ (4.8).

Deliverable: the reference passes Sod (shock position and plateau to the present tolerance)
and holds a Gresho vortex at $`M = 10^{-3}`$, with the choices above recorded here.

### 4.3 Advective flux and predictor

- **Flux.** A new Riemann-slot module, `hydro/riemann/advect.wgsl`: the flux of the
  pressureless system, upwinded on the face velocity (its only wave speed), with a
  local Lax-Friedrichs term of speed $`|v_n|`$. It fills the same `Flux` struct; `out.vn` is
  the face velocity, and the momentum flux carries no pressure. Viscosity and conduction are
  added to the faces as now (`add_diffusion`).
- **Predictor.** `hydro/predictor.wgsl` advances the primitives half a step with the full
  Jacobian. A compile-time constant `PSPLIT` drops the pressure-gradient term of the velocity
  row and the $`\rho c^2\nabla\cdot\mathbf{v}`$ term of the pressure row, which would be
  unstable at a step many times the acoustic limit. Whether to put $`\nabla p^n`$ back
  explicitly for second order in time is a question for 4.2.
- **Wall images.** The Riemann problem between a state and its mirror image has no advective
  flux at all; the wall's pressure force now comes from the correction pass (4.5).
- **Shocks.** Kwatra et al. show the split scheme captures shocks with the advective
  timestep. The carbuncle fix (HLL beside flagged shocks) has no meaning for this flux and is
  compiled out with it.

### 4.4 Timestep

`hydro/rates.wgsl` with `PSPLIT`: the rate is $`|v_x| + |v_y| + |v_z|`$ plus the diffusion
speed, without the sound speed. Kwatra et al. add a limit for the acceleration the pressure
gives within a step; the code already has that form for gravity,
$`C\sqrt{\Delta x/\max|\mathbf{g}|}`$, and the second reduction channel can take
$`|\mathbf{g} - \nabla p/\rho|`$ in place of $`|\mathbf{g}|`$.

A flow at rest has no advective limit at all, so the mode needs a ceiling: a multiple of the
acoustic step (a setting, default 20 or so, to be fixed in 4.2), below `tEnd` and the output
times as now.

### 4.5 The step on the GPU

Today the update writes parity $`1-p`$ and gravity's second half kick
(`gravity/correct.wgsl`) reads it and writes back to parity $`p`$. The pressure correction is
the same kind of pass, so the two merge:

| Pass | Shader | Change |
|---|---|---|
| Rates, clock | `hydro/rates.wgsl`, `hydro/clock.wgsl` | no sound speed; pressure-acceleration limit |
| Predictor | `hydro/predictor.wgsl` | `PSPLIT` |
| Update | `hydro/update.wgsl` | advective flux; writes $`U^*`$ to parity $`1-p`$ |
| Species | unchanged | they ride on the recorded mass fluxes, which the pressure step does not change |
| Gravity solve | unchanged | needs only $`\rho^{n+1}`$, which is final |
| Pressure source | `pressure/source.wgsl` (new) | reads $`U^n`$ and $`U^*`$; writes the right-hand side and the coefficient texture |
| Pressure solve | `pressure/*.wgsl` (new) | V-cycles until converged (4.6) |
| Correct | `pressure/correct.wgsl` (new), absorbing `gravity/correct.wgsl` | second gravity half kick, then the pressure terms; writes parity $`p`$ |

The correction is a gather over nearest neighbors: each cell forms $`p_f`$ and $`\hat v_f`$
on its own faces from $`p^{n+1}`$, $`\rho^{n+1}`$ and $`\mathbf{v}^*`$ of the two cells. The
tiled update's lesson applies (`architecture.md`, Performance notes): two call sites for one
face may differ in the last bit under Metal's relaxed math, so compute each face with one
piece of code, or extend the flux-determinism test to these faces and see.

The second gravity kick must be in $`\mathbf{v}^*`$ before the divergence is taken, so the
pressure source pass applies it on the fly when it reads the velocity. With point masses the
order is: drift, mass kick, gravity field, pressure source, solve, correct.

MHD is refused in this mode at first (`mhdUnsupported` is the place); its two-stage step would
need the split in both stages.

### 4.6 The solver

A second multigrid class, `src/pressure/multigrid.ts`, written beside `gravity/multigrid.ts`
and sharing its restriction, prolongation and reduction code where the shaders allow. It
differs in five ways.

**Variable coefficients.** Per level, one `rgba32float` texture: the three upper-face
coefficients $`\beta = 1/\rho_f`$ and the diagonal term
$`\alpha = 1/(\rho c^2\Delta t^2)`$. The smoother reads a cell's own texel and one neighbor's
per axis. Coarse levels: a coarse face takes the mean of the fine faces that tile it (the
arithmetic mean across the face, the harmonic mean of the two layers along the normal);
$`\alpha`$ is averaged. A separate small pass writes each level's coefficients, so no texture
is read and written in one pass.

**Walls as coefficients.** A face next to a solid cell, or on a reflecting box face, gets
$`\beta = 0`$: a Neumann condition. Coarsening then carries obstacles to every level with no
mask logic. A cell with no open face and $`\alpha = 0`$ keeps its value. Box faces:

| Hydro boundary | Pressure condition |
|---|---|
| periodic | wrap |
| reflecting, solid | Neumann ($`\beta = 0`$) |
| outflow, outflow-only | Dirichlet at the face: $`p' = 0`$ (ghost $`= -p'_c`$), the form `gravity/stencil.wgsl` uses for isolated boundaries |

**Two dimensions.** The gravity multigrid is 3D only (`sim.ts` keeps a $`2^3`$ placeholder in
2D). Most low-Mach tests are 2D, so the pressure solver needs a 2D form from the start: a
5-point stencil, restriction over 4 cells, levels $`n^2 \to 2^2`$ as one-layer textures,
$`16\times16\times1`$ workgroups, red-black coloring on $`x + y`$. One source with a
compile-time `FLAT` constant, as the hydro passes do with `P.dims.z == 1`.

**Convergence.** Gravity runs a fixed two cycles and tolerates a percent. Here the residual is
the divergence left in the flow. Port the gate of the block solver (`amr/gravity/gate.wgsl`):
after each cycle a reduction takes the largest residual; later cycles are dispatched
indirectly and collapse to nothing once it is below a tolerance relative to the largest
source term. Start at $`10^{-4}`$ and tune in 4.2. A hard cap (8 cycles) keeps a frame
bounded; the count of capped steps is a diagnostic. The solve is warm-started from the last
step's $`p'`$.

**The pure-projection limit.** With $`\alpha = 0`$ and no Dirichlet face the operator is
singular. Subtract the mean of the right-hand side, as the periodic gravity solve does with
the mean density. With any $`\alpha > 0`$ this is not needed.

Memory: 16 bytes per cell for the coefficients and 12 for $`p'`$, the right-hand side and the
residual, times 8/7 for the levels: about 540 MB at $`256^3`$, half as much again as the
state's four textures (64 bytes per cell, 1.07 GB). The budget in `gpu/memory.ts` has to count
it, and a run with self-gravity holds both solvers. If that is too much, the coefficients can be formed on the
fly from a one-channel density texture per level at the cost of more loads in the smoother.

### 4.7 Well-balanced pressure

A star or column at rest must stay at rest. The explicit scheme gets this from the trapezoid
rule of the well-balanced reconstruction (`hydro/q.wgsl`). In the split scheme the pressure
force is the difference of the $`p_f`$ and the gravity force is the cell kick, and the two do
not cancel by construction.

The fix is to solve for the deviation from balance: write $`p = p_h + p'`$ with $`p_h`$
satisfying the same trapezoid rule against the potential, cell by cell, and let the gravity
kick act only on what $`p_h`$ does not balance. This is a sub-stage of its own (2b), after the
unstratified tests pass. Until it is done, the mode refuses self-gravity and uniform fields
on equations of state with an energy.

### 4.8 Single precision

- The solve is for $`p' = p - p_\text{ref}`$, with $`p_\text{ref}`$ a constant of the run (the
  initial mean pressure). The solution texture then holds the fluctuation at full precision.
- The state's own pressure enters only through $`\alpha\,(p^a - p_\text{ref})`$. Its rounding
  noise is about $`6\times10^{-8}\,p`$, comparable to the signal at $`M = 10^{-3}`$, but
  $`\alpha`$ falls as $`1/c^2`$, so the term fades as the Mach number does and the solve tends
  to a projection driven by $`\nabla\cdot\mathbf{v}^*`$ alone.
- The energy $`E`$ cannot hold a pressure fluctuation below $`M \approx 3\times10^{-3}`$, and
  in this scheme it does not have to. Whether the rounding of $`E`$ feeds sound back into the
  flow through $`p^a`$ is the last question of 4.2, answered with `Math.fround`.
- If it does: for the liquid of stage 1, $`p^a`$ comes from the density alone and
  $`\alpha`$ is a constant, which removes the energy from the loop. The liquid is therefore
  the first equation of state to bring up in this mode.

### 4.9 Configuration and interface

- `SimConfig.pressure: 'explicit' | 'implicit'`, default explicit; `psplit=1` in a link; a
  checkbox under Hydro, "Implicit pressure".
- `SimConfig.pressureTol`, `pressureCycles` (the cap), `acousticCfl` (the ceiling of 4.4).
- Diagnostics: cycles used, largest $`|\nabla\cdot\hat{\mathbf{v}}|\,\Delta x/|\mathbf{v}|`$
  over the faces, steps that hit the cap.
- The "Velocity divergence" display field is a centered difference of cell velocities. This is
  an approximate projection: the face velocities are divergence-free to the tolerance, the
  cell-centered ones only to $`O(\Delta x^2)`$. The field's tooltip should say so, or a second
  field should show the face divergence.

### 4.10 Validation tests

| id | Criterion (thresholds from 4.2) |
|---|---|
| `psplit-sod` | L1 density error within 20% of the explicit scheme's; shock position within a cell |
| `psplit-gresho` | at $`M = 0.1, 10^{-2}, 10^{-3}`$: kinetic energy kept after two time units above a floor that does not fall with $`M`$; steps taken independent of $`M`$ |
| `psplit-conservation` | mass, momentum and energy to round-off over 100 steps of a periodic flow |
| `psplit-faces` | both cells of every face agree on $`p_f`$ and $`\hat v_f`$ bit for bit |
| `psplit-obstacle` | flow past a solid block: no mass through solid faces; face divergence below the tolerance in every gas cell |
| `psplit-2d-3d` | a 2D flow and the same flow in one layer of a 3D grid agree to round-off |
| `psplit-acoustic` | a sound wave resolved at an acoustic Courant number of 0.5 keeps its speed to 1%; at 10 it is damped and nothing grows |
| `psplit-cavity`, `psplit-taylor-green` | the stage 1 criteria, at the same accuracy, in at most a fifth of the steps |
| `psplit-balance` (2b) | the stratified column of the `well-balanced` test, to the same $`10^{-3}`$ in Mach number |

### 4.11 Order of work

1. Float64 reference and its unit tests (4.2).
2. The solver alone, tested against a manufactured solution with variable $`\beta`$, walls
   and both dimensions, before any hydro is attached.
3. The liquid equation of state through the whole step, uniform grid, 2D.
4. 3D; the ideal gas; Sod.
5. Viscosity, species, point masses.
6. 2b: well-balanced pressure, then self-gravity and stars.
7. `numerics.md` section, and timings against 2.2's prediction.

## 5. Stage 3: implicit pressure on the adaptive mesh

### 5.1 What the block gravity solver gives, and what it lacks

`amr/gravity.ts` solves each tree level in turn, coarsest first, with multigrid inside the
blocks (16, 8, 4, 2, 1 cells). A fine level takes Dirichlet values interpolated from the
coarse solution, and never corrects it (`amr/gravity/common.wgsl`: "one-way"). For gravity
the error is 0.3%.

For pressure that is not enough. The coarse solution under a fine patch does not know the
fine solution, so the flux $`\beta\,\partial_n p`$ through a refinement face differs between
the two sides. That mismatch is divergence left in the flow at every refinement boundary,
every step. Three things are needed.

### 5.2 A composite operator

At a refinement face the coarse cell's stencil uses the sum of the fine fluxes through the
$`2^{d-1}`$ fine faces that tile its own, recomputed from the fine pressures and the fine
ghost values. This is the pressure's version of what the hydro update already does
(`numerics.md`, "Ghost cells and conservation across levels"). The fine ghost value across the
face comes from limited interpolation of the coarse level, as the block gravity does now
(`amr/gravity/ghost.wgsl`); the interpolation must use the coarse cell being updated, so the
two sides see the same flux.

### 5.3 Two-way coupling

Options, in order of cost:

1. **Iterate the existing cycle with restriction.** Each cycle, after the fine levels relax,
   restrict $`p'`$ into the parents, and form the coarse right-hand side under refined regions
   from the restricted fine residual. With the composite fluxes of 5.2 the fixed point is the
   composite solution. All levels still relax in the same dispatches. The risk is slow
   convergence: this is a Schwarz-like iteration between levels, and the count of cycles may
   grow with depth.
2. **A true composite multigrid** (FAC, as in [Martin & Cartwright 1996][Martin1996] and
   [Almgren et al. 1998][Almgren1998]): V-cycles over the tree levels, with the in-block
   sublevels only below the coarsest tree level. Converges at a rate independent of depth,
   and gives up the "every level in one dispatch" property the gravity solver was built
   around.

Plan: build option 1, measure the cycles to a $`10^{-4}`$ composite residual on a 3-level
Gresho vortex and a cylinder wake. Go to option 2 only if that count is more than about twice
the uniform grid's.

The gate (`amr/gravity/gate.wgsl`) must measure the composite residual, coarse-fine faces
included, not each level's own.

### 5.4 The correction pass on blocks

`amr/correct.wgsl` gains the pressure terms. A coarse cell at a refinement face sums the fine
$`p_f`$ and $`p_f\hat v_f`$, recomputed; fine ghosts come from the assemble pass like every
other ghost. Momentum and energy are then conserved across levels to round-off, tested as
the validation suite tests the hydro fluxes across levels now.

### 5.5 Per-block timesteps

A single implicit solve couples every cell at one time level, and per-block timesteps
(`amr/bins_*.wgsl`) advance blocks to different times. The two do not combine without a
multilevel-in-time solve, which is research rather than engineering. The mode refuses
`tbins`. The cost is small: with the sound speed out of the timestep, the bins would differ
only by level, and the finest level usually holds most of the cells.

### 5.6 Regridding

- A new fine block's velocity comes from limited prolongation and is not divergence-free. The
  next step's solve removes that; no separate projection after a regrid is planned. Check that
  a regrid leaves no visible pressure pulse (a test: Gresho vortex with a refinement patch
  moving across it).
- The warm start $`p'`$ is prolonged into new blocks and restricted into new parents, as
  `amr/gravity/new_blocks.wgsl` does for the potential.

### 5.7 Two dimensions on blocks

`amr/gravity/common.wgsl` is 3D only. The pressure solver needs 2D blocks ($`16^2`$, sublevels
to 1 cell) from the start, as in 4.6. The refinement indicators (`amr/flag.wgsl`, Löhner on
density and pressure) see almost nothing in a flow of constant density; add vorticity as an
indicator for this mode.

### 5.8 Validation tests

| id | Criterion |
|---|---|
| `psplit-amr-poisson` | manufactured solution on a 3-level static mesh: second-order convergence, flux mismatch at refinement faces at round-off |
| `psplit-amr-gresho` | a vortex centered on a refinement corner keeps its kinetic energy as well as on a uniform grid at the coarse resolution or better |
| `psplit-amr-conservation` | mass, momentum, energy to round-off with regridding on |
| `psplit-amr-divergence` | face divergence below the tolerance on every leaf face, refinement faces included |
| `psplit-amr-regrid` | the moving-patch test of 5.6 |
| `psplit-amr-wake` | Strouhal number of the cylinder wake within 3% of the uniform grid's at the fine resolution |

### 5.9 Order of work

1. Composite operator and gate on a static mesh, against the manufactured solution.
2. Option 1 coupling; measure; decide on option 2.
3. Correction pass; conservation tests.
4. Regridding and warm start.
5. 2D blocks.
6. Self-gravity and stars with 2b, if 2b is done.

## 6. Not covered

- **Free surfaces** (water under air, dam breaks). They need an interface between two fluids
  with density ratios near 1000. The species machinery mixes ideal gases with different
  $`\gamma`$; a liquid-gas pair needs a stiffened-gas mixture rule and an interface treatment,
  and in stage 2 a solver that holds a 1000:1 jump in $`\beta`$. A stage 4 of its own.
- **Surface tension, buoyancy by the Boussinesq approximation.** Stratified flow is served by
  the compressible form with 2b.
- **Implicit viscosity.** The explicit limit binds only at low Reynolds number on fine grids.
  The pressure solver of stage 2 is most of what an implicit viscous solve would need, if it
  is wanted later.
- **MHD with implicit pressure.**
- **Low-Mach stellar formulations** (anelastic, pseudo-incompressible, as in MAESTRO). Stage 2
  with 2b covers slow stratified flow without changing the equations.

## 7. Open decisions

| Decision | Where it is settled | Default if nothing argues otherwise |
|---|---|---|
| Is stage 2 worth it for speed, or only for accuracy? | 2.2 | build it for accuracy regardless |
| Advective flux of the split scheme | 4.2 | upwind with a local Lax-Friedrichs term |
| $`p^a`$ from the pressure equation or the advected internal energy | 4.2 | the pressure equation |
| Explicit $`\nabla p^n`$ in the predictor | 4.2 | no |
| Solver tolerance and cycle cap | 4.2, then on the GPU | $`10^{-4}`$, 8 |
| Coefficient textures or coefficients formed on load | 4.6, by timing | textures |
| One merged correction pass or two | 4.5 | merged |
| Composite multigrid on blocks | 5.3 | option 1 |

## 8. References

- [Kwatra, Su, Grétarsson & Fedkiw (2009)][Kwatra2009], "A method for avoiding the acoustic
  time step restriction in compressible flow", J. Comput. Phys. 228, 4146.
- [Thornber et al. (2008)][Thornber2008], "An improved reconstruction method for compressible
  flows with low Mach number features", J. Comput. Phys. 227, 4873.
- [Bell, Colella & Glaz (1989)][Bell1989], "A second-order projection method for the
  incompressible Navier-Stokes equations", J. Comput. Phys. 85, 257.
- [Almgren, Bell, Colella, Howell & Welcome (1998)][Almgren1998], "A conservative adaptive
  projection method for the variable density incompressible Navier-Stokes equations",
  J. Comput. Phys. 142, 1.
- [Martin & Cartwright (1996)][Martin1996], "Solving Poisson's equation using adaptive mesh
  refinement", UC Berkeley ERL memorandum M96/66.
- [Ghia, Ghia & Shin (1982)][Ghia1982], "High-Re solutions for incompressible flow using the
  Navier-Stokes equations and a multigrid method", J. Comput. Phys. 48, 387.
- Gresho & Chan (1990), Fleischmann et al. (2020), Käppeli & Mishra (2014): as cited in
  `docs/numerics.md`.

The citations above are from memory and have not been checked against the publishers' records;
check each before any of this reaches `numerics.md` or the paper.

[Kwatra2009]: https://doi.org/10.1016/j.jcp.2009.02.027
[Thornber2008]: https://doi.org/10.1016/j.jcp.2008.01.036
[Bell1989]: https://doi.org/10.1016/0021-9991(89)90151-4
[Almgren1998]: https://doi.org/10.1006/jcph.1998.5890
[Martin1996]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/1996/ERL-96-66.pdf
[Ghia1982]: https://doi.org/10.1016/0021-9991(82)90058-4
