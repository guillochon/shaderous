# Numerics

This page describes every part of the Shaderous solver: the equations, how each term is
discretized, the constants the code uses, and where the method comes from. How the scheme
is laid out on the GPU is in [architecture.md](architecture.md). The source file for each
part is named in its section.

Shaderous is a finite-volume Godunov code. The scheme is standard, and each source is cited
where it is used. The changes we made, mostly to keep the scheme accurate in single
precision, are pointed out where they occur.

**Contents**

1. [Equations solved](#1-equations-solved)
2. [Grid and variables](#2-grid-and-variables)
3. [The timestep](#3-the-timestep)
4. [MUSCL-Hancock integration](#4-muscl-hancock-integration)
5. [Reconstruction and limiters](#5-reconstruction-and-limiters)
6. [Riemann solvers](#6-riemann-solvers)
7. [Robustness in single precision](#7-robustness-in-single-precision)
8. [Well-balanced reconstruction](#8-well-balanced-reconstruction)
9. [Equations of state](#9-equations-of-state)
10. [Optically thin cooling and heating](#10-optically-thin-cooling-and-heating)
11. [Viscosity and thermal conduction](#11-viscosity-and-thermal-conduction)
12. [Passive scalar and species](#12-passive-scalar-and-species)
13. [Self-gravity](#13-self-gravity)
14. [Coupling gravity to the gas](#14-coupling-gravity-to-the-gas)
15. [Point masses and sinks](#15-point-masses-and-sinks)
16. [Initial states in discrete balance](#16-initial-states-in-discrete-balance)
17. [Adaptive mesh refinement](#17-adaptive-mesh-refinement)
18. [Local (per-block) timesteps](#18-local-per-block-timesteps)
19. [Accelerating frames and damping zones](#19-accelerating-frames-and-damping-zones)
20. [Nuclear burning network](#20-nuclear-burning-network)
21. [Ideal MHD](#21-ideal-mhd)
22. [Rotating frames, symmetry and coordinates](#22-rotating-frames-symmetry-and-coordinates)
23. [Validation](#23-validation)
24. [Radiation transport](#24-radiation-transport)
25. [ANEOS materials and planets](#25-aneos-materials-and-planets)
26. [Bodies in a stream](#26-bodies-in-a-stream)
27. [References](#references)

---

## 1. Equations solved

Shaderous solves the compressible Euler equations with gravity, optional diffusion and
optically thin heating and cooling:

```math
\begin{aligned}
\partial_t \rho + \nabla\cdot(\rho\mathbf{v}) &= 0,\\
\partial_t (\rho\mathbf{v}) + \nabla\cdot\left(\rho\mathbf{v}\mathbf{v} + p\,\mathsf{I} - \boldsymbol{\tau}\right) &= -\rho\nabla\Phi,\\
\partial_t E + \nabla\cdot\left[(E + p)\mathbf{v} - \boldsymbol{\tau}\cdot\mathbf{v} + \mathbf{q}\right] &= -\rho\mathbf{v}\cdot\nabla\Phi + \mathcal{H} - \mathcal{C},\\
\nabla^2\Phi &= 4\pi G\rho \quad (+\ \text{point masses}),
\end{aligned}
```

where $`\rho`$ is the density, $`\mathbf{v}`$ the velocity, $`p`$ the pressure,
$`E = \rho e + \tfrac12\rho v^2`$ the total energy density ($`e`$ the specific internal energy),
$`\Phi`$ the gravitational potential, $`\boldsymbol{\tau}`$ the viscous stress, $`\mathbf{q}`$ the
conductive heat flux, and $`\mathcal{H}`$, $`\mathcal{C}`$ the heating and cooling rates per unit
volume. The pressure comes from an equation of state $`p(\rho, \rho e)`$ chosen per problem
([§9](#9-equations-of-state)).

The code also evolves:

- the **internal energy density** $`\rho e`$, advected with the gas, for the dual-energy formalism ([§7](#7-robustness-in-single-precision));
- a **passive scalar** $`s`$, advected as $`\partial_t(\rho s) + \nabla\cdot(\rho s\mathbf{v}) = 0`$;
- any number of **species mass fractions** $`X_i`$ with $`\sum_i X_i = 1`$ ([§12](#12-passive-scalar-and-species)).

A problem can also use a uniform external acceleration $`\mathbf{g}`$ (the Rayleigh-Taylor
problem does) in place of self-gravity.

## 2. Grid and variables

The domain is a Cartesian box of $`n^3`$ cubic cells ($`n^2`$ in 2D) with cell-averaged conserved
variables

```math
\mathbf{U} = (\rho,\ \rho v_x,\ \rho v_y,\ \rho v_z,\ E,\ \rho e,\ \rho s).
```

A 2D run is a single periodic layer of the same 3D scheme, so 2D and 3D use the same code.

Boundaries are set per face: periodic, outflow (zero gradient), outflow without inflow (the
normal velocity is clamped so gas can leave but not enter), and reflecting. They are applied
in the function that maps a cell index to a texel, so the uniform grid has no ghost cells
([architecture.md](architecture.md#boundaries-without-ghost-cells)). Obstacles drawn in the
interface are solid cells that act as reflecting walls.

Everything is computed in 32-bit floating point (WGSL has no 64-bit float type). Problems are
set up in code units in which the density, velocity and length are of order one. Problems
with physical units (the white dwarfs, the thermal instability, the stellar presets) keep a
scale for each unit and convert to cgs or SI only for display (`src/units/`).

## 3. The timestep

The timestep is computed on the GPU every step (`hydro/rates.wgsl`, `hydro/clock.wgsl`) and is
the smallest of

```math
\Delta t = \min\left[
C\,\frac{\Delta x}{\max\left[\sum_d\left(|v_d| + c_s\right) + s_\text{diff}\right]},\quad
C\sqrt{\frac{\Delta x}{\max|\mathbf{g}|}},\quad
\Delta t_\text{pair},\quad
\Delta t_\text{max}
\right],
```

where $`C = 0.8`$ is the CFL number, $`c_s`$ the sound speed, the sum is over the axes (the update is unsplit, so their fluxes add in one
step), and the maximum is over cells. Each term is:

- **Diffusion** adds a signal speed $`s_\text{diff} = 2.5\,d\,D/\Delta x`$, where $`d`$ is the number of
  dimensions and $`D`$ the larger of $`\tfrac43\nu`$ and the thermal diffusivity ([§11](#11-viscosity-and-thermal-conduction)).
  Where diffusion dominates, the step falls as $`\Delta x^2`$, as an explicit scheme requires.
- **Gravity** limits the step so that no cell is accelerated across more than a fraction of
  itself in one step: self-gravity, point masses, a rotating frame's forces and a uniform field
  alike (until 2026-11-18 a uniform field alone did not, and cold gas under it fell two cells
  in one step through the walls of a closed box). Curved grids add their geometric
  (centrifugal) accelerations, over each axis's own length ([§22](#22-rotating-frames-symmetry-and-coordinates)).
- **Point-mass pairs** ([§15](#15-point-masses-and-sinks)): each pair limits the step to
  $`\tfrac{C}{16}\min\left(r/v,\ \sqrt{r^3/GM}\right)`$, where $`r`$ is the softened separation, $`v`$ the
  relative speed and $`M`$ the pair's total mass. That is about 125 steps per circular orbit.
- **The problem's own maximum** $`\Delta t_\text{max}`$, and the time left until the next output
  or checkpoint, so that outputs fall exactly on their times.

Cooling does not limit the step; it is substepped instead ([§10](#10-optically-thin-cooling-and-heating)).

## 4. MUSCL-Hancock integration

The integrator is the unsplit MUSCL-Hancock scheme of [van Leer (1979)][vanLeer1979], in the
form given by [Toro (2009)][Toro2009]. It is second order in time and takes one Riemann solve
per face per step. It has two passes.

**Predictor** (`hydro/predictor.wgsl`). Each cell reconstructs limited differences
$`\Delta\mathbf{W}_d`$ of its primitive variables $`\mathbf{W} = (\rho, \mathbf{v}, p, s)`$
along each axis ([§5](#5-reconstruction-and-limiters)) and advances itself half a step in primitive form:

```math
\mathbf{W}^{n+1/2} = \mathbf{W}^n - \frac{\Delta t}{2\Delta x}\sum_d \mathsf{A}_d(\mathbf{W}^n)\,\Delta\mathbf{W}_d
+ \frac{\Delta t}{2}\,\mathbf{S}(\mathbf{W}^n),
```

where $`\mathsf{A}_d`$ is the Jacobian of the primitive equations along axis $`d`$, and
$`\mathbf{S}`$ holds half a gravity kick. For a general equation of state the pressure row uses
$`\rho c_s^2`$ in place of $`\gamma p`$. The predictor also stores a shock flag and a
first-order flag for each cell.

**Update** (`hydro/update.wgsl`, `hydro/flux.wgsl`). The update rebuilds face states from
the half-step values, $`\mathbf{W}_{i\pm1/2}^{\mp} = \mathbf{W}^{n+1/2}_i \pm \tfrac12\Delta\mathbf{W}_i`$,
solves a Riemann problem on each of the cell's $`2d`$ faces, and applies the conservative
update

```math
\mathbf{U}_i^{n+1} = \mathbf{U}_i^n - \frac{\Delta t}{\Delta x}\sum_d\left(\mathbf{F}_{i+1/2,d} - \mathbf{F}_{i-1/2,d}\right) + \Delta t\,\mathbf{S}_i .
```

Each cell computes the fluxes on all of its faces, so every face flux is computed twice, once
by each of its two cells. The two results are bit-identical, which is what conserves mass,
momentum and energy to round-off. This is explained in
[architecture.md](architecture.md#gathered-fluxes-and-bit-identical-faces).

## 5. Reconstruction and limiters

### Default: third order, limited only where the flow is not smooth

`hydro/recon.wgsl`. For a cell value $`q_c`$, let $`\delta_\text{near}`$ be the difference toward a
face and $`\delta_\text{far}`$ the difference away from it. The unlimited face value is that of the
parabola through the cell and its two neighbors:

```math
q_\text{face} = q_c + \tfrac16\left(\delta_\text{far} + 2\,\delta_\text{near}\right),
```

which is third-order accurate on the same three-cell stencil a slope limiter uses. A
three-cell limiter cannot tell a smooth extremum from a jump and flattens both, which holds
limited schemes to second order or less at extrema. Following the idea of
[Čada & Torrilhon (2009)][Cada2009], Shaderous limits only where a quantity changes by more
than a fraction $`\varepsilon = 0.01`$ of its natural scale $`S`$ across the cell. $`S`$ is the
density itself, the pressure itself, or the sound speed for the velocities. With

```math
\eta = \frac{\delta_\text{far}^2 + \delta_\text{near}^2}{(\varepsilon S)^2},
\qquad w = \mathrm{clamp}(1.5 - \eta,\ 0,\ 1),
```

the increment is $`w`$ times the unlimited one plus $`(1 - w)`$ times the limited one. The
limited increment is that of [Koren (1993)][Koren1993] with the bounds of the monotonized
central limiter:

```math
\Delta_\text{lim} = \begin{cases}
\mathrm{sgn}(\delta_\text{near})\,\min\left(2|\delta_\text{far}|,\ 2|\delta_\text{near}|,\ \tfrac13|\delta_\text{far} + 2\delta_\text{near}|\right) & \delta_\text{far}\,\delta_\text{near} > 0,\\
0 & \text{otherwise.}
\end{cases}
```

A smooth peak keeps its height: the largest error in the smooth-advection test is 2.5 to 5
times smaller than with the MC limiter. The cost is that kinks and jumps smaller than 1% are
not limited and can ring by a small fraction of their own height. Shock-tube errors along an
axis are a few percent higher than with MC.

### Classic slope limiters

`hydro/limiters/`. Each can be chosen in the interface. With $`a`$ and $`b`$ the backward and
forward differences, each gives a slope $`\sigma`$ with $`q_\text{face} = q_c \pm \tfrac12\sigma`$:

| Limiter | Slope $`\sigma`$ (zero where $`ab \le 0`$) | Reference |
|---|---|---|
| Monotonized central (MC) | $`\mathrm{sgn}(a)\min\left(2\lvert a\rvert,\ 2\lvert b\rvert,\ \tfrac12\lvert a+b\rvert\right)`$ | [van Leer (1977)][vanLeer1977] |
| van Leer | $`\dfrac{2ab}{a+b}`$ | [van Leer (1974)][vanLeer1974] |
| minmod | $`\mathrm{sgn}(a)\min\left(\lvert a\rvert,\ \lvert b\rvert\right)`$ | [Roe (1986)][Roe1986] |

The MC slope is also used wherever a single slope is needed: in prolongation on the adaptive
mesh, and for the species ([§12](#12-passive-scalar-and-species)).

## 6. Riemann solvers

`hydro/riemann/`. Every solver uses the wave-speed estimates of [Davis (1988)][Davis1988]:

```math
S_L = \min(u_L - c_L,\ u_R - c_R), \qquad S_R = \max(u_L + c_L,\ u_R + c_R),
```

which need only the pressure and sound speed on each side. The solvers therefore work
unchanged with any equation of state.

### HLL

The HLL solver of [Harten, Lax & van Leer (1983)][Harten1983] averages over the whole fan:

```math
\mathbf{F}^\text{HLL} = \frac{S_R\mathbf{F}_L - S_L\mathbf{F}_R + S_L S_R(\mathbf{U}_R - \mathbf{U}_L)}{S_R - S_L},
```

with $`\mathbf{F} = \mathbf{F}_L`$ if $`S_L \ge 0`$ and $`\mathbf{F} = \mathbf{F}_R`$ if $`S_R \le 0`$.
Barotropic gas always uses HLL, since it has no contact wave for HLLC to resolve.

### HLLC (default)

HLLC ([Toro, Spruce & Speares 1994][Toro1994]) restores the contact wave. Its speed is

```math
S_* = \frac{p_R - p_L + \rho_L u_L(S_L - u_L) - \rho_R u_R(S_R - u_R)}{\rho_L(S_L - u_L) - \rho_R(S_R - u_R)},
```

and the intermediate states, for $`K = L, R`$, are

```math
\mathbf{U}_{*K} = \rho_K\frac{S_K - u_K}{S_K - S_*}
\begin{pmatrix} 1 \\ S_* \\ v_K \\ w_K \\ \dfrac{E_K}{\rho_K} + (S_* - u_K)\left[S_* + \dfrac{p_K}{\rho_K(S_K - u_K)}\right] \end{pmatrix},
\qquad \mathbf{F}_{*K} = \mathbf{F}_K + S_K(\mathbf{U}_{*K} - \mathbf{U}_K).
```

The internal energy and the passive scalar are carried upwind with the mass flux.

### HLLC with a low-Mach correction

A Godunov flux damps a velocity jump at a rate set by the sound speed, so slow flow is damped
much faster than it moves. Following [Thornber et al. (2008)][Thornber2008], the velocity jump
passed to HLLC is scaled by the local Mach number:

```math
\mathbf{v}_{L,R} \to \tfrac12(\mathbf{v}_L + \mathbf{v}_R) \pm \tfrac{z}{2}(\mathbf{v}_L - \mathbf{v}_R),
\qquad z = \min\left[1,\ \max\left(z_\text{min},\ \tfrac{|\mathbf{v}_L|}{c_L},\ \tfrac{|\mathbf{v}_R|}{c_R}\right)\right].
```

The floor is $`z_\text{min} = 0.3`$. With one predictor and one update and the unlimited
third-order reconstruction, the scheme relies on some of the solver's damping: a
[Gresho & Chan (1990)][Gresho1990] vortex at Mach 0.01 gains kinetic energy without bound for
$`z_\text{min} \le 0.1`$. With 0.3, that vortex on $`64^2`$ cells loses 6% of its kinetic energy in
two time units, against 15% with plain HLLC. Scaling the acoustic wave speeds instead, as in
HLLC-LM ([Fleischmann et al. 2020][Fleischmann2020]), also removes the damping of pressure
jumps and was unstable in this scheme at Mach 0.03. Above Mach 1, shocks included, the solver
is plain HLLC.

### Carbuncle suppression

HLLC is prone to the carbuncle and odd-even decoupling at grid-aligned shocks
([Quirk 1994][Quirk1994]). The predictor flags a cell as a shock where the flow converges
($`\nabla\cdot\mathbf{v} < 0`$) and the relative pressure jump across it,
$`\max_d |p_{i+1} - p_{i-1}|/\min(p_{i+1}, p_{i-1})`$, exceeds 0.5. The jump is measured against
hydrostatic balance ([§8](#8-well-balanced-reconstruction)), so a steep atmosphere at rest is
not flagged. On any face next to a flagged cell the flux is HLL.

## 7. Robustness in single precision

`hydro/dual_energy.wgsl`, `hydro/pfloor.wgsl`.

**Dual energy.** In single precision, $`E - \tfrac12\rho v^2`$ loses the pressure of gas whose
kinetic energy dominates. Following [Bryan et al. (1995)][Bryan1995], the internal energy is
also evolved with its own equation,

```math
\partial_t(\rho e) + \nabla\cdot(\rho e\,\mathbf{v}) = -p\,\nabla\cdot\mathbf{v} + \text{(viscous heating)} + \mathcal{H} - \mathcal{C},
```

and after each update the pair is synchronized. With $`e_\text{th} = E - \tfrac12\rho v^2`$:

- if $`e_\text{th} > \eta E`$ with $`\eta = 10^{-3}`$, and $`e_\text{th}`$ is above the equation of state's least internal energy, then $`\rho e \leftarrow e_\text{th}`$ and $`E`$ is kept exactly as the conservative update left it;
- otherwise the advected $`\rho e`$ is used, and $`E`$ is rebuilt as $`\tfrac12\rho v^2 + \rho e`$.

Degenerate gas is an exception (`EOS_KEEP_TOTAL`). Its pressure is defined (that of the cold
electrons, at least) even where $`E - \tfrac12\rho v^2`$ is negative. In the cell ahead of each
strong shock of a white dwarf collision that difference dips below zero, and rebuilding $`E`$
there added 2.3% to the total energy at the greatest compression. Shaderous therefore keeps
$`E`$ and takes only the pressure from $`\rho e`$. The total energy then changes by 0.6% through
the collision, and is conserved exactly without gravity until gas leaves the box. The least
internal energy is also set per equation of state: for degenerate gas it is the
zero-temperature energy, which cold gas sits right on.

**Positivity fallback.** If a reconstructed face state has a non-positive density or
pressure, the cell drops to first order (zero slopes) for that step.

**Floors.** Density and pressure floors (by default $`10^{-10}`$ and $`10^{-14}`$ in code units)
catch what remains. The number of floored cells is a diagnostic shown in the interface.

**Jeans floor** (optional). Adds pressure where needed so that the Jeans length
$`\lambda_J = \sqrt{\pi c_s^2/(G\rho)}`$ is resolved by at least $`N_J \ge 4`$ cells, the
condition of [Truelove et al. (1997)][Truelove1997].

### Deviations from a reference

`hydro/ref_on.wgsl`, `hydro/refstate.ts`; `physics.reference` in a setup. Single precision holds
a number to a relative $`2^{-24}`$, so a pressure difference smaller than that part of the
pressure itself is lost when the energy is stored. Slow flow lives in such differences: a
vortex at Mach number $`M`$ is held by pressure differences of $`M^2`$ of the pressure, one
float32 step of it at $`M = 3.4\times10^{-4}`$, and the hydro-proofs failure maps found the
Gresho vortex lost below Mach 0.0014 however fine the grid, with implicit pressure as without.
A static atmosphere fails the same way over many decades of density: rounding of order
$`10^{-7}`$ in its initial state sends sound waves up the column whose speed grows as
$`\rho^{-1/2}`$, reaching Mach $`10^{-3}`$ near a total contrast of $`10^{-10}`$
([§8](#8-well-balanced-reconstruction)).

With a reference state the gas is stored as its deviation from a state at rest in hydrostatic
balance, as the well-balanced low-Mach scheme of [Edelmann et al. (2021)][Edelmann2021] splits
its state:

```math
\mathbf U = (\rho - \rho_0,\ \rho\mathbf v,\ E - e_0,\ \rho e - e_0,\ \rho s), \qquad e_0 = p_0/(\gamma - 1),
```

and the primitives the reconstruction and the predictor carry are $`\rho - \rho_0`$ and
$`p - p_0 = (\gamma - 1)(\rho e - e_0)`$, formed with no large pressure on the way. The
reference's own forces cancel exactly and are never computed: the momentum fluxes take
$`p - p_0`$ (the contact speed of HLLC too), the gravity kick is $`(\rho - \rho_0)\mathbf g`$,
and the balanced reconstruction's add-back is $`-(\rho - \rho_0)\Delta\Phi`$. The reference
enters only where its size is what matters: the densities of the mass fluxes, the sound speed,
and the energy fluxes, which carry $`(E + p)\mathbf v`$ and so are small where the flow is.
Each face takes the reference's density there (minmod-limited along its axis) and its pressure
carried to the face along the balance; the half step advects the reference's gradients; the
dual-energy switch decides on the whole energies but never forms the stored ones from them.

The reference is built once from the initial state in double precision: **uniform**, the
state's mean (a run without gravity), or a **column**, the mean of each layer across a uniform
field, its pressure integrated from the top layer's down with the trapezoid rule of
[§8](#8-well-balanced-reconstruction), which makes it balanced exactly. The deviations are taken
in double precision before anything is rounded, so a column that starts at rest has deviations
of zero. The display, the diagnostics and `readState` show the whole state.

So far it is for the ideal gas with explicit pressure on the uniform grid, in Cartesian
coordinates, and without the physics whose passes read the state as it is (cooling, burning,
diffusion, sources, damping, driving, species, radiation, a magnetic field, bodies, point
masses, self-gravity, rotating frames): `refUnsupported` in `src/sim.ts` says which.

Measured, against the same runs without it (2D, $`\gamma = 1.4`$). The isothermal columns of
the hydro-proofs maps, $`h`$ cells per scale height, with floors below their top cells; the
largest Mach number after a sound crossing:

| Column | Contrast | Without | With |
|---|---|---|---|
| $`h = 2.85`$, 64 cells | $`2\times10^{-10}`$ | $`3.6\times10^{-4}`$ | $`4\times10^{-17}`$ |
| $`h = 1.2`$, 64 cells | $`5\times10^{-25}`$ | 1.0 (destroyed) | $`2\times10^{-10}`$ |
| $`h = 0.7`$, 32 cells | $`8\times10^{-25}`$ | 0.72 | $`7\times10^{-11}`$ |
| $`h = 0.6`$, 24 cells | $`1\times10^{-24}`$ | 0.24 | $`7\times10^{-9}`$ |

The Gresho vortex of those maps at $`t = 0.2`$, with HLLC's low-Mach correction; the mean
velocity error:

| Mach number, grid | Without | With |
|---|---|---|
| $`3\times10^{-4}`$, $`128^2`$ | 25% | 7.1% |
| $`10^{-4}`$, $`128^2`$ | 41% | 13% |
| $`10^{-4}`$, $`256^2`$ | (about 50% at any grid) | 4.7% |

What is left with it falls with the grid: the scheme's dissipation, not the rounding. A column
whose top lies below single precision's smallest normal number ($`10^{-38}`$) is beyond
reach either way: the reference itself cannot hold it. The presets *Deep atmosphere* (40 scale
heights: Mach $`3\times10^{-10}`$ after a sound crossing with the reference, 1.4 without) and
*Gresho vortex, gas at Mach 0.001* show it, with the reference among their featured settings.

Slow convection is limited by the scheme's buoyancy as well as by precision. A polytropic
layer in discrete balance whose temperature gradient is $`\delta`$ off the adiabat, seeded with
convection cells at Mach $`10^{-3}`$ (2D, $`\gamma = 5/3`$, its temperature halving over a
box of 128 cells, about 100 cells per pressure scale height at the bottom). With the density
reconstructed as it is, a layer stable by $`\delta = -10^{-5}`$ gained 20 times its kinetic
energy by $`t = 1000`$, to Mach 0.008, as did every layer from $`\delta = 0`$ to $`10^{-5}`$,
with the reference or without (on 256 cells the neutral layer's grew 1.9 times); one stable by
$`10^{-3}`$ lost 89% of it. The faces' entropy was off by about $`(\Delta x/H)^2`$; with the
density reconstructed along the adiabat ([§8](#8-well-balanced-reconstruction)), the default for
the ideal gas (as first made, in its unbounded form), the layer stable by $`10^{-5}`$ keeps its kinetic energy and the neutral one gains
1.4 times. The *Convection* preset is this layer at $`\delta = 0.01`$: its cells settle at Mach
0.07 to 0.1 with a kinetic energy of $`4.8\times10^{-4}`$ on 128 cells ($`6.8\times10^{-4}`$
on 256 with the plain density), and the speed goes as $`\sqrt{\delta}`$: Mach 0.27 at
$`\delta = 0.1`$, 0.02 to 0.03 at $`10^{-3}`$ (kinetic energy $`3\times10^{-5}`$), 0.0075 at
$`10^{-4}`$ ($`1.8\times10^{-6}`$, a few times what the neutral layer gains). Below about
$`10^{-4}`$ on 128 cells the grid's error still sets the flow; with the plain density that
floor lay between $`10^{-3}`$ and $`10^{-2}`$, the cells at $`10^{-3}`$ peaking at Mach 0.045 and
sinking to it.

The columns also needed HLLC to be exact at rest ([§6](#hllc-default)): it formed its star
state's density as $`\rho(S - u)/(S - S^*)`$ and the flux from that less $`\rho`$, which at a
face at rest is $`\rho S/S - \rho`$, up to $`10^{-7}`$ of the density, every step. It now forms
the difference directly, $`\rho(S^* - u)/(S - S^*)`$, zero there, and likewise for the
momentum and energy: without it the $`h = 0.6`$ column above reached Mach 1.7, and the
Rayleigh-Taylor layers at rest, exactly still before, moved at Mach $`10^{-6}`$ once the density
was reconstructed along the adiabat (whether $`\rho S/S - \rho`$ rounds to zero depends on the
bits of $`\rho`$).

## 8. Well-balanced reconstruction

`hydro/q.wgsl`. In hydrostatic balance the pressure itself is not smooth across a stratified
cell, but its balance with gravity is. Following [Käppeli & Mishra (2014)][Kappeli2014], the
quantity that is limited and reconstructed is each neighbor's deviation from that balance:

```math
\delta p_n = (p_n - p_c) + \tfrac12(\rho_c + \rho_n)(\Phi_n - \Phi_c),
```

where $`c`$ is the cell and $`n`$ a neighbor. The balanced part of the step to the face,
$`-\rho_c\,\Delta\Phi_\text{face}`$, is added back unlimited. Where the cells satisfy this
trapezoid rule with $`\delta p = 0`$:

1. the two cells that share a face agree on its pressure, and
2. a cell's two face pressures differ by exactly the momentum that the gravity kick adds.

The discrete state is then stationary to round-off however steep it is. Walls and obstacles
are also exact: the image cell beyond a wall has the cell's own potential, and the wall face
is reached along the true potential. The shock detector ([§6](#carbuncle-suppression))
measures pressure jumps against the same balance, and so does the lower bound on face
pressures (`hydro/flux.wgsl`): no face state may fall below half the smaller of its two
cells' pressures, nor below half of either one carried to the face along the balance. At a
wall the other cell is this one's image, with its pressure, and the balanced wall pressure
$`p\,(1 - 1/2h)`$ of an isothermal column with $`h`$ cells per scale height is below $`p/2`$
once $`h < 1`$. Before the bound followed the balance, it raised that pressure, and the cell
under a lid fell at Mach 0.002 ($`h = 0.99`$) to 0.08 ($`h = 0.7`$) in a sound crossing; now an
8-cell column holds at $`5\times10^{-7}`$ down to $`h = 0.85`$ and $`5\times10^{-6}`$ at
$`h = 0.7`$, the balance existing only above $`h = 1/2`$. (Found by the failure maps of the
hydro-proofs project.)

What remains is precision. A column of many scale heights fails once its total density
contrast falls below about $`10^{-10}`$, at any $`h`$: rounding of order $`10^{-7}`$ in the
initial state sends sound waves up the column, and their velocity grows as $`\rho^{-1/2}`$
as they climb, about $`\sqrt{\text{contrast}}`$ in all: Mach $`5\times10^{-4}`$ at the top of
a 64-cell column with $`h = 2.85`$ (contrast $`2\times10^{-10}`$), $`7\times10^{-7}`$ in 16
cells of the same $`h`$.

Measured: an isothermal column whose density falls by a factor $`e`$ every 4 cells (nine
million over the box) keeps its largest Mach number at $`4\times10^{-5}`$ after a sound crossing,
against 0.35 without the balanced reconstruction. The background of the Rayleigh-Taylor
problem is 240 times quieter, and the error in the Jeans growth rate 12 times smaller. It is
on by default for every equation of state with an energy; `wb=0` in a link turns it off for
comparison.

**Density along the adiabat.** The balanced reconstruction makes the pressure smooth in a
stratified layer, but the density was reconstructed as it is, and in a layer about a hundred
cells per scale height deep it changes by a part in a hundred per cell, where the limiter's
switch to limiting lies ([§5](#5-reconstruction-and-limiters)). The faces then carried a
density and a pressure off the cell's adiabat by about $`(\Delta x/H)^2`$, and gas carried up
or down arrived with the wrong entropy: a spurious buoyancy. A layer at the adiabat, seeded
with convection cells at Mach $`10^{-3}`$ on 128 cells, gained 17 times its kinetic energy by
$`t = 1000`$, and so did one stable by $`10^{-5}`$ of the adiabatic gradient
([§7](#deviations-from-a-reference)); the more compressive the limiter, the more (at
$`t = 300`$: Koren 2.9 times, MC 1.4, van Leer 1.1; minmod damped it to 0.24).

So for the ideal gas the density is reconstructed the way the pressure is: what is limited is
each neighbor's density brought to the cell's pressure along the adiabat,
$`\rho_n (p_c/p_n)^{1/\gamma}`$, flat in a layer at the adiabat, and the face density is that
reconstruction carried to the face's pressure along the adiabat (`hydro/q.wgsl` `isen_inc`, on
whole densities and pressures with a reference state too; the CPU reference of
`src/reference/hydro.ts` does the same). It is a simpler relative of the local steady-state
reconstruction of [Grosheintz-Laval & Käppeli (2020)][GrosheintzLaval2020], which reconstructs
density, velocity and pressure about a cell's own steady adiabatic flow (constant entropy, mass
flux and Bernoulli constant) and takes gravity's force from that flow; this one treats the
density alone, at rest, and keeps the balanced pressure, and the face density is kept between
the densities of the cell and its neighbor beyond the face.

Where the gas is not on one smooth adiabat it does harm: across a shock, in a reaction zone, in
cold gas falling at several times its sound speed, where the pressure (and so the density tied
to it) is the least certain part of the state. Unchecked, it made a Sedov blast 1.70 cells out
of round (1.10 with the plain density), drifted the Evrard collapse's energy 4.8% (0.23%) and
slowed a detonation front by 1.2% of its Chapman-Jouguet speed (0.12%). So what it adds to the
plain density's increment is bounded (`isen_blend`): by a quarter of the smaller density step
to a neighbor, or, where the gas is smooth, by 1% of the density, which is far more than the
correction there, about $`(\Delta x/H)^2`$ of the density. The gas is smooth where the
pressure changes by less than 20% from cell to cell, the neighbors' densities at the cell's
pressure lie within 1% of the cell's (the limiter's own measure of smoothness; a layer
$`10^{-2}`$ off the adiabat, 50 cells per scale height deep, departs by $`10^{-4}`$ per cell)
and the velocity changes by less than a tenth of the sound speed (gas falling onto a sink at
Mach 3 changes by 0.2 to 0.3 per cell). Each of the three takes the room from 1% of the density
down to nothing between its bound and twice it, continuously.

It is bounded rather than switched off, or blended with the plain density by a weight, because
both of those failed. A switch turns a difference of rounding into one of method in the cells
where it flips: a Sedov blast in an octant differed from the whole box by
$`1.5\times10^{-5}`$ in density, against $`2\times10^{-6}`$ with the plain density. A weight,
however smooth, feeds back through the difference between the two reconstructions, which is
largest at a shock: in a sink's converging flow, where the infall reaches Mach 3 at its
accretion shock, the whole box's own mirror asymmetry grew to $`5\times10^{-3}`$ by step 64 with
switches and $`7.5\times10^{-3}`$ with weights, against $`1.4\times10^{-6}`$ with the plain
density (which grows too, slowly, there: $`3.8\times10^{-5}`$ by step 218). A bound acts only
where it binds, and there its gain is that of the bound itself, at most 1% of the density over
the guards' bounds: $`1.8\times10^{-4}`$ by step 218. At 5% of the density the converging flow
was unstable again, or with the velocity guard the Sedov octant differed from the whole box by
$`3.2\times10^{-5}`$; with no guards on the room, a detonation's reaction zone was left in part
at a coarser level of an adaptive mesh; at half the density step, the Evrard collapse's energy
drifted 0.63%.

| | Plain density | Along the adiabat |
|---|---|---|
| Sedov blast, out of round | 1.10 cells | 1.07 |
| Evrard collapse, energy drift | 0.23% | 0.31% |
| Detonation front, speed error | 0.12% | 0.075% |
| Sedov octant against the whole box | $`2\times10^{-6}`$ | $`2.5\times10^{-6}`$ |
| Convecting layer at $`\delta = 10^{-6}`$, kinetic energy by $`t = 1000`$ | 14 times | 2.0 times |
| At $`\delta = 10^{-4}`$, kinetic energy by $`t = 1000`$ | 1.7 times (it decays) | 10 times |

(The layers are the *Convection* preset, 128 cells, with its reference state: at
$`\delta = 10^{-4}`$ the plain density's own error in the buoyancy damps the convection, which
the reconstruction along the adiabat keeps going. Before $`t = 700`$ the bounded reconstruction
follows the unbounded one at $`\delta = 10^{-6}`$, which reaches 1.5 times by $`t = 1000`$.)
Elsewhere the changes are small: the Rayleigh-Taylor layers' kinetic energy grows 47.54 times
against 47.53, the Kelvin-Helmholtz layer's transverse kinetic energy 11.17 against 11.11, a
polytrope's energy drifts 2.0% against 2.0%. A shearing sheet's epicycles keep their energy to
$`6\times10^{-5}`$ either way. With a symmetric limiter the lower face of a cell first took
the plain density and the upper face this one; both take it now. `physics.isentropicRecon`
(link key `isen`) turns it off.

## 9. Equations of state

`src/eos/`. Every equation of state implements the same four WGSL functions, so the hydro
passes do not depend on it:

```wgsl
fn eos_pressure(rho: f32, eint: f32) -> f32;      // eint = internal energy density
fn eos_sound_speed(rho: f32, p: f32) -> f32;
fn eos_eint_from_p(rho: f32, p: f32) -> f32;
fn eos_temperature(rho: f32, eint: f32) -> f32;
```

Two more return pairs, so that an equation of state which solves for its temperature does so
once for both: `eos_prim(rho, eint)` gives a cell's pressure and sound speed, and
`eos_face(rho, p)` a face state's sound speed and internal energy.

The equation of state is chosen when the shaders are assembled, and each module has a
float64 TypeScript mirror for testing (`tests/unit/eos*.test.ts`).

| Equation of state | Pressure | Notes |
|---|---|---|
| **Ideal gas** (`ideal.wgsl`) | $`p = (\gamma - 1)\rho e`$ | $`c_s^2 = \gamma p/\rho`$ |
| **Barotropic** (`barotropic.wgsl`) | $`p = K\rho^{\Gamma}`$ (isothermal for $`\Gamma = 1`$) | The energy equation is not used; $`\rho e`$ follows the density; HLL fluxes |
| **Liquid** (`liquid.wgsl`) | $`p = c_0^2(\rho - \rho_0)`$ | Weakly compressible: a constant, chosen sound speed; the pressure takes both signs; no energy equation; HLLC fluxes (below) |
| **Gas + radiation** (`gas_radiation.wgsl`) | $`p = \dfrac{\rho k T}{\mu m_u} + \tfrac13 a T^4`$ | $`T`$ is the root of $`\rho e = \dfrac{\rho kT}{(\gamma - 1)\mu m_u} + aT^4`$, from 8 Newton steps started above the root (`quartic.wgsl`) |
| **Degenerate** (`degenerate.wgsl`) | $`p = p_\text{deg}(\rho) + p_\text{ion} + p_\text{rad}`$ | Zero-temperature electrons at any degree of relativity ([Chandrasekhar 1935][Chandrasekhar1935]), plus ideal ions and radiation |
| **White dwarf matter** (`helmholtz.wgsl`) | $`p = p_\text{deg}(\rho) + p_{e,\text{th}} + p_\text{ion} + p_\text{rad} + p_\text{C}`$ | The electron-positron gas at any degeneracy and temperature from a table of its free energy, in the manner of Helmholtz ([Timmes & Swesty 2000][TimmesSwesty2000]); ideal ions, radiation, and the ions' Coulomb energy |
| **Ionizing hydrogen** (`ionizing.wgsl`) | $`p = \dfrac{\rho k T}{m_u}\left[h(1 + x) + o\right]`$ | Ionized fraction $`x`$ of the hydrogen from Saha equilibrium ([Saha 1920][Saha1920]); $`h`$ and $`o`$ are the particles per $`m_u`$ from hydrogen nuclei and from the other species |
| **Hydrogen and helium** (`hhe.wgsl`) | $`p = \dfrac{\rho kT}{m_u}\dfrac{\partial h}{\partial\ln\rho} + \tfrac13 aT^4`$ | H₂, H, H⁺, He, He⁺, He⁺⁺ and electrons in chemical equilibrium, from a table of their free energy $`h = F m_u/kT`$ (the model of [Tomida et al. 2013][Tomida2013]); the radiation term is optional |
| **Rock, iron and water** (`aneos.wgsl`) | $`p(\rho, T)`$ and $`e(\rho, T)`$ of ANEOS, tabulated | Condensed matter with melting and vaporization, one material or two in a cell; §25 |

**Degenerate electrons.** With $`x = p_F/(m_e c)`$ the dimensionless Fermi momentum and
$`n_e = \rho Y_e/m_u`$,

```math
p_\text{deg} = \frac{\pi m_e^4 c^5}{3h^3}\left[x(2x^2 - 3)\sqrt{1 + x^2} + 3\sinh^{-1}x\right],
\qquad x^3 = \frac{3h^3 n_e}{8\pi m_e^3 c^3}.
```

The density scale follows from $`Y_e`$, the electrons per nucleon, taken from the problem's
composition. Each part is thermodynamically consistent on its own, so the sound speed is
$`c_s^2 = dp_\text{deg}/d\rho + \Gamma_1 p_\text{th}/\rho`$. For $`x < \tfrac12`$ the closed forms
cancel to nothing in single precision, so the pressure and energy are summed as series in
$`x^2`$ there. The series agree with the closed forms to $`10^{-7}`$ for $`0.3 \le x < 0.5`$.

**White dwarf matter.** The electrons' thermal free energy is tabulated in
$`\log\rho Y_e`$ and $`\log T`$ with its first derivatives and interpolated by bicubic Hermite
polynomials, so their pressure and energy, being its derivatives, satisfy the first law
(`tools/eos/NOTES.md` has the table's construction and accuracy). The temperature comes from
the energy by Newton's method, started from a table of the inverse.

The ions' Coulomb energy comes from the free energy of the one-component plasma, as Helmholtz
has it (the fits of Yakovlev & Shalybkov 1989). With the coupling
$`\Gamma = \bar Z^2 e^2/(a kT)`$ and $`a`$ the ion-sphere radius, the energy per ion is
$`kT\,u(\Gamma)`$ and the pressure a third of the energy density:

```math
u = \begin{cases}
a_1\Gamma + b_1\Gamma^{1/4} + c_1\Gamma^{-1/4} + d_1, & \Gamma \ge 1\\
-\tfrac{\sqrt3}{2}\Gamma^{3/2} + a_2\Gamma^{b_2}, & \Gamma < 1
\end{cases}
\qquad a_1 = -0.898 .
```

The term $`a_1\Gamma kT`$ does not depend on temperature (it is the ion spheres' electrostatic
energy) and joins the zero-temperature electrons: it lowers the cold pressure by
$`0.0046\,\bar Z^{2/3}`$ of the electrons' where they are relativistic (1.5% for carbon) and by
more where they are not, so a 1 M☉ carbon-oxygen white dwarf is 6% smaller
([Hamada & Salpeter 1961][Hamada1961]). The rest is thermal, positive, and rises with
temperature, which the solve for $`T`$ needs; near crystallization it more than doubles the
ions' heat capacity. In thin matter the cold term alone would make the pressure negative
(below $`0.18\,\bar Z^2/Y_e`$ g/cm³), and fully ionized ion spheres are the wrong model there in
any case. Helmholtz drops the Coulomb terms where the total pressure would be negative; here
the whole Coulomb free energy is faded by $`w = r/(1 + r)`$, $`r = \rho/\rho_0`$, with
$`\rho_0`$ 64 times that density (1100 g/cm³ for carbon-oxygen). The fade is a function of
density applied to the free energy, so pressure and energy remain consistent. The white dwarf
builder and its settling use the same cold curve; built on the electrons' pressure alone, a
star would read the missing pressure as heat (10⁸ K and more). `physics.eosParams.helmCoulomb`
(link key `coulomb`) multiplies the whole term; 0 switches it off. Every white dwarf preset
uses this equation of state. The plain degenerate one remains a choice (`eos=degenerate`) and
is about twice as cheap per step where nothing burns (`tools/eos/NOTES.md`), at the price of
reading shocked gas too cold. The burning network was
trained with the heat capacity of the gas without it, a difference of a few percent at burning
temperatures, where $`\Gamma`$ is near one.

**Hydrogen and helium.** From molecular clouds to stellar envelopes the gas is hydrogen and
helium in seven forms, and what it costs to turn one into another is most of its heat capacity:
4.48 eV to dissociate an H₂ molecule, 13.6 eV to ionize an atom, 24.6 and 54.4 eV for helium.
The module is one free energy, that of an ideal mixture (`src/eos/hhe_model.ts`): each species
has its translational states and its internal ones (for H₂ the rotation, with ortho and para
hydrogen held at 3:1, and the vibration), and energies are counted from the ground states of H₂
and He, so that the energy is zero at $`T = 0`$. The numbers of the species are those that
minimize it at fixed nuclei, which gives the dissociation and Saha equations, solved for one
unknown, the electron density. Because the free energy is at a minimum, its first derivatives
need none of the composition's:

```math
\frac{p}{\rho} = \frac{kT}{m_u}\frac{\partial h}{\partial\ln\rho},\qquad
e = -\frac{kT}{m_u}\frac{\partial h}{\partial\ln T},\qquad h = \frac{F m_u}{kT}.
```

$`h`$ is tabulated in $`\log\rho`$ ($`10^{-20}`$ to $`10^3`$ g/cm³, 0.1 dex) and $`\log T`$ (10 to
$`10^8`$ K, 0.01 dex) with its exact first derivatives and its cross derivative, and
interpolated by bicubic Hermite polynomials: the lookup and the solve for the temperature are
the white dwarf module's (`eos/table.wgsl`). Two things differ. $`h`$ itself runs to ±100,
which single precision would round to $`10^{-4}`$ of the pressure and $`10^{-2}`$ of the sound
speed, so the table holds its differences between neighbouring nodes instead (only derivatives
are ever needed). And Newton's method starts from a table of the gas's own inverse at the
table's full resolution, which is what carries it through dissociation and ionization, where
the energy rises steeply at nearly constant temperature; as the shader runs it (at most 10
evaluations, a last step under $`3\times10^{-4}`$ in $`\ln T`$ taken without another), it takes
1.1 evaluations on average and 3 at most on the unit test's 1200 solves below 0.1 g/cm³
(`tests/unit/hhe.test.ts`, which asks for under 3 and under 8). Between the nodes the table gives the model's pressure to $`6\times10^{-5}`$, its
energy to $`4\times10^{-4}`$ and $`c_s^2`$ to 0.6% at worst, 0.15% at the 99th percentile (on
the fronts at the lowest densities, where they are sharpest; the 1500 states of
`tests/unit/hhe.test.ts` and the 4000 of `tools/eos/NOTES.md`); on the GPU, in single precision, pressure to energy and
back agrees to $`10^{-5}`$.

$`\Gamma_1`$ is 5/3 below 50 K, where H₂ does not turn, 7/5 at a few hundred kelvin, and falls
to 1.1 where H₂ dissociates (2000 K at $`10^{-8}`$ g/cm³) and again where hydrogen ionizes:
below the 4/3 a self-gravitating ball needs, which is what sets off the second collapse of a
protostar (the `second-collapse` preset). The adiabat of gas that turned opaque at
$`10^{-13}`$ g/cm³ and 10 K reaches 2060 K at $`10^{-8}`$ g/cm³, the track of
[Masunaga & Inutsuka (2000)][Masunaga2000].

A table is for one composition; two ship, the code's solar gas (X = 0.7381, its metals
counted as helium) and hydrogen, chosen by `physics.composition`, and a run with another is
refused. The radiation term $`aT^4`$ is analytic and optional
(`physics.eosParams.hheRad`, link key `hherad`): 1 for a star's interior, 0 for a cloud, which
its radiation leaves. What the model lacks sets where it holds: there are no degenerate
electrons and no Coulomb terms, and pressure ionization is a stand-in, each bound electron
costing $`kT\rho/\rho_\text{pi}`$ with $`\rho_\text{pi} = 0.33`$ g/cm³ (hard spheres of a Bohr
radius). That term is part of the free energy, so pressure and energy stay consistent; it
leaves gas below 0.01 g/cm³ alone and ionizes it above about 1 g/cm³, as a star's interior is
(bare Saha equations leave a third of the hydrogen at the Sun's center neutral). So the table
holds below about 0.1 g/cm³ at any of its temperatures, and above that where the gas is hot
enough to be ionized. Cool gas denser than about 1 g/cm³ is outside it: there the pressure
need not rise with temperature or density, and the sound speed has a floor.
`tools/eos/NOTES.md` has the model's choices and the table's construction.

**Saha hydrogen.** The ionization fraction solves

```math
\frac{x^2}{1 - x} = \frac{m_u}{X\rho}\left(\frac{2\pi m_e kT}{h^2}\right)^{3/2} e^{-\chi/kT},
\qquad \chi = 13.6\ \text{eV},
```

where $`X`$ is the hydrogen mass fraction, and the internal energy includes $`\chi`$ per ionization. $`T`$ comes from Newton's method kept
inside a bracket. The adiabatic index $`\Gamma_1`$ is computed exactly and dips below 1.3 in the
ionization zone.

**Weakly compressible liquid.** For flow that should be incompressible, the explicit scheme is
kept and the fluid is given an artificial sound speed $`c_0`$, about ten times its fastest flow
(the approach of smoothed-particle and lattice-Boltzmann codes):

```math
p = c_0^2\,(\rho - \rho_0), \qquad c_s = c_0 .
```

The density then departs from $`\rho_0`$ by about $`M^2`$ ($`0.78\,M^2`$ in the Gresho vortex,
at Mach numbers from 0.1 to 0.01), which is the error of calling the flow incompressible; the
diagnostics panel shows the largest departure. The cost is the timestep, which follows $`c_0`$:
$`1 + 1/M`$ times the steps the flow alone would need.

- *Why not an isothermal gas with a large sound speed.* Its pressure is $`c_0^2\rho`$, a hundred
  times the differences of order $`\rho v^2`$ that drive the flow at Mach 0.1. The linear form
  stores only the departure, as one exact subtraction of two nearby single-precision
  densities.
- *The pressure takes both signs.* A run with this equation of state has a negative pressure
  floor, which the shared passes read as "do not floor a pressure, divide by one, or judge a
  jump against one" (`p_signed`, `hydro/common.wgsl`): the shock detector and the limiter's
  smoothness scale take $`\rho c_0^2`$ in its place, the predictor's positivity check tests the
  density alone, and the refinement indicator measures pressure ripples against
  $`\rho c_0^2`$. Magnetic fields, self-gravity and comoving coordinates are refused with it.
- *HLLC, not HLL.* A barotropic gas has no contact wave, and takes HLL. But HLL also has no
  shear wave: it damps a jump in the transverse velocity at the sound speed, which in slow
  flow is most of the dissipation. A Gresho vortex at Mach 0.1 on $`64^2`$ cells loses 8.4% of
  its kinetic energy in four turns with HLL on scaled velocity jumps and 2.2% with the
  low-Mach HLLC. The liquid therefore keeps HLLC (blended toward HLL beside flagged shocks, as
  for a gas with an energy), and then loses what an ideal gas at the same Mach number loses, to
  four digits: 1.0%, 2.5% and 6.0% by $`t = 2`$ at Mach 0.1, 0.03 and 0.01.
- The "internal energy" carried is the elastic energy of compression,
  $`\rho_0 c_0^2\,[(1+x)\ln(1+x) - x]`$ with $`x = p/\rho_0 c_0^2`$, to third order in $`x`$, so
  that the total energy is conserved as far as the scheme conserves it. The temperature is
  shown as $`c_0^2`$.

Measured (`src/validation/liquid.ts`, an Apple GPU): the viscous Taylor-Green vortex decays at
1.0035 times $`4\nu k^2`$ on 64 cells per wavelength; the lid-driven cavity at Reynolds number
100 on $`128^2`$ cells is within 0.005 of the lid speed of [Ghia, Ghia & Shin (1982)][Ghia1982]
on the vertical centerline; on a three-level mesh a vortex keeps its mass to $`10^{-6}`$ and its
momentum to $`10^{-7}`$ of the sum of its magnitudes; the cylinder wake preset at Reynolds number
100 sheds at a Strouhal number of 0.174 with 16 cells across the cylinder (0.165 in an
unbounded stream; the box's periodic sides are 16 diameters apart). Presets: Taylor-Green
vortex, double shear layer, cylinder wake, lid-driven cavity (liquid), Gresho vortex.

**Implicit pressure for the liquid.** With the pressure explicit, a liquid at Mach number
$`M`$ takes $`1 + 1/M`$ times the steps its flow needs. The implicit step (`src/pressure/`,
the setting *Pressure: implicit*, `psplit=implicit` in a link) takes the flow's own: advection,
viscosity and driving are explicit as before, and the sound is taken implicitly. In backward
Euler form, which the time-centered step below generalizes,

```math
\rho' = \rho - \Delta t\,\nabla\cdot\mathbf{m}', \qquad
\mathbf{m}' = \mathbf{m}^* - \Delta t\,\nabla p', \qquad
p' = p(\rho) + c_0^2\,(\rho' - \rho),
```

with $`\mathbf{m}^*`$ the momentum the explicit update left. Eliminating $`\rho'`$ and
$`\mathbf{m}'`$ gives one equation for the new pressure,

```math
\frac{p'}{c_0^2\,\Delta t^2} - \nabla^2 p' = \frac{p(\rho)}{c_0^2\,\Delta t^2} - \frac{\nabla\cdot\mathbf{m}^*}{\Delta t},
```

which becomes the pressure Poisson equation of a projection method as $`c_0 \to \infty`$. A
step is then:

1. *Rates and clock.* The sound speed enters the timestep divided by the acoustic steps a step
   may span (a setting, 1000 by default), so it only bounds the step of a liquid at rest.
2. *Predictor.* The half step of advection alone: the acoustic terms are dropped, and the
   velocity takes the gradient of the last step's solved pressure in their place. Without that
   gradient the face velocities are off by half a step of the pressure force, and a Gresho
   vortex on $`32^2`$ cells keeps 0.78 of its kinetic energy by $`t = 2`$ where it keeps 0.97
   with it (float64 prototype, `src/reference/psplit.ts`).
3. *Update,* with the advective flux (`hydro/riemann/advect.wgsl`): the mean of the two
   sides' $`(\rho, \rho\mathbf{v}, E)\,v_n`$ less a local Lax-Friedrichs term of speed
   $`|v_n|`$. Between a state and its image in a wall it is zero.
4. *Source* (`pressure/source.wgsl`): the right-hand side, from the old density and the
   divergence of face momenta (each the mean of its two cells'), and the finest level's
   coefficients.
5. *Solve* (`pressure/multigrid.ts`): V-cycles, 3 by default, warm-started. Red-black
   Gauss-Seidel, restriction by averaging, trilinear prolongation, as for gravity (§13), with
   three differences. The operator is $`\alpha p - \nabla\cdot(\beta\nabla p)`$ with a texel
   of coefficients per cell on every level: a face nothing crosses (a wall of the box, the
   surface of an obstacle, the face of an inlet) has $`\beta = 0`$, and a coarse face takes the
   mean of the fine faces that tile it, so obstacles reach every level with no mask. The
   levels are one cell deep in 2D. And the coarsest level (at most 16 cells) is solved in one
   invocation, which also, where no face of the box is open, sets the mean of the pressure from
   the sum of the equation: Gauss-Seidel would only creep toward it when $`\alpha h^2`$ is
   small. An open (outflow) face holds $`p = 0`$. After the cycles the residual is measured
   and tallied against a tolerance ($`10^{-4}`$ of the largest right-hand side), to be read on
   request; the number of cycles is fixed, because sizing later cycles from the residual by
   indirect dispatch (as the block gravity solver does) cost more in passes than it saved in
   cells on every grid tried.
6. *Correct* (`pressure/correct.wgsl`): the momentum takes the difference of face pressures
   and the density the difference of face mass fluxes $`\bar m^*_f - \Delta t\,\partial_n p'`$,
   so mass and momentum are conserved to round-off. The density the explicit update advected is
   not used.

*Time-centered* (`pressure/theta.wgsl`; `splitCentered` in `src/reference/psplit.ts`). Backward
Euler damps sound it resolves (a wave at an acoustic Courant number of 0.5 lost 14% of its
amplitude a period) and makes the whole step first order. The step's pressure force is instead
that of $`p_\theta = \theta p' + (1-\theta)p^n`$, and the divergence its solve sees that of
$`\theta\mathbf m' + (1-\theta)\mathbf m^n`$, with $`p^n`$ the last step's solved pressure and
$`\mathbf m^n`$ the face momenta the step starts from. Eliminating as before,

```math
\frac{x}{\theta^2 c_0^2\Delta t^2} - \nabla^2 x
= \frac{p(\rho)}{\theta\,c_0^2\Delta t^2} + \frac{(1-\theta)\,p^n}{\theta^2 c_0^2\Delta t^2}
- \frac{\theta\,\nabla\cdot\mathbf m^* + (1-\theta)\,\nabla\cdot\mathbf m^n}{\theta\,\Delta t},
\qquad x = p_\theta,
```

the same operator with $`\alpha/\theta^2`$ for $`\alpha`$: one solve a step, started from
$`p^n`$, with the same three cycles (a larger $`\alpha`$ only helps the smoother). The correction
takes its force from $`x`$ and the mass fluxes
$`\theta(\bar m^*_f - \Delta t\,\partial_n x) + (1-\theta)\bar m^n_f`$, and hands
$`p' = (x - (1-\theta)p^n)/\theta`$ on to the next step and the predictor. Each cell takes

```math
\theta = \max\big(\chi,\ \min(1,\ \tfrac12 + 0.1\,C)\big),\qquad C = \frac{c_s\Delta t}{\Delta x},
\qquad \chi = \min\big(1,\ \tfrac12 + 4\,\Delta t\max(0, -\nabla\cdot\mathbf v)\big),
```

from its own acoustic Courant number and compression. Sound the step resolves is carried nearly
undamped (Crank-Nicolson at $`\theta = \tfrac12`$); sound it does not resolve, $`C \ge 5`$, which
is every flow at a low Mach number, is damped by backward Euler as before. ($`\theta = \tfrac12`$
everywhere leaves grid-scale sound ringing: a vortex at Mach 0.03 in the float64 prototype did
not converge in the step.) $`\chi`$ is for shocks, where Crank-Nicolson rings: with $`\theta`$
near $`\tfrac12`$ the gas behind the Sod tube's shock overshot its speed by 40% and oscillated
behind it; with $`\chi`$, cells a shock compresses within the step take backward Euler and the
rest stay centered. Where the step resolves sound, the predictor's half-step pressure also takes
the compression $`-\rho c^2\nabla\cdot\mathbf v`$, weighted by $`2(1-\theta)`$
(`hydro/predictor.wgsl`), so the advected energy's flux sees the pressure half way through the
step. The cost: no pass, solve or cycle more; a few more texture reads in the source, the
correction and the predictor; 20 bytes a cell more for the liquid and 12 for a gas (the pressure
and face momenta or velocities at the start of the step, the solved pressure).

Measured in the float64 prototype (each run against the same scheme at a fortieth of the
default step, so that only the error in the step is counted): a simple wave of amplitude 0.05 in
gas moving at Mach 0.42 converges at second order in the step (the error falls 3.9 and 3.6 times
a halving; backward Euler: 2.0), and its error is a sixth of backward Euler's at the default
Courant number of 0.8 and a twelfth at 0.4. A vortex at Mach 0.03 converges at second order when the grid and the step are
refined together (3.6 times a halving; backward Euler 2.2 times): what remains first order at a
fixed grid is a mixed term of the step and the cell size. The Sod tube's error is first order
in the step with either scheme, as a shock's is.

For a gas whose pressure is a function of its density the mass flux has to be corrected like
this: the published form of the method ([Kwatra et al. 2009][Kwatra2009]), which leaves the
density as advection made it, is explicit in such a gas's pressure and unstable for it
(`tests/unit/psplit.test.ts`).

*An ideal gas* (uniform grid; `pressure/source_gas.wgsl`, `pressure/correct_gas.wgsl`) takes
the published scheme with one addition. The equation is for $`p' = p - p_\text{ref}`$, the
mean pressure the run started with, so that single precision holds the part that varies:

```math
\frac{p'}{\rho c^2\,\Delta t^2} - \nabla\cdot\left(\frac{1}{\rho}\nabla p'\right)
= \frac{p^a - p_\text{ref}}{\rho c^2\,\Delta t^2} - \frac{\nabla\cdot\mathbf{v}^*}{\Delta t},
```

with $`p^a`$ the pressure advected over the step (first-order upwind), $`\rho c^2`$ from the
state at the start of the step, a face's velocity the density-weighted mean of its two
cells' and its $`1/\rho`$ from the mean of their densities. The correction takes
$`-\Delta t\,\nabla p`$ from face pressures
$`p_f = (\rho_R p_L + \rho_L p_R)/(\rho_L + \rho_R)`$, puts
$`-\Delta t\,\nabla\cdot(p\,\hat{\mathbf v})`$ into $`E`$ and
$`-\Delta t\,p\,\nabla\cdot\hat{\mathbf v}`$ into the internal energy, with
$`\hat v_f = v^*_f - \Delta t\,(p_R - p_L)/(\rho_f\Delta x)`$: all differences of face values.

The addition: advection carried the mass, momentum and energy on the predictor's face
velocities $`u_f`$, which are divergence-free only to first order in the step, and unlike a
liquid a gas has nothing that pushes a density error back (it is an entropy error). So each
face also moves $`(\hat v_f - u_f)`$ times the upwind advected state. The density and the
energy then see the divergence the solve left. In the float64 prototype that takes the density
error of a Gresho vortex from 7% to 0.07%, and removes the ceiling of some 20 acoustic steps
that the published form needs to stay stable below Mach 0.01. Time-centered, the solve is for
$`e = \theta(p' - q)`$, the change of the pressure the step's force is from (with $`\theta`$ the
cell's, the operator stays symmetric:
$`(\alpha/\theta^2)\,e - \nabla\cdot(\beta\nabla e)`$ equals the backward Euler right-hand side
with $`\theta\nabla\cdot\mathbf v^* + (1-\theta)\nabla\cdot\mathbf v^n`$ for the divergence, over
$`\theta`$). And the velocity that carries the mass, momentum and energy through a face, and
that the pressure and the fields do their work on, is
$`v_c = v^n_f + \chi_f(\hat v_f - v^n_f)`$, with $`\chi_f`$ the larger of the two cells' switches:
the mean of the old and the corrected face velocity in smooth flow, whatever $`\theta`$, the
corrected one in a shock. Carried on $`\hat v_f`$ alone, as backward Euler has it, the step was
first order even where the flow is all but incompressible; with the mean, the kinetic energy a
face's kick and pressure difference give its gas is the work they do on it. An open face carries no pressure
gradient (the gas leaves at its own pressure and velocity). The timestep is also limited by
the pressure's own acceleration, $`C\sqrt{\Delta x\,\rho/|\nabla p'|}`$, which is what bounds
the first step of a shock tube at rest. For a gas the flow's limit and this one act together:
the step is also at most the solution of $`(v + a\,\Delta t)\,\Delta t = C\,\Delta x`$, with $`v`$ and $`a`$ the largest
rate and acceleration on the grid: the correction carries the gas at up to the velocity the
step ends with (in a shock; the mean of it and the starting one elsewhere). Gas moving at the flow's limit that the
pressure also pushes would otherwise cross more than a cell in a step, with no sound speed in
the rate to cover it: a hot spot in dense gas moving at 0.7 of a cell a step emptied the cells
ahead of it in its first step (`psplit-moving-jump`).

*The solve is for the change of the pressure*, $`d = p - q`$, from the last step's solved
pressure $`q`$, which the correction keeps: the right-hand side gains
$`\nabla\cdot(\tfrac1\rho\nabla q)`$ and the solve starts from zero. Its terms are then
differences of neighboring values, each good to single precision of its own size. Not from the
state's own pressure: at Mach 0.001 that is noise as large as the flow's pressure, and its
Laplacian blows the run up. The reference pressure is taken only where the pressure varies by
less than a factor of 4 over the grid; a stratified gas has none.

*A uniform field, for the gas.* Each face's velocity is kicked by $`\Delta t\,g`$, so a face is
at rest where $`p_R - p_L = g\,\Delta x\,(\rho_L + \rho_R)/2`$: the trapezoid balance of the
explicit scheme's well-balanced reconstruction (§8). With the density-weighted face pressures
a cell of a column in that balance has $`p_\text{hi} - p_\text{lo} = \rho g\,\Delta x`$ exactly,
which the cell's kick cancels; a wall's face pressure is the cell's carried half a cell down
the field; the field's work is $`g`$ times the mass the cell's two faces carry, halved (none at
rest). The predictor's lagged pressure force is formed the same way, face by face (a centered
difference over the cell's own density is off by the density's curvature, 1.6% where it falls
by $`e`$ in four cells, and that alone drove the steep column to Mach 0.1). Columns whose
density falls by $`e`$ every 16 cells and every 4 (a factor of $`9\times10^6`$ in all) then
hold at a Mach number of $`1.2\times10^{-6}`$ through a sound crossing, with the energy to
$`2\times10^{-7}`$ (`psplit-gas-balance`); the explicit scheme holds them to $`10^{-4}`$.

*Self-gravity, for the gas.* The same construction with the potential in place of
$`g\,\Delta x`$: a face is kicked by $`-\Delta t\,(\phi_R - \phi_L)/\Delta x`$, with the
half-step potential the explicit scheme's reconstruction also differences (§14), the cell's
kick is its density times the mean of its two faces', and the work is each face's kick times
the mass it carries (the update's own flux form of the work, §14, is off with implicit pressure,
on the uniform grid as on blocks: left on, it counted the work on moving gas twice, and a star
crossing the grid left its heat behind and tripled its central density in a dynamical time),
shared between its two cells by their densities: a cell takes half its
density times the face's kick times the face's velocity, the kinetic energy its half of the kick
gives it. (Halved evenly, the first cell of near-vacuum beyond a star's surface took half the
work done on the star's gas: with the gas around a star at $`10^{-5}`$ of its central density
or less, that cell was $`3\times10^5`$ times hotter after one step and the run was not finite
after ten. `psplit-gas-star-vacuum` holds the star at a contrast of $`10^7`$.) The potential is
solved for the advected density, before the pressure passes, and the faces are kicked with it as
it is: the gas the kicks act on is then the gas the potential is of. (Kicked with the potential
of the step before extrapolated half a step, a star moving 0.8 of a cell a step was pulled back
toward where its field lagged it, and lost 10% of its momentum in a dynamical time.) The timestep keeps gravity's limit,
$`C\sqrt{\Delta x/|\mathbf g|}`$: without it the columns above took five steps a sound crossing
and lost their balance. A settled polytrope on $`64^3`$ cells holds its central density to
0.5% over two dynamical times in 27 steps, where the explicit scheme takes 436 and drifts 2%;
its total energy drifts 0.8% against the explicit 0.2% (`psplit-gas-polytrope`). That is not
the steps' length: at a half, a quarter and an eighth of the Courant number it is 0.70%, 0.60%
and 0.53%, and the mass, 0.22% (the thin gas around the star), does not change with the step. Collapse and other dynamic self-gravity have not
been tested with it. A star that moves is held: at 0.3 of its central sound speed, 6.4 cells
in a dynamical time, its central density stays within 0.6%, its momentum within 0.2% and its
energy within 0.4% (`psplit-gas-star-moving`; M4 Pro, 2026-11-19). Two limits remain. On a mesh
of two levels the same star gains about 1% of its momentum in a dynamical time, as the explicit scheme does there at a
hundredth of the size with steps a tenth as long: not looked into. And two stars 8 cells in
radius, of a gas softer than their structure, set to collide in near-vacuum (random setup 70293)
collapse as they move under either scheme; the explicit one stays finite through it and the
implicit step, which is for slow flow, does not.

*Species* that ride with the gas are moved after the correction and scaled to the corrected
density: mass fractions sum to one and a uniform species stays uniform to round-off. On the
uniform grid the correction leaves the mass fluxes of the whole step, advection and correction,
in the update's record of them (`pressure/mf_on.wgsl`), and the species move with those: each
species' total mass is kept to $`10^{-7}`$, in the liquid as in the gas (`psplit-species`). On
blocks the correction does the same to each block's records of its faces' mass fluxes, which
the species pass reads (`amr/pressure/mf_on.wgsl`): a liquid's species keep their mass to
$`3\times10^{-7}`$ on a mesh that re-meshes as it runs. (Moved with the advective fluxes and
scaled, a species' mass changed by the order of the density's correction: $`2\times10^{-3}`$ in
a liquid at Mach 0.1 on the uniform grid, $`5\times10^{-4}`$ on blocks.) Species that set the equation of state,
and burning, are refused.

Measured (`psplit-gas-gresho`, `psplit-sod`; M4 Pro, 2026-11-19, time-centered): the Gresho
vortex on $`64^2`$ cells to $`t = 2`$ takes 219 steps at Mach 0.1 and 0.01 (explicit low-Mach
HLLC: 3 422 at Mach 0.1) and keeps 0.9924 of its kinetic energy (backward Euler: 0.9908;
explicit: 0.9896), with the density within $`3\times10^{-4}`$ of 1 and mass and energy
conserved to $`10^{-7}`$. At Mach 0.001 single precision shows: 0.985 kept (backward Euler:
0.974), the density off by up to 1.4% (2%; the pressure there is the difference of an energy of
$`10^6`$ and a kinetic energy of order 1). The Sod tube on 128 cells conserves mass and energy
to $`10^{-7}`$ and puts the shock within 0.6 of a cell in 38 steps (explicit: 111), with an L1
density error of 0.0097 (backward Euler: 0.016 in 37 steps; explicit: 0.004), 0.0078 at half
the Courant number (0.011). The rest is not the step's: in the prototype, at 128 cells, the
error tends to 0.007 as the step goes to zero, the pressure's part of the flux having no
upwinding but what backward Euler or the switch gives it. A shock is not what the step is for.

Measured (`src/validation/psplit.ts`, an Apple GPU): a Gresho vortex on $`64^2`$ cells to
$`t = 2`$ takes 212 steps at Mach 0.1, 0.01 and 0.001 alike, where the explicit scheme takes
3 411 and 32 198 at the first two (and by its timestep's scaling ten times that again at the third), and keeps 0.9922 of its kinetic energy at each (explicit, Mach 0.1:
0.9896), with the density within $`0.8\,M^2`$ of the reference. The Taylor-Green vortex decays
at 1.0026 times the exact rate in 237 steps (explicit: 1.0035 in 1 837). A sound wave resolved
at an acoustic Courant number of 0.5 returns after a period within 0.18% of a wavelength at
0.984 of its amplitude, in 129 steps ($`\theta = 0.55`$; backward Euler: 0.858); at a Courant
number of 10 ($`\theta = 1`$) it is damped to 0.125, its density never above the starting
amplitude, as before. The solver on its own reaches a manufactured solution to $`10^{-5}`$ in eight
cycles from zero, with an obstacle and variable coefficients, cutting the residual 5 times a
cycle in 2D.

*A uniform field.* Its kick is not the update's but the correction's, formed on the faces:
a face's mass flux takes $`\Delta t\,g`$ times the mean of its two cells' densities, the cell's
momentum the mean of its two faces' kicks, and a wall's face pressure is the cell's carried
half a cell down the field. A column whose pressure differences obey the trapezoid rule
$`p_{j+1} - p_j = g\,\Delta x\,(\rho_j + \rho_{j+1})/2`$ then has no mass flux and no change of
momentum, to round-off: it holds at a Mach number of $`10^{-6}`$ through 60 sound crossings,
and a column started at uniform density settles to that balance (`psplit-balance`). The
solved pressure starts from the state's, $`p(\rho)`$, so the first step is balanced too.

*On the adaptive mesh* (`src/amr/pressure.ts`, `amr/pressure/*.wgsl`). The same step on the
leaves of the tree. The equation is solved by multigrid over the tree's levels, with three
things particular to it.

- *The operator is conservative across levels.* Where the region beyond a fine cell's face is
  coarser, the fine cell's ghost lies on the line between the two centers,
  $`p_g = p_i + \tfrac23(p_c - p_i)`$. A coarse cell beside a refined block takes through that
  face the mean of the fine faces' fluxes, read from the fine cells: the same numbers. The
  correction then moves mass and momentum with those same face values, so both are conserved
  across levels to round-off. Using a merely consistent flux on the coarse side instead would
  leave a density error at every refinement boundary each step, and a liquid turns a density
  error into a pressure $`1/M^2`$ times as large.
- *Full approximation.* The pressure lives on every block, parents holding the mean of their
  children. Each level relaxes the plain operator (the parent's value across a refined face),
  with the difference between the plain and the conservative operator, frozen when the finer
  level was last restricted, added to its right-hand side. A cycle goes from the finest level
  down (two red-black sweeps, then restriction of the pressure and the residual) to the root
  level, which is a uniform grid and is solved by the uniform grid's multigrid (two V-cycles),
  and back up (the parent's change added to its children, piecewise constant, then two
  sweeps). At convergence the leaves satisfy the conservative equation. Three cycles a step.
- *The solved pressure rides in the state.* A liquid does not use its internal-energy slot, so
  the correction stores the solved pressure there. It is then regridded and its ghosts filled
  with the rest of the state, the predictor takes its gradient, and the next solve starts from
  it. (The pressure of the state's own density would not do: at Mach 0.001 its single-precision
  noise is as large as the flow's pressure, and the run blows up within ten steps.)

*Refinement in a liquid.* Density and pressure are nearly uniform in a liquid whatever it
does, so for one the refinement indicator (§17) also takes the same normalized second
derivative of each velocity component, with ripples measured against the largest speed of the
initial conditions (so that liquid at rest, whose velocity is round-off, asks for nothing).
A gas with implicit pressure, which is there for low Mach numbers, takes it too.
It finds under-resolved shear layers and the edges of vortices: a double shear layer a fifth
of a root cell thick is refined to three levels, at the start and as it moves
(`psplit-amr-shear`). A layer three cells thick reads 0.3, under the default threshold of 0.8,
and the kinks of the Gresho vortex's velocity read 0.6 to 0.7.
The mesh a run starts on is chosen the same way (`amr/init_cpu.ts`), with the velocity's
ripples measured against the largest speed sampled in each block, and a liquid's pressure
against $`\rho c_0^2`$ as on the GPU. (Before that the starting mesh measured a liquid's
pressure against its own magnitude, which is near zero: it refined wherever the pressure
curved at all, and the first regrid merged those blocks again.)

*An ideal gas on blocks* (`amr/pressure/*_gas.wgsl`). The uniform grid's gas step, level by
level: the solve is for the change from the carried solved pressure, which has its own atlas
with ghosts (`lag_fill.wgsl`) because the gas's internal-energy slot is in use; the face
velocities the update keeps go to the source; and the carried fluxes of the correction are
matched at refinement faces like the liquid's. Its coefficients vary from cell to cell:
$`\alpha = 1/(\rho c^2 \Delta t^2)`$ and $`1/\rho`$ on faces. A leaf takes them from the
advected state's primitives. A parent takes the means of its children's, restricted level by
level once a step (`coef_restrict.wgsl`). The primitives of a parent would not do: a block
whose children are refined themselves holds a placeholder state, and the levels below such a
block would solve another equation. With them the cycle reduced the residual by 0.65 a cycle
on a three-level mesh and a run at Mach 0.01 blew up within 25 steps; with restricted
coefficients it is 0.04 a cycle, as on a two-level mesh and as for the liquid. Two things
follow the uniform grid: an open face carries no pressure gradient, on the root level's solver
too, and the carried pressure's acceleration limits the step (`amr/pressure/acc_gas.wgsl`,
in the primitives pass), without which a shock tube starts with one step far too long. The
Sod tube on a two-level mesh that follows its waves takes 73 steps (the uniform grid at the
same finest cell: 75), conserves mass and energy to $`10^{-7}`$, and its L1 density error is
0.0061 (backward Euler: 0.0093 in 71 steps; `psplit-amr-sod`).

*Time-centered on blocks* (`amr/pressure/theta.wgsl`). The uniform grid's step, with three
things particular to the mesh, and no pass, solve or cycle more.

- *A gas's theta, cell by cell.* The source writes each leaf cell's switch $`\chi`$, and every
  pass that forms the cell's $`\theta`$ takes it from there with the cell's sound speed and
  its own block's cell size. The solve is for $`e = \theta d`$ with $`\alpha/\theta^2`$ on the
  diagonal, and the parents take the means of their children's $`1/(\rho c^2\theta^2)`$, so the
  coarse levels solve the leaves' equation; the source runs before that restriction now. The
  liquid takes one $`\theta`$ for the whole mesh, that of the finest cell it may have: with
  $`\theta`$ varying from level to level it would enter the faces' coefficients, and the
  coarse levels' plain operator of a liquid, $`\beta = 1`$, would solve another equation.
- *The state the step started from.* The source keeps it, $`(\rho, \rho\mathbf v)`$ of every
  leaf cell, for the correction (the correction writes over it). A face's starting velocity or
  momentum is formed from its two cells, the coarse cell and the fine one at a refinement face,
  so the two sides of every face, fine faces included, form the same bits, and mass, momentum and
  energy stay conserved across levels.
- *Refinement faces carry with the larger of their two cells' $`\theta`$.* There the starting
  velocity, from the two cells' centers, is not formed as the face's own velocity is, from the
  fine cell and its ghost, and the difference is sound that the step must damp. Carried on the
  mean of the two, a gas vortex at Mach 0.01 across refinement boundaries kept a density error
  of $`6\times10^{-3}`$ (backward Euler: $`3.3\times10^{-3}`$); with the larger $`\theta`$,
  which is 1 where sound is not resolved, $`3.0\times10^{-3}`$. On the corrected velocity alone
  at every refinement face, the Sod tube's waves, which the step resolves, were smoothed there
  enough that the mesh stopped following them.

Measured on blocks (`psplit-amr-*` and the block halves of the other tests; M4 Pro,
2026-11-19), against backward Euler: the gas vortex keeps 0.99929 of its kinetic energy
(0.99919) in the same 276 steps, its residual after ten steps $`6\times10^{-5}`$
($`3\times10^{-5}`$), no solve above the tolerance; a mass moving through gas is within 0.37%
of the explicit run's velocity at an eighth of the Courant number (1.3%); the polytrope on two
levels holds its central density to 0.07% (0.22%) in 63 steps (62), its energy drifting 0.64%
(0.56%); the liquid's runs at low Mach number, where $`\theta = 1`$, are as before to round-off,
and the cylinder wake's solves end above the tolerance 0.03% of the time (0.5%). An unrefined
mesh keeps the uniform grid's kinetic energy to $`10^{-7}`$ in the same steps.

*Obstacles, inlets and a uniform field on blocks.* A face on an obstacle, in this block or the
next, on the coarser side or among the fine faces of a refined one, takes no pressure flux and
no mass; a cell held at its state sends its own momentum through. Parents hold no marks, so
the plain operator the coarse levels relax treats their cells as gas: that changes how much
the coarse levels help near an obstacle, not what the solve converges to (the leaves'
equations fix that). A uniform field's kick is formed on the faces as on the uniform grid,
with the fine faces' own densities at a refinement boundary. The cylinder wake preset on a
three-level mesh, 32 cells across the cylinder, sheds with a period of 1.40 (the uniform grid
at 16 cells across: 1.43), with 0.2% of its steps' solves ending above the tolerance
(`psplit-amr-wake`); a liquid column with a refinement boundary across it holds at a Mach
number of $`2\times10^{-6}`$ (`psplit-amr-balance`).

*A uniform field at refinement boundaries* (`psplit-amr-field`). Three things make a column
at rest stay at rest where the mesh changes level.

- The coarse cell's center is not straight across a refinement face from a fine cell's: it
  lies half a fine cell to one side, and along the field that is a pressure difference,
  $`\rho g`$ times the offset, that has nothing to do with the face. So the fine cell's ghost
  is $`p_g = p_i + \tfrac23(p_c - p_i - T)`$ with $`T`$ that difference (`ap_lift`). Without
  it, two fine cells one above the other would both have to match the coarse cell's pressure,
  and a liquid column flowed at Mach 0.01 along every boundary that runs along the field; with
  it, $`8\times10^{-6}`$. This was wrong for the liquid from the first version of the block
  solver; its test had only a boundary across the field.
- For the gas, the two cells share the face's pressure, so the fine cell's pressure carried
  half a fine cell along the field and the coarse cell's carried half a coarse cell back must
  meet there. With the face's weights as between cells of one level that fixes the ghost's
  density, $`\rho_g = (4\rho_c - \rho_i)/3`$: the face's density is then the mean along the
  line between the centers. The mesh's own interpolated ghost density left the coarse cell
  feeling 0.3% of the field.
- For the gas, $`T`$ takes the coarse cell's density, so that the fine cells' offsets cancel
  in the coarse cell's face pressure, and the carried pressure's ghosts (which the predictor
  differences) are filled the same way.

An isothermal gas column whose density falls by $`e`$ over the box then stays at Mach
$`3\times10^{-6}`$ with a boundary across the field, $`6\times10^{-5}`$ with a refined strip
from wall to wall and $`8\times10^{-4}`$ with a refined patch (the explicit scheme on the same
meshes: $`6\times10^{-4}`$, $`8\times10^{-3}`$, $`1.8\times10^{-3}`$). What the patch leaves
is at its corners, where the conditions for rest around a fine corner cell and its three coarse
neighbors disagree at first order in the cell size: there is no state of rest there. And where
a boundary meets a wall the liquid is left with Mach $`1.6\times10^{-3}`$: the mesh's
interpolation of ghost cells has no slope along the wall's normal there, and the predictor
reads those ghosts.

*The gas's own gravity on blocks* (`psplit-amr-polytrope`). As on the uniform grid: the
potential's difference across each face kicks the face velocities (the block's own ghost of
the potential beyond a block face, the fine faces' own at a refinement boundary), a wall or an
open face takes the cell's acceleration, the offset term above takes the field of the gravity
atlas, and the potential is solved for the corrected state after the pressure passes. The
polytrope preset with the star one level finer than the box around it (128 cells across at the
finest) holds its central density to 0.2% over two dynamical times in 58 steps and 3 s; the
explicit scheme takes 737 steps and 11 s and holds it to 0.8%. Mass changes by 0.23% (0.22%
explicit: the thin gas around the star leaving) and total energy by 0.6% (0.3%). Only a star
at rest is tested, as on the uniform grid.

*Point masses* (`psplit-masses`; ideal gas, uniform grid and blocks). Their pull kicks the
face velocities like self-gravity's: by the difference of their softened potential across the
face, taken half a step ahead along their motion (`gravity/ext.wgsl` forms the difference
without subtracting two large potentials), and by the cell's own pull at walls and open faces.
The masses then move through the step pulled by the corrected state. An isothermal atmosphere
in balance with a mass, its central density $`e^2`$ times the outer, stays at Mach
$`1.2\times10^{-3}`$ over a sound crossing in 40 steps (explicit: $`1.3\times10^{-3}`$ in 380)
with a tenth of the explicit scheme's energy drift; on a mesh refined at the center,
$`4.7\times10^{-3}`$, at the refined patch's corners. Three things to know:

- *The masses are given exactly what the gas was given.* The gas is kicked once, by the masses
  half a step along their way, so the masses' step is drift-kick-drift here (kick-drift-kick
  in the explicit scheme): half a step, the whole step's kick, the other half
  (`gravity/bh_kick.wgsl`). The kick is from the reduction of the gas's own kicks: each cell's
  advected density times the mean of its faces' kicks by that mass, formed face for face as
  the correction formed them, fine faces and wrap-around faces included
  (`gravity/bh_moment.wgsl`, `amr/bh_moment.wgsl`). Two masses moving through uniform gas
  then conserve the momentum of gas and masses together to $`6\times10^{-7}`$ of what they
  exchange, on the uniform grid and across refinement boundaries, as the explicit scheme does.
  (Kicked by the gas as it is at the step's two ends instead, with the usual reduction, they
  lost 2.4% at a quarter of the default Courant number and 30% at the default.) The step is
  still long for a mass that moves: at the default Courant number its velocity after a
  crossing differs from the explicit run's by 2.8% (backward Euler: 7.9%), and by 1.1%, 0.82%,
  0.47% and 0.25% at a half, a quarter, an eighth and a sixteenth of it: first order in the
  step, in the coupling of the masses and the gas (not looked into further).
- *The face that wraps around a periodic box* takes the two cells' pulls instead: the masses'
  potential is not periodic, and its difference to a point outside the box left 2% of the
  exchanged momentum unaccounted for whatever the step.
- *Sinks are refused.* What a sink takes is removed by the explicit scheme's second gravity
  kick, which this step does not run; a point mass's sink radius must be 0. And a liquid with
  point masses is refused.

The masses' work on the gas is in flux form here (kick times the mass the faces carry), where
the explicit passes use the kick form for it: right for slow flows, and what the explicit
scheme's note on cold, fast debris warns against.

Measured (`psplit-amr-*`): a Gresho vortex on a three-level mesh whose refinement boundaries
run through it (512 cells across at the finest, refined on its velocity at a threshold of
0.6) takes 273 steps to $`t = 1`$ at Mach 0.1, 0.01 and 0.001 alike, against the explicit
scheme's 12 900 at Mach 0.1, and keeps 0.9993 of its kinetic energy at each (explicit, to
$`t = 0.5`$: 0.9994); mass is conserved to $`2\times10^{-7}`$ and momentum to
$`4\times10^{-8}`$ of the sum of its magnitudes. The liquid's composite residual after three
cycles is 4 to $`10\times10^{-4}`$ of the right-hand side. That is single-precision round-off
of the operator at this cell size, not the solver: at Mach 0.1 it is $`3\times10^{-4}`$ after
2, 4 and 8 cycles alike ($`1.2\times10^{-4}`$ at 256 cells). The ideal gas on the same mesh
(`psplit-amr-gas`, Mach 0.1 and 0.01) keeps 0.9992, conserves mass to $`6\times10^{-8}`$ and
total energy to $`5\times10^{-8}`$, and its residual, of a solve for the change alone, is 1 to
$`3\times10^{-5}`$ early in the run and under $`10^{-3}`$ throughout; its density moves by
0.35% in the first steps, a sound wave the setup's pressure sheds at the refinement
boundaries. A mesh that
merges its fine blocks as it runs leaves the density within $`0.78\,M^2`$: no pressure pulse.
Without refinement the block grid reproduces the uniform grid's run to $`10^{-7}`$ in kinetic
energy. In 3D it runs and conserves; it has not been measured beyond that.

Limits. The liquid and the ideal gas only, on a Cartesian grid, without sinks, cooling,
damping zones, frames, mirror symmetry, burning or radiation transport (and the liquid without
self-gravity or point masses); on the adaptive mesh also
without per-block timesteps (one solve couples every cell at one time). Each is refused with the reason. And on
small grids the step costs more than it saves: the $`128^2`$ cavity at Reynolds number 100
takes a third of the explicit scheme's steps (viscosity sets its step, not the flow) and three
times its wall-clock time, because a V-cycle's many small passes cost more than its cells. A
step costs 6.6 explicit steps at $`512^2`$ and 4.2 at $`128^3`$ (three cycles; the Gresho vortex
of an ideal gas on an Apple M4 Pro, 200 steps, the best of three: 2.9 against 0.44 ms and 18
against 4.4 ms), against the $`1 + 1/M`$ it replaces: it pays below a Mach number of about 0.2
in 2D and 0.3 in 3D.

## 10. Optically thin cooling and heating

`src/cooling/`, `hydro/cooling.wgsl`. Two functions are available, chosen when the shaders are
assembled:

- **Power law with uniform heating:** $`\mathcal{C} - \mathcal{H} = \rho^2\Lambda_0 T^\alpha - \rho\Gamma_0`$, with no cooling below a temperature floor ($`\alpha = 1/2`$ is bremsstrahlung).
- **Interstellar:** the fit of [Koyama & Inutsuka (2002)][Koyama2002], as corrected by
  [Vázquez-Semadeni et al. (2007)][VazquezSemadeni2007]:

```math
\frac{\Lambda(T)}{\Gamma} = 10^{7}\exp\left(\frac{-1.184\times10^5}{T + 1000}\right) + 1.4\times10^{-2}\sqrt{T}\exp\left(\frac{-92}{T}\right),
```

with $`\mathcal{C} - \mathcal{H} = n^2\Lambda - n\Gamma`$, where $`n`$ is the number density of
hydrogen atoms in cm⁻³, $`T`$ the temperature in kelvin, and $`\Gamma`$ the heating rate per
atom, a parameter of the problem.

Cooling acts on the internal energy at fixed density at the end of the update, in up to 12
second-order substeps that each change the energy by at most a tenth. A substep that would
cross thermal balance stops at the balance point. A cooling time far shorter than the
timestep is therefore stable and does not overshoot. The timestep is not shortened for
cooling: a cell that cools faster than the substeps can follow loses at most 70% of its
energy in one step. In code units the temperature is $`p/\rho`$; the interstellar curve
converts it to kelvin and reads the density as atoms per cm³.

## 11. Viscosity and thermal conduction

`hydro/diffusion_core.wgsl` (shared with MHD, [§21](#21-ideal-mhd)),
`hydro/diffusion.wgsl`, `hydro/diff_rate.wgsl`. The stress and heat flux are

```math
\boldsymbol{\tau} = \rho\nu\left[\nabla\mathbf{v} + (\nabla\mathbf{v})^\mathsf{T} - \tfrac23(\nabla\cdot\mathbf{v})\mathsf{I}\right],
\qquad
\mathbf{q} = -\kappa\rho\nabla T,
```

with constant kinematic viscosity $`\nu`$ and conductivity per unit density $`\kappa`$. These
match Athena++'s `nu_iso` and `kappa_iso`
([Stone et al. 2020][Stone2020]). The temperature of an ideal gas diffuses at
$`(\gamma - 1)\kappa`$.

Both are explicit fluxes from the state at the start of the step, added to each face's
Riemann flux: $`-\boldsymbol{\tau}\cdot\hat{\mathbf{n}}`$ to the momentum and
$`-(\boldsymbol{\tau}\cdot\hat{\mathbf{n}})\cdot\mathbf{v} + \mathbf{q}\cdot\hat{\mathbf{n}}`$ to the energy. Derivatives
along the face normal are differences of the face's two cells. Derivatives across the face
are the mean of the centered differences in the two cells. The density in $`\rho\nu`$ and
$`\kappa\rho`$ on a face is the smaller of its two cells'. With their mean, a cell of thin gas
beside dense gas diffused as fast as the ratio of the densities times $`\nu`$ or $`\kappa`$,
past any timestep the diffusion limit allows, and blew up. With the smaller density no cell
diffuses faster than at its own density. Because they are part of the face
flux, they inherit its properties: both cells of a face compute the same bits, and they are
matched across refinement and time-bin boundaries like the hydro flux, so momentum and
energy are conserved to round-off. On the rim of a block face on the adaptive mesh, where the
edge ghost cells are never filled, both blocks use one-sided differences.

The heat that the stress dissipates goes into the separately evolved internal energy: the
energy that the faces carry in, less what the cell's kinetic energy gains. An obstacle face
is a no-slip, adiabatic wall at rest. The box faces keep their hydro boundary (a reflecting
face is a plane of symmetry without shear stress).

There is no implicit or super-time-stepping solve, so strong diffusion on a fine mesh is
slow ([§3](#3-the-timestep)). Viscosity makes the flux update 1.6 times as long in 2D and 1.8
times in 3D, because its cross derivatives need 4 and 8 more loads per face. Conduction alone
adds no measurable time.

Measured accuracy: the kinetic energy of a shear wave decays as $`\exp(-2\nu k^2 t)`$ to 0.1% on
64 cells per wavelength; between two no-slip walls to 0.05%. A temperature wave at constant
pressure decays at $`\kappa k^2(\gamma - 1)/\gamma`$ to 0.3%.

In the thermal-instability problem conduction ($`\kappa = 3\times10^{-4}`$) sets the size of the
clouds, as [Koyama & Inutsuka (2004)][Koyama2004] showed it must: the scale of the structure
at $`t = 0.6`$ changes by 6% between $`512^2`$ and $`1024^2`$ cells, against a 20% fall without
conduction.

## 12. Passive scalar and species

The **passive scalar** is reconstructed and advected with the hydro variables in the same
passes.

**Species** (`src/species/species.ts`, `hydro/species*.wgsl`, `amr/species*.wgsl`) are stored
apart from the hydro state as $`\rho X_i`$ in groups of four. The hydro passes never read them.
The update records the mass flux $`F`$ through every face, and a separate pass moves every
species with it:

```math
(\rho X_i)^{n+1} = (\rho X_i)^n - \frac{\Delta t}{\Delta x}\sum_d\left(F X_{i,\text{face}}\big|_{+} - F X_{i,\text{face}}\big|_{-}\right).
```

Here $`X_{i,\text{face}}`$ is the upwind cell's fraction at the half step plus half its limited
slope toward the face: the same MUSCL-Hancock reconstruction the scalar gets. Each species
gets its own MC slope, so a sharp front in one species does not flatten the others. The slopes
then need not sum to zero, and the face values must sum to one. The excess is taken back from
the slopes of the sign that has too much, each in proportion to its size. This only shrinks
slopes, so no face value leaves the range of its neighbors, and a species that is uniform is
left alone. (Rescaling face values to sum to one, as in the consistent multi-fluid advection
of [Plewa & Müller 1999][Plewa1999], would change such a species.) With face fractions that
sum to one, $`\sum_i\rho X_i`$ moves exactly as $`\rho`$ does. Mass fractions therefore stay in
$`[0, 1]`$, keep summing to one, and a uniform species stays uniform. Rounding is removed by
scaling the new values to the new density.

A problem without species pays nothing. A problem with many pays only for the species pass.

**Composition in the equation of state** (`src/species/mixture.ts`). When every species is a
nuclide (`h1`, `he4`, `c12`, ...) or `air` (dry air as one species, for laboratory flows), the
equation of state follows each cell's own mixture. Each species may have its own adiabatic
index $`\gamma_i`$ (`Problem.speciesGamma`, else the run's). Each module needs at most four
numbers, and each is a mass-weighted sum over the species, so they mix linearly:

```math
k_1 = \mu_\text{ref}\sum_i \frac{X_i(1 + z_i)}{A_i},\quad
k_2 = \frac{1}{Y_{e,\text{ref}}}\sum_i \frac{X_i Z_i}{A_i},\quad
k_3 = \mu_\text{ref} X_\text{H},\quad
k_4 = \mu_\text{ref}\sum_i \frac{X_i(1 + z_i)}{A_i(\gamma_i - 1)}.
```

The problem's own composition is the reference: it keeps the kelvin scale and says how many
free electrons $`z_i`$ each species gives. $`k_1 = \mu_\text{ref}/\mu`$ scales the gas constant
and $`k_4`$ the heat capacity, so the gas has $`p = k_1\rho T`$, $`e = k_4\rho T`$ and
$`\gamma = 1 + k_1/k_4`$: heat capacities summed by mass, the mixing rule of ideal gases at one
temperature. It is exact where two gases meet at one pressure and temperature, and it gives
Quirk & Karni's helium bubble with 28% air (density 0.182, $`\gamma = 1.648`$) from the two gases
alone. $`k_2 = Y_e/Y_{e,\text{ref}}`$ divides the degenerate electrons' density scale, and $`k_3`$
and $`k_1 - (1 + z_\text{H})k_3`$ are the ionizing gas's $`h`$ and $`o`$. The moments
live in one texture beside the state. The species passes write them whenever they write
$`\rho X_i`$, and on the adaptive mesh the primitives pass gathers them for every ghost, as it
gathers the state. Every pass sets the cell's composition before calling the equation of
state, and each face state of the Riemann problem takes the composition of the cell it was
reconstructed in.

Without species, or with tracers only (names that are not nuclides), the shaders are built
with the constant `COMP = false` (`eos/comp_off.wgsl`) and all of this folds away: such runs
time the same as before it existed. Mixing by mass fraction keeps an ideal gas of one
$`\gamma`$ exactly in pressure balance across a contact, since its pressure does not depend on
the composition. The other equations of state err by about 1% in the cells where two gases
mix (the `species-eos` test).

**Different gammas** are the hard case. A conservative scheme carrying mass fractions makes
the pressure oscillate where two gases with different $`\gamma`$ meet
([Karni 1996][Karni1996]; [Quirk & Karni 1996][QuirkKarni1996]). Here the second-order
reconstruction is the cause: the density and the mass fractions are reconstructed separately
(in different passes), so at a face they disagree on how much of each gas crossed it. A
helium-air contact at one temperature, carried across the box, erred by 3.7% in pressure and
8% in velocity. So where $`\gamma - 1`$ jumps between neighbors, both the hydro and the species
go toward first order. The predictor blends its half-step state toward the cell's own state by
a weight that rises from 0 where the neighbors differ by 1% to 1 where they differ by 5%, and
the species passes blend their face values by the same weight. The contact then errs by 0.2%
in pressure. At unequal temperatures the mixing rule itself is no longer exact, and the error
is 2.8% for helium twice as hot as the air. The weight is continuous: an on/off switch turned
round-off into differences between mirror-image cells (5% in the shock-bubble problem against
$`10^{-6}`$ now). A threshold of 0.1% made every mixed cell first order, and the mixing layer
smeared itself ever wider.

## 13. Self-gravity

### Multigrid on the uniform grid

`src/gravity/`. The potential is solved every step by geometric multigrid
([Brandt 1977][Brandt1977]) on the 7-point Laplacian:

```math
\frac{1}{\Delta x^2}\sum_d\left(\Phi_{i+1} - 2\Phi_i + \Phi_{i-1}\right) = 4\pi G(\rho_i - \bar\rho_\text{per}).
```

- V-cycles with red-black Gauss-Seidel smoothing: 2 sweeps before restriction, 2 after
  prolongation, 8 on the coarsest $`2^3`$ level.
- Restriction averages 8 cells; prolongation is trilinear.
- Warm-started from the previous potential extrapolated linearly in time,
  $`\Phi^n + (\Delta t_n/\Delta t_{n-1})(\Phi^n - \Phi^{n-1})`$ (at most twice the last step
  ahead; `gravity/advance.wgsl`, the uniform grid). The default is one V-cycle per step on the
  uniform grid (3 on the adaptive mesh, where later cycles run only while they still reduce the
  residual).
- *How converged.* After two cycles the largest residual is at single precision's floor,
  about $`10^{-5}`$ of the source (thirty more cycles do not lower it), but a smooth part of the
  error is not: it is what the total energy and momentum see. A converged solve responds symmetrically to
  the density, $`\sum\rho^n\Phi^{n+1} = \sum\rho^{n+1}\Phi^n`$, and the departure from it,
  $`\tfrac12\sum(\rho^n\Phi^{n+1} - \rho^{n+1}\Phi^n)`$ summed over the steps, is energy lost or
  gained. On the Evrard collapse at $`64^3`$ to $`t = 1.5`$ it came to $`-1.5\times10^{-2}`$ of
  $`|E|`$ with one cycle a step from the previous potential, $`-1.1\times10^{-3}`$ with two,
  $`-1\times10^{-4}`$ with three and $`-8\times10^{-6}`$ with four. From the extrapolated guess:
  $`-4.6\times10^{-4}`$ with one cycle, $`-3\times10^{-5}`$ with two. Momentum says the same: on
  the isothermal collapse (`isothermal-collapse`, $`64^3`$, periodic) to $`t = 2`$ the total
  momentum's largest drift, as a fraction of $`\sum|\rho\mathbf v|`$, was $`2.5\times10^{-3}`$
  with one cycle from the previous potential (why two were the default) and
  $`2.2\times10^{-4}`$ with two, but $`2.4\times10^{-5}`$ with one from the guess and
  $`9\times10^{-6}`$ with two. One cycle from the guess is the default: it is more converged
  than two were, and a cycle is 1.5 ms of a $`128^3`$ step's 12.7 (M4 Pro).
- **Periodic** boundaries subtract the mean density, $`\bar\rho_\text{per} = \langle\rho\rangle`$.
- **Isolated** boundaries: a reduction sums the gas's moments $`\sum m\,x^a y^b z^c`$ about the box
  center for $`a + b + c \le 4`$ (35 of them), a pass moves them to the center of mass, and the
  boundary potential is the multipole expansion about it to $`l = 4`$
  (`gravity/multipole_eval.wgsl`):

```math
\Phi_\text{bdry}(\mathbf{x}) = -G \sum_{l=0}^{4} \frac{1}{r^{l+1}} \sum m\,|\mathbf{y}|^l
P_l\!\left(\hat{\mathbf{n}}\cdot\hat{\mathbf{y}}\right),
```

where $`\mathbf{d} = r\,\hat{\mathbf{n}}`$ is the position relative to the center of mass and
$`\mathbf{y}`$ the gas's. Each $`|\mathbf{y}|^l P_l`$ is a polynomial of degree $`l`$ in
$`\hat{\mathbf{n}}\cdot\mathbf{y}`$ and $`|\mathbf{y}|^2`$, so the sum comes from the moments. Up to the
quadrupole only, as it was, two white dwarfs a third of the box from its center felt its edge
pull them toward the middle with about a third of each other's pull, and could not be settled
into balance (`problems/settle.ts`).

### Multigrid on the adaptive mesh

`src/amr/gravity/`. The potential lives on every block, parents included. The density is
averaged into the parents. A level's ghost cells come from its neighbors or are interpolated
from the level above. A multigrid cycle runs inside the blocks (16, 8, 4, 2 and 1 cells across).
The coupling between levels is one-way: a fine level never corrects the coarse potential
beneath it. A level's boundary values are trilinear in the coarser potential; where one of the
coarse cells the interpolation needs cannot be reached (beyond an isolated face of the box, or
where the coarser level itself ends) it is extended linearly through the nearest coarse cell.
Taken as that cell's value, as it once was, the interpolation fell to first order along the
axis: the potential of a uniform sphere was off by 0.17% (two levels at 128) and 0.26% (three
at 256) at the edges of the refined regions, against 0.02% elsewhere, and so got worse as
levels were added. It is now 0.03% and 0.05%, below the 0.15% of 64^3 blocks with no
refinement; re-solving from scratch with 12 V-cycles a level changes none of these, so the
iteration has converged. A new mesh is
solved level by level from the coarsest down. During a run all levels relax in the same
dispatches, each against the boundary values its coarser level had at the start of the cycle,
so the cost does not grow with the depth of the tree.

## 14. Coupling gravity to the gas

**Kick-drift-kick.** Gravity enters as half a kick with the old potential inside the update
pass and the other half after the new potential is solved (`gravity/correct.wgsl`):

```math
(\rho\mathbf{v})^{n+1} = (\rho\mathbf{v})^{*} - \tfrac{\Delta t}{2}\left(\rho^n\nabla\Phi^n + \rho^{n+1}\nabla\Phi^{n+1}\right),
```

where $`(\rho\mathbf{v})^{*}`$ is the momentum after the flux update. Applying the whole kick after
the flux update leaks $`\tfrac12\rho g^2\Delta t^2`$ of internal energy per step from gas in
hydrostatic balance, which slowly cools a star.

**Energy source.** Two forms are offered (`gravEnergyFlux` in `SimConfig`):

1. **Flux form** (default), after [Springel (2010)][Springel2010]. The work on a cell is the mass
   flux through each face times the potential difference across it, at the half step:

   ```math
   \Delta E_i = -\frac{\Delta t}{2\Delta x}\sum_d\left[F_{i+1/2}\left(\Phi_{i+1} - \Phi_i\right) + F_{i-1/2}\left(\Phi_i - \Phi_{i-1}\right)\right]^{n+1/2}.
   ```

   Summed over the cells, this work is the negative of the change in $`\sum\rho\Phi`$ that the
   same mass fluxes cause, so gas plus gravitational energy is conserved to round-off in a fixed
   potential, and with self-gravity to the accuracy of the solve once $`\Phi^{n+1/2}`$ is the
   mean of the two solves (below). It can overheat low-density cells next to a sharp density contrast.
2. **Kick form.** Each kick $`\mathbf{k}`$ adds exactly the kinetic energy it creates,
   $`\mathbf{k}\cdot(\mathbf{m} + \mathbf{k}/2)/\rho`$. Gravity then never changes the internal
   energy, but total energy is not conserved where the flow is under-resolved.

The two forms apply to self-gravity and the uniform field. The pull of the point masses and of
the accelerating frame ([§15](#15-point-masses-and-sinks)) always takes the kick form, whichever
is chosen, from the very acceleration the kick applies. Debris torn from a star is cold and fast
(thermal energy about a thousandth of the total), and each step the tidal field changes its
kinetic energy by hundreds of times its internal energy. Taken in flux form, the
truncation-level mismatch between that work and the kick lands in
$`E - \tfrac12\rho v^2`$: it cut the entropy of the leading half of a disrupted $`n = 3`$ star
a hundredfold, after which the dual-energy switch kept the drained value. In kick form the
star's entropy holds to 0.1 dex. These fields are external, so the gas's own energy is not
conserved under them in either form.

**The half-step potential.** Exact conservation needs the work charged with the mean of the two
solves: with a symmetric solve,

```math
\tfrac12\sum\left(\rho^{n+1}\Phi^{n+1} - \rho^n\Phi^n\right) = \sum\left(\rho^{n+1} - \rho^n\right)\tfrac12\left(\Phi^n + \Phi^{n+1}\right),
```

and the change of density is the divergence of the very mass fluxes the work uses. The update
cannot have $`\Phi^{n+1}`$, so on the uniform grid it charges the work with $`\Phi^n`$
(`gravity/field.wgsl`), and once the new potential is solved the pass of the second kick
(`gravity/correct.wgsl`) moves each cell's work to the mean:

```math
E_i \mathrel{-}= \tfrac12\left(\rho_i^{n+1} - \rho_i^n\right)\left(\Phi_i^{n+1} - \Phi_i^n\right).
```

A cell's share of the correction differs from the flux form's share of the faces' by a flux of
energy between cells, so the total is the same. The source pass keeps the step's change of
density, which is gone by then (a one-channel texture, 4 bytes a cell; `gravity/rhs.wgsl`). A
star at rest is untouched (its density does not change, and $`\Phi^n`$ is what it was charged
with before), and the correction comes before the dual-energy synchronization of the same pass,
so that sees the corrected energy. It is on with the multigrid's self-gravity in flux form and a
gas with an energy, not with implicit pressure (whose faces take the work) or comoving
coordinates (kick form). Without it, and on the block mesh, the work is charged with the
potential extrapolated from the last two solves, $`\tfrac32\Phi^n - \tfrac12\Phi^{n-1}`$: off by
$`\tfrac12\Delta t^2\ddot\Phi`$ at equal steps, and first order after the potential is set
without a step (the first step, a restart, a change of $`G`$) and after steps of
$`\Delta t = 0`$ past an output time, where $`\Phi^{n-1} = \Phi^n`$. (Charging the extrapolated
potential and correcting from it, as first done, gave the same energies, but kept twice the
bytes and cost a quarter more.)

With per-block timesteps ([§18](#18-local-per-block-timesteps)) the block mesh's extrapolation
adds the potential's curvature in time, from the rates of the block's last two steps: in a
collapse the mean of the two solves runs ahead of a linear extrapolation by that curvature.

**Where the energy error of a self-gravitating run comes from.** Summed over the cells, the
change of total energy each step splits exactly into: the half-step potential (above), the
solve's asymmetry ([§13](#13-self-gravity)), and everything that changes $`E`$ besides the hydro
fluxes and the work. Measured at $`64^3`$, with the box's faces made reflecting to keep out what
crosses them:

| | before (2 cycles) | now (1 cycle) |
|---|---|---|
| Evrard, $`t = 1.5`$: half-step potential | $`-2.7\times10^{-3}`$ | $`\approx 0`$ |
| solve's asymmetry | $`-1.1\times10^{-3}`$ | $`-4.6\times10^{-4}`$ |
| dual-energy synchronization after the second kick | $`+4.6\times10^{-3}`$ | $`+4.1\times10^{-3}`$ |

The last is where cold gas falls in supersonically: the kick gives a cell slightly more kinetic
energy than the flux-form work gave its total energy, $`E - \tfrac12\rho v^2`$ goes negative,
and the synchronization ([§7](#7-robustness-in-single-precision)) rebuilds $`E`$ from the
advected internal energy. It does not change with the timestep (the same at a CFL number of
0.8, 0.4 and 0.2, while the potential's term fell fourfold for each halving) and falls fast
with resolution: 1.6%, 0.46% and 0.034% of $`|E|`$ at $`32^3`$, $`64^3`$ and $`128^3`$. Before the
correction the first two terms happened to cancel part of it. With the box's own outflow faces,
the steps that land on output times and the steps of $`\Delta t = 0`$ past them made the
potential's term larger. The tests' largest drifts: `evrard` 0.82% to 0.23%, `sym-evrard`'s
whole box 0.44% to 0.28% and octant 0.69% to 0.13%, `cyl-evrard` 0.11% to 0.04%. Stars
at rest (the polytropes, the white dwarf, the planet) have no error from gravity to speak of
(their half-step and asymmetry terms are below $`10^{-5}`$): what their total energy does is
gas crossing the box's outflow faces. An $`n = 1`$ polytrope gained $`7\times10^{-4}`$ of
$`|E|`$ per dynamical time in its open box and $`4\times10^{-5}`$ in a closed one; the $`n = 3`$
star with radiation in its equation of state in an octant (`rad-star`), 8.9% against 0.11% in a
closed box.

## 15. Point masses and sinks

`gravity/bh*.wgsl`, `gravity/sink.wgsl`. Up to 32 point masses act on the gas through
Plummer-softened gravity ([Plummer 1911][Plummer1911]),

```math
\Phi_\text{pm}(\mathbf{x}) = -\sum_k\frac{GM_k}{\sqrt{|\mathbf{x} - \mathbf{x}_k|^2 + \epsilon_k^2}},
```

The gravity-field texture holds self-gravity alone. Every pass that needs the masses' pull or
potential (the kicks, the predictor, the timestep, the well-balanced reconstruction, the debris
tally) adds it from the masses as they are when it runs (`gravity/ext.wgsl`). Each mass's
potential step between neighboring cells is formed as
$`GM(q_1 - q_0)/(r_0 r_1 (r_0 + r_1))`$ with $`q = r^2 + \epsilon^2`$, not as the difference
of two potentials thousands of times larger. Each step the masses drift. A reduction over the updated gas then gives each mass
its pull from the gas (the reaction to its pull on the gas) and what its sink takes: one
reduction for every four masses in the run (`gravity/mass_sums.ts`), since a reduction's
record has to stay small. The
second half-kick pass removes exactly that from the cells. The pull between two masses is
softened by the larger of their two softening lengths.

A sink whose rate is set to zero takes nothing: the mass then only orbits.

A mass can be flagged **massless** (`PointMass.massless`): it carries a token $`GM`$, a
billionth of the gas's, so that it has an orbit to follow and pulls on nothing. Its sink's rate
is then given per unit time, and what it takes is counted but changes neither its mass nor its
velocity. A mass flagged **fixed** (`PointMass.fixed`) is never moved: it still pulls, takes
gas and grows. The sink tool places massless particles when its Mass is 0, and fixed ones when
Fixed is on.

A source of gas (`hydro/sources.wgsl`) may ride a massless particle (`SourceSpec.carrier`): it
is then wherever the particle is, and the gas it adds arrives with the particle's velocity.
The source tool places its sources so unless Fixed is on.

**Sinks.** Inside the sink radius $`r_s`$, gas is removed at a rate that falls from the local
orbital frequency $`\sqrt{GM/r^3}`$ at the center to zero at $`r_s`$. The removed gas takes its
momentum to the mass. Mass and momentum are conserved between gas and masses to $`10^{-7}`$ in
the validation test. Sinks in grid codes go back to [Bate et al. (1995)][Bate1995],
[Krumholz et al. (2004)][Krumholz2004] and [Federrath et al. (2010)][Federrath2010]. Removal at a
local orbital rate is closer to the sinks used in circumbinary-disk studies such as
[Farris et al. (2014)][Farris2014]. By default the softening length and sink radius are two
cells: of the finest level if the problem keeps the blocks around its masses there, otherwise
of the coarsest level.

**Pair timestep.** See [§3](#3-the-timestep). Two masses on an orbit of eccentricity 0.9 in
near-empty gas keep the orbit's energy to 0.7% over two periods with the limit; without it they
lose 94%.

## 16. Initial states in discrete balance

`problems/settle.ts`. A star set up from its analytic profile is in balance with the analytic
potential, not with the potential the Poisson solver finds on the grid, and it rings in
response to the mismatch. As [Zingale et al. (2002)][Zingale2002] do for stellar models mapped
onto a grid, Shaderous settles every star before the first step. It iterates

```math
h(\rho) + \Phi = C
```

with the grid's potential: set the density from this relation (with $`h`$ the specific
enthalpy of the star's cold gas and $`C`$ chosen so the star keeps its mass), solve the
potential again, and repeat until the density changes by less than $`10^{-5}`$. A polytrope
takes 6 to 8 rounds. $`C`$ is found by regula falsi, started from the previous round's value.
Several stars settle together, each in its own potential (the grid's potential less the pull
of the other stars as point masses), and a moving star keeps its velocity. For white dwarfs
each round is over-relaxed by 1.6 and the rounds stop at $`10^{-3}`$: 7 to 12 rounds, taking
1.6 to 5.5 s.

*Stars in a circular binary* (`OrbitPart` of type `binary`, `problems/roche.ts`). A star held
on a circular orbit is not falling freely, so it is settled in the whole potential of the frame
turning with the orbit at $`\Omega`$: the grid's (both stars' gas), the point masses' and
$`-\tfrac12\Omega^2 R^2`$, the Roche potential. It starts tidally stretched toward its
companion, reaching no farther than its L1 point, and turning with the orbit (tidally locked).
Along the line of centers the tide pulls a star away from its place, and the rounds slid a
star that fills its lobe half its radius toward L1. So each round adds a uniform pull
$`\boldsymbol\lambda\cdot(\mathbf x - \mathbf x_\star)`$ to its potential, as
self-consistent-field binaries fix their boundary points ([Hachisu 1986][Hachisu1986]): it cancels the rest of
the binary's pull at the star's center of mass, with proportional and integral feedback (gains
0.3 and 0.05 of $`GM/R^3`$ per round) on how far that center has strayed. A Sun-like star
filling its lobe beside a $`0.7\,M_\odot`$ point mass settles within $`10^{-3}`$ of its
radius of where the orbit puts it. `fill` sets the separation from [Eggleton's (1983)][Eggleton1983]
lobe radius,

```math
\frac{r_L}{a} = \frac{0.49\,q^{2/3}}{0.6\,q^{2/3} + \ln(1 + q^{1/3})}, \qquad q = \frac{M_1}{M_2}.
```

*A frame that follows a binary* (`OrbitPart.track`, `gravity/bh_kick.wgsl` `follow_binary`). A
corotating frame turns at the orbit's starting $`\Omega_0`$, and an orbit that changes drifts
round the grid. Following it, the frame's angular velocity is the GPU's own: after every step
the point masses' pass measures the binary's axis: that of the quadrupole of its mass in the
orbital plane, the gas denser than where the stars end and the point masses, about their center
of mass. With $`A = I_{xx} - I_{yy}`$ and $`B = 2 I_{xy}`$ the axis lies at
$`\tfrac12 \operatorname{atan2}(B, A)`$, and it turns in the frame at
$`w = (A \dot B - B \dot A) / 2(A^2 + B^2)`$, the moments' rates summed from the momenta
(`gravity/frame_cell.wgsl`, a reduction of eleven sums over the grid each step). No gas has to be
told apart as one star's or the other's. Its angle $`\varphi`$ from where it started (taken before
the first step) and $`w`$ set

```math
\dot\Omega = 2k\,w + k^2\varphi, \qquad k = \Omega_0 / 2,
```

critically damped, so the line returns to its place in about a third of an orbit without
swinging past it. The new $`\Omega`$ is copied into the shaders' `P.rot` as the scale factor of
comoving coordinates is, and every velocity feels the Euler force $`-\dot{\boldsymbol\Omega}
\times \mathbf x`$: the masses' as an impulse in that pass, the gas's in the external
acceleration of the next step's kicks (`gravity/ext.wgsl`). It is not a potential, so the
well-balanced reconstruction does not see it. Two point masses in a frame started 10% slower
than their orbit (`rot-follow`): after two orbits the frame turns at the orbit's rate to
$`6\times10^{-4}`$ and the line of centers is $`0.2^\circ`$ from where it started, having swung
at most $`1.6^\circ`$ in the second orbit; the separation holds to $`10^{-3}`$. In the Roche lobe
overflow preset, whose orbit slows about 12% in two orbits as gas leaves the donor, the line of
centers stays within $`4.5^\circ`$ of its place, where a frame at the starting rate let it turn
$`32^\circ`$. Two stars without cores (n = 1.5 polytropes of masses 1 and 0.6 and radii 1 and
0.6, the first filling its lobe, on a $`128^3`$ grid with three levels): followed, their axis
stays within $`4.2^\circ`$ over two orbits and their separation holds (2.15 to 2.45 from 2.40, as
the density peaks give it to a cell); at the starting rate the axis turned $`51^\circ`$ by
$`t = 28`$ and the separation fell to 1.99, and the pair had nearly merged by $`t = 36`$. The
physics is the same in either frame, so the difference is numerical: most likely the orbital
angular momentum a binary moving across the grid loses to the advection's error, which one
held still does not.

*Releasing it after a merger* (`OrbitPart.release`). Once one star is torn apart over the
other, $`A^2 + B^2`$ falls toward 0 and $`w`$ divides by it: the frame turns at random. In the
magnetized double white dwarf at $`1024^3`$ its rate rose from 0.32 to 0.36 as the orbit
shrank, fell to 0.07 by $`t = 104`$ as the donor came apart, and was back at 0.13 two units
later. Released, the frame watches how much the mass still looks like a binary,
$`e = \sqrt{A^2 + B^2} / (I_{xx} + I_{yy})`$ of the same gas and masses: 1 for two points, 0 for
a body symmetric about z, 0.71 for two touching uniform spheres. In that run it was 0.90 to 0.94
until the donor broke up, 0.38 at $`t = 100`$ and 0.22 at $`t = 106`$. Smoothed over half a
starting orbit and never let rise ($`\bar e`$), it weighs the axis law against slowing down:

```math
\dot\Omega = W\,(2k\,w + k^2\varphi) - (1 - W)\,\operatorname{sign}(\Omega)\,\frac{\Omega_0^2}{2\pi},
\qquad W = \operatorname{smoothstep}(0.3, 0.7, \bar e),
```

the axis law left out once $`W < 10^{-3}`$. The frame slows by $`\Omega_0`$ per starting orbit,
a linear ramp that stops it within an orbit of the handover's end, where an exponential approach
would take several orbits to get close. When the step would take it past 0 with $`W = 0`$, it
stops for good: $`\Omega = 0`$, the gas feels that last turn's Euler force in the next step, and
the run is inertial from then on. The energy and momentum in the diagnostics are then the true
ones. The test was two n = 1.5 polytropes (masses 1 and 0.6) with the donor overfilling its lobe
1.2 times, on a $`64^3`$ grid with two levels in a box 8 across, merging within an orbit
($`P_0 = 13.7`$). Released, $`\bar e`$ fell from 0.85 to 0.31 between $`t = 6`$ and 18, the
frame slowed at the ramp's $`0.0335`$ and stopped at $`t \approx 24`$, and the remnant sat
compact at the center to $`t = 80`$, keeping 68% of the mass in the box. Followed without the
release, the frame swung between $`\Omega = -0.22`$ and 0.47, the total energy jumped from
$`-0.63`$ to $`-0.28`$ and back to $`-0.47`$, and only 52% of the mass stayed, in a lopsided remnant.

*A star with a point-mass core* (a giant; `StarPart.core`, `problems/cored_polytrope.ts`). Part
of the star's mass is a point mass at its center with the point masses' softened potential,
$`-G M_c / \sqrt{r^2 + \epsilon^2}`$, and the rest a polytropic envelope in balance with its own
gravity and the core's pull. With $`h = h_c\theta`$, $`x = r/R`$ and
$`\nu = A\,m(x)/M_\mathrm{env}`$,

```math
\theta' = -\frac{\nu + A\,q_c\,x^3 (x^2 + e^2)^{-3/2}}{x^2}, \qquad \nu' = P\,x^2\theta^n,
```

with $`A = G M_\mathrm{env} / (h_c R)`$, $`P = 4\pi G R^2 \rho_c / h_c`$,
$`q_c = M_c / M_\mathrm{env}`$ and $`e = \epsilon / R`$. The envelope's radius and mass,
$`\theta(1) = 0`$ and $`\nu(1) = A`$, fix $`A`$ and $`P`$ (two nested bisections); without a core
only their product matters and this is the Lane-Emden equation. The settling adds the core's
potential to the grid's, which holds the gas's alone, and runs on the CPU. A giant of one solar
mass with a core of 0.4 softened by a tenth of its radius, on a mesh with 32 finest cells across
the radius, holds its central density within 6% over six dynamical times, with a kinetic energy
under $`2\times10^{-3}`$ of its internal energy.

Settling rewrites a star as cold gas. A problem that lights a star (the sub-Chandrasekhar
supernova) puts its heat in afterward (`Problem.heat`): the internal energy that each cell of a
hot spot should have, written into the settled state on the uniform grid and the adaptive mesh.

For an $`n = 1`$ polytrope this relation is exactly the trapezoid rule of
[§8](#8-well-balanced-reconstruction). Unsettled, a $`1\,M_\odot`$ white dwarf (whose nearly
relativistic core has $`\Gamma_1`$ close to 4/3 and so little restoring force) swung by 30% in
central density at a finest resolution of 256 cells on two levels. Settled, it stays within
0.5% (`paper/paper/figures/make_balance.py`).

## 17. Adaptive mesh refinement

`src/amr/`. The mesh is a tree of blocks of $`16^3`$ cells ($`16^2`$ in 2D) with two ghost layers.
A block splits into $`2^d`$ children of half the cell size, and face neighbors differ by at most
one level. The data layout is described in [architecture.md](architecture.md#adaptive-mesh).

**Depth.** Up to 7 levels below the root grid (`MAX_AMR_LEVELS` in `amr/tree.ts`), 128 times
finer than the roots, as long as a whole root block of 16 cells is left per side: 7 levels from
2048 cells across (one root block), 6 from 1024. 3D grids go to 4096³ at the finest level, on the
adaptive mesh only. The leaf index, which finds the leaf over any point of the finest level, has
one 4-byte entry per finest block, $`(n/16)^d`$: 8 MB at 2048³ and 64 MB at 4096³, counted in a
grid's memory. Deeper than 7 the tree would still work (its keys stay exact in a double), but
the root level would hold nothing in 3D, and with steps no longer than 32 times the finest
([§18](#18-local-per-block-timesteps)) the coarse levels would take ever more steps than they
need.

### Refinement criteria

`amr/flag.wgsl`. The main indicator is the normalized second derivative of
[Löhner (1987)][Lohner1987], in the form FLASH uses ([Fryxell et al. 2000][Fryxell2000]), applied to the
density and pressure:

```math
\chi^2 = \frac{\sum_d\left(q_{i+1} - 2q_i + q_{i-1}\right)^2}
{\sum_d\left(|q_{i+1} - q_i| + |q_i - q_{i-1}| + \epsilon\left(|q_{i+1}| + 2|q_i| + |q_{i-1}|\right)\right)^2},
\qquad \epsilon = 0.01.
```

It is close to 1 at a discontinuity and falls as $`\Delta x/L`$ across a smooth profile of scale
$`L`$. A block is refined where $`\max\chi > 0.8`$, and $`2^d`$ siblings merge where it is below 0.2
in all of them (FLASH's defaults; both can be changed during a run). An indicator based on
first differences refines smooth gradients as readily as shocks, and kept most of the binary's
disk at the finest level in our tests. Further criteria:

- **Jeans length:** with self-gravity, refine where $`\lambda_J < 16\,\Delta x`$ (`jeansRefine`).
- **Dense gas:** a problem can keep every block with gas above a density at the finest level
  (`refineDense`). The white dwarf problems do this, because a star's smooth interior does not
  raise $`\chi`$.
- **Point masses:** a problem can keep every block near its masses at the finest level.
- **Composition:** with species, a jump in any mass fraction between neighboring cells counts
  as $`8\,|\Delta X|`$, so a jump of 0.1 reaches the refinement threshold. Here a first
  difference is the right measure, since a mass fraction is nearly piecewise constant. To
  $`\chi`$, an interface spread over a few coarse cells is a smooth profile, which is never
  refined back. In the shock-bubble problem the bubble drifted onto coarse blocks that way,
  ending with 24% of its helium at the finest level; with this criterion, 99%.
- **Thin gas** below a density cutoff never asks for refinement, so ripples in the ambient
  medium around a star do not refine it.

The mesh is rebuilt every 8 steps from a per-block indicator read back without blocking. New
leaves are filled by copying, by restriction (averaging the $`2^d`$ children, which conserves),
or by prolongation with MC-limited linear interpolation, which also conserves. The total energy
is not interpolated with its own slope: a new cell gets its own kinetic energy (from its
interpolated density and momentum), its interpolated internal energy, and one offset shared by
its siblings that keeps their mean $`E`$ the coarse cell's. The offset is the kinetic energy
that the spread of velocities among the children adds (never positive, as $`|\mathbf{m}|^2/\rho`$
is convex), taken from the heat. Interpolated on its own, $`E`$ left $`E - KE`$ off by terms of
the order of the kinetic energy's variation across the coarse cell, which in cold, fast gas is
many times its heat, and where the flow converges (so that $`E`$ is little more than $`KE`$)
the dual-energy switch took that for heat. A Zel'dovich pancake refined at its caustic was
heated $`10^8`$-fold there ([§22](#comoving-coordinates)). With a magnetic field $`E`$ is
rebuilt after a re-mesh in any case.

**A full block budget.** The atlas holds a fixed number of blocks, set by the GPU memory budget
([architecture.md](architecture.md#gpu-memory-budget)). When a re-mesh wants more than it holds,
the most strongly flagged blocks are refined first, and room is made for them by merging the
siblings whose largest indicator is lowest (those closest to merging anyway), provided it is
below 0.8 of the indicator of the block that needs the room (`EVICT_MARGIN` in
`amr/tree.ts`); the margin keeps blocks flagged about equally from trading slots back and forth.
Blocks next to the refinement, or that balance would split for it, are not merged, and a block
merged to make room is not split again in the same re-mesh: its children would be filled from
the slot of a parent, which holds no data. Dense-gas blocks are flagged infinitely and never
make room for one another.

### Ghost cells and conservation across levels

Ghost cells are computed as they are loaded, not exchanged
([architecture.md](architecture.md#ghost-cells-computed-on-load)). At a refinement boundary the
coarse cell does not use its own flux on that face. It recomputes the $`2^{d-1}`$ fine fluxes
that cover the face, from the same fine states and with the same arithmetic as the fine cells,
and sums them. This is the effect of the flux correction of
[Berger & Colella (1989)][Berger1989] without storing any flux. Mass, momentum and energy are
conserved to round-off, and the validation suite checks that fine and coarse fluxes agree to
$`10^{-6}`$.

## 18. Local (per-block) timesteps

Off by default ("Local timesteps" under Hydro, `tbins` in a link). The subcycling of
[Berger & Oliger (1984)][Berger1984] lets fine levels step more often than coarse ones.
Shaderous assigns steps per block instead, in power-of-two time bins as in particle codes such
as GADGET-2 ([Springel 2005][Springel2005]).

- Time advances in **cycles** of up to 32 substeps of the smallest step $`\Delta t_\text{min}`$
  (`tbins` 1 to 5: cycles of 2 to 32). The mesh is checked again only between cycles, so a
  cycle runs whole on a mesh flagged at its start. At a third of a cell per substep, 32
  substeps carry a feature some 11 cells, inside the 16-cell buffer block kept around flagged
  ones. Longer cycles would need the finer levels re-meshed within a cycle. With self-gravity
  cycles stop at 16 substeps: over 32 the potential carried forward between solves (below)
  lagged an Evrard collapse (128 finest, 3 levels, to $`t = 0.8`$) enough to lose
  $`2.3\times10^{-3}`$ of its energy, against $`5.7\times10^{-4}`$ with one timestep and
  $`6.0\times10^{-4}`$ with cycles of 16.
- At the start of a cycle each block gets a bin $`k`$ from its own signal speed, on the GPU.
  A block in bin $`k`$ advances $`2^k\Delta t_\text{min}`$ at a time.
- A first sweep lowers every block to the bin of its fastest face neighbor (once, so each fast
  region widens by one block). Further sweeps enforce two rules: face neighbors are at most
  one bin apart, and a block is never slower than a coarser neighbor.
- Across a face whose other side steps twice as often, the slower block takes the faster side's
  fluxes, recomputed from stored states: half when its own step starts and half at its
  midpoint. A faster block reads a slower neighbor that is part way through its step from that
  neighbor's half-step prediction. Nothing is stored, and conservation holds to round-off.
- A cycle keeps the smallest step it starts with, so a run begins with 16 single steps.
  Initial conditions out of balance can shrink the step 30-fold within two steps (gas at rest
  falling onto a point mass at the inner radius of an $`(r, \theta)`$ run), and a cycle begun
  at $`t = 0`$ took all its substeps at the step of the gas at rest and went non-finite. Whether
  to take a cycle is decided before each one, so a long call of steps goes on into cycles once
  the single steps are taken (decided once per call, a validation run of many steps in one call
  took none at all).
- With self-gravity the potential is solved on every block at the end of a cycle. In between it
  is solved only on blocks finishing a step, and carried forward elsewhere at the rate at which
  it last changed. Point masses move every substep. A block's half-step potential, for the
  energy's work, is $`\Phi + \tfrac{\Delta t}{2}(2r_1 - r_0)`$ with $`r_1`$ and $`r_0`$ the rates
  over its last two steps (taken as equal in length): the mean of $`\Phi`$ and a quadratic's
  next value. When the run goes over from single steps to cycles, the rate starts from the last
  single step rather than from nothing.

The extra passes cost about 10% per block update without gravity and 50 to 70% with it (3D).
So the share of updates actually made is read back every few cycles, and the run falls back to
one timestep while that share is above 80% (65% with gravity). Measured against one timestep:
a 3D Sedov blast at 256 cells with 3 levels makes 2.2 times fewer block updates and runs 1.9
times faster; an isothermal collapse at 256 with 3 levels makes 2.3 times fewer and runs 1.6
times faster. With gravity, energy is conserved as well as with one timestep: an Evrard collapse
(128, 2 levels, to $`t = 0.8`$) loses $`5.7\times10^{-4}`$ of it against $`5.8\times10^{-4}`$. It
lost $`1.2\times10^{-3}`$, which had been put down to slow blocks extrapolating their
potential further. That was not it: with every block forced into the fastest bin the loss was
the same. Two things were. The potential's history was dropped when the cycles began, so the
first cycle's half step was the potential itself. And at the start of each cycle the half step
was extrapolated linearly over the new step, which lags the mean of the step's two potentials
in a collapse; single steps lag it as well, but their $`\tfrac32\Phi^n - \tfrac12\Phi^{n-1}`$
over-extrapolates when the step shrinks, which happens to offset it (skipping that
re-extrapolation, cycles of two substeps reproduced the single steps' loss to four digits).

## 19. Accelerating frames and damping zones

**Gas rest frame** (tidal disruption). As in [Guillochon et al. (2009)][Guillochon2009] and
[Guillochon & Ramirez-Ruiz (2013)][Guillochon2013], the frame accelerates with the gas's center
of mass, so the star stays at the box center while the perturber's orbit is integrated
alongside the gas. Every cell receives the opposite of the frame's acceleration.

**Point-mass rest frame** (binary). The frame accelerates with the mean pull of the gas on the
masses, and the gas feels the opposite acceleration. After each step the velocity of the
masses' center of mass is reset to zero, which removes the slow drift from rounding in their
mutual pull and the momentum of accreted gas.

**Damping zone** (binary disk), of the kind used by [de Val-Borro et al. (2006)][deValBorro2006].
Beyond 4.5 binary separations the density, velocity and specific internal energy relax toward
the initial disk, at a rate that rises from zero there to three times per local orbital period
at the box edge (6 separations). Without the zone, gas flowed in through the corners, and the
disk grew until its uneven pull pushed the binary out of the box.
With a magnetic field (the magnetized accretion disk) the zone acts on the gas alone: B^2 / 2 is
taken out of E before it and put back after, and the field is left to the induction equation.

**Faces held at an exact solution** (the Noh, uniform-collapse and Coggeshall presets). A
zone's target is a function of position and of time, the end of the step, so a flow with an
exact solution can supply one (`FlowDef.damping`, `setup/parts/flows/converging.ts`). The three
cells next to each open face then relax at a rate of $`10^{30}`$, that is, are set to the
solution at the end of every step. That is a Dirichlet boundary built from the zone, with no
change to the loaders, on the uniform grid and on blocks. A converging flow needs it: an open
face copies the cell inside it, and cannot push gas in as the gas beyond the box would.
Without it, a rarefaction from the faces of the uniform collapse reached the center by
$`t = 0.3`$, and in the Cartesian Noh implosion the gas coming in formed dense sheets along the
axes that reached the shock. Such a zone also runs in spherical (r, θ) runs, which refuse the
disk's and the tunnel's: it takes their positions and velocities as (r, θ) and
$`(v_r, v_\theta, v_\phi)`$.

**Turbulence driving** (`hydro/drive.ts`, `hydro/drive.wgsl`, `hydro/drive_ou.wgsl`; the
driven-turbulence presets). A random acceleration stirs the gas at the largest scales: the
Ornstein-Uhlenbeck forcing of [Eswaran & Pope (1988)][EP88], with the solenoidal and
compressive projection of [Federrath et al. (2010)][Federrath2010]. The idea and the presets
follow the WebGPU reduced-MHD page of [Mallet & Abraham (2026)][Mallet2026]. The modes are the
wavevectors $`\mathbf{k} = 2\pi\mathbf{n}/L`$ with integer $`\mathbf{n}`$ in a band
$`k_\text{min} \le |\mathbf{n}| \le k_\text{max}`$ (one of each pair $`\pm\mathbf{n}`$, in the
plane in 2D; at most 64, the longest first: 16 in 3D for the default band of 1 to 2, 61 for 1 to 3).
The acceleration is

```math
\mathbf{a}(\mathbf{x}) = \sum_m \mathbf{a}^r_m\cos(\mathbf{k}_m\cdot\mathbf{x}) - \mathbf{a}^i_m\sin(\mathbf{k}_m\cdot\mathbf{x}),
```

summed directly at each cell's position, so it needs no FFT and works on any grid and any
level. Over a time $`h`$ each amplitude becomes
$`f\,\mathbf{a} + \sigma\sqrt{1 - f^2}\,\mathsf{P}\boldsymbol{\xi}`$, with $`f = e^{-h/\tau}`$,
$`\boldsymbol{\xi}`$ unit normal deviates and
$`\mathsf{P} = \zeta\mathsf{P}_\perp + (1 - \zeta)\mathsf{P}_\parallel`$ ($`\zeta = 1`$
solenoidal, 0 compressive). $`\sigma`$ is set so that the box mean of $`|\mathbf{a}|^2`$ is the
requested rms acceleration squared in the stationary state, and the first amplitudes are drawn
from that state, so the driving is at full strength from the start. The step is taken on the
GPU by the clock pass, over the step it has just set (over the whole cycle with per-block
timesteps, so every block of a cycle sees the same amplitudes); the deviates come from a hash of
the seed, the step count, the mode and the draw, so a run is the same from the same seed. (The
preview of a step not yet taken, the dt the panel shows before the first step, runs the clock
pass without advancing the amplitudes. It once advanced them, so a run's driving depended on
how often the page had read its diagnostics before its first step, and a page's first run
differed from the same setup loaded again.) The
updates add $`\rho\,\mathbf{a}\,\Delta t`$ to the momentum and its kinetic energy
$`\mathbf{k}\cdot(\mathbf{m} + \mathbf{k}/2)/\rho`$ to the total energy (in MHD, each stage
over its own step). The acceleration has zero mean but its force $`\rho\mathbf{a}`$ does not
where the density varies, and nothing removes it: the box picks up a net momentum, about a tenth
of its mass times the rms speed after four crossing times at Mach 1. Restarting from a
snapshot draws the amplitudes afresh. In `driving`: the isothermal preset at $`32^3`$ holds an
rms speed of 1.0 to 1.2 (the sound speed is 1) from $`t = 3`$; the same seed gives the same kinetic
energy and another seed one 20% apart; with the force frozen, unrefined blocks whose root grid is
the uniform one follow it to $`10^{-7}`$ in kinetic energy at $`t = 0.3`$; and with a guide field
in 2D the field is amplified with $`\nabla\cdot\mathbf{B}`$ at zero. Not yet compared against the injected power or a measured spectrum (no spectra yet).

## 20. Nuclear burning network

`src/burn/network.ts`, `hydro/burn_net_core.wgsl`; trained offline in `tools/burn/`. The
13-isotope α-chain (aprox13, [Timmes et al. 2000][Timmes2000]) is stepped by a small neural
network rather than an implicit integrator, wherever the gas is hotter than $`10^8`$ K and
denser than $`10^4`$ g cm$`^{-3}`$ (the range of its training set). A hydrodynamic step longer
than 10 ms is split into network steps.

- **Reference network.** `burner.py` integrates aprox13 (the network module is generated with
  pynucastro, [Smith Clark et al. 2023][Smith2023], from REACLIB rates with pynucastro's
  screening) at constant density with self-heating, the temperature following from the
  tabulated electron-positron equation of state as in the simulator. The shipped network was
  trained on a first reference that kept only the alpha exits of C12 + C12, C12 + O16 and
  O16 + O16 (55%, about 40% and 21% of their totals) and dropped the proton exits with the
  protons. Against aprox13 it burns carbon 1.8 times and oxygen 3 to 4 times more slowly, so
  half the carbon takes 1.8 times longer to burn and half the oxygen 1.8 to 2.8 times longer
  (as Doron Kushnir pointed out). `build_net.py` now builds pynucastro's aprox13, which folds
  each proton exit through Na23, Al27 or P31, for the next training. The inputs stay the rate
  terms of the first reference (`aprox13_features.py`).
- **Network.** Inputs: $`\log\rho`$, $`\log T`$, $`\log\Delta t`$, the mass fractions, and the
  logarithm of every screened rate term of the reference (61 in all). The rate terms are inputs
  because they rise with temperature far more steeply than a network this size could resolve
  from $`\log T`$ alone. Four SiLU hidden layers, then two stages of 27 outputs. Each stage moves
  mass as $`\Delta X = (1 - e^{-k\Delta t})(M g - X s)`$: every species loses a share
  $`s_i`$ of itself, and what is lost, $`M = \sum_i X_i s_i`$, is shared out by a softmax $`g`$.
  So mass is conserved exactly and no fraction goes negative, whatever the weights. The energy
  released is the change in nuclear binding energy, never predicted. Weights are stored in half
  precision.
- **DAgger.** `dagger.py` collects the states that the network itself visits when applied step
  after step, labels them with the reference network, and retrains on them
  ([Ross, Gordon & Bagnell 2011][Ross2011]). After the equation of state changed, every stored
  trajectory was burned again and the rounds continued; round 35b is the trained model.
- **Weight edits.** Further rounds stopped helping, so the trained weights were edited directly
  (`tools/burn/weightprog/`). The worst failure was a runaway in silicon-sulfur ash near
  $`4\times10^9`$ K: about 10 times too fast a burn, whose heat carried the gas to $`^{56}`$Ni.
  Each stage's burn rate is one output ($`\log_{10}k = 5 + 5z`$), so a fix can act on the rate
  alone. Two units are added to the last hidden layer, so the hidden layers are 130 wide, and
  no trained weight changes. Each is a gate $`g = \mathrm{relu}(\mathbf w\cdot\mathbf a + b)`$ of
  that layer's inputs $`\mathbf a`$ (realized as $`\mathrm{silu}(20x)/20`$), which lowers both
  stages' $`z`$ by $`\alpha g`$. $`\mathbf w`$ and $`b`$ are least-squares fits to the shift of
  $`z`$ that best fixes each state's step, from exact reference burns of 1 ms, 0.1 ms and
  10 μs and from the network's own rollout states. One unit ($`\alpha = 2`$) fires on hot
  carbon-oxygen and slows it mildly. The other ($`\alpha = 1.5`$) is held at zero on gas with
  $`X_{\rm C} + X_{\rm O} > 0.2`$, above $`4.4\times10^9`$ K and at 10 μs steps, and slows the
  runaway. Compared trajectory by trajectory with 35b (`weightprog/paired.py`, 600 held-out
  trajectories in each of four hard regimes), the sum of the four regimes' 90th percentiles of
  each trajectory's worst error fell by 0.69, or 29%, at 1 ms steps (90% bootstrap interval 0.46-0.89),
  and by 0.52 and 0.56 (about a quarter) at 3 ms and 0.3 ms steps; the silicon-sulfur regime's
  fell from 1.58 to 0.87, and the others did not change. Results and the methods that failed
  are in the docstrings of `weightprog/twounit.py`, `rateedit.py` and `lpedit3.py`.
- **Limiter.** Optionally, no cell takes in burning energy faster than $`f_b\,e/t_s`$
  ([Kushnir et al. 2013][Kushnir2013]); each network step is then scaled down as a whole.
  Off by default. It spreads the burning over about $`\Delta x/f_b`$, and holds where that is
  much smaller than every other length of the problem ([Kushnir & Katz 2020][KushnirKatz2020]).
  A detonation then has to be lit over a region about $`\Delta x/f_b`$ across
  ([Kushnir, Wygoda & Sharon 2020][Kushnir2020], Sec. 2.2): in the C/O tube at $`f_b = 0.05`$
  the default layer at $`4\times10^9`$ K drives only a shock, up to 200 km wide, while a layer at
  $`8\times10^9`$ K lights a detonation at the unlimited speed.
- **Heating rate.** The burn pass writes each cell's energy released over the step, divided by
  the step, to a texture of its own: the view "Nuclear heating" (per unit volume) and the
  diagnostic L nuc (summed over the gas) read it.
- **No neutrino losses.** Neither the reference network nor the simulator takes energy out as
  thermal neutrinos (pair, photo-, plasma and bremsstrahlung). In a detonation they are
  negligible: some $`10^{13}`$-$`10^{15}`$ erg g$`^{-1}`$ s$`^{-1}`$ at $`5\times10^9`$ K and
  $`10^7`$-$`10^8`$ g cm$`^{-3}`$, against a few $`10^{17}`$ erg g$`^{-1}`$ released within
  the burning time, under a percent over the second or so the hot ash takes to expand. They
  matter over longer times (carbon simmering before a Chandrasekhar-mass explosion, the
  cooling of a merger remnant), and electron captures, not thermal neutrinos, matter in
  nuclear statistical equilibrium above about $`10^9`$ g cm$`^{-3}`$.

## 21. Ideal MHD

Presets with a magnetic field (the "Magnetic fields" group: Orszag-Tang, the MHD blast, the
rotor, Brio-Wu, the field loop, the Alfvén wave, the Alfvén-wave collision and the Elsasser
packets, and the tearing mode and island coalescence (with resistivity); Kelvin-Helmholtz's
"Magnetic field" variant; and the polytrope's "Dipole field" and
"Random field" variants) solve ideal MHD, with gravity, point masses, sinks and cooling (below).
It runs on the uniform grid and on the adaptive mesh, which re-meshes with the field (below),
with per-block timesteps, and draws its field lines (below). Species, burning, viscosity and
conduction act with a field too: every physics option of the gas alone couples to one. Units: the magnetic pressure is $`B^2/2`$, so
$`B_\text{Gauss} = \sqrt{4\pi}\,B`$ and plasma $`\beta = 2p/B^2`$. The total energy
$`E`$ includes $`B^2/2`$.

- **Constrained transport.** The field lives on the cell faces: one texture holds $`b_x`$,
  $`b_y`$, $`b_z`$ on each cell's lower faces, with one more texel per axis for the grid's
  upper faces. Each face is updated from the electric fields on its four edges,

  ```math
  b_x^{n+1} = b_x^n - \frac{\Delta t}{\Delta x}\left[(E_z^{j+1/2} - E_z^{j-1/2}) - (E_y^{k+1/2} - E_y^{k-1/2})\right],
  ```

  so every edge enters the faces around it with opposite signs, and the divergence of every cell
  ($`\sum`$ over its six faces) does not change. The cell-centered field is the mean of each
  axis's two faces.
- **Integrator.** The van Leer predictor-corrector of [Stone & Gardiner (2009)][SG09], as in
  Athena++. Stage 1 takes half a step with first-order fluxes from the cells' own states.
  Stage 2 takes the full step from the start, with fluxes from a limited linear reconstruction
  of the half-step primitives (the run's limiter on $`\rho`$, $`\mathbf{v}`$, $`p`$, the scalar
  and the two tangential field components). Each stage has three passes:
  - fluxes, each face computed once into storage buffers, so the two cells sharing it read the same bits;
  - edge electric fields, stored as the stage's impulses $`(\Delta t/\Delta x)\,E`$;
  - the update of cells and faces.

  The passes run slab by slab along z, so the flux and edge buffers hold one slab of layers
  (at most 192 MB together, `slabLayers`), not the whole grid. Faces and edges on the plane
  between two slabs are computed by both, with the same pipelines from the same inputs, so the
  result is bit for bit the same as with one slab (the `mhd-slabs` test).

  The half-step state uses the hydro predictor's textures. The internal energy's
  $`p\,\nabla\cdot\mathbf{v}`$ takes the pressure at the start in stage 1 and at the half step in
  stage 2.
- **Edge electric fields.** The construction of [Gardiner & Stone (2005)][GS05], in Athena++'s
  form. By duality, a face normal to $`a`$ carries $`E_{t_2} = -F_a(B_{t_1})`$ and
  $`E_{t_1} = F_a(B_{t_2})`$. The edge value is the mean of the four face values around it plus
  their differences from the cell-centered reference $`-\mathbf{v}\times\mathbf{B}`$, upwinded
  by the sign of the mass flux through the faces. This is what keeps a field loop carried by a
  uniform flow intact. In 2D the in-plane components $`E_x`$, $`E_y`$ are face values.
- **Riemann solvers.** HLLD ([Miyoshi & Kusano 2005][MK05]) by default, used for the HLLC and
  low-Mach HLLC settings. It resolves contacts and Alfvén waves. HLLE, with Davis estimates from
  the fast magnetosonic speeds, is used for the HLL setting. The face's own $`b_n`$ is the
  normal field on both sides. Next to shocks HLLD falls back to HLLE, as HLLC does to HLL:
  stage 1 flags the cells where the flow converges and the total pressure $`p + B^2/2`$ jumps
  by more than the shock threshold between neighbors, and stage 2's fluxes take HLLE where
  either side is flagged. The flag is in the half-step state's fourth channel, which the
  "Shock" field shows.
- **Timestep.** The signal speed is $`|v_x| + |v_y| + |v_z| + d\,c_f`$, with $`d`$ the number of
  dimensions and $`c_f = \sqrt{c_s^2 + B^2/\rho}`$ the fastest magnetosonic speed.
- **Dual energy and floors.** The synchronization of §7 uses $`E - KE - B^2/2`$, where
  $`B`$ is the new cell-centered field.
- **Boundaries.** Periodic axes map the upper face to the lower face of cell 0. At an outflow
  face the boundary faces are evolved like any other, with edge fields from ghost cells that
  copy the edge cells. At a reflecting face the ghost's normal field changes sign, as its
  normal velocity does: the field lies along the wall. The tangential face indices of ghost
  cells are mapped like cells.

  A **line-tied wall** (kind 4) holds field lines that cross it, as the closed,
  flux-conserving boundaries of X-point collapse studies do ([Tsiklauri & Haruki
  2007][Tsiklauri2007]; [Thurgood et al. 2018][Thurgood2018]). The gas sticks to it: the
  ghost's whole velocity is reversed (no slip, which also makes it a no-slip wall for viscosity
  without a field). The ghost's field is the edge cell's, unreversed, so the field normal to
  the wall is continuous; and the electric field along the wall is zero on its edges, so the
  faces in the wall never change and the flux through it is fixed to the bit. Nothing crosses
  it: no mass, and no Poynting flux, ideal or resistive. Because the ghost copies the edge
  cell's field, a field with a gradient across the wall sees a small spurious current there.
  In `mhd-line-tied`: a stressed X-point, $`\mathbf{B} = (y, 1.44x)`$ with
  $`\eta = 10^{-3}`$, keeps every wall face to the bit, conserves mass exactly and total energy
  to $`7\times10^{-8}`$, uniform and on blocks, and collapses (peak density 3.5 by $`t = 1`$);
  the current-free null, $`\mathbf{B} = (y, x)`$, picks up kinetic energy of
  $`8\times10^{-6}`$ of its magnetic energy by $`t = 1`$, from the walls' spurious current; and
  for the gas alone, a shear flow between two such walls decays as
  $`\exp(-2\nu\pi^2 t)`$ to $`2\times10^{-5}`$.

  A **moving wall** (kind 5) is a line-tied wall that slides along itself at the velocity set
  for its face (`domain.wallVelocity`; the part normal to the wall is dropped). The ghost's
  velocity is $`2\mathbf{v}_\text{wall} - \mathbf{v}`$, so the gas at the wall moves with it (no
  slip), and the electric field along the wall is $`-\mathbf{v}_\text{wall}\times\mathbf{B}_n`$,
  with $`B_n`$ the mean of the two wall faces beside each edge, so the footpoints of the field
  lines move with the wall and the flux through it is carried along it exactly. The wall does
  work on the gas and the field, so the total energy is not conserved. In `moving-wall`: the
  lid-driven cavity (the preset, Reynolds number 100, Mach 0.08, $`128^2`$) settles by
  $`t = 12`$ to a vertical-centerline velocity within $`0.0065`$ of the lid speed of the table of
  [Ghia, Ghia & Shin (1982)][Ghia1982] at every height, its vortex core at $`u = -0.214`$ against
  their $`-0.211`$; a field $`B_y = 1 + 0.2\sin 2\pi x`$ through a wall moving at 0.5 is carried
  along it to $`2.5\times10^{-3}`$ of its amplitude by $`t = 0.5`$, while the faces of the wall
  at rest opposite stay to the bit and the flux through the moving one holds to
  $`5\times10^{-9}`$; and on unrefined blocks the magnetic energy follows the uniform grid to
  $`10^{-5}`$ after some 1700 steps.
- **Initial fields.** A preset gives a vector potential $`\mathbf{A}`$, sampled at the edge
  centers, and $`b`$ is its discrete curl, so the divergence starts at zero up to rounding. Or
  it gives the face values of a field that is divergence-free by construction.
- **Divergence in float32.** Each edge impulse is one stored number that the four faces
  around the edge add with opposite signs, so in exact arithmetic no cell's divergence changes.
  In float32 each face's sum rounds on its own, and over many steps the rounding would walk
  the divergence away from zero (to $`3.5\times10^{-6}`$ of $`\max|B|`$ after 750 steps of
  Orszag-Tang, before this was fixed). So each face is a pair: the field $`b`$ and a
  compensation $`c`$ (a second face texture) holding what $`b`$'s rounding dropped, and the
  impulses are added with error-free sums (Knuth's TwoSum). The pair carries the face's exact
  sum of impulses, and $`\max|\nabla\cdot B|\,\Delta x/\max|B|`$ stays at a few roundings for
  good: $`7\times10^{-8}`$ after 751 steps and $`9\times10^{-8}`$ after 3483. The diagnostics
  report it (`magnetic.divB`, and the field "Magnetic divergence"), as the panel's ∇·B row,
  with the magnetic energy, the largest |B| and the least plasma beta.
- **Gravity, point masses, sinks and cooling** (plan section 7b). Kick-drift-kick as in the
  hydro step: stage 1 adds the half step's whole source, so the half-step state the stage-2
  fluxes come from feels it; stage 2 adds the first half kick, in kinetic form or, with
  Conserve energy on, as flux-form work from its own mass fluxes; after the step the potential
  is solved for the updated state, the masses move, and the correction pass applies the second
  half kick. That pass writes the cells back into the state's parity, and the faces are copied
  there after them (the leaves' faces, on blocks). E holds $`B^2/2`$, which stays out of what
  the correction rebuilds and of what a sink takes: a sink removes a share of the gas's mass,
  momentum and kinetic and internal energy, and leaves the field where it is (flux cannot be
  removed without breaking constrained transport). Cooling acts at the end of stage 2 and
  changes E by exactly the change of $`e_\text{int}`$. On blocks, per-block timesteps carry on
  with self-gravity and point masses as without a field. A star's settling keeps the field's energy. In
  `mhd-gravity` and `mhd-gravity-blocks`: a polytrope threaded by a uniform field (beta 1000)
  is as balanced as without it, a stratified column with a horizontal field as still, cooling
  follows the analytic decay to $`10^{-5}`$, and a sink keeps the mass budget to $`10^{-7}`$
  with the field's energy left. (MHD's reconstruction has no well-balanced pressure: the column
  is compared with the hydro column without it.)
- **Obstacles and inflows** are refused with a field: the MHD fluxes know no solid or fixed
  cell, and gas flowed into an obstacle (a random setup between walls gained 0.14% of its mass
  in 200 steps). The obstacle tool is off in an MHD run.
- **Species** move by the same pass as without a field (§16), from the face mass fluxes that
  the MHD update's stage 2 records (on blocks, each block's own fluxes before the matching at
  refinement boundaries, which the species pass does itself); the block grid's species
  predictor reads the MHD assembly's velocities. Burning flips the state's parity as a brush
  does, and the faces follow it. In `mhd-species` the species test's bump, carried twice around
  the box in a field at beta 4, keeps its mass fractions in [0, 1] summing to 1, and returns
  more accurately than the passive scalar the MHD fluxes carry (2.7e-4 against 3.5e-4 at
  $`128^2`$; 1.0e-3 against 1.3e-3 on unrefined blocks at 64).
- **Viscosity and conduction** (`hydro/diffusion_core.wgsl`, `mhd/diffusion.wgsl`), isotropic as
  without a field ([§11](#11-viscosity-and-thermal-conduction); no Braginskii anisotropy). The
  stress and heat flux are added to every MHD face flux in both stages, from that stage's state
  (the start's, then the half step's), so they are integrated to second order in time where the
  gas alone takes them from the start of the step. On blocks they come from the assembled state,
  whose edge ghosts give every face its transverse neighbors (no one-sided rims), and the flux
  register matches them at refinement boundaries with the rest of the flux. The update adds the
  viscous heat to the separately evolved internal energy, working the faces' stresses out again
  from the same state. The diffusive speed limits the timestep as in §3. In `mhd-diffusion`:
  shear waves across $`B_z`$ decay as $`\exp(-2\nu k^2 t)`$ to 0.08% (2D) and 0.3% (3D), on blocks
  too; a standing Alfvén wave along $`B_x`$, damped by viscosity alone, follows the damped
  oscillator $`V'' + \nu k^2 V' + k^2 v_A^2 V = 0`$ in its wave energy to 0.03%; a temperature
  wave across $`B_y`$ decays at $`\kappa k^2 (\gamma - 1)(1 + B^2)/(\gamma + B^2)`$ (with
  $`\rho = p = 1`$: the frozen-in field shares the pressure balance and stiffens it) to 0.1%;
  and a magnetized Kelvin-Helmholtz run on two levels with per-block timesteps conserves
  energy to $`4\times10^{-8}`$.
- **Resistivity** (`mhd/ohm.wgsl`, `mhd/corner.wgsl`): a constant magnetic diffusivity
  $`\eta`$ (`physics.resistivity`, link key `eta`; Athena++'s `eta_ohm`), so that the induction
  equation gains $`\eta\nabla^2\mathbf{B}`$. Each edge field gains $`\eta\mathbf{J}`$,
  $`\mathbf{J} = \nabla\times\mathbf{B}`$, before it becomes an impulse, so constrained transport
  keeps the divergence where it was. The total energy takes the matching Poynting flux
  $`\eta(\mathbf{J}\times\mathbf{B})\cdot\hat{\mathbf{n}}`$ through every face, added to the face
  flux like the viscous stress (and so matched at refinement boundaries), and is conserved to
  round-off; the separately evolved internal energy takes the Ohmic heat $`\eta J^2`$ of each
  cell. $`\mathbf{J}`$ comes from the cell-centered field everywhere: at an edge from the four
  cells around it, at a face from its two cells and their transverse neighbors, at a cell from
  its six neighbors. The face fields' grid-scale checkerboard, which those differences do not
  see, is left to the scheme's own dissipation. The field's diffusivity limits the timestep as
  viscosity does ([§3](#3-the-timestep)). In `mhd-resistivity`: a field wave at rest, held by the
  gas pressure, loses magnetic energy as $`\exp(-2\eta k^2 t)`$ to 0.08% along an axis, 0.3%
  along the diagonal and 0.6% in 3D, on blocks too, with the field out of the plane (through the
  in-plane edge fields of 2D) as well; total energy holds to $`4\times10^{-7}`$, and a magnetized
  Kelvin-Helmholtz run on two levels with per-block timesteps conserves it to
  $`4\times10^{-7}`$. A Harris sheet of half-width $`a = 0.05`$ at $`S = a v_A/\eta = 500`$
  (512², the `tearing` preset seeded at $`10^{-7}`$) tears at about $`0.3`$ to $`0.4`$ per unit
  time while the sheet is still thin, falling as it diffuses wider; constant-$`\psi`$ theory
  ([Furth, Killeen & Rosenbluth 1963][FKR63]) gives about $`0.7`$ for the initial sheet. This is
  not yet a validation of the growth rate.
- **Magnetized tidal disruptions** (plan section 11). The tidal-disruption family's "Partial,
  dipole field", "Partial, random field" (beta 0.7) and "Full, random field" (beta 1.5) set the
  field as Guillochon & McCourt (2017) did in FLASH (their MultiTidalPoly
  `Simulation_initBlock.F90`, and the notebook that wrote its `mags.dat`): A = sqrt(2 p / beta_M)
  times a shape inside 0.98 of the radius, p the star's pressure. Their beta_M is 1 / Az, so the
  plasma beta is about the same through the star (the dipole's is beta_M / 4 at the center),
  not set at the center as the paper says. The dipole's shape is (-y, x, 0); the random one
  their lattice, |FFT| of amplitudes k^(-17/6) with Gaussian moduli and random phases on 100^3,
  scaled to a largest value of 1, stretched across the star's diameter. (Their code also
  multiplied it by the cell size, which leaves a field 800 times weaker than the paper's figure
  1; without it the energy is that figure's.) At beta_M = 10^4 the dipole starts with 3.3e44
  erg (8.65e-5 code units, the exact integral of the formula) and the random field with 8.4e44
  erg on the presets' grid (6.8e44 from their own lattice at theirs); the least beta, about
  0.1, is in the edge cells just inside the cut. `mhd-tidal-disruption` checks the dipole's
  energy and central beta and runs the grazing encounter against its hydro control.
- **Magnetized accretion disk** (`problems/mri_disk.ts`, "Accretion disks" group). A globally
  isothermal disk (c_s = 0.15, so H / r = 0.15 r^(1/2)) around one Plummer-softened point mass
  with a sink, in exact vertical and radial balance with the softened potential: the density is
  rho_mid(R) exp(-(Phi(R, z) - Phi(R, 0)) / c_s^2), with rho_mid proportional to R^(-3/2) (a flat
  surface density) tapered inside R = 0.75 and outside R = 3, and the rotation
  v_phi^2 = R^2 / (R^2 + eps^2)^(3/2) + c_s^2 dln(rho_mid) / dln(R), the same at every height. The
  disk holds a hundredth of the central mass. The field is four poloidal loops with no net flux,
  as in [Hawley (2000)][Hawley2000] but laid along R: A_phi proportional to
  max(exp(-z^2 / 2H^2) - e^-2, 0) sin(pi ln(R / 0.75) / ln 1.35), closing at |z| = 2H, scaled so
  that beta = 50 at the midplane where two loops meet. The MRI's fastest wavelength there,
  2 pi v_A / Omega = 1.3 H, spans some 6 cells at r = 1 on the default 256^3 mesh (2 levels) and
  15 at r = 1.85; [Hawley, Guan & Krolik (2011)][Hawley2011] ask for 8 to 10. The finest level
  holds the gas denser than 5e-5 (about 2H from the midplane) and the sink. The damping zone of
  section 19 holds the outer disk beyond R = 3.25; with a field it acts on the gas alone and
  leaves B^2 / 2 in E as it was. Field that rises into the corona, and field a sink leaves
  behind, meet the Alfven ceiling below. `mhd-disk` runs it on both grids.
- **Disk diagnostics** (`mhd/mri.wgsl`; `magnetic.disk` in the API; the panel's α M, α R, Q z,
  Q φ rows and the views "MRI quality Q_z" and "MRI quality Q_φ"; for presets with
  `Problem.disk`). About the first point mass, in cylinders about its z axis, with the softened
  Keplerian Omega_K. The quality factors Q = 2 pi |B_i| / (sqrt(rho) Omega_K dx) of the field's
  vertical and azimuthal parts count the cells across the MRI's fastest-growing wavelength
  ([Hawley, Guan & Krolik 2011][Hawley2011]: above about 8 for Q_z and 20 for Q_phi the
  instability grows, and its turbulence saturates, as at higher resolution). Where beta < 1 the
  MRI does not grow, and Q is 0 there, in the view and the averages; the panel's are
  mass-weighted over the rest of the disk's body. The alphas are volume sums over the disk's
  body, between the sink and the damping zone and within two scale heights c_s / Omega_K of
  the midplane, over the summed pressure: the Maxwell stress -B_R B_phi, and the Reynolds stress
  rho dv_R dv_phi with the velocities measured from the mean flow of each of eight annuli
  (log-spaced), after the Keplerian speed is taken from v_phi. All 41 channels ride on the
  magnetic diagnostics' one reduction. On the default disk, after three orbits at r = 1: alpha_M
  0.026, alpha_R 0.003, Q_z 4.2, Q_phi 15; marginal, as expected at 256^3. Before the turbulence
  (one orbit) alpha_R is negative, -0.015: the inner edge's laminar adjustment, gas flowing in
  at half the sound speed with its orbital speed swinging about Keplerian, is a real flux of
  angular momentum, read before the flow turns turbulent.
- **Alfven ceiling** (`hydro/fluff.wgsl` `alfven_floor`; `Problem.fluff.va`, the panel's "Max
  Alfven speed", link key `fva`). Where |B| / sqrt(rho) would pass the ceiling after a cell's
  update, mass is added at rest until it does not: momentum stays, so the added gas takes kinetic
  energy but not heat. Off (0) except in the disk, where it is 3, a little above the orbital
  speed at the sink's edge. Not conservative: the added mass is the price of a timestep that a
  few nearly empty magnetized cells would otherwise set.
- **Fields inside stars** (`mhd/star_fields.ts`, plan section 8). A dipole along z,
  $`\mathbf{A} = A_0 \max(p - p_\text{cut}, 0)\,(-y, x, 0)`$, normalized by $`B_z = 2A_0(p_c -
  p_\text{cut})`$ at the center to the requested beta; or a random field, a sum of 64 Fourier
  modes of A between $`2\pi/R`$ and 8 times that with $`|A_k| \propto k^{-17/6}`$ (a Kolmogorov
  $`E_B(k) \propto k^{-5/3}`$), tapered the same way and scaled so that the mean $`B^2`$ within
  $`R/5`$ of the center gives the beta. The taper takes the preset's own pressure profile, not
  the settled one, except for a star in a binary's Roche potential: there it takes the pressure
  of the star as it will settle (`problems/roche_scf.ts`), solved before the run in the
  settling's own balance, $`h + \phi = C`$ in its own potential (by FFT on 32 cells across its
  reach), the companion's as a point, the centrifugal term and the settling's hold, by
  self-consistent-field rounds. A star that nearly fills its lobe is stretched to a point at L1,
  out where the round profile has no pressure; tapered by the round profile, the field stopped
  short of the tip at L1 that feeds the stream. Against the settled 0.6 M☉ donor of a run at
  fill 0.97 the solved pressure is within 10% everywhere, out to the tip. With the taper linear in p, B falls about as fast as p toward the surface
  (in a cold white dwarf a little faster, about $`p^{1.4}`$), so the plasma beta climbs as
  $`1/p`$ or more: in a 0.6 solar mass white dwarf with beta $`10^5`$ at the center, it is
  $`10^8`$ by $`6\times10^4`$ g/cm³. The taper `sqrt` (`field.taper`) takes its square root
  instead, as Guillochon & McCourt's fields do, so B follows $`\sqrt{p}`$ and the beta stays
  about the central one out to the field's edge. The faces are the line integrals of A as for every other field, so coarse and
  fine agree and $`\nabla\cdot B`$ is at rounding. In `mhd-star-fields` the dipole polytrope
  (beta 1000) keeps its central density to 0.5% (0.1% on blocks) and its magnetic energy to 2%
  for two dynamical times; the random field loses 15% of its energy in one, its small scales
  being a few cells across at $`64^3`$.
- **Field brushes** (`edit/field_brush_core.wgsl`; the tools "Field strength" and "Pull
  field", offered in runs with a field). Both change the field by the curl of a field on the
  edges, taken on each face as its circulation around the face, so every cell's divergence stays
  as it was, to rounding. For the strength brush that is a change of the vector potential, the
  field's own in the radial gauge about the brush's center $`x_0`$,
  $`\mathbf{A}(x) = \int_0^1 t\,\mathbf{B}(x_0 + t d) \times d\,dt`$ with $`d = x - x_0`$,
  integrated exactly: the field in a cell is taken as the scheme holds it, each component linear
  along its own axis between the cell's two faces, so $`t\mathbf{B}`$ is quadratic along the ray
  within a cell, and Simpson's rule cell by cell is exact. That field is divergence-free, with
  its normal component continuous across faces, so the curl of its potential is the field. (A
  field constant in each cell is neither: its potential's curl missed the field by the jump
  between neighboring cells, which between the coarse cells of a turbulent 2D run at 1024 with 5
  levels was as large as the field, and a pull left cell-sized patches of strong and vanishing
  field.) (In 2D the
  in-plane field's $`A_z = \int_0^1 (\mathbf{B} \times d)_z\,dt`$, without the $`t`$: the plane
  has no $`\partial A_y / \partial z`$ to give the other half of $`B_x`$.) With $`w`$ the
  brush's weight:
  - Field strength: $`\delta\mathbf{A} = (f - 1)\mathbf{A}`$, $`f = e^{\pm w \ln 1.5}`$ (Shift
    weakens), so the field becomes $`f\mathbf{B} + \nabla f \times \mathbf{A}`$: 1.5 times
    stronger in the brush's hard core, with the flux it gains closing back around its rim.
  - Pull field: the induction equation, $`\delta\mathbf{B} = \nabla\times(\mathbf{v}\times
    \mathbf{B})`$, for the displacement $`\mathbf{v} = g\,w\,\delta`$ of a drag $`\delta`$ and a
    grip $`g`$ (the tool's Grip, 50% by default): the field lines near the center move by
    $`g\delta`$ and less of them toward the edge, and the gas stays where it is. A drag is cut
    into substeps that move the field at most half a finest cell, and $`\mathbf{B}`$ is read
    upstream of each edge, half the substep's displacement back, from the cell the field comes
    from (donor-cell constrained transport, with the cell's linear profile along the motion).
    Moving the potential instead, $`\delta\mathbf{A} = \mathbf{A}(x - \mathbf{v}) -
    \mathbf{A}(x)`$, was unstable on an adaptive mesh: where a coarse cell's face meets the finer
    faces beside it, the potential's curl is not quite the field, and each of the hundred small
    pulls of a slow drag added that difference again, shifted. Replaying 59 pulls recorded from a
    slow drag on the 2D turbulence (1024, 5 levels) grew the largest $`|B|`$ from 2.9 to 2100
    that way; it now stays at 2.9.
    Held fast ($`g = 1`$), the field the brush sweeps up would come to rest in it, its speed
    relative to the brush falling to zero, and pile up without bound: a fast drag across a
    turbulent field made sheets whose magnetic pressure blew the gas apart. With $`g < 1`$ the
    field slips through the brush at $`(1 - g)\delta`$, and conserving flux holds it to
    $`1/(1 - g)`$ times what it was along the drag (in `mhd-field-brush`, a sweep across the
    box: 2.6 times at 50%, where the lines also bend around the brush; 20 times at 95%).
    Across the rim the drag squeezes the field ahead by $`1/(1 - g\,|\nabla w|\,\delta)`$ in a
    frame: a frame's drag is kept to 0.15 of the rim's width (at least 0.3 of the radius) over
    the grip, and to 30 finest cells, which holds that to 1.3, and a fast drag catches up over
    the frames after.

  Edges are integrated at the midpoints of their finest-level segments, at positions exact in
  float32 and in a fixed order: the copies of a shared face get the same bits, and a coarse
  face changes by the mean of its fine faces' changes. Each face adds its four edges' terms to
  its field and compensation with error-free sums, as the update adds its impulses: the
  potential grows with the distance from the brush's center while a face's change is a
  difference of it across a cell, and a plain sum left $`\nabla\cdot B`$ at that distance
  over the cell's side times float32's rounding. An edge on a boundary face that is not
  periodic does not change, so neither does the flux through the boundary. The total energy of
  each cell then changes by that of its $`B^2/2`$; kinetic and internal energy stay. In
  `mhd-field-brush` the brush's hard core on the rotor's uniform field comes out 1.5 times
  stronger to $`4\times10^{-8}`$, and every face in it on Orszag-Tang at $`64^2`$, whose field
  changes from cell to cell, to $`6\times10^{-6}`$; a frame's pull on Orszag-Tang's coarse mesh
  raises the largest $`|B|`$ by 9%, and sixty small pulls across its refinement boundaries
  (512 finest, 2 levels) leave it as it was; on both grids $`\nabla\cdot B`$ stays below
  $`1.4\times10^{-7}`$, the
  total energy changes by the magnetic energy's change to $`4\times10^{-4}`$ of it, and shared
  faces agree bit for bit (coarse the mean of fine to $`1.4\times10^{-7}`$). The gas brushes keep
  the field as it was; on blocks they had left the leaves' faces behind in the other parity,
  which a commit pass now carries along.
- **First-order flux correction** (`mhd/face_flux.wgsl`). Stage 2 applies the half-step
  state's fluxes to the state at the start. Next to the edge of a star in near-vacuum, stage 1
  fills a nearly empty cell (350 times its density in half a step); stage 2 then pushes the
  half-step cell's gas out through its other faces, more than the start held, and the update
  empties it while the energy that flowed in stays. The result was floored pressures, speeds
  of 5 to 8 in the ambient gas that cut the timestep sixfold against the hydro run, and a
  passive scalar below zero. A cell whose density stage 1 changed by more than a factor of 4
  either way (more than a strong shock compresses gas) is suspect, and every face of a suspect
  cell takes the first-order HLLE flux of the state at the start: for that cell, the
  first-order step over dt, which stays positive. The decision is the face's own, from its two
  cells, so the cells, slabs or blocks that share a face compute the same flux: mass, momentum
  and energy stay conserved, and the edge fields, made from these fluxes, keep the divergence.
  (Athena++ decides cell by cell after the update and redoes it; deciding before needs no
  second update pass.) On blocks, each stage assembles into its own textures, so stage 1's
  assembly, ghosts included, is still there for stage 2. The test in the 1D float64 reference
  shows the scheme itself is right: the failure needs a cell next to a convex edge, which has
  more faces to empty through than the one that filled it. A settled gamma = 5/3 star on its
  tidal-disruption grid now steps like the hydro control (no floored cells to t = 2, mass lost
  only by outflow).
- **First-order flux correction after the update** (`mhd/fofc.wgsl`; on blocks
  `mhd/amr_fofc.wgsl`). The test above misses a cell that stage 2 empties without stage 1
  having moved its density much: in the thin corona over a magnetized disk (issue #3: midplane
  density 25, ambient $`10^{-5}`$, loops of $`\beta = 100`$), the half-step fluxes of the
  neighbors carried out more mass than the cell held, at speeds of $`10^6`$ after the first step,
  and the run went non-finite within ten steps on either grid. So stage 2 also checks every cell
  after its update, as Athena++ does: one left with less than 1/8 of its density, or with no
  internal energy, is listed. The faces of a listed cell then take the first-order HLLE flux of
  the start in their mass, momentum and energy, and the listed cells and their neighbors are
  updated again. The field's fluxes and the edge fields stay those of the second-order step, so
  the field and $`\nabla\cdot B`$ are untouched. With first-order energy fluxes beside a
  second-order field, $`E - KE - B^2/2`$ can take the difference in the field's energy as heat (in
  $`\beta \ll 1`$ gas near a sink, $`10^7`$ times the cell's temperature; the block runs gained
  13% of their energy and stalled). So a cell updated again beside a corrected face whose
  $`E - KE - B^2/2`$ is more than 4 times its advected internal energy keeps the advected energy
  and rebuilds $`E`$ from it; elsewhere $`E`$ stays as the fluxes made it. Keeping the advected energy
  in every such cell was worse: in the thin gas over the poles it drifted cold, and on blocks the
  corona drained to the floor (thousands of floored cells at $`64^3`$, tens of thousands at
  $`128^3`$).

  On the uniform grid, a fix pass rewrites the listed cells' faces in the slab's flux buffers
  and a redo pass updates the cells again. A neighbor that the redo leaves all but emptied in its
  turn is listed too and corrected in a second round (without it the disk went non-finite at
  $`128^3`$ by step 10); round 0 judges cells on the update's own list alone, so which cells are
  listed does not depend on the slabs. The list runs through all slabs of a step: each slab
  fixes the faces its buffers hold and redoes the cells whose faces and field are at hand (its
  own but the top layer, and the two layers below), and slabs of 4 layers agree with one bit for
  bit, with outflow faces and with $`z`$ periodic (the last slab then holds $`z`$ face $`n + 1`$, the
  twin of face 1). The passes run 16 workgroups that step through the list and end at once when
  it is empty; there is no indirect dispatch, because the update has no storage buffer to spare
  under WebGPU's limit of 8.

  On blocks, the update marks a cell in the fourth channel, otherwise unused, of its new face
  texel, and its block at a corner of its ghost padding, which nothing reads. A scan lists each
  marked leaf and its face neighbors (with time bins, those that start a step: one at its
  midpoint takes the corrected register in its deposit), reading one texel per unmarked leaf.
  Stage 2's flux and update passes then run again over that list: a face of a marked cell takes
  the first-order flux, into the scratch and, on the block's surface, the register; the block's
  other surface faces take back the register's own entry, the very flux the neighbor used; inner
  faces are computed again. With several chunks every chunk's fluxes go first, as in the step,
  since a coarser or slower neighbor reads the corrected register. A second round follows, as on
  the uniform grid: a pass over the leaves of the first marks, in a texture of one float a texel
  (the face texture, of four, cannot be read and written in one pass), the cells the first round
  left with less than 1/8 of their density and adds their leaves' face neighbors to the list,
  and the fluxes and update run again over the whole list with both marks. It judges by density alone: it sees the internal energy after its floor, and marking
  every cold cell at the floor, step after step, left the cold gas over the poles at the pressure
  floor in a thousand cells and more. One chunk per leaf agrees with one chunk bit for bit.

  Measured on an RTX 3080, paired runs against the step without the correction: on the uniform
  grid Orszag-Tang at $`1024^2`$ 8% faster (the update's cell code, now one function, compiles
  better), an MHD blast at $`128^3`$ and $`256^3`$ 3% slower, the MRI box at $`128^3`$ within 2%; on
  blocks an MHD blast at $`128^3`$ with one level within 1%, and a field loop at $`1024^2`$ with two
  levels, whose step takes 1.1 ms, 11.5% slower: the two rounds add seven small passes a step
  (scan, fluxes, update, recheck, fluxes, update, reset of the list), about 0.13 ms in Chrome on
  Windows however little they do. They were nine (13%) until round 1 kept round 0's list, adding
  to it from the recheck, instead of emptying it and scanning again.
  Issue #3's disk at $`128^3`$ now runs 400 steps on the uniform grid with at most 5 floored cells,
  and on blocks with one and two levels and with time bins with at most 58 (from tens of
  thousands before the second round and the narrower energy rule), all with $`\nabla\cdot B`$ at
  $`10^{-7}`$. The gas around its sink thins, and the step falls with it, as the setup check warns.
  With the sink's point
  mass every block of that disk takes the fastest bin; the correction beside slower blocks is
  tested by a dense blob (density 10, speed 3) through gas of $`10^{-6}`$ threaded by a weak field
  across its path, at $`128^3`$ with 2 bins (`mhd-near-vacuum-bins`): some 1,900 leaves corrected
  in 120 steps while the bins save 30% of block updates, mass conserved to the last bit, and one
  chunk per leaf as one chunk.
- **Compile time.** GPU compilers inline every function at every call, so a lookup called at
  several places, or in a loop they unroll, is compiled that many times. The block grid's
  assembly took 4 s to compile in Chrome (far more in Firefox), with the face lookup inlined
  about a hundred times. The passes now make such calls from one loop whose bound the compiler
  cannot see through (`loop_count`, `amr/grid.wgsl`), with the same arithmetic in the same
  order: assembly 0.45 s, update 0.26 s, re-meshing 0.9 s. The uniform grid's cell-centered
  field reads its six faces directly instead of through the boundary mapping (edge fields
  0.45 s to 0.14 s). A magnetized run builds no hydro update or prediction, which it never
  runs (the hydro update was the slowest compile of a load). The same holds off the MHD
  passes: the uniform grid's hydro update compiled its fallback for a face tile that does not
  fit (two face fluxes per axis) and the shearing box's four sampled faces at separate call
  sites, which kept loops now hold once each (Sedov 11 s to 4 s, the Helmholtz detonation
  83 s to 27 s); and the block grid's multigrid stencil evaluated the isolated boundary's
  multipole expansion inside every neighbor lookup, inlined some 120 times in the residual and
  restriction, where it is now filled in from one loop for the cells on the domain's edge
  (11 s to 1.7 s). With a costly equation of state (Helmholtz, ANEOS) what counts is how many
  times its solve is inlined. Every load that may reach beyond a face converted a sheared
  ghost's second cell too, a second solve, compiled into every run: whether there is a
  shearing box is now a constant (`SHEAR_BOX`, the slot `<geometry>`), which halved the update
  again (the detonation 27 s to about 12 s). The prediction's shell, a rim two cells thick,
  makes its six loads from one kept loop, in the interior's order (5 to 10 s to 2 to 3 s); the
  interior keeps them inline. All give the same bits as before. `tools/compile_times.mjs`
  times each pipeline of a setup, alone, on a cold shader cache; the machine's other load
  moves its figures by up to a factor of two, so compare builds run by turns.
- **Memory.** Five face textures (field and compensation at both parities, and the half step)
  add 80 bytes per cell to the uniform grid's 112 (`MhdPasses.cellBytes`), plus the slab
  buffers.
- **Snapshots.** A file of a magnetized run holds the cell-centered field (`Bcc1`-`Bcc3`, as
  Athena++ names them) for analysis. For a restart, a uniform grid's file adds each cell's lower
  faces (`Bf1`-`Bf3`) with the upper boundary faces in the metadata. A block grid's file adds
  every face of every leaf, upper faces included, in one more dataset (`ShaderousFaces`, which
  other readers ignore). Either way a restart on the same grid or mesh rebuilds the face field
  bit for bit. The mesh comes back from the file's levels; a leaf the file lacks (another root
  grid, another block size) refuses the restart, as does a uniform grid of another size. The
  compensations restart at zero (`mhd-restart`: densities agree with the original run to
  $`4\times10^{-7}`$ three steps later, on blocks as on the uniform grid).
- **Block grid** (`mhd/amr_*.wgsl`, `mhd/amr_mhd.ts`; plan section 4b). The same three passes per
  stage, block by block, after an assembly that fills every leaf's ghosts. That includes the
  edge ghosts beside its edges, where one coordinate is up to two cells out and a second is one
  cell out. Every ghost is found from its global position through the leaf index (`IDX`):
  copied from a leaf at its level, restricted from a finer one, or prolonged from a coarser
  one. A face on a refinement boundary is taken from the block on its own level. Prolonged faces
  carry limited slopes across them (without them, the jagged fine ghost field set off
  odd-even noise along refinement boundaries with HLLD). At a refinement boundary:
  - a coarse cell takes the mean of the fine fluxes, from the fine blocks' flux register;
  - every edge takes the value of the finest blocks around it (looked up through `IDX`, so
    blocks that share only an edge or a vertex are covered too), from their edge atlas;
  - so each coarse face stays the mean of the fine faces on it.

  The tree keeps 2:1 balance across edges (3D) and vertices (2D) for runs with a field. Initial
  faces come from line integrals of $`\mathbf{A}`$ summed over the finest level's segments, so
  a coarse face starts as exactly the mean of its fine ones. The flux scratch holds a chunk of
  leaves at a time (256 MB), with fluxes computed again for the update when there is more
  than one chunk. In tests:
  - an unrefined block grid matches the uniform grid to $`10^{-7}`$, and every face that two
    blocks share is bit for bit the same;
  - on refined meshes (rotor 2D, blast 3D) energy is conserved to $`2\times10^{-7}`$, and each
    coarse face equals its fine mean to $`6\times10^{-8}`$;
  - magnetic energy is within 0.3% (rotor) and 0.1% (blast) of the uniform grid at the finest
    resolution;
  - chunking changes no bit.
- **Re-meshing** (`mhd/amr_regrid.wgsl`, plan section 6), beside the cells' re-meshing and with
  the same table:
  - A copied block keeps its faces and their compensations.
  - A merged block's face is the mean of the old children's faces on it. On its boundary it
    takes instead the copy (and compensation) of an old leaf at its level, so the two copies
    agree bit for bit.
  - A new child first gets the fine faces on each coarse cell's faces. Where an old leaf at the
    fine level held one, it is copied with its compensation (never interpolated where fine data
    exist). Otherwise the coarse face is used, with limited slopes across it that keep its mean.
    The faces inside each coarse cell then start from the mean of the two outer faces in line
    with them. They are corrected by the least change, $`\delta = -A^\top L^{+} r`$, that makes
    every fine cell divergence-free. Here $`r`$ is the fine cells' divergence, $`A`$ the
    incidence of interior faces on cells, and $`L = AA^\top`$ the Laplacian of the hypercube of
    $`2^d`$ fine cells. The Walsh functions diagonalize it (eigenvalue $`2|s|`$), so $`L^+`$ is a
    transform and a division. The construction aims where [Tóth & Roe (2002)][TR02] do (exactly
    divergence-free, conservative, existing fine faces kept), but it is not their formulas.
  - The total energy of re-meshed cells is rebuilt from their kinetic and internal energy and
    the new $`B^2/2`$.

  The tree and leaf index of the old mesh are copied before the new one is uploaded, for these
  lookups. In the `mhd-amr-moving` test, a dense field loop carried once around the box is
  re-meshed 144 times (the `field-loop-dense` preset). $`\nabla\cdot B`$ stays at
  $`1.5\times10^{-7}`$, same-level shared faces are bit for bit, coarse faces equal their fine
  mean to $`6\times10^{-10}`$, and 96% of the magnetic energy is kept (uniform 256²: 96%).
- **Time bins** (`cfg.timeBins`, the `tbins` key; plan section 5b). Blocks take steps of
  $`2^k`$ times the smallest, as for the gas alone (§18), with the field's own rules:
  - Neighbors across faces, edges and (2D) vertices are at most one bin apart, and a block is
    never slower than a coarser one. So beside a block that starts a step, a neighbor is either
    starting one too or exactly at its midpoint.
  - The assembly reads a neighbor's ghosts as a blend of its state and its half-step store,
    weighted for the time the receiving block's stage needs: the neighbor's start, its
    midpoint, its end (extrapolated, from a faster neighbor in stage 2), or 3/4 of its step.
  - Each edge is defined by the finest leaf around it and, among those, the fastest. A slower
    block takes a faster neighbor's first-substep fluxes (times 1/2) and edge impulses in its
    own step, and the second ones in a deposit at its midpoint (`LD`, the leaves at their
    midpoint with a faster neighbor across a face, edge or vertex). So both copies of a shared
    face see the same impulses.
  - The passes run over lists built on the GPU, with dispatch sizes made there too: the
    assembly, the commit and the deposits from `amr/bins_args.wgsl`, the fluxes, edges and
    update chunk by chunk of the flux scratch from `mhd/amr_chunk_args.wgsl`.

  An edge whose cells lie beyond an outflow or reflecting boundary takes nothing from there: the
  definer search skips them (mapped back into the box, they named a neighbor's edge one cell
  off). In the `mhd-amr-bins` test (2 bins, kept on), the rotor, the dense field loop and the
  2D and 3D blasts keep $`\nabla\cdot B`$ at $`2\times10^{-7}`$, every shared face bit for bit, energy
  as with one timestep, and magnetic energy within $`10^{-4}`$ of the run with one timestep.
  The saving depends on the contrast in signal speed. The 2D blast at $`t = 0.05`$ makes 85%
  of the block updates of one timestep. The rotor, the loop and the 3D blast make 98-100%:
  their signal speeds differ by less than the factor of 2 between bins almost everywhere. With
  3 bins the 2D blast makes 89%: neighbors are pulled into the fastest bin for a longer cycle.
  Chunked substeps change no bit (`mhd-amr-chunks`).
- **Field lines** (`render/fieldlines.ts`; plan section 10; the view's Field lines switch,
  on by default with a field). Each frame the cell-centered field and the velocity are
  resampled onto a small volume (up to $`128^3`$ in 3D, $`1024^2`$ in 2D; `render/bsample.wgsl`,
  `amr/bsample.wgsl`, from the leaf cell under each voxel on blocks).
  - In 2D the lines are contours of the flux function $`A_z`$ ($`B_x = \partial_y A_z`$,
    $`B_y = -\partial_x A_z`$), integrated from the face field itself along the bottom row and
    up each column, on a grid of up to $`2048^2`$ corners (`render/aflux.wgsl`). The faces are
    divergence-free to round-off, so the flux to a corner does not depend on the path, and
    neighboring columns agree. On blocks, a face of the finest level inside a coarser cell takes
    the normal component interpolated linearly between that cell's faces, which keeps the field
    divergence-free there. (Integrated first from the resampled cell-centered field, each column
    drifted on its own and the contours came out jagged at 2048^2.) The same flux passes between any two lines, so they
    crowd where the field is strong; and since $`A_z`$ moves with the gas in ideal MHD, so do
    they. The count is the number of levels, evenly spaced across the range of $`A_z`$.
  - Colors, everywhere: the magnetic pressure $`B^2/2`$ on a log scale, over at most six decades
    below the strongest field shown (3D: from the weakest on the lines; 2D: from as far below the
    mean of $`\log|B|`$ in the plane as the strongest is above it), in a rainbow from red
    (weak) through yellow, green, cyan and blue to magenta (strong), as
    [Guillochon & McCourt (2017)][GM17] drew theirs. The colors set the lines apart from the
    view's contour lines.
  - In the plane (2D, and the slice view of a 3D run) the lines are drawn 1.5 pixels wide,
    gently shaded. Where flux-function levels come closer than about twice that width they fade
    out.
  - In 3D the lines are traced from seeds (`render/fieldlines.wgsl`): from each seed both ways
    along $`\hat{B}`$ with midpoint steps of 1/256 of the box (1/512 in a plane), up to a box
    length each way, until it leaves the box, closes on itself, or the field fades below 1% of
    its largest value. The seeds move with the gas (the clock's time since the last frame times
    the velocity), so each stays on its line as the field evolves and the picture does not
    flicker. A seed is placed again, where it leaves the box or where its field falls below 2%
    of the largest, as one of 24 random points drawn with weight $`|B|^2`$ (magnetic energy),
    within the bounding box of the voxels where the field is at least 2% of the largest (a star
    a few radii across in a box of sixteen fills a thousandth of it: drawn over the whole box,
    nearly every seed would miss it).
  - Volume view: each segment is a capsule 0.0008 of the box in radius (at least 0.6 of a pixel
    at its distance, so a far tube stays a continuous line rather than breaking into dashes)
    drawn on a camera-facing quad, found and lit per pixel (ray-capsule intersection), colored as
    above. The tubes go into color, distance and depth targets first; the ray-marcher stops at a
    tube's distance and lays the gas in front of it over it, so tubes behind dense gas fade. The
    block outline drawn after the gas is tested against the tubes' depth, so it passes behind
    the tubes in front of it; the tools' guides stay on top.
  - Slice view of a 3D run: the lines follow the field's components in the slice plane, from
    seeds in the plane weighted by $`|B|`$, drawn as lines over the slice. (The field in a
    plane of a 3D run is not divergence-free, so it has no flux function.)

  The `api.spec.ts` test draws them on Orszag-Tang (2D) and the blast in 3D, uniform and on
  blocks, in both views.
- **CPU reference.** `src/reference/mhd.ts` is the same scheme in float64. After 30 steps of
  Orszag-Tang at 128² and 15 of a 3D blast the GPU agrees with it to $`10^{-7}`$ in density,
  energy and face field.

Measured (RTX 3080, Chrome), all in the validation suite:

| Test | Result |
| --- | --- |
| Brio-Wu, 256 cells | Compound wave at $`x = -0.03`$, slow shock at 0.146, $`\min v_y = -1.593`$; energy to $`2\times10^{-7}`$; HLLD and HLLE agree to 0.002 in mean density |
| Alfvén wave, one period | $`L_1(v_z) = 7.9\times10^{-5}`$ at 128², $`1.97\times10^{-5}`$ at 256²: order 2.00 |
| Field loop, 128², $`t = 1`$ | 93% of the magnetic energy kept (MC limiter 94%); peak $`|B|`$ 8% high (see below) |
| Orszag-Tang, 256², $`t = 0.5`$ | Energy to $`9\times10^{-8}`$, $`\nabla\cdot B`$ $`7\times10^{-8}`$ |
| Blast, 64³ | Energy to $`10^{-6}`$, stretched twice as far along the field as across it |

**The field loop's peak.** $`|B|`$ overshoots its initial value on the loop's leading edge,
and at its center, where the field's direction is undefined. The overshoot is 8% at 128² with the
default limiter, 4% with MC or minmod, and 12% at 256². It is the same with HLLE as with HLLD,
so neither the reconstruction nor the Riemann solver causes it. It comes from the edge fields,
and the edge-field formula has been compared line by line with Athena++'s
`field/calculate_corner_e.cpp` (the same weights, grouping and signs). In 1D it reduces
exactly to the Godunov flux. So this is a property of the scheme, not a bug.

**Lee's upwind averaging, tried.** [Lee (2013)][Lee13] argues that for flow mostly along one
axis the edge estimates from faces downstream of the edge carry no information about it, and
keeps only the upwind ones (upwind-MEC). That selection was built as an option of the edge
pass, applied to Gardiner & Stone's four estimates. Against the mean, at $`t = 1`$:

| Field loop | Mean | Upwind |
| --- | --- | --- |
| Peak $`|B|`$ / initial, 128² and 256² | 1.08, 1.12 | 1.15, 1.17 |
| Peak $`|B|`$, advected at 0.57° to x (Lee's hard case), 128² | 1.08 | 1.27 |
| Largest $`|B|`$ outside the loop / initial | 0.17 to 0.25 | 0.26 to 0.38 |
| Mean error of $`|B|`$ and of its azimuthal direction inside | 0.015 to 0.024 | 0.032 to 0.074 |

Orszag-Tang and the Alfvén wave are unchanged to a percent. So the selection alone makes things
worse, and the mean shows no small-angle trouble here.

**Lee & Deane's estimates, tried.** Lee's upwind-MEC selects among the estimates of
[Lee & Deane (2009)][LD09], not Gardiner & Stone's. Each face's field is carried to the edge by
a Taylor series, $`E + \tfrac{\Delta}{2}E' + \tfrac{\Delta^2}{8}E''`$, with central differences
of the face values along their line (their eqs. 54 to 59). That is the parabola through three
faces in a row, $`(6E_j + 3E_{\rm toward} - E_{\rm away})/8`$. Both were built as options on
the uniform grid; they need the faces two along each line. The four constructions at $`t = 1`$:

| Field loop | GS05 mean | GS05 upwind | LD09 MEC | Lee upwind-MEC |
| --- | --- | --- | --- | --- |
| Magnetic energy kept, 128² / 256² | 93.3% / 96.1% | 93.3% / 96.2% | 94.5% / 96.7% | 94.6% / 96.9% |
| Peak $`|B|`$ / initial, 128² / 256² | 1.08 / 1.12 | 1.15 / 1.17 | 1.21 / 1.26 | 1.17 / 1.19 |
| Stray $`|B|`$ outside, 128² / 256² | 0.17 / 0.25 | 0.26 / 0.38 | 0.22 / 0.33 | 0.21 / 0.31 |
| Interior error, 128² / 256² | 0.019 / 0.015 | 0.054 / 0.032 | 0.031 / 0.021 | 0.028 / 0.020 |
| At 0.57°: kept, peak, stray, interior error | 94.2%, 1.08, 0.17, 0.024 | 94.4%, 1.27, 0.35, 0.074 | 95.6%, 1.23, 0.34, 0.059 | 97.1%, 1.19, 0.16, 0.019 |

Orszag-Tang and the Alfvén wave move by under 0.2% in every column. Lee's full upwind-MEC is
the least dissipative. In the small-angle case it was designed for, it also keeps the loop's
shape best, with the least stray field and the smallest interior error. But it raises the peak
overshoot, and at 45° the plain mean is cleaner on every shape measure. The code keeps only the
mean (Gardiner & Stone, as Athena++); the other three were removed in November 2026 and are in
the history (commit 9513236).

## 22. Rotating frames, symmetry and coordinates

### Rotating frame

A setup may run in a frame turning at a constant angular velocity $`\boldsymbol\Omega`$ about
an axis through the box center (`physics.rotation`; `gravity/rot.wgsl`). The gas then feels

```math
\mathbf{a} = -2\,\boldsymbol\Omega\times\mathbf{v} - \nabla\Phi_c,
\qquad \Phi_c = -\tfrac12\,|\boldsymbol\Omega\times\mathbf{x}|^2 ,
```

and all velocities, in the initial state, the outputs and the files, are those relative to the
turning frame (`problems/rotating.ts` converts an inertial setup). The centrifugal force can be
switched off, leaving the Coriolis force alone (an f-plane). In 2D only the part of
$`\boldsymbol\Omega`$ along $`z`$ is used.

**Centrifugal force.** $`\Phi_c`$ is added to the external potential of the point masses and
the accelerating frame (`gravity/ext.wgsl`), so every pass that sees that potential sees it: the
half kicks (with the energy in kick form, as for the masses), the gravity limit of the
timestep, the well-balanced reconstruction, which then holds gas at rest in the rotating frame,
and the potential view, which shows the effective potential. Its difference across a face is
formed directly,
$`-(\boldsymbol\Omega\times\mathbf{x})\cdot(\boldsymbol\Omega\times\mathbf{d}) - \tfrac12|\boldsymbol\Omega\times\mathbf{d}|^2`$,
not as the difference of two potentials. The diagnostics add $`\rho\Phi_c`$ to the potential
energy, so the total energy reported is the Jacobi integral, which is what the frame conserves.

**Coriolis force.** Over a time $`\tau`$ it turns the momentum by $`-2|\boldsymbol\Omega|\tau`$
about $`\boldsymbol\Omega`$, which Rodrigues' formula does exactly (with $`1 - \cos\theta`$
formed as $`2\sin^2(\theta/2)`$). A turn keeps $`|\mathbf{m}|`$, so the force does no work and
the total energy needs no source term. The turn is split around the step,

```
turn(dt/2)  kick(dt/2)  fluxes  kick(dt/2)  turn(dt/2)
```

which is symmetric in time; the predictor adds $`-2\boldsymbol\Omega\times\mathbf{v}`$ to its
half kick. The turns live in a shader slot of their own, the frame slot
(`gravity/frame_off.wgsl`, `frame_rot.wgsl`), applied by every update (uniform grid, blocks
with or without per-block timesteps, and both MHD stages, where stage 1 takes its half step's
whole share) and after the second kick, so a run without a rotation compiles without it and is
unchanged bit for bit. The induction equation is the same in a rotating frame, so the field
needs nothing. Point masses turn their velocities in their drift and kick passes and feel the
centrifugal force. The step is limited to a tenth of a radian of frame rotation, about 63
steps a turn; the turn is exact, so this bounds the splitting error, not stability.

Debris energies and the disk stresses add $`\boldsymbol\Omega\times\mathbf{x}`$ back to the
velocities, which they need in the inertial frame.

**Rotating stars.** In a frame with the centrifugal force the stars of the initial conditions
are settled ([§16](#16-initial-states-in-discrete-balance)) at rest in it, spinning rigidly at
$`\Omega`$: each is balanced in the centrifugal potential of its spin about its own center
$`\mathbf{x}_c`$ as well,

```math
h(\rho) + \phi - \tfrac12\,|\boldsymbol\Omega\times(\mathbf{x}-\mathbf{x}_c)|^2 = C ,
```

which flattens it into a rotating polytrope (`Settling.omega`, `inFrame` in `problems/settle.ts`;
`amr/settle.wgsl` on the GPU). For a star on the axis this is the frame's own $`\Phi_c`$; off
it, $`\Phi_c`$ differs by a part linear in $`\mathbf{x}-\mathbf{x}_c`$, a uniform force that
moves the star whole and is left out as its companions' pull is. (The stars of a circular
binary are settled in the Roche potential instead, which has the frame's whole centrifugal
term: [§16](#16-initial-states-in-discrete-balance).) The star's gas ends at the
ridge of its effective potential, $`w^3 = GM/\Omega^2`$ from its axis (its mass $`M`$ taken as
a point): past it the centrifugal potential wins, and the condition would fill the box's far
corners. The `polytrope-rotating` preset is the $`n = 1`$ polytrope at $`\Omega = 0.4`$ (its
second moments' axis ratio 0.908). On $`128^3`$ cells, settled on the CPU (uniform grid) or the
GPU (blocks, one level), it holds its central density to 0.2% and its axis ratio to $`10^{-3}`$
over four dynamical times, its gas denser than a tenth of the center slower than 0.004 in the
frame, as still as a star that does not spin; the two settlings agree to $`10^{-5}`$. Settled
without the spin, the same star changes its axis ratio by 0.15 (`rot-star`). On $`64^3`$ the
equator's thin outer layers stir to most of their sound speed within a few dynamical times,
though the star keeps its shape to 0.01.

Tests (`validation/frames.ts`): uniform flow on an f-plane turns at $`2\Omega`$ with its phase
right to $`3\times10^{-7}`$ rad and its kinetic energy to $`10^{-6}`$; isothermal gas held by
the centrifugal force stays below Mach $`10^{-5}`$ for more than a sound crossing (Mach 0.02
without the well-balanced reconstruction); a blob carried by gas moving straight in the
inertial frame lands within a sixth of a cell of where it should after a quarter turn; a binary
at rest in its corotating frame keeps its separation to $`10^{-10}`$ for two orbits; a
uniform field along the axis is left exactly as it was; and a rotating polytrope holds its
shape (above).

### Mirror symmetry

A setup symmetric under mirrors through the origin (`domain.symmetric`) can be run on the part
above its planes of symmetry (`domain.mirror`, link key `sym`): an octant in 3D or a quadrant in
2D, and on any one or two axes. Cells stay cubic. Where every axis in use is mirrored, each
mirrored axis's domain is $`[0, L]`$ instead of $`[-L/2, L/2]`$, with $`L`$ the box side, so an
octant at $`n^3`$ has the cell size of the whole box at $`(2n)^3`$, for an eighth of the cells.
Where only some are (a half, or a quadrant in 3D), the grid is **rectangular**: a mirrored axis
has $`n/2`$ cells over $`[0, L/2]`$ and the others $`n`$ over $`[-L/2, L/2]`$, the same cube as
without mirrors at the same cell size, for half or a quarter of the cells (`gridDims`,
`src/domain.ts`). The uniform grid's textures, passes and readbacks take three counts; the
multigrid's levels halve each axis until the shortest has two cells (a coarsest level of $`2\times2\times4`$ and the like), with each face's boundary values
only as wide as that face. On blocks the tree keeps its cube of coordinates and creates only
the root blocks of the computed part, its upper face along a halved axis then the domain's
boundary, so the atlas holds half or a quarter of the blocks; a halved axis needs an even
number of root blocks, or the grid stays a cube. Fractions of the box (what the probe, the
brushes, the API and the display hand over) are of the computed extent along each axis: of
half the box along a halved one. Magnetic runs keep the cube for now. Against the whole box at
$`64^3`$, a Sedov blast mirrored about $`z`$ alone ($`64\times64\times32`$) or about $`x`$ and
$`y`$ ($`32\times32\times64`$) agrees in density to $`3\times10^{-6}`$ with the same unfolded
mass, and a uniform sphere's potential to $`2\times10^{-6}`$; on two levels of blocks the half
has half the leaves and the whole's totals.

**Gas.** The lower face of a mirrored axis is reflecting, so every reflecting path applies.
Two things differ from a wall. The Riemann problem between a state and its image has zero mass
flux in exact arithmetic but not in float32, so the flux through the plane is set to what it
must be, the normal momentum flux alone, and the mass on the grid is conserved to the bit. And
the well-balanced reconstruction treats the plane as an interior face between a cell and its
image (the potential is even across it), not as a wall the potential continues through.

**Gravity.** A plane of symmetry is a zero-gradient face for the potential, on every multigrid
level of both grids: the ghost is the cell's own image in the stencils, the prolongation, the
field's differences and the ghost layers of the blocks, where a block next to the plane finds
its coarse neighbors' images too. The other faces stay isolated. The multipole boundary
unfolds the grid's moments to the whole configuration's: the moment
$`\sum m\,x^i y^j z^k`$ of the whole is the grid's times 2 for each mirrored axis whose power is
even, and zero if any is odd. So the mass doubles per mirror, the center of mass lies on the
planes, and the quadrupole (and every term to $`l = 4`$) is that of the whole.

**Point masses** must lie and move on every plane, where each is its own image and its pull on
the gas is unchanged. Their reduction is unfolded where it is read: masses and scalar totals
times $`2^k`$, a vector's mirrored components zero and the others doubled per mirror, and an
angular momentum's component kept only along the one axis every mirror is normal to.
**Settling** needs nothing: the star's mass is measured on the grid, which holds its part.

**Magnetic fields.** A field is a pseudovector: across a mirror its normal component is kept
and its tangential ones change sign, the opposite of the conducting wall a reflecting face
otherwise is. Every place the field is mirrored (the cell-centered ghosts on both grids, a
tangential face beyond the plane, a block's ghost faces) takes that parity at a plane of
symmetry, and the normal face on the plane is a real face, updated from the edge fields on it.
A setup says its field qualifies (`domain.symmetricField`): its vector potential, an ordinary
vector, must be odd in its normal component and even in the others across each plane. A field
of $`\mathbf{A} = (x y^2 f(r), 0, 0)`$ pushing gas about gives, in an octant, the whole box's
density to $`4\times10^{-7}`$ and $`\nabla\cdot\mathbf{B}`$ at round-off; with a wall's parity
instead the density is off by $`5\times10^{-3}`$.

**Diagnostics** report the whole configuration: extensive totals times $`2^k`$, momentum and
the center of mass zero along mirrored axes. Snapshots hold the computed part with its true
bounds, so yt reads an octant as an octant, and carry the mirrors in their settings.

**The view** shows the whole configuration of an octant (or a 2D quadrant): the volume and
slice views fold their sample positions back across the planes (`|2d - 1|` in fractions of
the view's box), the overlays and the pointer's tools map between the computed box and the
view's (a fraction $`f`$ of the one is $`\tfrac12 + \tfrac f2`$ of the other), and the
`mirror` view setting turns it off. A rectangular grid always shows its images, which make the
cube of the box again (without them its part would be drawn stretched); its slice map stays
square, two texels to a cell along a halved axis. Field lines are drawn in the computed part
only.

What the mirrors would get wrong is refused with a message: a uniform field or a frame
rotation that breaks the symmetry, a periodic upper face, periodic self-gravity, turbulence
driving, a field not declared symmetric, a point mass or star off a plane, and more than one
star.

Tests: the mass flux through each plane is zero to the bit; a Sedov blast in an octant and in a
2D quadrant matches the whole box at the same cell size to $`3\times10^{-6}`$ in density; the
potentials of a uniform sphere and of a triaxial ellipsoid in an octant match the whole box's
to $`2\times10^{-6}`$; a settled polytrope's central density and the Evrard collapse's energy
behave as in the whole box (on blocks with per-block timesteps too); and a sink at the origin
takes in the same mass to $`10^{-7}`$.

### Comoving coordinates

A setup with `physics.cosmology` runs gas in an expanding background, in the form Enzo uses
([Bryan et al. 1995](https://doi.org/10.1016/0010-4655(94)00191-4)): positions are comoving, so
the grid does not change; densities and pressures are comoving ($`a^3`$ times the proper ones),
so the mass on the grid is constant; velocities are proper peculiar velocities and specific
energies proper ones, so temperatures and sound speeds are physical as they stand. With scale
factor $`a(t)`$,

```math
\begin{aligned}
\partial_t\rho + \tfrac1a\nabla\cdot(\rho\mathbf{v}) &= 0,\\
\partial_t(\rho\mathbf{v}) + \tfrac1a\nabla\cdot(\rho\mathbf{v}\mathbf{v} + p\,\mathsf{I}) &= -\tfrac{\dot a}{a}\rho\mathbf{v} - \tfrac{\rho}{a}\nabla\Phi,\\
\partial_t(\rho e) + \tfrac1a\nabla\cdot(\rho e\,\mathbf{v}) &= -\tfrac{p}{a}\nabla\cdot\mathbf{v} - 3\tfrac{\dot a}{a}p,\\
\nabla^2\Phi &= \tfrac{4\pi G}{a}\left(\rho - \bar\rho\right).
\end{aligned}
```

Time is in units of $`1/H_0`$ and the mean comoving density is 1, so $`G = 3\Omega_m/8\pi`$.
The clock starts at zero at the starting redshift; the end time follows from the ending one.

**The scale factor** is advanced on the GPU, since the timestep never visits the CPU: the
clock pass takes one RK4 step of $`\dot a = aH(a)`$,
$`H = \sqrt{\Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_\Lambda}`$, each step and keeps $`a`$ at
its start, middle and end with $`H`$ at the middle. A step changes $`a`$ by at most 1%. Over
$`a = 0.01`$ to 1 (466 steps) the float32 value stays within $`3\times10^{-7}`$ of float64.

**Every gradient carries $`1/a`$.** The passes work with the proper cell size $`a\,\Delta x`$
(the middle of the step's for the fluxes), so one scalar covers the flux update, the
predictor, the species and the timestep's CFL and gravity limits. Both cells of a face use
the same $`a`$, so their fluxes still agree to the bit. Positions keep the comoving size.

**The expansion drag** is solved exactly and split around the step, through the frame slot
(`gravity/frame_cosmo.wgsl`): before the fluxes from $`a`$ to the middle of the step, after the
second kick on to its end. The momentum scales as $`1/a`$, an ideal gas's $`\rho e`$ as
$`a^{-3(\gamma-1)}`$ (formed from series in $`a_1/a_0 - 1`$: GPU `pow` was biased enough to move
the pressure by $`10^{-4}`$ over a few hundred steps), and $`E`$ changes by exactly what the
two did. The predictor adds the drag's rates to its half step.

**Gravity.** The Poisson source is divided by $`a`$ at the end of the step it is solved for,
and the field stored is the proper acceleration $`-\nabla\Phi/a`$, so the kicks, the predictor
and the timestep use it unchanged. The energy takes the kick form (what the kick adds to the
kinetic energy) rather than the flux-form work: the flux form charged the energy with the
potential's differences while the kick took the field, and in the cold gas of the pancake
preset (temperature $`10^{-10}`$) $`E - KE`$ was then that disagreement, which the dual-energy
switch took for heat: the temperature rose $`10^5`$-fold in the first step. With the kick
form it falls as $`a^{-2}`$, as adiabatic expansion has it, to 0.4% after 47 steps.

The well-balanced reconstruction is off in comoving runs. It measures pressure deviations
from hydrostatic balance, and for cold gas falling far from balance those deviations, as large
as $`\rho\,\Delta\Phi`$, act as spurious pressure: a 0.2% density wave grew 32% faster than
linear theory with it and within 1% without. (Without expansion, a cold wave in a static
periodic box grows 13% further with it than without; that case is not one it was built for.)

**Equations of state** relate proper quantities, while the state holds comoving densities and
pressures. An ideal gas's $`p = (\gamma-1)\rho e`$ is the same in both, so it is used as it
stands. Any other is compiled behind wrappers (`comovingEosSlot`, `src/eos/eos.ts`): its
functions are renamed and called at $`\rho/a^3`$ and $`p/a^3`$ or $`\rho e/a^3`$, with
pressures and energy densities multiplied by $`a^3`$ on the way out, so every caller (the
update, the predictor, the floors, the diagnostics, the display) goes through them. A pass that
holds the clock sets $`a^3`$ for the states it works on (the middle of the step in the update);
the others read the scale factor the step encoder copies from the clock into the parameters
after every step. The expansion drag on $`\rho e`$ uses the gas's own $`g = p/\rho e`$: over a
step $`\rho e`$ goes as $`a^{-3g}`$ with $`g`$ taken half way through (second order). Uniform gas
plus radiation, their pressures equal at $`a = 0.1`$, expanding to $`a = 0.3`$ ends within
$`2\times10^{-7}`$ in temperature of a float64 integration of $`de = -p\,d(1/\rho)`$, where an
ideal gas would be off by a factor 2.6. The tabulated equations of state (white dwarf matter,
hydrogen and helium, ANEOS) are refused.

**Cooling** works on proper quantities: the state's densities are comoving, so the cooling
function is called at the proper density $`\rho/a^3`$ (at the middle of the step) and its rate,
per unit proper volume, multiplied by $`a^3`$; the temperature comes through the wrappers. Uniform
gas cooling by $`\Lambda\propto T^{1/2}`$ while it expands from $`a = 0.1`$ to 0.3 ends within
$`10^{-5}`$ of a float64 integration.

**Units.** A run is scale free unless `physics.cosmology.boxMpc` gives the box's comoving side
in Mpc/h (with `h`, default 0.7): then lengths are in that side over the box, times in
$`1/H_0`$, and a density of 1 is the mean matter density $`\Omega_m\rho_\text{crit}`$, which is
what $`G = 3\Omega_m/8\pi`$ says in code units. Readouts in physical units show proper
densities, pressures and energy densities ($`a^{-3}`$ times the state's) and proper lengths
($`a`$ times), at the scale factor last read; velocities and temperatures are physical as they
stand. Code units show the state's comoving values.

The first version takes a periodic box of gas, with periodic self-gravity or none, on a uniform
grid or on blocks with one timestep for all; anything else is refused with a message.
The `zeldovich-pancake` preset (Gravitational collapse group) runs the pancake of
[Zel'dovich (1970)](https://ui.adsabs.harvard.edu/abs/1970A%26A.....5...84Z) through its
caustic at $`z = 1`$.

Tests: the scale factor on the GPU against float64 (Einstein-de Sitter and flat
$`\Lambda`$CDM, $`3\times10^{-7}`$); uniform gas whose velocity falls as $`1/a`$ and comoving
pressure as $`a^{3 - 3\gamma}`$ ($`5\times10^{-7}`$, uniform grid and blocks); a 0.2% wave that
grows as $`a`$ (0.8%); a traveling sound wave whose phase, over 25 radians from $`a = 0.1`$ to
0.3, follows $`\int k c_s/a\,dt`$ with the proper sound speed falling as $`1/a`$ (0.07%, its
comoving wavelength fixed); gas and radiation expanding (above); and the pancake against its
exact solution at half its caustic time (density 0.6%, velocity 0.25%; on blocks 0.7% and 0.2%
on a mesh refined as it runs, and 0.7% and 0.3% on one fixed with the caustic in its fine half).
The blocks were off by 6.5% in density at the caustic until a re-mesh built each new cell's
total energy from its own kinetic energy ([§17](#17-adaptive-mesh-refinement)): the cold gas
there was split into finer blocks after the first steps, the interpolated $`E - KE`$ was
$`10^{-4}`$ against a heat of $`10^{-12}`$, and the dual-energy switch kept it, so the caustic
formed from gas $`10^8`$ times too hot. Ruled out on the way: a mesh fine everywhere from the
start, or coarse everywhere, reproduces the uniform grid at that size to every digit; the error
followed the caustic when it was moved off the blocks' faces; and with no re-mesh the coarse-fine
boundary, a quarter of the box from the caustic, costs nothing measurable.

### Shearing box

`physics.shear` sets up the local model of a differentially rotating disk: a Cartesian box
corotating at $`\Omega`$ about $`z`$ with shear parameter $`q = -d\ln\Omega/d\ln R`$ (3/2 for
Keplerian), $`x`$ radial, $`y`$ along the orbit and $`z`$ vertical. The gas feels

```math
\mathbf{a} = -2\,\boldsymbol\Omega\times\mathbf{v} - \nabla\Phi_t,
\qquad \Phi_t = -q\,\Omega^2x^2 \;\left(+\tfrac12\Omega^2z^2 \text{ if stratified}\right),
```

the rotating frame's exact Coriolis turn with the tidal potential in place of the centrifugal
one. The background flow $`v_y = -q\Omega x`$ is steady. The well-balanced reconstruction sees
only the vertical part of $`\Phi_t`$: the radial tidal pull is balanced by the Coriolis force on
the background flow, not by pressure.

**The shearing boundary.** Along $`x`$ the box is periodic with a shift that grows in time,

```math
f(x + L, y, z) = f(x,\ y + q\Omega L\,t,\ z), \qquad v_y \to v_y - q\Omega L ,
```

which is what a pattern carried by the background flow, $`f_0(x,\ y + q\Omega x\,t)`$, satisfies
([Hawley et al. 1995](https://doi.org/10.1086/175311)). (The first version shifted the ghosts the
other way, which a uniform state, a mass count and epicycles cannot see; the shearing wave below
found it, its amplitude 54% off and its phase right.) The clock carries the offset
$`\Delta y = q\Omega L\,t \bmod L`$, summed step by step with
Kahan compensation (formed from $`t`$ it would lose its digits over a long run). A ghost beyond
an $`x`$ face lies between two cells sheared along $`y`$ and is blended linearly from both, its
$`v_y`$ shifted by the jump of the background flow; the half-step state takes the offset half a
step on.

**Conservation.** The two cells that share a shearing face compute its flux from different
ghosts, so they disagree and mass would not be conserved. Each takes instead the mean of its own
flux and the one the other side computes, remapped and boosted into its frame (as the mesh
recomputes its neighbors' fluxes at refinement boundaries):
$`F_\text{lo}(j) = \tfrac12\left[F_\text{own}(j) + (1 - \epsilon)F_\text{other}(j + s) + \epsilon F_\text{other}(j + s + 1)\right]`$,
with $`s`$ and $`\epsilon`$ the whole and fractional cells of the offset. Summed over $`j`$ both
faces give the same total, so mass and the $`x`$ and $`z`$ momenta are conserved to rounding
(the total, not each face). A dense blob carried across the boundary for two orbits keeps its
mass to $`10^{-8}`$.

**Rounding.** In float32 the background flow is not held exactly: each step adds the Coriolis
and tidal increments to velocities of the order of the shear speed, and in a steady state the
rounding of those additions is the same every step, so it adds up, to about $`10^{-4}`$ of the
shear speed an orbit. Orbital advection (evolving the departure from the background) would
remove it and the shear speed's limit on the timestep; it is not implemented yet. A density
that sits on a power of two is a trap of the same kind: float32 spacing is $`6\times10^{-8}`$
just below 1 and $`1.2\times10^{-7}`$ just above, so a small flux increment survives rounding
on one side and is lost on the other, a one-sided drain. A uniform sheet at density and
pressure 1 lost $`2.3\times10^{-6}`$ of its mass over ten orbits on Metal; at 1.3, 1.4 or 1.75
(the same sound speed and steps) it keeps it to $`10^{-7}`$, so the test runs at 1.4. The Coriolis
turn takes its sines from their series for small angles: the GPU's `sin` is accurate in absolute
terms only, and its error over the small angle of one step was biased enough to shift the phase
of the rotating-frame test by $`7\times10^{-6}`$ (now $`3\times10^{-7}`$).

**Magnetic fields.** The face field keeps both $`x`$ planes of $`b_x`$ (texels 0 and $`n`$),
sheared images of each other, instead of one shared face. Ghost cells are blended as above, and
a ghost's tangential faces likewise from the two sheared faces. After each stage's flux pass and
after its edge pass, a small pass (`mhd/shear.wgsl`) replaces what each $`x`$ plane computed by
the mean of its own and the other plane's, remapped and boosted: the face fluxes as in the
hydrodynamic update, and the edge impulses $`E_y`$ and $`E_z`$ on the two planes (Athena's
remap of the electric field), $`E_z`$ with the frame change $`u\,B_x`$ of $`-\mathbf v\times\mathbf B`$
when $`v_y`$ is boosted by $`u = \pm q\Omega L`$. Every face around such an edge then changes by the
same number, so the divergence stays at round-off and the two planes' $`b_x`$ stay in step; and
the sums of the remapped impulses over a plane agree, so the net vertical and radial fluxes
through the box are conserved. Resistivity and viscosity are refused with a field. The
MRI's disk diagnostics (`mhd/mri.wgsl`) take the local frame there: Maxwell stress
$`-B_xB_y`$, Reynolds stress from the velocity less the background flow, and the quality factors
with $`\Omega`$ the box's.

The first version runs on a uniform grid, periodic along $`y`$ (and $`z`$ unless stratified);
blocks (whose neighbors across the boundary change with time), self-gravity and diffusion are
refused. The `shearing-wave` preset swings a leading density wave round to trailing; `mri-box`
grows the magnetorotational instability from noise in a vertical field. Tests: the background
over ten orbits at density 1.4 (`shear-steady`: velocity within $`8.8\times10^{-4}`$ of the
shear speed, density within $`2.4\times10^{-4}`$, mass kept to $`8.5\times10^{-8}`$; the test
asks for $`2\times10^{-3}`$, $`10^{-3}`$ and $`10^{-6}`$), the blob, epicycles at
$`\kappa = \sqrt{2(2 - q)}\,\Omega`$ (back to $`5\times10^{-5}`$ after one, epicyclic energy to
$`5\times10^{-5}`$); the compressive shearing wave of
[Johnson & Gammie (2005)](https://doi.org/10.1086/426637) at $`128^2`$, projected on the sheared
wave and against a float64 integration of the linear equations through its swing (amplitude
0.3%, phase $`2\times10^{-5}`$ rad, where it grows 1.9 times); a magnetized box with net
vertical field and a field crossing the $`x`$ faces, whose net vertical and radial fluxes keep to
$`5\times10^{-10}`$ over two orbits with $`\nabla\cdot B`$ at $`10^{-7}`$; and the MRI's fastest
channel mode growing at $`0.762\,\Omega`$ against $`0.75`$ ($`32^3`$, $`\beta = 50`$).

### Axisymmetric (R, z) runs

`domain.geometry: "cylindrical"` makes a 2D run a body of revolution about the $`y`$ axis. The
2D grid is one periodic layer of the 3D scheme and already carries the third velocity
component, so the layer becomes the azimuth:

| Code axis | Meaning |
|---|---|
| x | $`R`$, from the axis (the lower $`x`$ face) to the box side |
| y | $`z`$ |
| z (the layer) | $`\phi`$: `v_z` is $`v_\phi`$ |

Swirl comes at no extra storage, and the constant field of 2D runs already points along
$`-y`$, which is $`-z`$ here. The scheme solves

```math
\partial_t\mathbf{U} + \frac{1}{R}\partial_R\left(R\,\mathbf{F}_R\right) + \partial_z\mathbf{F}_z = \mathbf{S}:
```

- **Fluxes.** Each radial face's flux is weighed by its area over the cell's volume per
  $`\Delta`$, $`R_-/R_c = i/(i + \tfrac12)`$ and $`R_+/R_c = (i + 1)/(i + \tfrac12)`$: pure index
  ratios, with no cancellation in float32. The axis face weighs exactly 0, so nothing crosses
  it. The face's flux itself is the same bits from both cells, so $`\sum R_c\,U`$ is conserved
  to rounding. The $`p\,\nabla\cdot\mathbf{v}`$ term takes the same weights on the face normal
  velocities, which gives the cylindrical divergence.
- **Angular momentum.** The $`\phi`$ momentum flux through radial faces is weighed by
  $`(R_\pm/R_c)^2`$, with no source: $`R\rho v_\phi`$ is then conserved to rounding.
- **Radial source,** $`(p + \rho v_\phi^2)/R`$, with $`p`$ the mean of the cell's two radial
  face pressures (each face's normal momentum flux less what its mass flux carries). With it
  the cylindrical balance reduces exactly to the Cartesian one: uniform pressure is in balance
  to rounding, and the well-balanced reconstruction holds a stratified column, and a star
  held up radially by its pressure, as it does in Cartesian coordinates.
- **Energy where the gas swirls.** The radial faces carry the total energy weighed by $`R`$ and
  the angular momentum by $`R^2`$, so the kinetic energy of the swirl that the updated angular
  momentum gives and the one the energy update holds differ, by a few percent of it at a steep
  edge. In cold gas carried round by its rotation, the swirl's energy a hundred times the heat,
  $`E - \tfrac12\rho v^2`$ is then mostly that difference. At a torus's inner edge it drained
  the pressure within a few hundredths of an orbit, the edge lost its support, and the gas
  spilling inward, flung back out by its angular momentum, piled up in the edge cell: the peak
  density of a torus around a point mass rose 2.8 times in an orbit at $`64^2`$ with the shock
  detector's HLL fallback off, 2.3 times with the minmod limiter, 1.5 times with the
  well-balanced reconstruction (each taking away some of the dissipation that had held it
  down), while the same torus in 3D, whose momentum and energy share their faces, stayed
  within 7%. So the dual-energy switch (hydro/dual_energy.wgsl) takes $`E - KE`$ for the
  internal energy only where that is also at least a tenth of the swirl's kinetic energy
  $`\tfrac12\rho v_\phi^2`$, and keeps the advected internal energy elsewhere. The same torus
  then holds to 1% in an orbit with any of the three; at $`256^2`$, with the well-balanced
  reconstruction on, a torus of constant angular momentum keeps its peak density within 0.3% for
  five orbits (with the old switch, and the radial balance off, it wobbled by 3 to 4%). (Measuring the radial balance against the effective
  potential, gravity's plus the centrifugal, was tried as well, the step between two cells
  density weighted and each cell's own swirl for its half step: it brought the first orbit to
  0.5% but was not needed, and is not used.)
- **Predictor:** the half step's geometric terms, $`-\rho v_R/R`$ in the density,
  $`-\rho c_s^2 v_R/R`$ in the pressure, $`+v_\phi^2/R`$ in $`v_R`$ and $`-v_Rv_\phi/R`$ in
  $`v_\phi`$.
- **The axis** reflects as a plane of symmetry does, with $`v_\phi`$ odd as well as $`v_R`$.

Cells are rings of volume $`2\pi R_c\Delta^2`$, so the diagnostics' masses and energies are true
ones, and the momentum they report is along $`z`$ alone. Snapshots are written in Athena++'s
cylindrical order $`(R, \phi, z)`$, one cell in $`\phi`$ spanning $`2\pi`$, which yt reads as
such; the reader takes such files back.

**Blocks.** On an adaptive mesh the same weights are formed from each cell's global radial
index on its level. Restriction and prolongation are volume weighted: a parent is the mean of
its children weighted by $`(i + \tfrac12)/(2I + 1)`$, and the prolonged children are laid about
the coarse ring's centroid rather than its middle, so that $`\sum R\,U`$ over them is the
parent's. Where a coarse cell takes a refined neighbor's fluxes across $`z`$, each fine face
weighs by its own radius. Per-block timesteps are refused for now. On two levels a swirling
ring keeps mass, energy and angular momentum to $`5\times10^{-7}`$ through its re-meshing.

**Self-gravity** (isolated) is solved in $`(R, z)`$ on the uniform grid by the same
multigrid, its levels one cell deep: the stencil drops $`z`$ and weighs the radial neighbors
by $`R_\pm/R_c`$, with the diagonal their sum, so cells stay square and red-black smoothing
is unchanged; the residual is restricted with the rings' volumes. The multipole boundary
needs no new evaluator: each ring's moments are filled in by symmetry about the axis
($`\sum x^2 = \sum z^2 = \tfrac12 m R^2`$ and so on to fourth order, the center of mass on
the axis), and the boundary is evaluated in the plane of the run. The flux-form
gravitational work takes the weighted mass fluxes. A uniform sphere's potential comes out
within $`7\times10^{-4}`$ of the analytic one (the 3D solve: $`1.5\times10^{-3}`$), and the
Evrard collapse in $`(R, z)`$ keeps its energy to 0.04% through bounce.

On blocks the per-level solver of §13 runs in 2D. Its blocks are one cell deep, laid out in
each sublevel's atlas as the hydro atlas is in 2D, and a dispatch's four invocations along
$`z`$ take four blocks of the list. The stencil is the uniform grid's (five points, the radial
neighbors weighed by $`R_\pm/R_c`$ from the cell's radial index at that sublevel); the residual
and the source are restricted with the rings' volumes, so a parent holds its children's mass;
the coarse error is prolonged, and the ghosts across a coarse-fine face interpolated,
bilinearly. The axis is a plane of mirror symmetry for the potential, as in a mirrored run,
and the multipole boundary takes the rings' moments as on the uniform grid. On three levels
($`128^2`$ finest) the uniform sphere's potential is within $`1.2\times10^{-3}`$ of the analytic
one in every leaf, and the Evrard collapse keeps its energy to 0.24% through bounce, its
thermal energy peaking at $`t = 1.125`$ as on the uniform grid. Per-block timesteps stay
refused.

**Stars** of a setup document may sit on the axis and move along it (an orbit, which puts its
bodies off it, is refused). They are settled as in 3D (section on settling above), on the
uniform grid and on blocks: the settling's sums are of volumes, and a ring's is
$`2\pi R\,\Delta x^2`$, with positions $`(R, z, 0)`$, so the star's mass and its balance are a
body of revolution's. On blocks they settle on the CPU (the GPU's sums are of cubes). An
$`n = 1`$ polytrope of 25 cells' radius ($`128^2`$) settles in 8 rounds and keeps its central
density to 1.2% over three sound crossings (`cyl-star`), as the 3D polytrope does.

**The timestep** of an $`(R, z)`$ run weighs a cell's radial signal rate by its outer face's
weight, $`(R_c + \Delta x/2)/R_c`$ (twice next to the axis, where the face is twice as large
as the cell is wide), and holds the centrifugal acceleration $`v_\phi^2/R_c`$ over that same
weighted length to the acceleration limit, as for gravity. Without them a fast swirl reaching
the axis (a random setup, $`v_\phi = 0.66`$ in gas of sound speed 0.01) emptied the axis cells
in its first step and went to NaN; with them it keeps its mass to round-off.

**Point masses** may sit on the axis and move along it. Their pull on the gas needs nothing,
since the distance from a mass on the axis to any point of a ring is that to the ring's cell
center; the gas's pull on them, and what their sinks take, are summed over rings (their true
volumes), with the parts across the axis dropped where the reduction is read, as each ring's
cancel. A sink on the axis takes in within 1% of what the same sink takes in 3D.

**Magnetic fields** (uniform grid). The face field keeps its layout: $`b_R`$ on the radial
faces, $`b_z`$ on the $`z`$ faces, $`B_\phi`$ in the layer. With $`E_\phi`$ on the edges,
$`\partial_t b_R = -\partial_z E_\phi`$ is the Cartesian update; $`\partial_t b_z =
\tfrac1R\partial_R(R E_\phi)`$ weighs each edge by its radius over the face's,
$`R_\pm/R_c`$; and $`B_\phi`$'s flux form, $`\partial_t B_\phi = \partial_z E_R - \partial_R
E_z`$, has no radii in it. The cylindrical divergence,
$`(R_+b_{R+} - R_-b_{R-})/(R_c\Delta) + \Delta_z b_z/\Delta`$, is then unchanged by every edge,
as the Cartesian one is. The cell's $`B_R`$ is its faces' mean weighed by their areas. The
radial faces' fluxes take the same weights as the gas's ($`R B_\phi`$'s stress with angular
momentum, by the square), and the radial source becomes
$`(p + B^2/2 + \rho v_\phi^2 - B_\phi^2)/R`$, its total pressure the mean of the two radial
faces' (the flux pass leaves each face's $`B_R`$ beside its flux for this), so that a uniform
field along the axis is in balance to rounding. On the axis $`E_\phi`$ and the flux of
$`B_\phi`$ vanish, and beyond it $`B_\phi`$ changes sign, as $`v_\phi`$ does. A vector
potential's $`b_z`$ is formed with the same radial weights. Resistivity, viscosity and blocks
are refused with a field.

The first version is hydrodynamics and ideal MHD (conduction works through the face weights);
viscosity (which needs the curvilinear stress), point masses off the axis, species and
per-block timesteps are refused. The `blast-axisymmetric` preset is a point explosion on the axis.
Tests: uniform gas at rest stays below Mach $`10^{-6}`$; a swirling ring carried along $`z`$
keeps mass, energy, $`z`$ momentum and angular momentum to $`10^{-7}`$; a point blast is
spherical, its radius along $`R`$ and $`z`$ within 1.5% of the 3D similarity solution; a
stratified column under gravity stays below Mach $`10^{-4}`$; the spherical Noh implosion at
$`512^2`$ (R to 2, so that no gas measured came in through a boundary) has its post-shock
density within 2.7% of 64 and the pre-shock within 0.1%, alike along the axis, the diagonal and
$`R`$; Bondi accretion onto a point mass on the axis within 3.5% of Bondi's rate; and a poloidal
field loop carried along $`z`$ keeps $`\nabla\cdot B`$ at $`3\times10^{-7}`$ and 96.2% of its
magnetic energy (the same loop in Cartesian coordinates: 96.0%), with a uniform axial field at
rest to Mach $`3\times10^{-7}`$; and a pressure-supported torus of constant angular momentum
around a point mass on the axis keeps its mass and the radius of its center of mass to 1% and
its peak density to 0.3% for five orbits. Its faces are open for outflow only: open both ways, they let the mass pull gas in
from outside the box, which then piled up at the mass without bound.


### Spherical (r, θ) runs

`domain.geometry: "spherical"` (with `domain.rMin`, the inner radius, and `domain.logR`) makes
a 2D run a body of revolution about the $`z`$ axis on a grid of shells:

| Code axis | Meaning |
|---|---|
| x | $`r`$, from $`r_\text{min} > 0`$ to the box side, evenly spaced in $`r`$ or in $`\ln r`$ |
| y | $`\theta`$, from 0 to $`\pi`$: both poles are in the grid |
| z (the layer) | $`\phi`$: `v_z` is $`v_\phi`$ |

The cells are not the same size along the two axes, and on a logarithmic grid not from one
shell to the next, so the grid's $`\Delta x`$ (the radial extent over the cell count) is only
a reference length. Every pass that divides by it takes the cell's own lengths as ratios to
it instead: $`h_r/\Delta x`$ and $`r_c\Delta\theta/\Delta x`$ scale the predictor's slopes and
the timestep's signal rates, and the update's face weights are areas over volumes times
$`\Delta x`$, so the update itself is the Cartesian one.

The timestep takes, per axis, the larger of the cell's length and its volume over its larger
face's area: a cell on a pole has half the volume per area of its polar face that
$`r_c\Delta\theta`$ says, and the innermost shell over a small $`r_\text{min}`$ a radial face
up to twice as large as $`h_r`$ suggests. It is also held to $`\text{CFL}\sqrt{L/a}`$ per
axis for the geometric acceleration, $`(v_\theta^2 + v_\phi^2)/r`$ along $`r`$ and
$`v_\phi^2\cot\theta/r`$ along $`\theta`$, as for gravity. Without both, a swirling blob
around the origin (a random setup, $`v_\phi = 0.06`$) emptied the pole cells next to $`r_\text{min}`$
in its first step and went to NaN; with them it keeps its mass and the axis drains smoothly.

- **Areas and volumes in closed forms.** $`V \propto \tfrac13\Delta(r^3)\,\Delta\cos\theta`$
  and the face areas are differences that cancel in float32 on a fine grid (at
  $`\Delta\ln r = 10^{-3}`$, $`r^3`$ changes by 0.3% across a cell). They are never formed as
  such: $`\tfrac13\Delta(r^3)`$ is $`r_c^2\Delta r + \Delta r^3/12`$ on a uniform grid and
  $`\tfrac23 r_c^3\sinh(\tfrac32\Delta)`$ on a logarithmic one, $`\Delta\cos\theta =
  2\sin\theta_c\sin(\Delta\theta/2)`$, and the constants of the logarithmic grid
  ($`e^{\Delta/2}`$, $`3/(2\sinh\tfrac32\Delta)`$, $`\sinh\Delta`$, $`2\sinh(\Delta/2)`$) are
  computed in double precision on the CPU. A radial face's weight is then
  $`(\Delta x/r_c)\,e^{\pm\Delta}\cdot3/(2\sinh\tfrac32\Delta)`$ (logarithmic) or
  $`\rho_\pm^2/(\rho_c^2 + \tfrac1{12})`$ with $`\rho = r/\Delta x`$ (uniform); a polar
  face's is $`\langle 1/r\rangle\Delta x\,\sin\theta_\pm/\Delta\cos\theta`$, with
  $`\sin\theta_\pm`$ formed from the nearer pole, so that both poles' faces weigh exactly 0.
- **Sources.** Radially $`(2p + \rho(v_\theta^2 + v_\phi^2))/r`$, in $`\theta`$
  $`\cot\theta\,(p + \rho v_\phi^2)/r`$. The $`-\rho v_rv_\theta/r`$ of the $`\theta`$ momentum
  and the whole source of $`v_\phi`$ are taken up by lever arms instead: the $`\theta`$ and
  $`\phi`$ momentum fluxes through a radial face are weighed by $`r_\pm/r_c`$ besides the
  area, and the $`\phi`$ momentum through a polar face by $`\sin\theta_\pm/\sin\theta_c`$, so
  that $`r\rho v_\theta`$ and $`r\sin\theta\,\rho v_\phi`$ are carried in conservation form.
  The pressure's share of each source is $`(h - w_-)\,p_- + (w_+ - h)\,p_+`$ from the two faces'
  own pressures, $`w_\pm`$ their weights and $`h`$ the cell's $`\Delta x`$ over its length along
  that axis. With equal face pressures it is $`(w_+ - w_-)\,p`$, the cell's exact average of
  $`2p/r`$ (or $`p\cot\theta/r`$), which the faces' own difference cancels to rounding; with
  them apart, the net force is $`h\,(p_- - p_+)`$, the Cartesian one. (The mean of the two face
  pressures, which made $`(R, z)`$ well balanced, is not enough here: its net force is
  $`\tfrac12(w_- + w_+)(p_- - p_+)`$, and $`\tfrac12(w_- + w_+)`$ is not $`h`$ in
  $`(r, \theta)`$, by $`1/(6\rho_c^2)`$ on a uniform grid.) The rest of each source is the
  half step's $`\rho v^2`$ times the same averages of $`1/r`$ and $`\cot\theta/r`$.
- **Predictor:** the half step's geometric terms, $`-\rho(2v_r + \cot\theta\,v_\theta)/r`$ in
  the density (and $`\rho c_s^2`$ times it in the pressure), $`+(v_\theta^2 + v_\phi^2)/r`$ in
  $`v_r`$, $`(\cot\theta\,v_\phi^2 - v_rv_\theta)/r`$ in $`v_\theta`$ and
  $`-(v_r + \cot\theta\,v_\theta)v_\phi/r`$ in $`v_\phi`$.
- **The poles** reflect, with $`v_\theta`$ and $`v_\phi`$ odd, as the $`(R, z)`$ axis does; the
  inner sphere is an ordinary face (reflecting, or open: outflow only for accretion).
- **Energy.** The dual-energy switch's guard against the swirl (above) applies as in
  $`(R, z)`$. Gravity's work goes into the energy in kick form: with the flux form and the
  well-balanced reconstruction together, a settled $`n = 1`$ polytrope's fundamental mode grew,
  its central density off by 3% after 18 time units from the analytic start and by 7.6% after
  16 from a settled one (either alone held it to 0.7%); in kick form it holds to 0.04%. Why the
  two together drive it was not found.
- **Gravity** is the monopole: each radial index's shell mass is summed over $`\theta`$ (one
  invocation per shell, Kahan-summed), a single invocation takes the prefix sum, and the field
  at each cell is $`-G\,M(<r_c)/r_c^2`$, the shell's own share inside $`r_c`$ included, with the
  potential $`-\sum G\,m/\max(r, r_\text{shell})`$ for the well-balanced reconstruction and the
  diagnostics. Point masses must sit at the origin, at rest: their $`GM`$ joins the enclosed
  mass and they are not kept as particles (no sinks; gas leaves through the inner face).
  Self-gravity beyond the monopole, MHD, viscosity, species, frames, shear, expansion,
  mirrors, driving and burning are refused.

The diagnostics' masses and energies are true ones (each cell's volume in its closed form), and
their momentum and center of mass are in $`(R, z)`$, along $`z`$ alone. Snapshots are written
with `Coordinates = spherical_polar`, which is already the grid's order $`(r, \theta, \phi)`$,
the radial faces from `RootGridX1`'s ratio and the centers volume weighted as Athena++ writes
them (a unit test checks the layout); such files are not read back. The slice view (and a 2D run's projection, the
same plane) draws the meridional half-plane, $`(R, z) = (r\sin\theta, r\cos\theta)`$: a
square of $`R`$ from 0 to $`2r_\text{max}`$ and $`z`$ from $`-r_\text{max}`$ to $`r_\text{max}`$,
the axis its left edge as in an $`(R, z)`$ run, so the grid's half-disk fills its left half. The
map stays the native $`(r, \theta)`$ grid; the slice shader takes each pixel to the
$`(r, \theta)`$ of its position (the cell fraction along $`r`$ logarithmic on a logarithmic
grid) and draws background inside $`r_\text{min}`$ and beyond $`r_\text{max}`$. The page's
pointer (probes, tools, contour labels) and its markers use the same mapping
(`src/spherical.ts` `sphFromView`, `sphToView`); obstacles drawn by hand do not (their outline
is a linear map of the screen). A native screenshot is the same picture at the map's size. The `blast-spherical` preset is a
point explosion at the origin.

**On blocks**, a level is the uniform grid at its
own resolution: each cell's geometry (areas, volumes, lever arms, sources, the predictor's
$`h_r/\Delta x`$ and $`h_\theta/\Delta x`$, the timestep's per-axis rates, both poles and the
inner sphere) comes from its global indices on its level, with that level's
$`r_\text{min}/\Delta x`$, $`\Delta\ln r`$ and $`\Delta\theta`$ (the finest level's, halved
or doubled per level). A coarser level's logarithmic constants come from the finest's
$`s = \sinh(\Delta/2)`$ by doubling, $`s' = 2s\sqrt{1 + s^2}`$, and then
$`e^{\Delta/2} = c + s`$, $`\sinh\Delta = 2sc`$, $`\sinh\tfrac32\Delta = s(3 + 4s^2)`$ with
$`c = \sqrt{1 + s^2}`$: no step cancels, and the finest level uses the CPU's constants as the
uniform grid does. Across levels:

- **Restriction** (on the fly and on merging) is the volume-weighted mean of the four children,
  each weighing its $`\tfrac13\Delta(r^3)\,\Delta\cos\theta`$ over their mean, so that the sum
  of $`V U`$ is kept; **prolongation** shifts the children's limited-slope offsets along $`r`$
  and $`\theta`$ by $`\tfrac14(v_+ - v_-)/(v_+ + v_-)`$ of the parent, $`v_\pm`$ the radial
  pair's volumes or the polar pair's $`\Delta\cos\theta`$, which centers them on the parent's
  centroid, so their volume-weighted mean is the parent (the $`(R, z)`$ prolongation's shift is
  this rule's radial one there).
- **Flux matching**: a coarse face's flux is the area-weighted mean of the two fine fluxes,
  along both axes: a radial face's fine faces weigh their $`\Delta\cos\theta`$, a polar face's
  their $`\int r\,dr`$ ($`r_c\Delta r`$, or $`r_c^2\sinh\Delta`$ on a logarithmic grid). The
  lever arms are the faces' own, the same on both sides.
- **Per-block timesteps** work with the geometry: a deposit (the second half of a faster
  neighbor's fluxes) weighs its fluxes by the cell's areas and lever arms but adds no source,
  and the sources take the pressures of the cell's own faces (with the bins a face shared with a
  faster neighbor carries half of the step's flux at first). The timestep is set by the
  smallest cells, at $`r_\text{min}`$ beside the poles ($`r_\text{min}\Delta\theta`$); the
  bins let the outer blocks step two or four times as long.
- **Gravity** is the monopole of the leaves, summed into the finest level's shells: each leaf's
  row of cells at one radial index (its mass Kahan-summed over $`\theta`$) hands each finest
  shell it covers that shell's share of its volume, and the shells are summed in leaf order. A
  cell's center at any level lies on a finest face, or at the finest level on a finest center,
  so the mass inside it (and the sum of $`m/r`$ over the shells outside, for the potential) is
  read at a half index of the finest shells, kept for the faces and centers of all of them.
  The field atlas, its two ghost layers included, is filled from that; with per-block
  timesteps the monopole is summed over the leaves as they stand at every substep and the
  potential written is the current one, not extrapolated half a step (the blocks' halves differ).

`sph-blocks`: on two levels, the Sedov blast (256² finest) has its shock 0.74% from the
similarity solution along every one of 64 rays (0.72% with two time bins), and the swirling blob
(128² finest, logarithmic) keeps mass, energy and angular momentum about the axis to
$`3\times10^{-7}`$, $`1.3\times10^{-7}`$ and $`1.0\times10^{-6}`$ through ten re-meshes (with two
bins $`2\times10^{-8}`$, $`1.2\times10^{-7}`$ and $`1.1\times10^{-6}`$, for 76% of a single
timestep's block updates). `sph-polytrope-blocks`: the polytrope on two levels (128² finest,
all three levels in use), settled as on the uniform grid, keeps the density of its innermost
cells to 0.08% over 10 dynamical times (0.09% with two bins). On one level of blocks at 128²
it holds to 0.040%, as on the uniform grid (0.039%); at the coarsest level alone (32²), 1.2%.

Tests (`sph-*`): uniform gas at rest between reflecting spheres at $`r = 0.1`$ and 1, both poles
in the grid, stays below Mach $`6\times10^{-6}`$ on uniform and logarithmic grids (64²), its
mass within $`3\times10^{-7}`$ of $`\tfrac43\pi(1 - 10^{-3})`$; a dense swirling blob thrown
across the grid keeps mass, energy and angular momentum about the axis to
$`5\times10^{-7}`$; a Sedov blast at the origin (256², $`r`$ from 0.005 to 0.5) has its shock
radius 0.74% from the similarity solution, the same along every ray to $`10^{-7}`$; an
$`n = 1`$ polytrope (128², logarithmic, $`r`$ from 0.05 to 2, the mass inside a point mass),
settled by stopping it every 0.2 for 6, keeps its central density to 0.04% over 10 dynamical
times; and isothermal Bondi accretion through an open inner sphere at $`r = 0.1`$ (128²,
logarithmic to 16, the outer two shells held at the isothermal atmosphere's state as a
reservoir) is within 1.5% of Bondi's rate in every shell inside the Bondi radius, averaged over
$`t = 40`$ to 80 (instantaneously within 3%: the flow inside the reservoir still breathes). With
the outer face merely open, the infall from the uniform start drew gas in through it without
bound, and the rate grew with it.


## 23. Validation

The validation suite (`src/validation/tests.ts`) has 223 tests. It runs on your own GPU at
[validate.html](https://guillochon.github.io/shaderous/validate.html) or with
`npm run test:gpu`. Reference solutions come from a float64 CPU implementation of the same
scheme (`src/reference/hydro.ts`) and an exact Riemann solver ([Toro 2009][Toro2009],
`src/reference/riemann_exact.ts`). The tests cover shock tubes along each axis and the diagonal,
a strong ($`10^5`$) shock tube, the Sedov-Taylor blast ([Taylor 1950][Taylor1950];
[Sedov 1959][Sedov1959]), smooth and uniform advection, bit-identical face fluxes, the Gresho
vortex, the well-balanced column, AMR versions of the shock problems, flux matching across
levels and time bins, the sphere potential, the Jeans instability ([Jeans 1902][Jeans1902]), a
polytrope, a cold white dwarf, the Evrard collapse ([Evrard 1988][Evrard1988]), tidal
disruption, cooling, viscosity, conduction, sinks, binary orbits, the equations of state
(hydrogen and helium among them, with the second collapse of a first core in spherical
coordinates), HDF5 round trips, GPU against CPU reference, the weakly compressible liquid and
implicit pressure for the liquid and the gas (§9), the MHD tests of §21, the frames and
coordinates of §22, radiation transport (§24), and the ANEOS materials, a planet at rest and
the moon-forming impact (§25). Each test's pass criterion is printed next to its result.

Six tests (`src/validation/exact.ts`) check presets against closed-form solutions from
ExactPack, Los Alamos's suite of exact solutions. Their formulas are in
`src/reference/converging.ts`, with the Riemann problems taken from the exact solver above, and
unit tests check that each one satisfies the Euler equations and its jump conditions:

- **Riemann problems** (`riemann-*` presets, at $`256^2`$): Einfeldt's 1-2-3 with its
  near-vacuum, Toro's stationary contact, Quirk's slow shock, shock-contact-shock and LeBlanc's
  tube with its pressure ratio of $`10^9`$. Their relative L1 density errors are 0.0095, 0.038
  (a shell six times denser than the gas around it, a few cells wide), 0.0039, 0.0027 and
  0.0085, with no floored cells. The states are those of ExactPack's own tests: its docstring
  table gives the slow shock's left density as 1 and the stationary contact's right pressure
  as 0.02, where the tests have 3.857143 and 0.01.
- **Noh implosion** ([Noh 1987][Noh1987]; `noh-*` presets, the `noh` flow, $`\gamma = 5/3`$,
  $`t = 0.6`$, at $`256^2`$, the open faces held at the exact solution as in §19). Planar:
  post-shock density 4.000 of 4. Cylindrical (R, z): 15.76 of 16. Spherical (r, θ): 61.3 of
  64, with the wall heating at the center left out of the average. In all three the shock is
  within 0.85 cells of $`r = 0.2`$, and the pre-shock density within $`10^{-5}`$ of
  $`(1 + t/r)^{k-1}`$ in L1 out to the held cells. On a 2D Cartesian grid
  ([Liska & Wendroff 2003][Liska2003]) it gets 15.89 of 16, with the shock within 0.33 cells of
  round along the axes and the diagonal.
- **Uniform collapse** (ExactPack's noh2; `uniform-collapse-*` presets, the
  `uniform-collapse` flow): $`u = -r/(1 - t)`$, with the density and the specific internal
  energy uniform along the adiabat. At $`t = 0.6`$, with the outer faces held, the L1 error in
  density, pressure and velocity is at most $`2.5\times10^{-6}`$ planar, $`6.6\times10^{-6}`$
  cylindrical and $`5.5\times10^{-5}`$ spherical.
- **Coggeshall's similarity solutions** ([Coggeshall 1991][Coggeshall1991]; `coggeshall-*`
  presets and flows, spherical, both radial faces held): problem 1, free homologous expansion
  with a uniform pressure; problem 2, homologous expansion against a pressure rising outward;
  problem 6, Kidder's isentropic compression ($`\gamma = 5/3`$), to 0.6 of the starting radii.
  L1 errors are at most $`10^{-4}`$, $`1.4\times10^{-4}`$ and $`5\times10^{-5}`$. Problems 1
  and 2 are singular at $`t = 0`$, so their runs start at $`t_0 = 1`$ of the solution. The other
  Euler-only problems need $`\gamma < 1`$ or a negative temperature (3, 4, 5), or are the Noh
  problem again (19). Problem 20, Noh with a uniform collapse added, is not consistent as
  ExactPack codes it: with its default $`u_0 = +2.3`$ the pre-shock density factor
  $`((r - u_0 t)/r)^k`$ goes negative between the shock and $`r = u_0 t`$, and with
  $`u_0 < 0`$ there is no shock. The rest need heat conduction with power-law conductivities.
- **Piston-driven detonations** (Fickett & Davis 1979, Fickett & Rivard 1974; the `mader` and
  `ehep` presets, $`\gamma = 3`$, at $`512^2`$). Mader's problem is a slab of explosive
  detonated from a wall at rest. Its exact solution burns instantly at the Chapman-Jouguet
  state; here a fast Arrhenius burn stands in, its reaction zone far thinner than a cell, and
  the run starts from the exact solution at 0.25 μs. Cells flagged as shocks do not burn, so
  the detonation keeps a von Neumann spike and starts 7.5 cells behind. Over the second half of
  the run it moves at 0.805 against the CJ speed of 0.8, and behind it the Taylor wave and the
  products at rest are within 1.7% (density) and 1.8% (pressure) in L1. Burning in shocked
  cells instead let the front outrun the box, as [Colella et al. (1986)][Colella1986] predict
  for unresolved fronts. The escape of the products (EHEP) starts from its exact solution once
  the explosive is all burned: a moving piston, an obstacle, cannot carry the species a burn
  needs. The products escape into vacuum and a rarefaction runs back and reflects from the
  piston; at the end the density is within 1.3%, the pressure 3.2% (it goes as
  $`\rho^3`$), and the velocity 0.17% of the CJ speed.

## 24. Radiation transport

Gray radiation hydrodynamics with the M1 closure (`src/radiation/`, `src/opacity/`), on the uniform grid and on blocks, in Cartesian coordinates. The setting is
`physics.radiation` (`m1` or `off`); off, a run builds none of it and steps as before, bit for
bit.

### Equations

The radiation energy density $`E`$ and flux $`\mathbf F`$ are evolved in the gas's frame
([Fornax][Skinner2019]), to first order in $`v/c`$ and without its terms in the flux equation:

```math
\begin{aligned}
\partial_t E + r\,\big[\nabla\cdot(\mathbf v E) + \nabla\cdot(\mathbf v\cdot\mathsf P) + \nabla\cdot\mathbf F\big] &= r\,\big[c\kappa_P\rho\,(aT^4 - E) - \mathbf v\cdot\mathbf g\big],\\
\partial_t \mathbf F + r\,\big[\nabla\cdot(\mathbf v\,\mathbf F) + c^2\,\nabla\cdot\mathsf P\big] &= -r c\kappa_R\rho\,\mathbf F,
\end{aligned}
```

and the gas gains the momentum $`\mathbf g = \kappa_R\rho\mathbf F/c`$ per unit time and the heat
$`c\kappa_P\rho(E - aT^4)`$. $`\kappa_P`$ is the absorption opacity, $`\kappa_R`$ the total
(absorption and scattering), $`\mathsf P = \mathsf D E`$ the pressure tensor with Levermore's
Eddington factor

```math
\chi(f) = \frac{3 + 4f^2}{5 + 2\sqrt{4 - 3f^2}},\qquad
\mathsf D = \tfrac{1-\chi}{2}\,\mathsf I + \tfrac{3\chi-1}{2}\,\hat n\hat n,\qquad f = \frac{|\mathbf F|}{cE}.
```

$`\nabla\cdot(\mathbf v\cdot\mathsf P) + \mathbf v\cdot\mathbf g`$ is the compression work
$`\mathsf P:\nabla\mathbf v`$, split so that what the radiation loses the gas's kinetic energy
gains. With radiation on, the equation of state leaves out a radiation term of its own (the gas
and radiation one, the degenerate one): $`E`$ carries it.

### The reduced speed of light, cell by cell

$`r`$ is the factor on the speed of light. Explicit transport at the true $`c`$ would cost a
step of $`\Delta x/c`$ everywhere; a reduced speed $`\hat c`$ everywhere gives radiation in
equilibrium with the gas $`c/\hat c`$ times its weight (26% too cold in `rad-equilibrium` at
$`\hat c = c/10`$), and trapped radiation in a lab-frame scheme lags the gas at $`(\hat c/c)\,v`$
([Skinner & Ostriker 2013][Skinner2013]; [Wibking & Krumholz 2022][Wibking2022]). Shaderous
reduces it only where a cell is too transparent to afford the true one:

```math
r = \min\!\Big(1,\ \frac{\hat c}{c}\,\max\!\big(1,\ \tau_c/\tau_0\big)\Big),\qquad \tau_c = \kappa_R\rho\Delta x,\quad \tau_0 = 2.5/3 .
```

In an opaque cell the step is limited by diffusion at $`D = rc/(3\kappa_R\rho)`$, and
$`\tau_0`$ is where its rate, $`2.5\,d\,D/\Delta x`$ as for conduction (§3), equals the streaming
rate $`d\,\hat c`$. With $`c/\hat c = 100`$, every cell of optical depth above about 80 has the
true speed, so the inside of a star has the right heat capacity and conserves energy exactly.
The one condition is $`\hat c \gg |v| + c_s`$ (the diagnostics count the cells where
$`|v| + c_s > \hat c/3`$), not $`\hat c \gg v\tau`$. Where $`r < 1`$ what is conserved is
$`E_\text{gas} + E/r`$. Every term but the time derivative carries $`r`$, the advection with
the gas included, so in a steady state $`r`$ drops out, in moving gas too. (With the advection
left at full speed, the supercritical shock of `rad-shock`, whose precursor's cells are reduced,
came out 6% too cool.) Where $`r = 1`$ the radiation moves exactly with the gas.

What a reduced speed cannot give is a radiative structure that moves across the grid through
reduced cells: the precursor of a shock of speed $`D`$ has only $`\hat c/D`$ rather than $`c/D`$
of a head start, and the state behind is off by as much as the precursor matters. In `rad-shock`
the temperature behind the supercritical shock is 3.2% low at $`\hat c = c/5`$, 1.1% at
$`c/2.5`$ and 0.06% at $`c`$. This rule is Shaderous's own; `radLocalC = false` gives the usual
uniform $`r = \hat c/c`$.

### A step

```
rates, clock     the step; the radiation's push adds (4/9) min(1, tau_c) E / rho to c_s^2,
                 and the step is at most N_r sweeps of 0.5 dx / (d chat)
predict, update  as without radiation; the update records the face mass fluxes
radiation/advect E / rho and F / rho moved with those mass fluxes, as species are, plus the
                 flux v . P; each cell's change times its r
gravity, correct
radiation/sweep  N_r times (8 by default, at most 16): transport, then the local exchange
radiation/commit the gas's side of the sweeps into the state
```

**Advection** is upwind on the sign of each face's mass flux, with the upwind cell's limited
slope over $`(1-\nu)/2`$, $`\nu`$ the face's Courant number: trapped radiation then moves exactly
as the gas does.

**Transport.** Each cell computes the fluxes of all its faces itself from the four cells across
each, so both cells of a face get the same bits (the gathered fluxes of
[architecture.md](architecture.md)). $`E`$ and the reduced flux $`\mathbf f = \mathbf F/(cE)`$
are reconstructed linearly: the slope is the central difference, at most $`\theta`$ times
either one-sided difference, with $`\theta = 2`$ (monotonized central) for $`\mathbf f`$ and
$`\theta = 1.2`$ for $`E`$. With $`\theta = 2`$ on $`E`$ and the single forward Euler stage, the
surface of an opaque body sent its light out in pulses: in the reference, 26% up and down with a
period of some 40 sweeps at 5 optical depths a cell, and still 3% with four times the cells. With
1.2 it is steady to 0.3% at 1 to 20 optical depths a cell, and the luminosity is 3% above the
converged one where minmod ($`\theta = 1`$) is 8% above. A face whose reconstructed
states have $`E \le 0`$ or $`|\mathbf f| > 1`$ takes its cells' own ([Wibking & Krumholz
2022][Wibking2022]). The flux is Rusanov's with the largest M1 wave speed along the face,
$`S = c\max(\sqrt{D_{nn}}, |f_n|)`$ of the two states (or $`c`$, global Lax-Friedrichs), and its
dissipation is scaled by $`\min(1, 1/\tau_\text{face})`$ ([Skinner et al. 2019][Skinner2019]), so
that in opaque gas the scheme diffuses at $`c/(3\kappa_R\rho)`$ whether or not cells resolve the
mean free path. The face flux uses the true $`c`$; each cell multiplies its divergence by its own
$`r`$. One forward Euler stage is stable to a Courant number of 0.5 with this reconstruction (at
0.8 a streaming pulse gained energy sevenfold in the reference), so the sweep's step is
$`0.5\,\Delta x/(d\hat c)`$.

**Exchange.** After the transport has made $`E^*`$ and $`\mathbf F^*`$, per cell, implicitly:

1. $`\mathbf F' = \mathbf F^*/(1+\theta_F)`$, $`\theta_F = rc\kappa_R\rho\Delta t`$; the gas
   gains $`(\mathbf F^* - \mathbf F')/(rc^2)`$ of momentum. In the gas's frame the equilibrium
   flux is zero, so nothing has to be iterated.
2. Backward Euler for the pair $`(\rho e, E)`$ eliminates $`E'`$:
   $`\rho e(T') + A\,aT'^4 = \rho e + A E^*`$, $`A = \theta/(r(1+\theta))`$,
   $`\theta = rc\kappa_P\rho\Delta t`$. Its root comes from Newton's method from the smaller of
   the two one-term solutions, an upper bound it descends from; then
   $`E' = E^*/(1+\theta) + aT'^4\,\theta/(1+\theta)`$. Both sides are sums of positive terms:
   nothing subtracts $`aT^4`$ from $`E`$, so float32 is enough at any optical depth
   (`rad-f32` holds $`E = aT^4`$ to $`10^{-6}`$ at $`\tau_c = 10^{10}`$ over $`10^4`$ sweeps).
   The larger of $`\rho e'`$ and $`E'/r`$ then takes what the root's rounding left, so the
   exchange conserves $`E_\text{gas} + E/r`$ exactly.
3. The work $`r\,\mathbf v\cdot\Delta(\rho\mathbf v)`$ comes off $`E'`$, at the gas's velocity half
   way through the sweep's kick: summed over the sweeps it is exactly the kinetic energy the
   commit adds.

The gas's side, $`(\rho e, \Delta\rho\mathbf v)`$, collects in a texture of its own through the
sweeps; the commit adds it to the state with the change of kinetic energy and runs the
dual-energy synchronization (§7).

**Initial states.** A problem may give $`E`$ and $`\mathbf F`$ (`Prim.erad`, `Prim.frad`); by
default $`E = aT^4`$ at the gas's temperature (with a radiation-bearing equation of state, the
problem's pressure is split between gas and radiation at the temperature the full equation of
state gives) and $`\mathbf F`$ is its diffusion value from the grid's centered differences.

**Opacities** (`<opacity>`, `src/opacity/`): constant absorption and scattering; electron
scattering $`0.2(1+X)`$ cm²/g with a small true absorption; Kramers' free-free
$`3.7\times10^{22}(1-Z)(1+X)\rho T^{-3.5}`$ cm²/g plus electron scattering. With physical units,
$`c`$, $`a`$ and these constants come from the units and composition.

**Output.** The view's Radiation group (energy, temperature, $`|\mathbf F|`$, $`f`$,
$`\tau_c`$, $`r`$); the diagnostics' radiation energy, $`\sum E/r`$, the share of $`E`$ in reduced
cells, the luminosity through the open faces, the energy that has left through them, and the
warning counts; `.athdf` files carry
`Er`, `Fr1`-`Fr3`, and a run restarts from one with its radiation bit for bit.

**The light curve.** The luminosity is the outward $`\mathbf F\cdot\hat n\,dA`$ of the cells on
the open faces of the box. After each step's sweeps one small pass (`radiation/escape.wgsl`,
`amr/rad_escape.wgsl`) adds it times the step to a running total of the energy that has left,
so the luminosity the Diagnostics panel shows (`L rad`, with `E rad`, `E esc`, `T rad` and the
count of cells too fast for $`\hat c`$) is that energy's change over the time between two
readouts: nothing between them is missed, and its history, in the panel's plot and its CSV, is
the bolometric light curve. It is the light at the box's edge, which left the gas a box's size
over $`\hat c`$ earlier, and in the gas's frame. The count starts again from zero on a restart.

### On the adaptive mesh

One timestep for all blocks (`src/amr/radiation.ts`, `src/amr/rad_*.wgsl`): with radiation on,
per-block timesteps are off, as with species.

```
amr/rad_assemble      (E, F) and kappa_R rho of every leaf's cells and two layers of ghosts
amr/assemble, clock   the step: the radiation's push, and its limit as each cell's own rate
predict, update       the update records the face mass fluxes, as for species
amr/rad_advect
gravity, correct
N_r times:  amr/rad_assemble, amr/rad_sweep
amr/rad_commit
```

**Ghosts** are computed on load, as the gas's are (§17): a same-level neighbor's cell, its
image at a boundary with the flux reversed across a wall, the mean of a finer block's cells, or
the coarser cell beyond the face with minmod slopes toward the ghost's quarter (which keep
$`E`$ positive) and its flux held to $`cE`$. $`\kappa_R\rho`$ is taken as it is from a coarser
cell and averaged from finer ones. The assembly runs before every sweep: 16 small passes a step
at the default 8 sweeps.

**Refinement boundaries.** A coarse cell takes, through a face it shares with finer cells, the
mean of their fluxes, each computed in the fine block from the fine block's cells and ghosts as
the fine cell computes it; the advection does the same with the fine blocks' recorded mass
fluxes. What leaves one level enters the other. Each cell multiplies the divergence by its own
$`r`$, and $`\tau_c`$ is the cell's own, so a coarse cell may have the true speed of light where
the fine cells beside it have a reduced one: across such a face $`E`$ is conserved, and
$`E/r`$ where both have the same $`r`$ (1 in opaque gas).

**The step.** A cell's sweeps are stable to $`N_r \cdot 0.5\,\Delta x/(d\hat c)`$ with its own
$`\Delta x`$; the limit is folded into each cell's rate, so the finest leaves set it.

**Re-meshing** (`amr/rad_regrid.wgsl`): an unchanged block is copied, a merged block takes the
mean of its children, and a new child its parent's cell with minmod slopes from the parent's
neighbors (assembled on the old mesh just before). $`E`$ and $`\mathbf F`$ are conserved.

**Initial states and settling.** Each block's $`\mathbf F`$ is the diffusion flux from
differences within the block (one-sided at its faces). A star is settled on the CPU with
radiation on, its pressure split between gas and radiation as on the uniform grid.

Memory: 84 bytes a texel on top of a block's 152, and 59 kB a block in 3D for the face mass
fluxes where no species already record them.

### Validation and cost

`src/reference/radiation.ts` is a float64 version of the scheme in 1D (unit tests in
`tests/unit/radiation.test.ts`), and `src/validation/radiation.ts` holds the GPU tests:

| Test | Result |
|---|---|
| `rad-equilibrium` | Closed box: with $`r = 1`$ the temperature is the root of $`e(T) + aT^4 = e_0 + E_0`$ to $`2\times10^{-7}`$, energy kept to $`5\times10^{-8}`$; one reduced speed $`c/10`$ everywhere misses it by 26%; the per-cell rule in an opaque box, by $`5\times10^{-8}`$ |
| `rad-f32` | Equilibrium at $`\tau_c = 10^{10}`$, $`10^4`$ sweeps: $`E/aT^4 - 1 < 1.3\times10^{-6}`$; total energy kept exactly at rest, to $`4\times10^{-5}`$ (about $`10^{-8}`$ a step) where the gas moves |
| `rad-streaming` | A pulse in vacuum moves at $`\hat c`$ to 0.1% with $`f = 1`$ |
| `rad-diffusion` | A Gaussian spreads at $`c/(3\kappa\rho)`$ to 1% at $`\tau_c = 1`$, 100 and $`10^4`$ |
| `rad-shadow` | Behind an opaque disc, 6% of the beam beside it |
| `rad-advect` | Trapped radiation at $`\beta = 10^{-2}`$, $`\beta\tau = 10^4`$, $`\hat c = c/100`$ moves with the gas to $`10^{-4}`$ of the box and keeps its width to 3.5% (the lab-frame reference lags at $`(\hat c/c)\,v`$) |
| `rad-marshak` | [Su & Olson (1996)][SuOlson1996]'s non-equilibrium Marshak wave: the GPU matches the float64 reference to $`2\times10^{-3}`$ of the incident energy at $`\tau = 1`$, 3.16 and 10, and both the float64 diffusion solution with the Marshak boundary to 0.03 at $`\tau = 3.16`$ ($`x \le 2`$) and $`\tau = 10`$ ($`x \le 5`$), beyond half a mean free path of the boundary, where M1 and diffusion part ways |
| `rad-star` | An $`n = 3`$ polytrope with a tenth of its pressure in radiation, over a dynamical time: it holds as the same star with its radiation in the equation of state does (rms speed inside 2.9% of the central sound speed, the baseline's 2.1%, at $`64^3`$), and crossing the box at Mach 10 it follows that baseline |
| `rad-shock` | Gas driven into a wall: the state behind the shock matches the Rankine-Hugoniot conditions with radiation to 0.1% (subcritical), and to 3% at $`\hat c = c/5`$ (supercritical; the limit above) |
| `rad-cooling-sphere` | A hot core behind a reduced-speed shell: the luminosity is steady to 0.03% over 30 steps (12% before the limiter above) and the same at $`\hat c`$ and $`2\hat c`$ to 5% |
| `rad-restart` | The radiation comes back from a snapshot bit for bit |
| `rad-off` | Off, a run has no radiation passes and its equation of state keeps its radiation term; on, the radiation constant moves from the equation of state to the transport |
| `rad-blocks-diffusion` | A Gaussian centered on the face between coarse and fine blocks (20 and 10 optical depths a cell) spreads as on the fine uniform grid: centroid to $`2\times10^{-4}`$ of the box, widths to 0.4%; the total energy changes by $`1.3\times10^{-5}`$, the uniform grid's by $`1.4\times10^{-5}`$ (rounding) |
| `rad-blocks-advect` | The pulse of `rad-advect` on a mesh that re-meshes four times as it moves: where the gas took it to $`10^{-4}`$ of the box, width kept to 2%, $`E_\text{gas} + E/r`$ to $`2\times10^{-6}`$ |
| `rad-blocks-cooling-sphere` | Only the blocks holding the shell refined: the luminosity through the box faces, on coarse blocks, is the fine uniform grid's to $`10^{-5}`$, and as steady |
| `rad-blocks-star` | The star of `rad-star` on three levels (128 cells across at the finest), settled on the mesh: over a dynamical time its central density rises 2% (the baseline's 1%), the rms speed inside is 1.1% of the central sound speed and the largest 2.9%, energy kept to 0.6%. Crossing the box at Mach 10 through four re-meshes: central density within 0.1% of the baseline's, the same rms speed about its motion, energy to 0.1% |
| `rad-blocks-restart` | From a snapshot: the same mesh and the same radiation in every cell, bit for bit |

On an RTX 3080 a sweep costs about a quarter of a hydro step (3D $`128^3`$: 2.1 ms a step
without radiation, 6.6 ms with 8 sweeps); at $`\hat c`$ about 15 times the gas's signal speed a
unit of simulated time costs 4.2 times what the gas alone does.

**Not yet.** Per-block timesteps with radiation (switched off); refinement on the radiation
energy itself; settling a star with radiation on the GPU; curvilinear coordinates, comoving
coordinates and the shearing box; magnetic fields; implicit pressure; more than one frequency
group; sources that shine. A run that asks for one of the coordinates, the field or implicit
pressure is refused with the reason (`radiationUnsupported`, `src/radiation/radiation.ts`).
The equation of state must have an energy and no radiation table
of its own: the ideal gas, gas and radiation, degenerate electrons or ionizing hydrogen, not white
dwarf matter, hydrogen and helium, ANEOS, the liquid or a barotropic gas. The sweep is
first order in time, and neither it nor the order of the sweeps and the hydro step is what
limits the tests (below), so [He, Wibking & Krumholz 2024][He2024]'s two-stage scheme, which
would take a second set of sweeps or halve their step, is not taken. Radiation pressure enters as
a source term the well-balanced reconstruction (§8) does not see, so stars held up largely by
radiation are not yet in discrete balance, and the Riemann solver sees the gas pressure's jumps
without the radiation's that balance them (`rad-advect`, below).

**What limits the accuracy** (M4 Pro, 2026-11-19). *Not the step.* The supercritical shock of
`rad-shock` at $`\hat c = c/5`$ is the same with 4, 8 and 16 sweeps a step (hydro steps of a
half, one and two times the default, the sweeps' own step fixed) and with the sweeps' step
halved: the shock at 0.9612 to 0.9617 of where its speed puts it, the density behind 1.0252 to
1.0255 times the jump's and the temperature 0.9676 to 0.9680 times. The 3 to 4% is the reduced
speed of light in the precursor, whose cells have $`\tau_c \approx 1`$ and so
$`r \approx 0.22`$: the radiation there diffuses at $`r c/(3\kappa\rho)`$ and the structure,
moving across the grid, sees $`1/r`$ times its radiation's heat capacity. It falls as
$`\hat c`$ grows (1.7%, 0.9% and 1.1% at $`c/2.5`$; 0.7%, 0.09%, 0.06% at $`c`$). The per-cell
rule cannot give these cells more: $`r = (\hat c/c)\,\tau_c/\tau_0`$ is where the explicit
transport's diffusion rate meets the sweep's streaming rate, so a larger $`r`$ needs a shorter
sweep. And 16 sweeps a step give the same shock in half the hydro steps here (2 710, against
5 419 at the default 8). *The radiation's pressure, unseen by the Riemann solver.* The pulse of
`rad-advect` narrows by 3.4% at the default Courant number and 2.0% at half of it, first order in
the hydro step, and the same whatever the sweeps' number or step; the same pulse with its
radiation in the equation of state (no transport) keeps its width to 0.3%. Within the pulse the
gas pressure is low where the radiation's is high; the hydro step sees only the gas's jumps, and
each step moves gas toward the pulse that the sweeps' force then stops. (Giving the predictor the
radiation's force as a half kick, as it has gravity's, moved the narrowing by 1.7% either way
with its sign, so it is not that.) The cure is the radiation's force in the well-balanced
reconstruction and in the update, the sweeps then giving only the rest of it: not a contained
change, and not made.

---

## 25. ANEOS materials and planets

`src/eos/aneos.ts`, `eos/aneos.wgsl`, `src/problems/planet.ts`, `src/debris/impact.ts`;
`tools/eos/aneos_table.py`. Rock, iron and water for planetary impacts, from ANEOS
([Thompson & Lauson 1972][ThompsonLauson1972]; [Melosh 2007][Melosh2007];
[Stewart et al. 2020][Stewart2020]): the equation of state `aneos` (`physics.eos`).

### Materials and their tables

| Material | Deck | Use |
|---|---|---|
| `pyrolite` | Pyrolite-ANEOS-SLVTv0.2G1 ([Stewart 2022][StewartPyrolite]) | a mantle |
| `fe85si15` | Fe85Si15-ANEOS-SLVTv0.2G1 ([Stewart 2020][StewartFe85Si15]) | a core |
| `water` | Water-ANEOS-SLVTv0.3G1 ([Stewart 2024][StewartWater]) | ice and water |
| `forsterite` | Forsterite-ANEOS-SLVTv1.0G1 ([Stewart et al. 2019][StewartForsterite]) | rock, as the older calculations have it |

Each material's published input deck is run through pyaneos ([Stewart 2026][pyaneos]), the
Python port of M-ANEOS, on one grid for all of them: 401 densities from $`10^{-8}`$ to
$`100\ \mathrm{g\,cm^{-3}}`$ and 173 temperatures from 50 K to $`10^6`$ K, uniform in the
logarithms at 0.025 dex. pyaneos reproduces each published table at its own nodes to
$`5\times10^{-9}`$, the precision the tables are printed to. The decks are run with pyaneos's
`NOTENSION`: the solid stretched below its reference density is tabulated in equilibrium with
its vapor, so the pressure is positive everywhere.

**What is tabulated is the pressure and the energy, not the free energy.** ANEOS is built as a
Helmholtz free energy, but in its fluid with ionization its pressure and entropy are not the
derivatives of the free energy it reports: at $`2\times10^4`$ K they differ from them by 1 to
13% in the iron alloy (the solid agrees to $`10^{-6}`$). A table of $`F`$ with $`p`$ and $`s`$
as its slopes is then inconsistent with itself, and a first version built that way was wrong by
3% at the median in the fluid. So the table holds what every code that uses these materials
uses, ANEOS's own $`p(\rho, T)`$ and $`e(\rho, T)`$, as two functions with their first and
cross derivatives at every node, for bicubic Hermite interpolation:

```math
\psi = \operatorname{asinh}\frac{p}{P_s},\quad P_s = 10^9\ \mathrm{dyn\,cm^{-2}};\qquad
\lambda = \ln\left[e - E_c(\rho)\right],\quad E_c(\rho) = F(\rho, T_0) + E_\text{off}
```

$`\psi`$ is linear in the pressure below $`P_s`$ (vapor, and a body's surface, where the
pressure falls to nothing across one cell of the table) and logarithmic above. $`E_c`$, the
cold curve, is the free energy of the lowest isotherm, so $`e - E_c`$ is $`T_0 s`$ there,
positive, and rises with $`T`$; $`E_\text{off}`$ makes $`E_c`$ positive. The node slopes are
ANEOS's own derivatives (differences of a part in $`10^4`$), limited as a monotone cubic's are
([Fritsch & Carlson 1980][FritschCarlson1980]): the interpolant then rises along $`T`$ wherever
the nodes do, which the solve for the temperature needs. One table is 2.2 MB, fetched when a
run asks for its material.

On the GPU the table is the texture every tabulated equation of state reads (`EOS_TAB`,
`eos/table_tex.wgsl`), loaded and bound as the white dwarf's and the hydrogen–helium tables
are (`src/eos/table.ts`), with their bicubic Hermite basis. Their lookup and their solve for the
temperature are not shared: those read one free energy, one column per temperature, and solve
for a derivative of it. Here a row holds $`\psi`$'s columns, then $`\lambda`$'s, then the cold
curve, a pair stacks its five mixtures along the rows and blends two before reading them, and
the solve is for the value of $`\lambda`$ or $`\psi`$ itself, by bisection over the nodes and
Newton's method on the cubic between them.

Against ANEOS at 6000 random states per material, between the nodes, by its phase flag:

| Phase | Pressure, median | 90th percentile | Energy, median | 90th percentile |
|---|---|---|---|---|
| Solid | $`10^{-8}`$ | $`10^{-5}`$ | $`3\times10^{-7}`$ | $`10^{-5}`$ |
| Liquid and gas | $`10^{-6}`$ | $`2\times10^{-4}`$ | $`10^{-5}`$ | $`8\times10^{-4}`$ |
| Vapor dome | rounding | 4 to 13% | $`10^{-5}`$ | up to the latent heat |

In the dome the saturation pressure rises by up to a factor of 20 from one temperature node to
the next, and along the line where the last liquid boils away the energy rises by the latent
heat across one or two nodes: there the table is right to about half a node, 3% in
temperature, and wrong by the size of the jump if read as an error at a point.

### Reading it

The hydro holds the density and the energy (or, at a face, the pressure), so the temperature
comes first. $`\lambda`$ (or $`\psi`$) rises along $`y = \log_{10}T`$ at the cell's density:
the two node rows on either side of the answer are found by bisection (8 steps for 173 rows,
each the cubic along the density through two texels), and the place between them by Newton's
method on the cubic through them, kept in its bracket by a step of false position (the Illinois
rule) where a limited slope is zero; 8 steps. The other quantity and the sound speed are then
read there. Single precision against the float64 mirror (`aneos-eos`): the pressure back from
its own energy to $`10^{-4}`$ at worst, the sound speed to $`3\times10^{-4}`$ at the 99th
percentile.

Against the published tables' own sound speed, at their nodes, in solid and liquid rock and iron
at or above the reference density: $`3\times10^{-6}`$ at the median and $`2\times10^{-4}`$ at the
90th percentile, and over 1% at 2 to 3% of the nodes (`tests/unit/aneos_published.test.ts`, which
runs where the published files are at hand; below the reference density the published solid is
under tension and ours is not).

The sound speed is $`c^2 = (\partial p/\partial\rho)_T + \Gamma\,(\partial p/\partial\ln T)_\rho/\rho`$
with $`\Gamma = (\partial p/\partial T)/(\rho\,\partial e/\partial T)`$ held between 0 and 5,
and $`c`$ between 30 m/s and 100 km/s. Every term is positive. (The form with the energy's
slope along the density, which is the exact speed of a system whose pressure is $`p(\rho, e)`$,
multiplies a latent heat by a ratio of two slopes along $`T`$ that are both next to nothing in
cold gas: cold dust came out at 26 km/s and set the timestep of a whole run.)

A state below the table's lowest isotherm is read as that isotherm's, and `eos_eint_floor` is
its energy: without the floor a cell that had landed there handed its faces more energy than it
held, step after step. Thinner than the table's least density the gas is ideal: the pressure
of the least density, in proportion.

### Two materials in a cell

A run holds one material or two. With two, they are its species (the mantle's first), and a
cell's mass fraction $`X`$ of the second reaches the equation of state as the composition's
moment $`y`$ (§12). In a cell that holds both, each takes the volume that brings the two to one
pressure at the cell's temperature:

```math
\frac{1 - X}{\rho_a} + \frac{X}{\rho_b} = \frac{1}{\rho},\qquad p_a(\rho_a, T) = p_b(\rho_b, T)
```

This closure is solved when the pair is first asked for, in the page (about a second), not in
the shader: at every node, by bisection on $`\rho_a`$, for $`X = \tfrac14, \tfrac12,
\tfrac34`$. The result is three more tables of the same form, between the two materials' own:
the mixture's pressure, and its energy $`(1 - X)e_a + Xe_b`$. A cell reads the two tables on
either side of its $`X`$, blended texel by texel, exactly as it reads one material. Between
tables the blend is linear in $`\psi`$ and $`\lambda`$, which is an interpolation of the
closure, not the closure: at the tables' own $`X`$ the pressure is each material's at its own
density to 0.5% (90th percentile, both condensed).

### Planets

A planet (`problems/planet.ts`; a star part of type `planet`) is given by its mass, its mantle's
material, optionally a core's material and share of the mass, and the temperature of its
surface (2000 K by default: a body still hot from its accretion, as the published impact
calculations start). Each layer lies along its own adiabat, $`de = p\,d\rho/\rho^2`$ with the
pressure the table gives, so that a layer moved whole stays on it; where two layers meet the
pressure and the temperature are continuous. The structure is integrated inward from the
surface for a trial radius (fourth-order Runge-Kutta, 2000 steps), and the radius is the one at
which the mass runs out at the center.

| Body | Radius | Central pressure | Central density |
|---|---|---|---|
| 1 M⊕, 30% Fe85Si15 under pyrolite | 1.024 R⊕ | 351 GPa (the Earth: 364) | 12.6 g/cm³ (13.1) |
| 0.12 M⊕, the same | 0.544 R⊕ | 73 GPa | 9.0 g/cm³ |

A planet is not settled (§16): its layers are not one barotrope. It starts on the integrated
structure, and at 16 cells to the radius rings by 3.6% in central density.

The gas around a planet is its mantle's material, thin, at the surface's temperature: vapor
and dust with next to no pressure, which falls onto the planet. It carries $`10^{-7}`$ of the
planet's mass.

### What an impact leaves

`debris/impact.ts` sorts the gas of a run as the moon-forming impact calculations do
([Canup 2004][Canup2004]). About the densest gas, every cell is on the Kepler orbit its
position and velocity give it around the planet's mass: escaping if the orbit is unbound, the
planet's if its pericenter is inside the planet's radius (that of a body of the planet's mass
at the Earth's mean density), the disk's otherwise; the planet's mass and radius are taken
again from what fell to it, until they settle. The disk's mass $`M_D`$ and angular momentum
$`L_D`$ give the moon it would gather into ([Ida et al. 1997][Ida1997];
[Kokubo et al. 2000][Kokubo2000]):

```math
M_\text{moon} = 1.9\,\frac{L_D}{\sqrt{G M_p a_R}} - 1.1\,M_D - 1.9\,M_\text{esc},\qquad a_R = 2.9\,R_p,\quad M_\text{esc} = 0.05\,M_D
```

**The moon-forming impact** (`moon-forming-impact`). Theia, 0.12 M⊕, strikes the proto-Earth,
0.9 M⊕, on a parabola whose pericenter is half the sum of their radii, so that they touch at
45° ([Canup 2004][Canup2004]); both have 30% Fe85Si15 cores under pyrolite. A run in a box of
24 R⊕ (the preset's is 16) at 256 finest cells on two levels of refinement (10.7 cells to an
Earth radius; 0.7 million cells at the start and 3.5 million by the end), with a time unit of
805.5 s:

| Time | Planet (M⊕) | Disk (M⊕) | Escaping (M⊕) | Disk from Theia | Iron in the disk | Moon (lunar masses) |
|---|---|---|---|---|---|---|
| 12 (2.7 h) | 0.989 | 0.032 | 0.00002 | 80% | 26% | 1.3 |
| 16 (3.6 h) | 0.991 | 0.030 | 0.00005 | 81% | 27% | 1.1 |
| 28 (6.3 h) | 0.986 | 0.035 | 0.00001 | 74% | 25% | 0.85 |
| 36 (8.1 h) | 0.981 | 0.036 | 0.004 | 72% | 24% | 1.25 |
| 48 (10.7 h) | 0.960 | 0.057 | 0.004 | 50% | 17% | 1.95 |

Contact is at t = 3; what is left of Theia comes round and strikes again near t = 20, when the
sorting means little. Total energy is conserved to $`4\times10^{-4}`$ over the run, and
$`1.3\times10^{-4}`$ of the mass leaves through the box's faces. The run takes 65 steps per
time unit.

The disk of the first hours, 0.03 M⊕ of which four fifths is Theia's and which would gather
into about one lunar mass, is the published result for this impact. Two things in the table are
the resolution's, not the impact's:

- **Iron.** The published disks hold a few percent of iron; this one a quarter. Theia's core
  is 6 cells across, and what the mesh mixes into its mantle goes where the mantle goes.
- **The disk's growth after t = 36**, at the planet's expense and with the Earth's rock. The
  planet is then hot, swollen and spinning near its limit, and its outer layers spread across
  the radius the sorting uses, which is a cold Earth's.

In the preset's own box at 128 finest cells on one level of refinement, 8 cells to an Earth
radius (the `giant-impact` test), the disk at t = 12 is 0.034 M⊕, 80% Theia's, and would gather
into 1.5 lunar masses. In that box at 256 cells on two levels (16 cells to the radius) total
energy drifts by 0.24% in the first four time units, before contact, and by 0.02% after; 0.2%
of the mass has left through the faces by t = 28.

### Limits

- **The table's cell at a body's surface is soft.** The pressure rises from the vapor
  pressure to GPa across the one cell of the table that holds the edge of the condensed phase
  (6% in density), where in ANEOS it rises from the edge itself. The outermost cells of a body
  are the ones in it.
- **Water's solid is not ice Ih**: the 2024 model has one solid, of 1.25 g/cm³. A cold icy
  body is about a third too dense near its surface.
- **No strength.** Bodies are fluids: right above a few hundred kilometers.
- **Refused with it**: comoving coordinates, radiation transport (§24) and implicit pressure
  (the liquid and the ideal gas only, §9). Magnetic fields are not refused; they and burning
  have not been tried.
- A run's two materials must be its two species. A third species, a tracer, turns the
  composition off: the dye (the passive scalar) tells bodies apart instead.

## 26. Bodies in a stream

What the code does with solid bodies in a flow: the force the gas exerts on each. The plan
this follows, with what is still to come (bodies from mesh files, a wind tunnel, moving
bodies), is [plans/wind-tunnel.md](../plans/wind-tunnel.md).

### Forces on bodies

A body is an obstacle of the setup: its cells are solid, and each face between a gas cell and
a solid one is a reflecting wall, made by the gas cell's mirror image (section 6, and
`hydro/flux.wgsl`); with viscosity the wall is also no-slip (section 11). The flux through a
wall face is the momentum the gas gives the wall there. So the force on a body is the sum,
over its wall faces, of the fluxes the update has formed anyway, times the faces' areas:

```math
\mathbf F = \sum_{\text{wall faces } f} \pm\, \mathbf F^{\text{mom}}_f A_f ,
```

with + for a gas cell's upper face and - for its lower one. This is the momentum the gas lost
to the body, exactly: a control volume around the body balances it to round-off, which test
`force-balance` checks (below). The torque is taken about each body's point (a sphere's
center, a box's) with each cell's force at the cell's center: exact for the pressure, whose
force on a face is along the face's normal, and off by half a cell's lever for viscosity's.

- **Which body.** A solid cell's mark (the fourth component of the second state texture) is
  1 plus its body's index: the setup's obstacles in order, the first four apart and any after
  them summed with the fourth (`MAX_BODIES`). Every test of the mark is > 0.5, so a mark of 2
  is as solid as one of 1. An obstacle drawn with the brush belongs to the first body.
- **Where it is summed.** The update writes each gas cell's wall flux into two textures laid
  out as the state (`hydro/walls_on.wgsl`, `amr/walls_on.wgsl`): the pressure's part (the flux's
  momentum less viscosity's) and viscosity's, each with the body's mark. A reduction sums
  them per body (`hydro/forces.wgsl`, `amr/forces.wgsl`, `src/hydro/forces.ts`), with the
  diagnostics. Without bodies none of this is compiled in.
- **Implicit pressure.** The update's wall flux is then advection's, which carries nothing
  through a wall, and viscosity's; the pressure's force is the correction's, which forms each
  wall face's pressure (the cell's, carried half a cell along a uniform field) and moves the
  momentum by it. The correction writes that part (`pressure/pwalls_on.wgsl`,
  `amr/pressure/pwalls_on.wgsl`). A gas's pressure is taken less the solve's reference
  pressure: a constant pressure has no net force on a closed body, and float32 keeps its
  digits for the part that does.
- **On blocks.** A body's surface is always at the finest level (section 17), so a wall face
  is never a refinement face and the sum over the leaves counts each once. With per-block
  timesteps a block writes only on its own steps (not in the passes that add a faster
  neighbor's flux half way through), so what is read is each block's last step's force.
- **Reported** as `bodies` in the diagnostics (the page's panel, the runner's `diag.jsonl`,
  the MCP `diagnostics` tool): per body its force, the pressure's and viscosity's parts, the
  torque, and, where the setup gives a reference (a density, speed, area and length), the drag
  and lift coefficients $`C = F / (\tfrac12 \rho U^2 A)`$ and the moment coefficients.

### A wind tunnel

The `tunnel` part (`setup/parts/tunnel.ts`) is a document transform and two pieces of the
solver that already exist: the box's faces (open upstream and downstream; slip walls, open or
periodic at the sides; a floor that may be a no-slip or a moving wall, section 2's kinds 4 and
5), held cells at the inlet (as an `inflow`), and a damping zone at the outlet (section 19's,
`hydro/damping.wgsl`, with a target written for the tunnel: the stream's state, at a rate
$`4U/\ell\,s^2`$ at depth $`s \in [0, 1]`$ into a zone of length $`\ell`$). Below Mach 0.3 the
gas is the liquid with implicit pressure; the damping then relaxes the advected state before
the solve, which projects it, and so is allowed with implicit pressure for the liquid (it is
still refused for a gas, whose solve also carries the energy). The force coefficients
(`Diagnostics.bodies`) are over $`\tfrac12\rho U^2 A`$, with $`A`$ the first obstacle's shadow
along the stream unless the tunnel gives one.

**A rotor's frame.** With `rotor` the run turns at $`\Omega`$ about the stream's axis (section
22, Rotating frame), and the stream is $`U\hat{\mathbf{a}} - \boldsymbol\Omega\times\mathbf{r}`$
at the inlet, in the sponge's target and at the start. Cells outside the cylinder the box holds
are held at it: the box's corners would otherwise carry gas at $`\Omega r`$ across open faces.
For this the liquid's implicit pressure now runs in a rotating frame: the update gives the cell
the centrifugal force's whole kick and the Coriolis force's second half-turn before the solve
(`hydro/update.wgsl`, `amr/update.wgsl`; the explicit step has them in its correction pass, which
an implicit step does not run). For a liquid of uniform density the centrifugal force is a
gradient, which the solve takes into the pressure. A gas with implicit pressure is still refused
in a rotating frame. Test `rotor-frame`: a two-bladed paddle in its frame, the torque about the
axis from the explicit scheme and from implicit pressure, -3.99 and -4.06 (1.7%).

### Bodies that move

An obstacle with a `motion` (`geometry/motion.ts` for its definitions; `hydro/moving.ts` for
the GPU's) moves on the uniform grid. Its pose and velocity are WGSL generated from the setup:
each step, after everything else, one invocation evaluates every moving body's pose at the
step's end time (the clock's, so that the GPU's own timestep sets it), and its velocity by
central differences of the poses (`hydro/moving_pose.wgsl`); a pass marks each cell from the
poses (`hydro/moving_stamp.wgsl`), and a third hands the gas of the cells that change kind to
their neighbors (`hydro/moving_exchange.wgsl`):

- **Covered** (gas, now in the body): its mass, momentum, energies and scalar go to its face
  neighbors that stay gas, shared by their densities.
- **Uncovered** (in the body, now gas): each neighbor that stays gas gives it $`1/(n+1)`$ of
  its state ($`n`$ such neighbors; at most half of what the neighbor holds goes, where it gives
  to several), and it moves at the body's velocity there.

Both are gathers, each cell working out what it gives and takes with the arithmetic its
neighbors use, so mass is conserved to the bit. A body's cell holds the body's velocity there
in its momentum slot (its density as it was). Its wall then moves: the mirror image of a gas
cell has the normal velocity $`2v_{s,n} - v_n`$, and the wall face carries no mass, only the
pressure $`p^*`$ of the Riemann problem between the cell and its image, and its work
$`p^* v_{s,n}`$; viscosity drags the gas toward $`\mathbf v_s`$. A body's cells count in the
timestep's rates, with its velocity, so that it crosses at most the Courant fraction of a cell
in a step. Test `piston` (a slab sliding at Mach 0.085 through a closed box): mass to the bit,
the compressed gas's mean pressure 0.9% from the adiabat, and the energy the gas gained equal
to the work the wall did (the force diagnostic's $`-\sum \mathbf F\cdot\mathbf V\,\Delta t`$)
to 0.06%. The staircase jumps a cell at a time, which leaves a thin band of lower density in
the cells a body uncovers and a bump ahead of it (plans/wind-tunnel.md 6.3).

With implicit pressure the wall's velocity enters the solve's source: a face onto a moving
body's cell carries the cell's density times the wall's normal velocity, so that the solve
sees the body push (`pressure/source.wgsl`); no mass crosses the face in the correction.

The force on a moving body also counts the momentum the exchange hands the gas where a cell
changes kind (an uncovered cell's $`m\mathbf v_s`$ less what its donors gave), so that it is
the momentum the body gave the gas, exactly: test `force-balance-moving` (a cylinder sliding
through a periodic box) closes the box's momentum balance to 6e-8. The viscous stress at a wall
takes the wall's velocity (a moving body's solid cell's).

**With the liquid and implicit pressure, gas fractions near moving bodies** (plans/wind-tunnel.md
8.1's cheap form): each cell near a moving body gets a gas fraction $`\phi`$ and its faces open
fractions $`a_f`$, from the bodies' signed distances at the step's end, stamped before the solve.
The solve takes the cell's pressure at its true density $`\rho/\phi`$, $`\phi`$ times its
compressibility term and $`\beta_f = a_f`$, and the correction the same $`a_f`$ on the pressure's
push in its mass flux, so that as a body sweeps through a cell the solve pushes the liquid out
of it; a cell becomes the body's only when $`\phi < 0.05`$. This moves an open-ended column of
liquid at the piston's speed to 10% with a force on the piston near nil (without it: 40% and
forces of up to 16).

Not yet good enough: test `galilean-drag` (a cylinder moving through still liquid against
liquid streaming past it at rest) gives a drag 87% higher for the moving body with the explicit
scheme and 164% with the fractions; the force history of a moving body is dominated by the
jumps of its cells. The cure is cut cells with state redistribution (plans/wind-tunnel.md 8.1,
whose float64 prototype comes first). Not built: moving bodies on blocks, moving mesh bodies
(their distance field on the GPU).

### Bodies from mesh files

A `mesh` shape (setups.md) is a triangle mesh read from a file (`src/geometry/`: STL, OBJ,
PLY, glTF and GLB, and outlines for 2D). Each file becomes a signed distance field in its own
frame, on a grid at the finest cell size (`geometry/sdf.ts`), which the painters sample: a
cell is solid where the distance at its center is negative.

1. **Winding made consistent.** Vertices at the same place (to a millionth of the mesh's size)
   are welded, and each triangle's neighbor across an edge of exactly two triangles is turned
   where the two run along it the same way. A closed piece is then turned outward (its enclosed
   volume positive); an open one keeps the way most of its triangles were given. An edge of
   three or more triangles, where pieces touch, joins nothing.
2. **A bounding-volume tree** over the triangles, with each node's area-weighted normal and
   centroid.
3. **Distance**, exact (to the nearest triangle) within a band of three cells of the surface,
   and clamped to the band's width beyond.
4. **Sign** by the generalized winding number ([Jacobson et al. 2013][Jacobson2013]), with a
   node more than two of its radii away counted as its dipole ([Barill et al. 2018][Barill2018]):
   inside above 1/2. Holes and pieces that pass through each other change the number a little
   and not the answer; a triangle wound the wrong way would subtract its solid angle, which is
   why the winding is made consistent first. For a single closed piece, a point whose nearest
   surface point is inside a face takes that face's side instead. Beyond the band a point takes
   the number of a coarse grid of the band's spacing: it is at least the band's width from the
   surface, and its coarse sample within 0.87 of that, so the two are on the same side.
5. **A report**: volume, area, bounds, the shadow along each axis (the drag's reference area),
   the thinnest part (from each triangle inward along its normal to the next surface), and how
   far the winding number strays from 0 or 1 away from the surface (how broken the mesh is).

Measured (unit tests `sdf-*`, `tests/unit/geometry.test.ts`): an icosphere of 5120 triangles
matches r - R within the band to 0.1 of a cell; the volume the field holds (each point counted
as clamp(1/2 - d/h, 0, 1) of a cell) converges to the mesh's at order 1.7 to 2.9 (relative
error 1.2e-2, 3.5e-3, 5.3e-4, 7.1e-5 at 10, 20, 40, 80 points across); a tenth of the triangles
removed, or half of them flipped, changes no point more than a cell from the surface; touching
pieces stay solid at their shared face. A field takes seconds on the CPU for meshes of tens of
thousands of triangles and under half a minute for a million (one core of a desktop CPU:
82 000 triangles on 129 cubed points in 2.7 s, 330 000 on 249 cubed in 19 s, 1.3 million on
249 cubed in 26 s). It is made where the setup is compiled, and kept for later runs at the
same spacing; on blocks each initialization worker makes its own.

**Validation.** `force-balance`: a periodic box with a cylinder and a slab of fluid moving
toward it, viscosity on, so that only the body changes the gas's momentum. Over 30 steps the
change of the gas's momentum is minus the sum of the force times dt to 3e-7 to 5e-6 of it,
in each of eight runs: liquid and ideal gas, explicit and implicit pressure, uniform grid and
two levels of blocks (the test's threshold is 1e-4).

## References

[Jacobson2013]: https://doi.org/10.1145/2461912.2461916
[Barill2018]: https://doi.org/10.1145/3197517.3201337
[ThompsonLauson1972]: https://github.com/isale-code/M-ANEOS/tree/master/docs
[Melosh2007]: https://doi.org/10.1111/j.1945-5100.2007.tb01009.x
[Stewart2020]: https://arxiv.org/abs/1910.04687
[StewartPyrolite]: https://doi.org/10.5281/zenodo.6968539
[StewartFe85Si15]: https://doi.org/10.5281/zenodo.3866550
[StewartWater]: https://doi.org/10.5281/zenodo.14226693
[StewartForsterite]: https://doi.org/10.5281/zenodo.3478631
[pyaneos]: https://doi.org/10.5281/zenodo.20736271
[FritschCarlson1980]: https://doi.org/10.1137/0717021
[Canup2004]: https://doi.org/10.1016/j.icarus.2003.09.028
[Ida1997]: https://doi.org/10.1038/38669
[Kokubo2000]: https://doi.org/10.1006/icar.2000.6496
[He2024]: https://arxiv.org/abs/2404.08247
[Skinner2013]: https://arxiv.org/abs/1306.0010
[Skinner2019]: https://arxiv.org/abs/1806.07390
[SuOlson1996]: https://doi.org/10.1016/0022-4073(96)84524-9
[Wibking2022]: https://arxiv.org/abs/2110.01792
[Bate1995]: https://doi.org/10.1093/mnras/277.2.362
[Berger1984]: https://doi.org/10.1016/0021-9991(84)90073-1
[Berger1989]: https://doi.org/10.1016/0021-9991(89)90035-1
[Brandt1977]: https://doi.org/10.1090/S0025-5718-1977-0431719-X
[Bryan1995]: https://doi.org/10.1016/0010-4655(94)00191-4
[Edelmann2021]: https://doi.org/10.1051/0004-6361/202140653
[GrosheintzLaval2020]: https://doi.org/10.1016/j.jcp.2020.109805
[Cada2009]: https://doi.org/10.1016/j.jcp.2009.02.020
[Chandrasekhar1935]: https://doi.org/10.1093/mnras/95.3.207
[Davis1988]: https://doi.org/10.1137/0909030
[deValBorro2006]: https://doi.org/10.1111/j.1365-2966.2006.10488.x
[Hawley2000]: https://doi.org/10.1086/308180
[Hawley2011]: https://doi.org/10.1088/0004-637X/738/1/84
[Eggleton1983]: https://doi.org/10.1086/160960
[Evrard1988]: https://doi.org/10.1093/mnras/235.3.911
[Farris2014]: https://doi.org/10.1088/0004-637X/783/2/134
[Federrath2010]: https://doi.org/10.1088/0004-637X/713/1/269
[Fleischmann2020]: https://doi.org/10.1016/j.jcp.2019.109004
[Fryxell2000]: https://doi.org/10.1086/317361
[Gresho1990]: https://doi.org/10.1002/fld.1650110510
[Guillochon2009]: https://doi.org/10.1088/0004-637X/705/1/844
[Guillochon2013]: https://doi.org/10.1088/0004-637X/767/1/25
[Hachisu1986]: https://doi.org/10.1086/191148
[Harten1983]: https://doi.org/10.1137/1025002
[Jeans1902]: https://doi.org/10.1098/rsta.1902.0012
[Kappeli2014]: https://doi.org/10.1016/j.jcp.2013.11.028
[Koren1993]: https://openalex.org/W2125064741
[Kwatra2009]: https://doi.org/10.1016/j.jcp.2009.02.027
[Koyama2002]: https://doi.org/10.1086/338978
[EP88]: https://doi.org/10.1016/0045-7930(88)90013-8
[Federrath2010]: https://doi.org/10.1051/0004-6361/200912437
[Mallet2026]: https://doi.org/10.5281/zenodo.21987739
[Thurgood2018]: https://doi.org/10.1063/1.5035489
[Tsiklauri2007]: https://doi.org/10.1063/1.2800854
[Ghia1982]: https://doi.org/10.1016/0021-9991(82)90058-4
[Koyama2004]: https://doi.org/10.1086/382478
[Krumholz2004]: https://doi.org/10.1086/421935
[Kushnir2013]: https://doi.org/10.1088/2041-8205/778/2/L37
[Kushnir2020]: https://doi.org/10.1093/mnras/staa3017
[KushnirKatz2020]: https://doi.org/10.1093/mnras/staa594
[Lohner1987]: https://doi.org/10.1016/0045-7825(87)90098-3
[Plewa1999]: https://arxiv.org/abs/astro-ph/9807241
[Plummer1911]: https://doi.org/10.1093/mnras/71.5.460
[Quirk1994]: https://doi.org/10.1002/fld.1650180603
[Noh1987]: https://doi.org/10.1016/0021-9991(87)90074-X
[Coggeshall1991]: https://doi.org/10.1063/1.858008
[Colella1986]: https://doi.org/10.1137/0907073
[Liska2003]: https://doi.org/10.1137/S1064827502402120
[Roe1986]: https://doi.org/10.1146/annurev.fl.18.010186.002005
[Ross2011]: https://arxiv.org/abs/1011.0686
[Saha1920]: https://doi.org/10.1080/14786441008636148
[Tomida2013]: https://doi.org/10.1088/0004-637X/763/1/6
[Masunaga2000]: https://doi.org/10.1086/308439
[Sedov1959]: https://ui.adsabs.harvard.edu/abs/1959sdmm.book.....S
[Smith2023]: https://doi.org/10.3847/1538-4357/acbaff
[Springel2005]: https://doi.org/10.1111/j.1365-2966.2005.09655.x
[Springel2010]: https://doi.org/10.1111/j.1365-2966.2009.15715.x
[Stone2020]: https://doi.org/10.3847/1538-4365/ab929b
[Taylor1950]: https://doi.org/10.1098/rspa.1950.0049
[Thornber2008]: https://doi.org/10.1016/j.jcp.2008.01.036
[Timmes2000]: https://doi.org/10.1086/313407
[TimmesSwesty2000]: https://doi.org/10.1086/313304
[Hamada1961]: https://doi.org/10.1086/147195
[Toro1994]: https://doi.org/10.1007/BF01414629
[Toro2009]: https://doi.org/10.1007/b79761
[SG09]: https://doi.org/10.1016/j.newast.2008.06.003
[Lee13]: https://doi.org/10.1016/j.jcp.2013.02.049
[LD09]: https://doi.org/10.1016/j.jcp.2008.08.026
[TR02]: https://doi.org/10.1006/jcph.2002.7120
[GS05]: https://doi.org/10.1016/j.jcp.2004.11.016
[FKR63]: https://doi.org/10.1063/1.1706761
[MK05]: https://doi.org/10.1016/j.jcp.2005.02.017
[Truelove1997]: https://doi.org/10.1086/310975
[vanLeer1974]: https://doi.org/10.1016/0021-9991(74)90019-9
[vanLeer1977]: https://doi.org/10.1016/0021-9991(77)90095-X
[vanLeer1979]: https://doi.org/10.1016/0021-9991(79)90145-1
[VazquezSemadeni2007]: https://doi.org/10.1086/510771
[Zingale2002]: https://doi.org/10.1086/342754
[Karni1996]: https://doi.org/10.1137/S106482759528003X
[QuirkKarni1996]: https://doi.org/10.1017/S0022112096007069

- Barill, G., Dickson, N. G., Schmidt, R., Levin, D. I. W., & Jacobson, A. (2018). Fast winding numbers for soups and clouds. ACM Trans. Graph. 37, 43. [doi:10.1145/3197517.3201337](https://doi.org/10.1145/3197517.3201337)
- Bate, M. R., Bonnell, I. A., & Price, N. M. (1995). Modelling accretion in protobinary systems. MNRAS 277, 362. [doi:10.1093/mnras/277.2.362](https://doi.org/10.1093/mnras/277.2.362)
- Berger, M. J., & Oliger, J. (1984). Adaptive Mesh Refinement for Hyperbolic Partial Differential Equations. Journal of Computational Physics 53, 484. [doi:10.1016/0021-9991(84)90073-1](https://doi.org/10.1016/0021-9991(84)90073-1)
- Berger, M. J., & Colella, P. (1989). Local Adaptive Mesh Refinement for Shock Hydrodynamics. Journal of Computational Physics 82, 64. [doi:10.1016/0021-9991(89)90035-1](https://doi.org/10.1016/0021-9991(89)90035-1)
- Brandt, A. (1977). Multi-level adaptive solutions to boundary-value problems. Mathematics of Computation 31, 333. [doi:10.1090/S0025-5718-1977-0431719-X](https://doi.org/10.1090/S0025-5718-1977-0431719-X)
- Bryan, G. L., et al. (1995). A piecewise parabolic method for cosmological hydrodynamics. Computer Physics Communications 89, 149. [doi:10.1016/0010-4655(94)00191-4](https://doi.org/10.1016/0010-4655(94)00191-4)
- Čada, M., & Torrilhon, M. (2009). Compact third-order limiter functions for finite volume methods. Journal of Computational Physics 228, 4118. [doi:10.1016/j.jcp.2009.02.020](https://doi.org/10.1016/j.jcp.2009.02.020)
- Canup, R. M. (2004). Simulations of a late lunar-forming impact. Icarus 168, 433. [doi:10.1016/j.icarus.2003.09.028](https://doi.org/10.1016/j.icarus.2003.09.028)
- Chandrasekhar, S. (1935). The Highly Collapsed Configurations of a Stellar Mass (Second Paper). MNRAS 95, 207. [doi:10.1093/mnras/95.3.207](https://doi.org/10.1093/mnras/95.3.207)
- Davis, S. F. (1988). Simplified Second-Order Godunov-Type Methods. SIAM Journal on Scientific and Statistical Computing 9, 445. [doi:10.1137/0909030](https://doi.org/10.1137/0909030)
- de Val-Borro, M., et al. (2006). A comparative study of disc-planet interaction. MNRAS 370, 529. [doi:10.1111/j.1365-2966.2006.10488.x](https://doi.org/10.1111/j.1365-2966.2006.10488.x)
- Eggleton, P. P. (1983). Approximations to the radii of Roche lobes. ApJ 268, 368. [doi:10.1086/160960](https://doi.org/10.1086/160960)
- Evrard, A. E. (1988). Beyond N-body: 3D cosmological gas dynamics. MNRAS 235, 911. [doi:10.1093/mnras/235.3.911](https://doi.org/10.1093/mnras/235.3.911)
- Farris, B. D., et al. (2014). Binary Black Hole Accretion from a Circumbinary Disk: Gas Dynamics inside the Central Cavity. ApJ 783, 134. [doi:10.1088/0004-637X/783/2/134](https://doi.org/10.1088/0004-637X/783/2/134)
- Federrath, C., et al. (2010). Modeling Collapse and Accretion in Turbulent Gas Clouds: Implementation and Comparison of Sink Particles in AMR and SPH. ApJ 713, 269. [doi:10.1088/0004-637X/713/1/269](https://doi.org/10.1088/0004-637X/713/1/269)
- Fleischmann, N., et al. (2020). A low dissipation method to cure the grid-aligned shock instability. Journal of Computational Physics 401, 109004. [doi:10.1016/j.jcp.2019.109004](https://doi.org/10.1016/j.jcp.2019.109004)
- Fritsch, F. N., & Carlson, R. E. (1980). Monotone Piecewise Cubic Interpolation. SIAM Journal on Numerical Analysis 17, 238. [doi:10.1137/0717021](https://doi.org/10.1137/0717021)
- Fryxell, B., et al. (2000). FLASH: An Adaptive Mesh Hydrodynamics Code for Modeling Astrophysical Thermonuclear Flashes. ApJS 131, 273. [doi:10.1086/317361](https://doi.org/10.1086/317361)
- Gresho, P. M., & Chan, S. T. (1990). On the theory of semi-implicit projection methods for viscous incompressible flow and its implementation via a finite element method that also introduces a nearly consistent mass matrix. II - Implementation. International Journal for Numerical Methods in Fluids 11, 621. [doi:10.1002/fld.1650110510](https://doi.org/10.1002/fld.1650110510)
- Guillochon, J., et al. (2009). Three-dimensional Simulations of Tidally Disrupted Solar-type Stars and the Observational Signatures of Shock Breakout. ApJ 705, 844. [doi:10.1088/0004-637X/705/1/844](https://doi.org/10.1088/0004-637X/705/1/844)
- Guillochon, J., & Ramirez-Ruiz, E. (2013). Hydrodynamical Simulations to Determine the Feeding Rate of Black Holes by the Tidal Disruption of Stars: The Importance of the Impact Parameter and Stellar Structure. ApJ 767, 25. [doi:10.1088/0004-637X/767/1/25](https://doi.org/10.1088/0004-637X/767/1/25)
- Hamada, T., & Salpeter, E. E. (1961). Models for Zero-Temperature Stars. ApJ 134, 683. [doi:10.1086/147195](https://doi.org/10.1086/147195)
- Hachisu, I. (1986). A Versatile Method for Obtaining Structures of Rapidly Rotating Stars. II. Three-dimensional Self-consistent Field Method. ApJS 62, 461. [doi:10.1086/191148](https://doi.org/10.1086/191148)
- Harten, A., Lax, P. D., & van Leer, B. (1983). On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws. SIAM Review 25, 35. [doi:10.1137/1025002](https://doi.org/10.1137/1025002)
- Hawley, J. F. (2000). Global Magnetohydrodynamical Simulations of Accretion Tori. ApJ 528, 462. [doi:10.1086/308180](https://doi.org/10.1086/308180)
- Hawley, J. F., Guan, X., & Krolik, J. H. (2011). Assessing Quantitative Results in Accretion Simulations: From Local to Global. ApJ 738, 84. [doi:10.1088/0004-637X/738/1/84](https://doi.org/10.1088/0004-637X/738/1/84)
- Ida, S., Canup, R. M., & Stewart, G. R. (1997). Lunar accretion from an impact-generated disk. Nature 389, 353. [doi:10.1038/38669](https://doi.org/10.1038/38669)
- Jacobson, A., Kavan, L., & Sorkine-Hornung, O. (2013). Robust inside-outside segmentation using generalized winding numbers. ACM Trans. Graph. 32, 33. [doi:10.1145/2461912.2461916](https://doi.org/10.1145/2461912.2461916)
- Jeans, J. H. (1902). The Stability of a Spherical Nebula. Philosophical Transactions of the Royal Society of London Series A 199, 1. [doi:10.1098/rsta.1902.0012](https://doi.org/10.1098/rsta.1902.0012)
- Käppeli, R., & Mishra, S. (2014). Well-balanced schemes for the Euler equations with gravitation. Journal of Computational Physics 259, 199. [doi:10.1016/j.jcp.2013.11.028](https://doi.org/10.1016/j.jcp.2013.11.028)
- Kokubo, E., Ida, S., & Makino, J. (2000). Evolution of a Circumterrestrial Disk and Formation of a Single Moon. Icarus 148, 419. [doi:10.1006/icar.2000.6496](https://doi.org/10.1006/icar.2000.6496)
- Koren, B. (1993). A robust upwind discretization method for advection, diffusion and source terms. In Numerical Methods for Advection-Diffusion Problems, Notes on Numerical Fluid Mechanics 45, 117. Vieweg. [OpenAlex](https://openalex.org/W2125064741)
- Koyama, H., & Inutsuka, S. (2002). An Origin of Supersonic Motions in Interstellar Clouds. ApJL 564, L97. [doi:10.1086/338978](https://doi.org/10.1086/338978)
- Koyama, H., & Inutsuka, S. (2004). The Field Condition: A New Constraint on Spatial Resolution in Simulations of the Nonlinear Development of Thermal Instability. ApJL 602, L25. [doi:10.1086/382478](https://doi.org/10.1086/382478)
- Krumholz, M. R., McKee, C. F., & Klein, R. I. (2004). Embedding Lagrangian Sink Particles in Eulerian Grids. ApJ 611, 399. [doi:10.1086/421935](https://doi.org/10.1086/421935)
- Kushnir, D., Katz, B., Dong, S., Livne, E., & Fernández, R. (2013). Head-on Collisions of White Dwarfs in Triple Systems Could Explain Type Ia Supernovae. ApJL 778, L37. [doi:10.1088/2041-8205/778/2/L37](https://doi.org/10.1088/2041-8205/778/2/L37)
- Kushnir, D., & Katz, B. (2020). An accurate and efficient numerical calculation of detonation waves in multidimensional supernova simulations using a burning limiter and adaptive quasi-statistical equilibrium. MNRAS 493, 5413. [doi:10.1093/mnras/staa594](https://doi.org/10.1093/mnras/staa594)
- Kushnir, D., Wygoda, N., & Sharon, A. (2020). Sub-Chandrasekhar-mass detonations are in tension with the observed t0-M_Ni56 relation of type Ia supernovae. MNRAS 499, 4725. [doi:10.1093/mnras/staa3017](https://doi.org/10.1093/mnras/staa3017)
- Kwatra, N., Su, J., Grétarsson, J. T., & Fedkiw, R. (2009). A method for avoiding the acoustic time step restriction in compressible flow. Journal of Computational Physics 228, 4146. [doi:10.1016/j.jcp.2009.02.027](https://doi.org/10.1016/j.jcp.2009.02.027)
- Löhner, R. (1987). An adaptive finite element scheme for transient problems in CFD. Computer Methods in Applied Mechanics and Engineering 61, 323. [doi:10.1016/0045-7825(87)90098-3](https://doi.org/10.1016/0045-7825(87)90098-3)
- Masunaga, H., & Inutsuka, S. (2000). A Radiation Hydrodynamic Model for Protostellar Collapse. II. The Second Collapse and the Birth of a Protostar. ApJ 531, 350. [doi:10.1086/308439](https://doi.org/10.1086/308439)
- Melosh, H. J. (2007). A hydrocode equation of state for SiO2. Meteoritics & Planetary Science 42, 2079. [doi:10.1111/j.1945-5100.2007.tb01009.x](https://doi.org/10.1111/j.1945-5100.2007.tb01009.x)
- Plewa, T., & Müller, E. (1999). The consistent multi-fluid advection method. A&A 342, 179. [arXiv:astro-ph/9807241](https://arxiv.org/abs/astro-ph/9807241)
- Plummer, H. C. (1911). On the problem of distribution in globular star clusters. MNRAS 71, 460. [doi:10.1093/mnras/71.5.460](https://doi.org/10.1093/mnras/71.5.460)
- Quirk, J. J. (1994). A contribution to the great Riemann solver debate. International Journal for Numerical Methods in Fluids 18, 555. [doi:10.1002/fld.1650180603](https://doi.org/10.1002/fld.1650180603)
- Colella, P., Majda, A., & Roytburd, V. (1986). Theoretical and numerical structure for reacting shock waves. SIAM Journal on Scientific and Statistical Computing 7, 1059. [doi:10.1137/0907073](https://doi.org/10.1137/0907073)
- Coggeshall, S. V. (1991). Analytic solutions of hydrodynamics equations. Physics of Fluids A 3, 757. [doi:10.1063/1.858008](https://doi.org/10.1063/1.858008)
- Fickett, W., & Davis, W. C. (1979). Detonation. University of California Press.
- Fickett, W., & Rivard, W. C. (1974). Test problems for hydrocodes. Los Alamos Scientific Laboratory report LA-5479.
- Noh, W. F. (1987). Errors for calculations of strong shocks using an artificial viscosity and an artificial heat flux. Journal of Computational Physics 72, 78. [doi:10.1016/0021-9991(87)90074-X](https://doi.org/10.1016/0021-9991(87)90074-X)
- Liska, R., & Wendroff, B. (2003). Comparison of several difference schemes on 1D and 2D test problems for the Euler equations. SIAM Journal on Scientific Computing 25, 995. [doi:10.1137/S1064827502402120](https://doi.org/10.1137/S1064827502402120)
- Roe, P. L. (1986). Characteristic-based schemes for the euler equations. Annual Review of Fluid Mechanics 18, 337. [doi:10.1146/annurev.fl.18.010186.002005](https://doi.org/10.1146/annurev.fl.18.010186.002005)
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
- Saha, M. N. (1920). Ionization in the Solar Chromosphere. Philosophical Magazine 40, 472. [doi:10.1080/14786441008636148](https://doi.org/10.1080/14786441008636148)
- Sedov, L. I. (1959). Similarity and Dimensional Methods in Mechanics. Academic Press. [link](https://ui.adsabs.harvard.edu/abs/1959sdmm.book.....S)
- Smith Clark, A. I., et al. (2023). pynucastro: A Python library for nuclear astrophysics. ApJ. [doi:10.3847/1538-4357/acbaff](https://doi.org/10.3847/1538-4357/acbaff)
- Springel, V. (2005). The cosmological simulation code GADGET-2. MNRAS 364, 1105. [doi:10.1111/j.1365-2966.2005.09655.x](https://doi.org/10.1111/j.1365-2966.2005.09655.x)
- Springel, V. (2010). E pur si muove: Galilean-invariant cosmological hydrodynamical simulations on a moving mesh. MNRAS 401, 791. [doi:10.1111/j.1365-2966.2009.15715.x](https://doi.org/10.1111/j.1365-2966.2009.15715.x)
- Stewart, S. T. (2020). Equation of State Model Fe85Si15-ANEOS: Development and documentation (Version SLVTv0.2G1). Zenodo. [doi:10.5281/zenodo.3866550](https://doi.org/10.5281/zenodo.3866550)
- Stewart, S. T. (2022). ANEOS-PYROLITE (v0.2.0). Zenodo. [doi:10.5281/zenodo.6968539](https://doi.org/10.5281/zenodo.6968539)
- Stewart, S. T. (2024). ANEOS Equation of State Model for Water (v0.3). Zenodo. [doi:10.5281/zenodo.14226693](https://doi.org/10.5281/zenodo.14226693)
- Stewart, S. T. (2026). pyaneos: an oracle-validated Python port of M-ANEOS (v1.0). Zenodo. [doi:10.5281/zenodo.20736271](https://doi.org/10.5281/zenodo.20736271)
- Stewart, S. T., et al. (2019). Equation of State Model Forsterite-ANEOS-SLVTv1.0G1: Documentation and Comparisons. Zenodo. [doi:10.5281/zenodo.3478631](https://doi.org/10.5281/zenodo.3478631)
- Stewart, S. T., et al. (2020). The shock physics of giant impacts: Key requirements for the equations of state. AIP Conference Proceedings 2272, 080003. [arXiv:1910.04687](https://arxiv.org/abs/1910.04687)
- Stone, J. M., et al. (2020). The Athena++ Adaptive Mesh Refinement Framework: Design and Magnetohydrodynamic Solvers. ApJS 249, 4. [doi:10.3847/1538-4365/ab929b](https://doi.org/10.3847/1538-4365/ab929b)
- Taylor, G. (1950). The Formation of a Blast Wave by a Very Intense Explosion. I. Theoretical Discussion. Proceedings of the Royal Society of London Series A 201, 159. [doi:10.1098/rspa.1950.0049](https://doi.org/10.1098/rspa.1950.0049)
- Thompson, S. L., & Lauson, H. S. (1972). Improvements in the Chart D radiation-hydrodynamic code III: Revised analytic equations of state. Sandia National Laboratories. [In the M-ANEOS repository](https://github.com/isale-code/M-ANEOS/tree/master/docs)
- Thornber, B., et al. (2008). An improved reconstruction method for compressible flows with low Mach number features. Journal of Computational Physics 227, 4873. [doi:10.1016/j.jcp.2008.01.036](https://doi.org/10.1016/j.jcp.2008.01.036)
- Timmes, F. X., & Swesty, F. D. (2000). The Accuracy, Consistency, and Speed of an Electron-Positron Equation of State Based on Table Interpolation of the Helmholtz Free Energy. ApJS 126, 501. [doi:10.1086/313304](https://doi.org/10.1086/313304)
- Timmes, F. X., Hoffman, R. D., & Woosley, S. E. (2000). An inexpensive nuclear energy generation network for stellar hydrodynamics. ApJS 129, 377. [doi:10.1086/313407](https://doi.org/10.1086/313407)
- Tomida, K., et al. (2013). Radiation Magnetohydrodynamic Simulations of Protostellar Collapse: Protostellar Core Formation. ApJ 763, 6. [doi:10.1088/0004-637X/763/1/6](https://doi.org/10.1088/0004-637X/763/1/6)
- Toro, E. F., Spruce, M., & Speares, W. (1994). Restoration of the contact surface in the HLL-Riemann solver. Shock Waves 4, 25. [doi:10.1007/BF01414629](https://doi.org/10.1007/BF01414629)
- Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, 3rd ed. Springer. [doi:10.1007/b79761](https://doi.org/10.1007/b79761)
- Truelove, J. K., et al. (1997). The Jeans Condition: A New Constraint on Spatial Resolution in Simulations of Isothermal Self-gravitational Hydrodynamics. ApJL 489, L179. [doi:10.1086/310975](https://doi.org/10.1086/310975)
- van Leer, B. (1974). Towards the Ultimate Conservation Difference Scheme. II. Monotonicity and Conservation Combined in a Second-Order Scheme. Journal of Computational Physics 14, 361. [doi:10.1016/0021-9991(74)90019-9](https://doi.org/10.1016/0021-9991(74)90019-9)
- van Leer, B. (1977). Towards the Ultimate Conservative Difference Scheme. IV. A New Approach to Numerical Convection. Journal of Computational Physics 23, 276. [doi:10.1016/0021-9991(77)90095-X](https://doi.org/10.1016/0021-9991(77)90095-X)
- van Leer, B. (1979). Towards the Ultimate Conservative Difference Scheme. V. A Second-Order Sequel to Godunov's Method. Journal of Computational Physics 32, 101. [doi:10.1016/0021-9991(79)90145-1](https://doi.org/10.1016/0021-9991(79)90145-1)
- Vázquez-Semadeni, E., et al. (2007). Molecular Cloud Evolution. II. From Cloud Formation to the Early Stages of Star Formation in Decaying Conditions. ApJ 657, 870. [doi:10.1086/510771](https://doi.org/10.1086/510771)
- Yakovlev, D. G., & Shalybkov, D. A. (1989). Degenerate cores of white dwarfs and envelopes of neutron stars: thermodynamics and plasma screening in thermonuclear reactions. Astrophysics and Space Physics Reviews 7, 311.
- Zingale, M., et al. (2002). Mapping Initial Hydrostatic Models in Godunov Codes. ApJS 143, 539. [doi:10.1086/342754](https://doi.org/10.1086/342754)
