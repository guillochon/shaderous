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
20. [Nuclear burning surrogate (in development)](#20-nuclear-burning-surrogate-in-development)
21. [Validation](#21-validation)
22. [References](#references)

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
C\,\frac{\Delta x}{\max\left(|v_d| + c_s + s_\text{diff}\right)},\quad
C\sqrt{\frac{\Delta x}{\max|\mathbf{g}|}},\quad
\Delta t_\text{pair},\quad
\Delta t_\text{max}
\right],
```

where $`C = 0.8`$ is the CFL number, $`c_s`$ the sound speed, and the maximum is over cells and
axes. Each term is:

- **Diffusion** adds a signal speed $`s_\text{diff} = 2.5\,d\,D/\Delta x`$, where $`d`$ is the number of
  dimensions and $`D`$ the larger of $`\tfrac43\nu`$ and the thermal diffusivity ([§11](#11-viscosity-and-thermal-conduction)).
  Where diffusion dominates, the step falls as $`\Delta x^2`$, as an explicit scheme requires.
- **Gravity** limits the step so that no cell is accelerated across more than a fraction of
  itself in one step.
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
measures pressure jumps against the same balance.

Measured: an isothermal column whose density falls by a factor $`e`$ every 4 cells (nine
million over the box) keeps its largest Mach number at $`4\times10^{-5}`$ after a sound crossing,
against 0.35 without the balanced reconstruction. The background of the Rayleigh-Taylor
problem is 240 times quieter, and the error in the Jeans growth rate 12 times smaller. It is
on by default for every equation of state with an energy; `wb=0` in a link turns it off for
comparison.

## 9. Equations of state

`src/eos/`. Every equation of state implements the same four WGSL functions, so the hydro
passes do not depend on it:

```wgsl
fn eos_pressure(rho: f32, eint: f32) -> f32;      // eint = internal energy density
fn eos_sound_speed(rho: f32, p: f32) -> f32;
fn eos_eint_from_p(rho: f32, p: f32) -> f32;
fn eos_temperature(rho: f32, eint: f32) -> f32;
```

The equation of state is chosen when the shaders are assembled, and each module has a
float64 TypeScript mirror for testing (`tests/unit/eos*.test.ts`).

| Equation of state | Pressure | Notes |
|---|---|---|
| **Ideal gas** (`ideal.wgsl`) | $`p = (\gamma - 1)\rho e`$ | $`c_s^2 = \gamma p/\rho`$ |
| **Barotropic** (`barotropic.wgsl`) | $`p = K\rho^{\Gamma}`$ (isothermal for $`\Gamma = 1`$) | The energy equation is not used; $`\rho e`$ follows the density; HLL fluxes |
| **Gas + radiation** (`gas_radiation.wgsl`) | $`p = \dfrac{\rho k T}{\mu m_u} + \tfrac13 a T^4`$ | $`T`$ is the root of $`\rho e = \dfrac{\rho kT}{(\gamma - 1)\mu m_u} + aT^4`$, from 8 Newton steps started above the root (`quartic.wgsl`) |
| **Degenerate** (`degenerate.wgsl`) | $`p = p_\text{deg}(\rho) + p_\text{ion} + p_\text{rad}`$ | Zero-temperature electrons at any degree of relativity ([Chandrasekhar 1935][Chandrasekhar1935]), plus ideal ions and radiation |
| **Ionizing hydrogen** (`ionizing.wgsl`) | $`p = \dfrac{\rho k T}{m_u}\left[h(1 + x) + o\right]`$ | Ionized fraction $`x`$ of the hydrogen from Saha equilibrium ([Saha 1920][Saha1920]); $`h`$ and $`o`$ are the particles per $`m_u`$ from hydrogen nuclei and from the other species |

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

**Saha hydrogen.** The ionization fraction solves

```math
\frac{x^2}{1 - x} = \frac{m_u}{X\rho}\left(\frac{2\pi m_e kT}{h^2}\right)^{3/2} e^{-\chi/kT},
\qquad \chi = 13.6\ \text{eV},
```

where $`X`$ is the hydrogen mass fraction, and the internal energy includes $`\chi`$ per ionization. $`T`$ comes from Newton's method kept
inside a bracket. The adiabatic index $`\Gamma_1`$ is computed exactly and dips below 1.3 in the
ionization zone.

## 10. Optically thin cooling and heating

`src/cooling/`, `hydro/cooling.wgsl`. Two functions are available, chosen when the shaders are
assembled:

- **Power law with uniform heating:** $`\mathcal{C} - \mathcal{H} = \rho^2\Lambda_0 T^\alpha - \rho\Gamma_0`$, with no cooling below a temperature floor ($`\alpha = 1/2`$ is bremsstrahlung).
- **Interstellar:** the fit of [Koyama & Inutsuka (2002)][Koyama2002], as corrected by
  [Vázquez-Semadeni et al. (2007)][VazquezSemadeni2007]:

```math
\Gamma = 2\times10^{-26}\ \text{erg s}^{-1}, \qquad
\frac{\Lambda(T)}{\Gamma} = 10^{7}\exp\left(\frac{-1.184\times10^5}{T + 1000}\right) + 1.4\times10^{-2}\sqrt{T}\exp\left(\frac{-92}{T}\right)\ \text{cm}^3,
```

with $`\mathcal{C} - \mathcal{H} = n^2\Lambda - n\Gamma`$.

Cooling acts on the internal energy at fixed density at the end of the update, in up to 12
second-order substeps that each change the energy by at most a tenth. A substep that would
cross thermal balance stops at the balance point. A cooling time far shorter than the
timestep is therefore stable and does not overshoot. The timestep is not shortened for
cooling: a cell that cools faster than the substeps can follow loses at most 70% of its
energy in one step. In code units the temperature is $`p/\rho`$; the interstellar curve
converts it to kelvin and reads the density as atoms per cm³.

## 11. Viscosity and thermal conduction

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
are the mean of the centered differences in the two cells. Because they are part of the face
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
at $`t = 0.6`$ changes by 8% between $`512^2`$ and $`1024^2`$ cells, against a 20% fall without
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
- Warm-started from the previous potential. The default is 2 V-cycles per step (3 on the
  adaptive mesh, where later cycles run only while they still reduce the residual).
- **Periodic** boundaries subtract the mean density, $`\bar\rho_\text{per} = \langle\rho\rangle`$.
- **Isolated** boundaries: a reduction computes the mass $`M`$, center of mass and second moments
  $`S_{ij}`$ of the gas, and the boundary potential comes from the monopole and quadrupole
  (the dipole vanishes about the center of mass):

```math
\Phi_\text{bdry}(\mathbf{x}) = -G\left[\frac{M}{r} + \frac{Q_{ij}d_i d_j}{2r^5}\right],
\qquad Q_{ij} = 3S_{ij} - S_{kk}\,\delta_{ij},
```

where $`\mathbf{d}`$ is the position relative to the center of mass and $`r = |\mathbf{d}|`$.

### Multigrid on the adaptive mesh

`src/amr/gravity/`. The potential lives on every block, parents included. The density is
averaged into the parents. A level's ghost cells come from its neighbors or are interpolated
from the level above. A multigrid cycle runs inside the blocks (16, 8, 4, 2 and 1 cells across).
The coupling between levels is one-way: a fine level never corrects the coarse potential
beneath it. On a uniform sphere the error stays below 0.3% with three levels. A new mesh is
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

   The same work leaves one cell and enters its neighbor, so total energy (gas plus field) is
   conserved. It can overheat low-density cells next to a sharp density contrast.
2. **Kick form.** Each kick $`\mathbf{k}`$ adds exactly the kinetic energy it creates,
   $`\mathbf{k}\cdot(\mathbf{m} + \mathbf{k}/2)/\rho`$. Gravity then never changes the internal
   energy, but total energy is not conserved where the flow is under-resolved.

The half-step potential $`\Phi^{n+1/2}`$ is extrapolated from the last two solves.

## 15. Point masses and sinks

`gravity/bh*.wgsl`, `gravity/sink.wgsl`. Up to four point masses act on the gas through
Plummer-softened gravity ([Plummer 1911][Plummer1911]),

```math
\Phi_\text{pm}(\mathbf{x}) = -\sum_k\frac{GM_k}{\sqrt{|\mathbf{x} - \mathbf{x}_k|^2 + \epsilon_k^2}},
```

and share the gravity-field texture with self-gravity, so the hydro passes do not know they
exist. Each step the masses drift. One reduction over the updated gas then gives each mass
its pull from the gas (the reaction to its pull on the gas) and what its sink takes. The
second half-kick pass removes exactly that from the cells. The pull between two masses is
softened by the larger of their two softening lengths.

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

For an $`n = 1`$ polytrope this relation is exactly the trapezoid rule of
[§8](#8-well-balanced-reconstruction). Unsettled, a $`1\,M_\odot`$ white dwarf (whose nearly
relativistic core has $`\Gamma_1`$ close to 4/3 and so little restoring force) swung by 25% in
central density. Settled, it stays within 3.5% over six dynamical times.

## 17. Adaptive mesh refinement

`src/amr/`. The mesh is a tree of blocks of $`16^3`$ cells ($`16^2`$ in 2D) with two ghost layers.
A block splits into $`2^d`$ children of half the cell size, and face neighbors differ by at most
one level. The data layout is described in [architecture.md](architecture.md#adaptive-mesh).

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
- **Thin gas** below a density cutoff never asks for refinement, so ripples in the ambient
  medium around a star do not refine it.

The mesh is rebuilt every 8 steps from a per-block indicator read back without blocking. New
leaves are filled by copying, by restriction (averaging the $`2^d`$ children, which conserves),
or by prolongation with MC-limited linear interpolation, which also conserves.

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

- Time advances in **cycles** of up to 8 substeps of the smallest step $`\Delta t_\text{min}`$.
- At the start of a cycle each block gets a bin $`k`$ from its own signal speed, on the GPU.
  A block in bin $`k`$ advances $`2^k\Delta t_\text{min}`$ at a time.
- Relaxation sweeps enforce three rules: face neighbors are at most one bin apart, a block
  next to a faster block steps as fast as it, and a block is never slower than a coarser
  neighbor.
- Across a face whose other side steps twice as often, the slower block takes the faster side's
  fluxes, recomputed from stored states: half when its own step starts and half at its
  midpoint. A faster block reads a slower neighbor that is part way through its step from that
  neighbor's half-step prediction. Nothing is stored, and conservation holds to round-off.
- With self-gravity the potential is solved on every block at the end of a cycle. In between it
  is solved only on blocks finishing a step, and carried forward elsewhere at the rate at which
  it last changed. Point masses move every substep.

The extra passes cost about 10% per block update without gravity and 50 to 70% with it (3D).
So the share of updates actually made is read back every few cycles, and the run falls back to
one timestep while that share is above 80% (65% with gravity). Measured against one timestep:
a 3D Sedov blast at 256 cells with 3 levels makes 2.2 times fewer block updates and runs 1.9
times faster; an isothermal collapse at 256 with 3 levels makes 2.3 times fewer and runs 1.6
times faster. With gravity, energy is conserved slightly less well, because a block that steps
less often has its half-step potential extrapolated further.

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

## 20. Nuclear burning surrogate (in development)

`tools/burn/` holds the offline part of a planned nuclear-burning module for the white dwarf
problems. It is not yet used by the simulator.

- **Reference network.** `burner.py` integrates the 13-isotope α-chain (aprox13,
  [Timmes 1999][Timmes1999]; the network module is generated with pynucastro,
  [Smith Clark et al. 2023][Smith2023]) at constant density with self-heating. At constant density
  the degenerate electrons' energy does not change, so

  ```math
  \frac{dT}{dt} = \frac{\epsilon_\text{nuc}}{c_v}, \qquad c_v = \frac{3k}{2\mu m_u} + \frac{4aT^3}{\rho}.
  ```

- **Surrogate.** `train.py` trains a small network that maps
  $`(\log\rho,\ \log T,\ \log\Delta t,\ X_i^{1/4})`$ to 13 logits. These are added to
  $`\log(X_i + 10^{-10})`$, and the new fractions are their softmax, so they are always positive
  and sum to one. The energy released is computed from the change in composition and the
  nuclear masses, never predicted, so energy and composition always agree.
- **DAgger.** `dagger.py` collects states that the surrogate itself visits when it is applied
  step after step, labels them with the reference network, and retrains on them
  ([Ross, Gordon & Bagnell 2011][Ross2011]). This targets the error that builds up when the
  surrogate is used in a rollout.

The species machinery of [§12](#12-passive-scalar-and-species) and the composition in
`src/units/composition.ts` are laid out so that a network can take over their state unchanged.

## 21. Validation

The validation suite (`src/validation/tests.ts`) has 36 tests. It runs on your own GPU at
[validate.html](https://guillochon.github.io/shaderous/validate.html) or with
`npm run test:gpu`. Reference solutions come from a float64 CPU implementation of the same
scheme (`src/reference/hydro.ts`) and an exact Riemann solver ([Toro 2009][Toro2009],
`src/reference/riemann_exact.ts`). The tests cover shock tubes along each axis and the diagonal,
a strong ($`10^5`$) shock tube, the Sedov-Taylor blast ([Taylor 1950][Taylor1950];
[Sedov 1959][Sedov1959]), smooth and uniform advection, bit-identical face fluxes, the Gresho
vortex, the well-balanced column, AMR versions of the shock problems, flux matching across
levels and time bins, the sphere potential, the Jeans instability ([Jeans 1902][Jeans1902]), a
polytrope, a cold white dwarf, the Evrard collapse ([Evrard 1988][Evrard1988]), tidal
disruption, cooling, viscosity, conduction, sinks, binary orbits, the equations of state, HDF5
round trips, and GPU against CPU reference. Each test's pass criterion is printed next to its
result.

---

## References

[Bate1995]: https://doi.org/10.1093/mnras/277.2.362
[Berger1984]: https://doi.org/10.1016/0021-9991(84)90073-1
[Berger1989]: https://doi.org/10.1016/0021-9991(89)90035-1
[Brandt1977]: https://doi.org/10.1090/S0025-5718-1977-0431719-X
[Bryan1995]: https://doi.org/10.1016/0010-4655(94)00191-4
[Cada2009]: https://doi.org/10.1016/j.jcp.2009.02.020
[Chandrasekhar1935]: https://doi.org/10.1093/mnras/95.3.207
[Davis1988]: https://doi.org/10.1137/0909030
[deValBorro2006]: https://doi.org/10.1111/j.1365-2966.2006.10488.x
[Evrard1988]: https://doi.org/10.1093/mnras/235.3.911
[Farris2014]: https://doi.org/10.1088/0004-637X/783/2/134
[Federrath2010]: https://doi.org/10.1088/0004-637X/713/1/269
[Fleischmann2020]: https://doi.org/10.1016/j.jcp.2019.109004
[Fryxell2000]: https://doi.org/10.1086/317361
[Gresho1990]: https://doi.org/10.1002/fld.1650110510
[Guillochon2009]: https://doi.org/10.1088/0004-637X/705/1/844
[Guillochon2013]: https://doi.org/10.1088/0004-637X/767/1/25
[Harten1983]: https://doi.org/10.1137/1025002
[Jeans1902]: https://doi.org/10.1098/rsta.1902.0012
[Kappeli2014]: https://doi.org/10.1016/j.jcp.2013.11.028
[Koren1993]: https://openalex.org/W2125064741
[Koyama2002]: https://doi.org/10.1086/338978
[Koyama2004]: https://doi.org/10.1086/382478
[Krumholz2004]: https://doi.org/10.1086/421935
[Lohner1987]: https://doi.org/10.1016/0045-7825(87)90098-3
[Plewa1999]: https://arxiv.org/abs/astro-ph/9807241
[Plummer1911]: https://doi.org/10.1093/mnras/71.5.460
[Quirk1994]: https://doi.org/10.1002/fld.1650180603
[Roe1986]: https://doi.org/10.1146/annurev.fl.18.010186.002005
[Ross2011]: https://arxiv.org/abs/1011.0686
[Saha1920]: https://doi.org/10.1080/14786441008636148
[Sedov1959]: https://ui.adsabs.harvard.edu/abs/1959sdmm.book.....S
[Smith2023]: https://doi.org/10.3847/1538-4357/acbaff
[Springel2005]: https://doi.org/10.1111/j.1365-2966.2005.09655.x
[Springel2010]: https://doi.org/10.1111/j.1365-2966.2009.15715.x
[Stone2020]: https://doi.org/10.3847/1538-4365/ab929b
[Taylor1950]: https://doi.org/10.1098/rspa.1950.0049
[Thornber2008]: https://doi.org/10.1016/j.jcp.2008.01.036
[Timmes1999]: https://doi.org/10.1086/313257
[Toro1994]: https://doi.org/10.1007/BF01414629
[Toro2009]: https://doi.org/10.1007/b79761
[Truelove1997]: https://doi.org/10.1086/310975
[vanLeer1974]: https://doi.org/10.1016/0021-9991(74)90019-9
[vanLeer1977]: https://doi.org/10.1016/0021-9991(77)90095-X
[vanLeer1979]: https://doi.org/10.1016/0021-9991(79)90145-1
[VazquezSemadeni2007]: https://doi.org/10.1086/510771
[Zingale2002]: https://doi.org/10.1086/342754

- Bate, M. R., Bonnell, I. A., & Price, N. M. (1995). Modelling accretion in protobinary systems. MNRAS 277, 362. [doi:10.1093/mnras/277.2.362](https://doi.org/10.1093/mnras/277.2.362)
- Berger, M. J., & Oliger, J. (1984). Adaptive Mesh Refinement for Hyperbolic Partial Differential Equations. Journal of Computational Physics 53, 484. [doi:10.1016/0021-9991(84)90073-1](https://doi.org/10.1016/0021-9991(84)90073-1)
- Berger, M. J., & Colella, P. (1989). Local Adaptive Mesh Refinement for Shock Hydrodynamics. Journal of Computational Physics 82, 64. [doi:10.1016/0021-9991(89)90035-1](https://doi.org/10.1016/0021-9991(89)90035-1)
- Brandt, A. (1977). Multi-level adaptive solutions to boundary-value problems. Mathematics of Computation 31, 333. [doi:10.1090/S0025-5718-1977-0431719-X](https://doi.org/10.1090/S0025-5718-1977-0431719-X)
- Bryan, G. L., et al. (1995). A piecewise parabolic method for cosmological hydrodynamics. Computer Physics Communications 89, 149. [doi:10.1016/0010-4655(94)00191-4](https://doi.org/10.1016/0010-4655(94)00191-4)
- Čada, M., & Torrilhon, M. (2009). Compact third-order limiter functions for finite volume methods. Journal of Computational Physics 228, 4118. [doi:10.1016/j.jcp.2009.02.020](https://doi.org/10.1016/j.jcp.2009.02.020)
- Chandrasekhar, S. (1935). The Highly Collapsed Configurations of a Stellar Mass (Second Paper). MNRAS 95, 207. [doi:10.1093/mnras/95.3.207](https://doi.org/10.1093/mnras/95.3.207)
- Davis, S. F. (1988). Simplified Second-Order Godunov-Type Methods. SIAM Journal on Scientific and Statistical Computing 9, 445. [doi:10.1137/0909030](https://doi.org/10.1137/0909030)
- de Val-Borro, M., et al. (2006). A comparative study of disc-planet interaction. MNRAS 370, 529. [doi:10.1111/j.1365-2966.2006.10488.x](https://doi.org/10.1111/j.1365-2966.2006.10488.x)
- Evrard, A. E. (1988). Beyond N-body: 3D cosmological gas dynamics. MNRAS 235, 911. [doi:10.1093/mnras/235.3.911](https://doi.org/10.1093/mnras/235.3.911)
- Farris, B. D., et al. (2014). Binary Black Hole Accretion from a Circumbinary Disk: Gas Dynamics inside the Central Cavity. ApJ 783, 134. [doi:10.1088/0004-637X/783/2/134](https://doi.org/10.1088/0004-637X/783/2/134)
- Federrath, C., et al. (2010). Modeling Collapse and Accretion in Turbulent Gas Clouds: Implementation and Comparison of Sink Particles in AMR and SPH. ApJ 713, 269. [doi:10.1088/0004-637X/713/1/269](https://doi.org/10.1088/0004-637X/713/1/269)
- Fleischmann, N., et al. (2020). A low dissipation method to cure the grid-aligned shock instability. Journal of Computational Physics 401, 109004. [doi:10.1016/j.jcp.2019.109004](https://doi.org/10.1016/j.jcp.2019.109004)
- Fryxell, B., et al. (2000). FLASH: An Adaptive Mesh Hydrodynamics Code for Modeling Astrophysical Thermonuclear Flashes. ApJS 131, 273. [doi:10.1086/317361](https://doi.org/10.1086/317361)
- Gresho, P. M., & Chan, S. T. (1990). On the theory of semi-implicit projection methods for viscous incompressible flow and its implementation via a finite element method that also introduces a nearly consistent mass matrix. II - Implementation. International Journal for Numerical Methods in Fluids 11, 621. [doi:10.1002/fld.1650110510](https://doi.org/10.1002/fld.1650110510)
- Guillochon, J., et al. (2009). Three-dimensional Simulations of Tidally Disrupted Solar-type Stars and the Observational Signatures of Shock Breakout. ApJ 705, 844. [doi:10.1088/0004-637X/705/1/844](https://doi.org/10.1088/0004-637X/705/1/844)
- Guillochon, J., & Ramirez-Ruiz, E. (2013). Hydrodynamical Simulations to Determine the Feeding Rate of Black Holes by the Tidal Disruption of Stars: The Importance of the Impact Parameter and Stellar Structure. ApJ 767, 25. [doi:10.1088/0004-637X/767/1/25](https://doi.org/10.1088/0004-637X/767/1/25)
- Harten, A., Lax, P. D., & van Leer, B. (1983). On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws. SIAM Review 25, 35. [doi:10.1137/1025002](https://doi.org/10.1137/1025002)
- Jeans, J. H. (1902). The Stability of a Spherical Nebula. Philosophical Transactions of the Royal Society of London Series A 199, 1. [doi:10.1098/rsta.1902.0012](https://doi.org/10.1098/rsta.1902.0012)
- Käppeli, R., & Mishra, S. (2014). Well-balanced schemes for the Euler equations with gravitation. Journal of Computational Physics 259, 199. [doi:10.1016/j.jcp.2013.11.028](https://doi.org/10.1016/j.jcp.2013.11.028)
- Koren, B. (1993). A robust upwind discretization method for advection, diffusion and source terms. In Numerical Methods for Advection-Diffusion Problems, Notes on Numerical Fluid Mechanics 45, 117. Vieweg. [OpenAlex](https://openalex.org/W2125064741)
- Koyama, H., & Inutsuka, S. (2002). An Origin of Supersonic Motions in Interstellar Clouds. ApJL 564, L97. [doi:10.1086/338978](https://doi.org/10.1086/338978)
- Koyama, H., & Inutsuka, S. (2004). The Field Condition: A New Constraint on Spatial Resolution in Simulations of the Nonlinear Development of Thermal Instability. ApJL 602, L25. [doi:10.1086/382478](https://doi.org/10.1086/382478)
- Krumholz, M. R., McKee, C. F., & Klein, R. I. (2004). Embedding Lagrangian Sink Particles in Eulerian Grids. ApJ 611, 399. [doi:10.1086/421935](https://doi.org/10.1086/421935)
- Löhner, R. (1987). An adaptive finite element scheme for transient problems in CFD. Computer Methods in Applied Mechanics and Engineering 61, 323. [doi:10.1016/0045-7825(87)90098-3](https://doi.org/10.1016/0045-7825(87)90098-3)
- Plewa, T., & Müller, E. (1999). The consistent multi-fluid advection method. A&A 342, 179. [arXiv:astro-ph/9807241](https://arxiv.org/abs/astro-ph/9807241)
- Plummer, H. C. (1911). On the problem of distribution in globular star clusters. MNRAS 71, 460. [doi:10.1093/mnras/71.5.460](https://doi.org/10.1093/mnras/71.5.460)
- Quirk, J. J. (1994). A contribution to the great Riemann solver debate. International Journal for Numerical Methods in Fluids 18, 555. [doi:10.1002/fld.1650180603](https://doi.org/10.1002/fld.1650180603)
- Roe, P. L. (1986). Characteristic-based schemes for the euler equations. Annual Review of Fluid Mechanics 18, 337. [doi:10.1146/annurev.fl.18.010186.002005](https://doi.org/10.1146/annurev.fl.18.010186.002005)
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
- Saha, M. N. (1920). Ionization in the Solar Chromosphere. Philosophical Magazine 40, 472. [doi:10.1080/14786441008636148](https://doi.org/10.1080/14786441008636148)
- Sedov, L. I. (1959). Similarity and Dimensional Methods in Mechanics. Academic Press. [link](https://ui.adsabs.harvard.edu/abs/1959sdmm.book.....S)
- Smith Clark, A. I., et al. (2023). pynucastro: A Python library for nuclear astrophysics. ApJ. [doi:10.3847/1538-4357/acbaff](https://doi.org/10.3847/1538-4357/acbaff)
- Springel, V. (2005). The cosmological simulation code GADGET-2. MNRAS 364, 1105. [doi:10.1111/j.1365-2966.2005.09655.x](https://doi.org/10.1111/j.1365-2966.2005.09655.x)
- Springel, V. (2010). E pur si muove: Galilean-invariant cosmological hydrodynamical simulations on a moving mesh. MNRAS 401, 791. [doi:10.1111/j.1365-2966.2009.15715.x](https://doi.org/10.1111/j.1365-2966.2009.15715.x)
- Stone, J. M., et al. (2020). The Athena++ Adaptive Mesh Refinement Framework: Design and Magnetohydrodynamic Solvers. ApJS 249, 4. [doi:10.3847/1538-4365/ab929b](https://doi.org/10.3847/1538-4365/ab929b)
- Taylor, G. (1950). The Formation of a Blast Wave by a Very Intense Explosion. I. Theoretical Discussion. Proceedings of the Royal Society of London Series A 201, 159. [doi:10.1098/rspa.1950.0049](https://doi.org/10.1098/rspa.1950.0049)
- Thornber, B., et al. (2008). An improved reconstruction method for compressible flows with low Mach number features. Journal of Computational Physics 227, 4873. [doi:10.1016/j.jcp.2008.01.036](https://doi.org/10.1016/j.jcp.2008.01.036)
- Timmes, F. X. (1999). Integration of nuclear reaction networks for stellar hydrodynamics. ApJS 124, 241. [doi:10.1086/313257](https://doi.org/10.1086/313257)
- Toro, E. F., Spruce, M., & Speares, W. (1994). Restoration of the contact surface in the HLL-Riemann solver. Shock Waves 4, 25. [doi:10.1007/BF01414629](https://doi.org/10.1007/BF01414629)
- Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, 3rd ed. Springer. [doi:10.1007/b79761](https://doi.org/10.1007/b79761)
- Truelove, J. K., et al. (1997). The Jeans Condition: A New Constraint on Spatial Resolution in Simulations of Isothermal Self-gravitational Hydrodynamics. ApJL 489, L179. [doi:10.1086/310975](https://doi.org/10.1086/310975)
- van Leer, B. (1974). Towards the Ultimate Conservation Difference Scheme. II. Monotonicity and Conservation Combined in a Second-Order Scheme. Journal of Computational Physics 14, 361. [doi:10.1016/0021-9991(74)90019-9](https://doi.org/10.1016/0021-9991(74)90019-9)
- van Leer, B. (1977). Towards the Ultimate Conservative Difference Scheme. IV. A New Approach to Numerical Convection. Journal of Computational Physics 23, 276. [doi:10.1016/0021-9991(77)90095-X](https://doi.org/10.1016/0021-9991(77)90095-X)
- van Leer, B. (1979). Towards the Ultimate Conservative Difference Scheme. V. A Second-Order Sequel to Godunov's Method. Journal of Computational Physics 32, 101. [doi:10.1016/0021-9991(79)90145-1](https://doi.org/10.1016/0021-9991(79)90145-1)
- Vázquez-Semadeni, E., et al. (2007). Molecular Cloud Evolution. II. From Cloud Formation to the Early Stages of Star Formation in Decaying Conditions. ApJ 657, 870. [doi:10.1086/510771](https://doi.org/10.1086/510771)
- Zingale, M., et al. (2002). Mapping Initial Hydrostatic Models in Godunov Codes. ApJS 143, 539. [doi:10.1086/342754](https://doi.org/10.1086/342754)
