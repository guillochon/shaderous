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
24. [References](#references)

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

Two more return pairs, so that an equation of state which solves for its temperature does so
once for both: `eos_prim(rho, eint)` gives a cell's pressure and sound speed, and
`eos_face(rho, p)` a face state's sound speed and internal energy.

The equation of state is chosen when the shaders are assembled, and each module has a
float64 TypeScript mirror for testing (`tests/unit/eos*.test.ts`).

| Equation of state | Pressure | Notes |
|---|---|---|
| **Ideal gas** (`ideal.wgsl`) | $`p = (\gamma - 1)\rho e`$ | $`c_s^2 = \gamma p/\rho`$ |
| **Barotropic** (`barotropic.wgsl`) | $`p = K\rho^{\Gamma}`$ (isothermal for $`\Gamma = 1`$) | The energy equation is not used; $`\rho e`$ follows the density; HLL fluxes |
| **Gas + radiation** (`gas_radiation.wgsl`) | $`p = \dfrac{\rho k T}{\mu m_u} + \tfrac13 a T^4`$ | $`T`$ is the root of $`\rho e = \dfrac{\rho kT}{(\gamma - 1)\mu m_u} + aT^4`$, from 8 Newton steps started above the root (`quartic.wgsl`) |
| **Degenerate** (`degenerate.wgsl`) | $`p = p_\text{deg}(\rho) + p_\text{ion} + p_\text{rad}`$ | Zero-temperature electrons at any degree of relativity ([Chandrasekhar 1935][Chandrasekhar1935]), plus ideal ions and radiation |
| **White dwarf matter** (`helmholtz.wgsl`) | $`p = p_\text{deg}(\rho) + p_{e,\text{th}} + p_\text{ion} + p_\text{rad} + p_\text{C}`$ | The electron-positron gas at any degeneracy and temperature from a table of its free energy, in the manner of Helmholtz ([Timmes & Swesty 2000][TimmesSwesty2000]); ideal ions, radiation, and the ions' Coulomb energy |
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
- Warm-started from the previous potential. The default is 2 V-cycles per step (3 on the
  adaptive mesh, where later cycles run only while they still reduce the residual).
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

   Summed over the cells, this work is the negative of the change in $`\sum\rho\Phi`$ that the
   same mass fluxes cause, so gas plus gravitational energy is conserved to round-off in a fixed
   potential, and to the accuracy of the solve and of $`\Phi^{n+1/2}`$ with self-gravity. It can overheat low-density cells next to a sharp density contrast.
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

The half-step potential $`\Phi^{n+1/2}`$ is extrapolated from the last two solves.

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

Settling rewrites a star as cold gas. A problem that lights a star (the sub-Chandrasekhar
supernova) puts its heat in afterward (`Problem.heat`): the internal energy that each cell of a
hot spot should have, written into the settled state on the uniform grid and the adaptive mesh.

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
or by prolongation with MC-limited linear interpolation, which also conserves.

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

- Time advances in **cycles** of up to 8 substeps of the smallest step $`\Delta t_\text{min}`$.
- At the start of a cycle each block gets a bin $`k`$ from its own signal speed, on the GPU.
  A block in bin $`k`$ advances $`2^k\Delta t_\text{min}`$ at a time.
- A first sweep lowers every block to the bin of its fastest face neighbor (once, so each fast
  region widens by one block). Further sweeps enforce two rules: face neighbors are at most
  one bin apart, and a block is never slower than a coarser neighbor.
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
With a magnetic field (the magnetized accretion disk) the zone acts on the gas alone: B^2 / 2 is
taken out of E before it and put back after, and the field is left to the induction equation.

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
  tabulated electron-positron equation of state as in the simulator.
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
  Off by default.
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
- **Magnetized accretion disk** (`problems/mri_disk.ts`, "Point masses" group). A globally
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
  the settled one. The faces are the line integrals of A as for every other field, so coarse and
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
- **Compile time.** GPU compilers inline every function at every call, so a lookup called at
  several places, or in a loop they unroll, is compiled that many times. The block grid's
  assembly took 4 s to compile in Chrome (far more in Firefox), with the face lookup inlined
  about a hundred times. The passes now make such calls from one loop whose bound the compiler
  cannot see through (`loop_count`, `amr/grid.wgsl`), with the same arithmetic in the same
  order: assembly 0.45 s, update 0.26 s, re-meshing 0.9 s. The uniform grid's cell-centered
  field reads its six faces directly instead of through the boundary mapping (edge fields
  0.45 s to 0.14 s). A magnetized run builds no hydro update or prediction, which it never
  runs (the hydro update was the slowest compile of a load). `tools/compile_times.mjs` times
  each pipeline of a setup, alone, on a cold shader cache.
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

Tests (`validation/frames.ts`): uniform flow on an f-plane turns at $`2\Omega`$ with its phase
right to $`3\times10^{-7}`$ rad and its kinetic energy to $`10^{-6}`$; isothermal gas held by
the centrifugal force stays below Mach $`10^{-5}`$ for more than a sound crossing (Mach 0.02
without the well-balanced reconstruction); a blob carried by gas moving straight in the
inertial frame lands within a sixth of a cell of where it should after a quarter turn; a binary
at rest in its corotating frame keeps its separation to $`10^{-10}`$ for two orbits; and a
uniform field along the axis is left exactly as it was.

### Mirror symmetry

A setup symmetric under mirrors through the origin (`domain.symmetric`) can be run on the part
above its planes of symmetry (`domain.mirror`, link key `sym`): an octant in 3D or a quadrant in
2D, and on any one or two axes. A mirrored axis's domain is $`[0, L]`$ instead of
$`[-L/2, L/2]`$, with $`L`$ the box side, so cells stay cubic and an octant at $`n^3`$ has the
cell size of the whole box at $`(2n)^3`$, for an eighth of the cells. (Mirroring some axes but
not others gains nothing on a cubic grid: the unmirrored axes still span the whole object.)

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
`mirror` view setting turns it off. Field lines are drawn in the computed part only.

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
and the timestep use it unchanged.

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
ideal gas would be off by a factor 2.6. The tabulated stellar equation of state is refused.

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
The `zeldovich-pancake` preset (Cosmology group) runs the pancake of
[Zel'dovich (1970)](https://ui.adsabs.harvard.edu/abs/1970A%26A.....5...84Z) through its
caustic at $`z = 1`$.

Tests: the scale factor on the GPU against float64 (Einstein-de Sitter and flat
$`\Lambda`$CDM, $`3\times10^{-7}`$); uniform gas whose velocity falls as $`1/a`$ and comoving
pressure as $`a^{3 - 3\gamma}`$ ($`5\times10^{-7}`$, uniform grid and blocks); a 0.2% wave that
grows as $`a`$ (0.8%); a traveling sound wave whose phase, over 25 radians from $`a = 0.1`$ to
0.3, follows $`\int k c_s/a\,dt`$ with the proper sound speed falling as $`1/a`$ (0.07%, its
comoving wavelength fixed); gas and radiation expanding (above); and the pancake against its
exact solution at half its caustic time (density 1%, velocity 0.3%; 3.6% in density on blocks,
whose coarse cells are twice as large).

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
remove it and the shear speed's limit on the timestep; it is not implemented yet. The Coriolis
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
through the box are conserved. Resistivity and viscosity are refused with a field.

The first version runs on a uniform grid, periodic along $`y`$ (and $`z`$ unless stratified);
blocks (whose neighbors across the boundary change with time), self-gravity and diffusion are
refused. The `shearing-wave` preset swings a leading density wave round to trailing; `mri-box`
grows the magnetorotational instability from noise in a vertical field. Tests: the background
over ten orbits (within $`10^{-3}`$ of the shear speed), the blob, epicycles at
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
  to rounding. The well-balanced reconstruction acts along $`z`$ alone, on the uniform grid
  and on blocks. Radially it made a torus held up by its rotation around a point mass pile up
  at its inner edge: at $`256^2`$ a torus with $`\ell\propto R^{1/4}`$ (stable to axisymmetric
  displacements) reached 2.2 times its peak density in two orbits, while the same torus in 3D
  at the same cell size, well balanced along $`x`$ and $`y`$, stayed within 2.5%, and with the
  radial balance off the axisymmetric one stayed within 1%. What was found:
  - The balance ignores rotation. For gas held up by its rotation the deviation from
    hydrostatic balance is the centrifugal term $`\rho v_\phi^2\Delta R/R`$, ten times the
    pressure or more near a torus's edges, where the limiter clips it and the unlimited
    hydrostatic step that goes back in gives the faces a pressure slope rotation does not hold
    up. (In 3D the rotation is carried by the advection of the swirl, which the reconstruction
    does not see.) Measuring against the effective potential $`\Phi - \int v_\phi^2/R\,dR`$
    needs it at two orders: between two cells, a step density weighted so that
    $`\tfrac12(\rho_a + \rho_b)`$ times it is the trapezoid rule on $`\rho v_\phi^2/R`$; and
    from a cell's center to its own face, its own $`v_\phi^2/R`$, so that its two faces differ by
    exactly what its kick and its geometric source cancel (with the pair's step there instead,
    they differ at first order where the density changes). With both, a torus with
    $`\ell\propto R^{1/4}`$ held for an orbit at $`256^2`$ (peak +0.3%, from +48%).
  - That is not all of it, and the rest is not the reconstruction's. With the radial balance
    off, the same torus piles up at its inner edge too whenever the scheme is less dissipative:
    with the minmod limiter (peak +129% in one orbit at $`64^2`$, against +1% with Koren's), or
    with the shock detector's HLL fallback switched off (+183%). The same torus in 3D at the same
    cell size, with the detector off, stays within 7%. So $`(R, z)`$ has an instability of its
    own at a rotating torus's inner edge, which dissipation holds down and the radial balance,
    taking dissipation away, lets through: the thin gas spilled inside the edge, carrying the
    torus's angular momentum, is flung outward (correctly) and gathers in the edge cell. It is
    not the predictor's geometric terms (without them it is worse). The next suspects are the
    geometric source's use of the cell's half-step $`v_\phi`$ where the face fluxes carry the
    angular momentum, and the cell-center values standing for ring averages.
  - So the radial balance stays off. With it on and both parts of the effective potential, the
    constant-$`\ell`$ torus of the tests rose 17% in three orbits; along $`z`$ alone it keeps
    within 2.6% for five. It still holds a stratified column along
  the axis as it does in Cartesian coordinates.
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
Evrard collapse in $`(R, z)`$ keeps its energy to 0.2% through bounce. Self-gravity on blocks
and settled stars are not done yet.

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
its peak density to 5% (it oscillates by a few percent, as the 3D torus's does) for five
orbits. Its faces are open for outflow only: open both ways, they let the mass pull gas in
from outside the box, which then piled up at the mass without bound.



## 23. Validation

The validation suite (`src/validation/tests.ts`) has 55 tests. It runs on your own GPU at
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
round trips, GPU against CPU reference, and the MHD tests of §21. Each test's pass criterion is printed next to its
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
[Hawley2000]: https://doi.org/10.1086/308180
[Hawley2011]: https://doi.org/10.1088/0004-637X/738/1/84
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
[EP88]: https://doi.org/10.1016/0045-7930(88)90013-8
[Federrath2010]: https://doi.org/10.1051/0004-6361/200912437
[Mallet2026]: https://doi.org/10.5281/zenodo.21987739
[Thurgood2018]: https://doi.org/10.1063/1.5035489
[Tsiklauri2007]: https://doi.org/10.1063/1.2800854
[Ghia1982]: https://doi.org/10.1016/0021-9991(82)90058-4
[Koyama2004]: https://doi.org/10.1086/382478
[Krumholz2004]: https://doi.org/10.1086/421935
[Kushnir2013]: https://doi.org/10.1088/2041-8205/778/2/L37
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
- Hamada, T., & Salpeter, E. E. (1961). Models for Zero-Temperature Stars. ApJ 134, 683. [doi:10.1086/147195](https://doi.org/10.1086/147195)
- Harten, A., Lax, P. D., & van Leer, B. (1983). On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws. SIAM Review 25, 35. [doi:10.1137/1025002](https://doi.org/10.1137/1025002)
- Hawley, J. F. (2000). Global Magnetohydrodynamical Simulations of Accretion Tori. ApJ 528, 462. [doi:10.1086/308180](https://doi.org/10.1086/308180)
- Hawley, J. F., Guan, X., & Krolik, J. H. (2011). Assessing Quantitative Results in Accretion Simulations: From Local to Global. ApJ 738, 84. [doi:10.1088/0004-637X/738/1/84](https://doi.org/10.1088/0004-637X/738/1/84)
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
- Timmes, F. X., & Swesty, F. D. (2000). The Accuracy, Consistency, and Speed of an Electron-Positron Equation of State Based on Table Interpolation of the Helmholtz Free Energy. ApJS 126, 501. [doi:10.1086/313304](https://doi.org/10.1086/313304)
- Timmes, F. X., Hoffman, R. D., & Woosley, S. E. (2000). An inexpensive nuclear energy generation network for stellar hydrodynamics. ApJS 129, 377. [doi:10.1086/313407](https://doi.org/10.1086/313407)
- Toro, E. F., Spruce, M., & Speares, W. (1994). Restoration of the contact surface in the HLL-Riemann solver. Shock Waves 4, 25. [doi:10.1007/BF01414629](https://doi.org/10.1007/BF01414629)
- Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, 3rd ed. Springer. [doi:10.1007/b79761](https://doi.org/10.1007/b79761)
- Truelove, J. K., et al. (1997). The Jeans Condition: A New Constraint on Spatial Resolution in Simulations of Isothermal Self-gravitational Hydrodynamics. ApJL 489, L179. [doi:10.1086/310975](https://doi.org/10.1086/310975)
- van Leer, B. (1974). Towards the Ultimate Conservation Difference Scheme. II. Monotonicity and Conservation Combined in a Second-Order Scheme. Journal of Computational Physics 14, 361. [doi:10.1016/0021-9991(74)90019-9](https://doi.org/10.1016/0021-9991(74)90019-9)
- van Leer, B. (1977). Towards the Ultimate Conservative Difference Scheme. IV. A New Approach to Numerical Convection. Journal of Computational Physics 23, 276. [doi:10.1016/0021-9991(77)90095-X](https://doi.org/10.1016/0021-9991(77)90095-X)
- van Leer, B. (1979). Towards the Ultimate Conservative Difference Scheme. V. A Second-Order Sequel to Godunov's Method. Journal of Computational Physics 32, 101. [doi:10.1016/0021-9991(79)90145-1](https://doi.org/10.1016/0021-9991(79)90145-1)
- Vázquez-Semadeni, E., et al. (2007). Molecular Cloud Evolution. II. From Cloud Formation to the Early Stages of Star Formation in Decaying Conditions. ApJ 657, 870. [doi:10.1086/510771](https://doi.org/10.1086/510771)
- Yakovlev, D. G., & Shalybkov, D. A. (1989). Degenerate cores of white dwarfs and envelopes of neutron stars: thermodynamics and plasma screening in thermonuclear reactions. Astrophysics and Space Physics Reviews 7, 311.
- Zingale, M., et al. (2002). Mapping Initial Hydrostatic Models in Godunov Codes. ApJS 143, 539. [doi:10.1086/342754](https://doi.org/10.1086/342754)
