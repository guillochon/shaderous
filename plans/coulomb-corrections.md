# Plan: Coulomb corrections for white dwarf matter

Status: built 2026-11-16 (steps 1 to 7). What building it found:

- **The white dwarf builder had to change too**, which this plan had missed. It built stars on
  the electrons' cold pressure alone. Under the corrected equation of state that pressure is
  above the cold curve, and the difference was read as heat: 2e8 K at 1e7 g/cm^3, 9e8 K at
  1e9. The builder, its enthalpy inverse and the GPU settling
  ([src/problems/white_dwarf.ts](../src/problems/white_dwarf.ts),
  [src/amr/settle.wgsl](../src/amr/settle.wgsl)) now use the same cold curve as the equation
  of state, and a unit test holds the two equal.
- **Stars are smaller**: 8586 to 8127 km at 0.6 M_sun, 5428 to 5094 km at 1 M_sun, 2771 to
  2290 km at 1.3 M_sun (the term lowers the Chandrasekhar mass by some 3%, and a star near it
  is very sensitive).
- **Presets**, at their default grids:
  - `wd-collision-burning`: the same collision, 0.5 time units (8%) earlier because the smaller
    stars start closer, and about 6% more energy released.
  - `sub-chandra`: with burning switched off the star sits as still as before (kinetic energy
    1e-4 of the binding energy either way), although its helium shell is built on the
    carbon-oxygen cold curve. With burning, a 1e9 K spot lights the denser shell where it did
    not before; the default 3e9 K spot detonates either way.
  - `detonation-co`: unchanged but for 3% less internal energy (the cold term).
- **Newton's starting point** needed the correction section 5 allowed for: one analytic step
  for the Coulomb heat at the tables' guess. 3.3 iterations on average, 5 at most (8 without
  the step, against a cap of 10 in the shader).
- **A star cache** in the setup builder was keyed on the electrons' scales alone and handed a
  star built without the term to a setup with it; the preset-equivalence tests caught it.
- The fit is Yakovlev & Shalybkov's as quoted in Helmholtz's source; the coefficients were
  checked by their continuity at Gamma = 1 and the Debye-Hueckel and ion-sphere limits, not
  against the paper.

## 1. Goal

Add the ions' Coulomb energy to the tabulated white dwarf equation of state
([src/eos/helmholtz.wgsl](../src/eos/helmholtz.wgsl), mirror in
[src/eos/helmholtz.ts](../src/eos/helmholtz.ts)), as FLASH's and Castro's Helmholtz do, from one
free energy so that pressure and energy stay consistent. On by default for every setup that
uses that equation of state.

What it changes physically:

- The cold pressure falls by 0.0046 Z^(2/3) of the electrons' where they are relativistic
  (1.5% for carbon, 1.8% for oxygen, 4% for nickel) and by more where they are not, as
  (rho_z / rho)^(1/3): 10% for carbon at 1e4 g/cm^3. A white dwarf of a given mass is a little
  smaller and denser.
- Strongly coupled ions hold more heat than an ideal gas (up to about 4 k per ion in place of
  1.5 k near crystallization), so cold dense matter is cooler for the same energy.

### Non-goals

- Crystallization: the liquid fit is used at every coupling, as in Helmholtz.
- Mixtures beyond the mean charge: Gamma is built from Zbar, as in Helmholtz, not from the
  linear mixing rule's mean of Z^(5/3).
- The `degenerate` equation of state (the `white-dwarf` and `wd-collision` presets). Its
  thermal part is solved as a quartic, which a Coulomb term would break; section 7.
- Retraining the burning network (section 6).

## 2. The free energy

The one-component plasma's excess free energy per ion, in units of kT, as a function of the
coupling Gamma = Zbar^2 e^2 / (a k T), with a the ion-sphere radius (4 pi a^3 n_i / 3 = 1).
These are the fits Helmholtz uses (Yakovlev & Shalybkov 1989), written as the free energy they
are the derivatives of:

| Range | f(Gamma) = F / (N_i k T) | u(Gamma) = Gamma f' = E / (N_i k T) |
|---|---|---|
| Gamma >= 1 | a1 G + 4 b1 G^(1/4) - 4 c1 G^(-1/4) + d1 ln G + K | a1 G + b1 G^(1/4) + c1 G^(-1/4) + d1 |
| Gamma < 1 | -2 c2 G^(3/2) + (a2 / b2) G^b2 | -3 c2 G^(3/2) + a2 G^b2 |

a1 = -0.898004, b1 = 0.96786, c1 = 0.220703, d1 = -0.86097, a2 = 0.29561, b2 = 1.9885,
c2 = 0.288675 (the Debye-Hueckel 1 / (2 sqrt 3)). K = -2.51928 makes f continuous at
Gamma = 1; u and u' already are, to four digits.

Since Gamma goes as rho^(1/3) / T, the energy per ion is k T u and the pressure is a third of
the energy density. Both follow from f, so the first law holds exactly.

## 3. Fitting it to the code's split

The equation of state is split into a zero-temperature part (functions of density alone) and a
thermal part that must be positive and grow with T, because the temperature is found from the
thermal energy by Newton's method. The Coulomb term splits the same way:

- **Cold**: a1 Gamma k T does not depend on T (it is a1 Zbar^2 e^2 / a per ion, the Madelung
  energy of the ion spheres). It joins the zero-temperature electrons. It is negative.
- **Thermal**: the rest, f_th = f - a1 Gamma and g = u - a1 Gamma. Both are positive at every
  Gamma, and so is the heat capacity g - Gamma g'. The solve for T stays well posed.

## 4. Low density

With the cold term as it stands, the cold pressure of carbon is negative below
rho_z = 0.18 Zbar^2 / Ye = 13 g/cm^3 (cold matter has a density at zero pressure), and its
derivative below half that. Fully ionized ion spheres in a uniform electron gas are the wrong
model there anyway: cold carbon is not pressure-ionized until some 1e3 to 1e4 g/cm^3. But the
code has gas at such densities (atmospheres, ambient gas), and a hydrostatic builder that
expects pressure to rise with density.

Helmholtz drops the Coulomb terms wherever the total pressure or energy would be negative,
which is a jump. Here the whole Coulomb free energy is faded instead, by a function of density
alone:

    F_C -> m w(rho) F_C,    w = r / (1 + r),   r = rho / rho_0,   rho_0 = 64 rho_z

with m the multiplier of section 5. Because the fade is applied to the free energy, pressure
and energy stay consistent: the energy takes the factor w, and the pressure gains the term
from w's slope,

    e = w (n_i k T / rho) u,        p / rho = w (n_i k T / rho) [u / 3 + (1 - w) f]

For C/O, rho_0 is 1.1e3 g/cm^3; for helium 90; for nickel 1.8e4. Above 100 rho_0 the fade is
within 1% of none. With the factor 64 the cold pressure is reduced by a third at most (near
rho_0 / 2) and rises with density everywhere, for every Zbar from 1 to 28: a unit test checks
that.

## 5. Design

- **`EosParams.helmCoulomb`**: a multiplier, as FLASH's `coulombMult`; 1 by default, 0 gives
  the equation of state as it was. An option `physics.eosParams.helmCoulomb` ("Coulomb
  corrections", 0 to 1, link key `coulomb`), shown for the Helmholtz equation of state.
- **Uniform slot**: `P.eos2.w`, which held the reference composition's ions per atomic mass
  unit. That number is 1 / mu - Ye, and mu and Ye are already in the block, so the shader
  derives it and the slot is free.
- **Mirror**: `helmCoulomb(rho, lnT, ye, ia, m)` returns the cold and thermal pieces with the
  derivatives the solver and the sound speed need; `helmThermo` and `helmSolve` take the
  multiplier as an optional last argument (0 when absent, so the table's own tests still
  compare against the bare electron gas).
- **Shader**: the same functions, in cgs like the rest of the module; `helm_cold(rho)` gives
  the cold pressure, energy and dp/drho with the Coulomb part, in place of `deg_pe` and
  `deg_c2` where the module's `eos_*` functions use them.
- **Newton's starting point**: the inverse tables were made without the Coulomb heat, so the
  first guess is up to a factor of two or so high in cold, strongly coupled matter. The
  iteration converges from there (the step is capped at one e-fold); the test of iteration
  counts is extended to this case and its bound set by what is found. Regenerating the
  inverse tables with the Coulomb term is the fix if the count matters.

## 6. What does not change

- The table (`helmholtz_table.bin`) and `make_table.py`.
- `tools/eos/helm.py`, which the burner's reference burns and training rollouts use for the
  heat capacity. The burning network was therefore trained on self-heating without the
  Coulomb heat. At burning temperatures Gamma is of order one (0.96 for C/O at 1e7 g/cm^3 and
  1e9 K), where the Coulomb heat capacity is 0.14 k per ion against 1.5 k for the ideal ions
  and more for the electrons: a few percent. Noted, not fixed.
- Saved `.athdf` files: they hold pressure, so a file written before loads with the same
  pressures and slightly different temperatures.

## 7. Steps

1. Mirror: `helmCoulomb`, the multiplier through `helmThermo`, `helmSolve` and
   `helmholtzEos`; `helmCoulomb` in `EosParams`, default 1.
2. Unit tests of the mirror:
   - pressure and energy against finite differences of the free energy (the first law);
   - continuity at Gamma = 1;
   - the limits: the cold reduction 0.0046 Z^(2/3) in relativistic matter, Debye-Hueckel at
     small Gamma;
   - the cold pressure and its slope positive for Zbar 1 to 28 over 1e-6 to 1e11 g/cm^3;
   - the thermal energy and pressure rising with T;
   - round trips of energy, pressure and temperature with the term on;
   - multiplier 0 reproduces today's numbers.
3. Shader, and the uniform slot.
4. GPU: the equation-of-state validation test compares shader and mirror with the term on,
   and once with it off.
5. Option, link key, codec; `describe()` lists it through the option table.
6. Run the white dwarf presets (`detonation-co`, `sub-chandra`, `wd-collision-burning`) and
   the validation tests that use them; compare a white dwarf's radius with and without.
7. Docs: `numerics.md`, `tools/eos/NOTES.md`, the comment at the head of the shader.

Later, if wanted: the cold term alone in the `degenerate` equation of state (a consistent
cold curve; its ideal-gas thermal part untouched), which would give the two non-burning white
dwarf presets the smaller radius without touching their quartic.
