# Plan: exact linear-wave tests for comoving coordinates

Status (2026-11-21): not begun. Nothing below has been run; the variable mapping in section 2
was derived by hand from the two sets of equations and should be confirmed by stage 1's first
test before anything else leans on it.

Goal: hold the comoving solver to exact solutions for the amplitude and the phase of linear
waves, with pressure and self-gravity together, for any gamma; and have the same tests ready,
written down, for magnetic fields and viscosity when comoving runs allow them.

Source: [Berlok (2022)](https://doi.org/10.1093/mnras/stac1882), *Hydromagnetic waves in an
expanding universe: cosmological MHD code tests using analytic solutions*, MNRAS 515, 3492,
with its Python reference code at <https://github.com/tberlok/comoving_mhd_waves>
(analytic solutions, a scipy ODE integration of the same linear system, and pytest tests that
hold one against the other to 1e-5).

| Stage | What it adds | Needs new physics? |
|---|---|---|
| 1 | A float64 reference: the linear ODE system, integrated in TypeScript | no |
| 2 | Standing compressible waves, pressure and gravity, gamma = 5/3 and 4/3, against stage 1 | no |
| 3 | The gamma = 4/3 closed form and its critical case; traveling waves | no |
| 4 | Comoving Alfvén and magnetosonic waves | comoving MHD |
| 5 | The viscous Alfvén wave | comoving viscosity |

Stages 1 to 3 test what exists today. Stages 4 and 5 are acceptance tests for lifting two of
the refusals in `cosmologyUnsupported` (`src/sim.ts`): "magnetic fields are not available in
comoving coordinates yet" and "viscosity and conduction are not available in comoving
coordinates".

---

## 1. What the code has

- Comoving hydrodynamics in the Enzo form (`docs/numerics.md`, section 22, "Comoving
  coordinates"): comoving positions, comoving density and pressure, proper peculiar velocity,
  time in 1/H0, mean comoving density 1, G = 3 Omega_m / (8 pi). The background in float64
  on the CPU is `src/cosmo.ts` (`hubble`, `timeOf`, `scaleFactor`, `edsScaleFactor`).
- Tests in `src/validation/cosmo.ts` (`COSMO_TESTS`, registered in `validation/frames.ts`):
  - `cosmo-wave`: a traveling sound wave, no gravity, gamma = 5/3, a from 0.1 to 0.3, 64 cells
    a wavelength in 2D. It checks the phase against the WKB integral of k c_s / a dt (0.07%)
    and that the wave stays in one mode. It does not check the amplitude.
  - `cosmo-growth`: a cold wave on its growing mode with self-gravity, 32^3, grows as a
    (0.8%). Pressure plays no part.
  - Nothing tests a wave where pressure and gravity are comparable, nor the amplitude's decay,
    nor the turn from oscillation to growth at the Jeans scale.
- Periodic self-gravity is 3D only (the multigrid solver), so every test with gravity is a
  cube of n^3 cells; a plane wave along x wastes the other two directions but costs little at
  n = 32 or 64.

## 2. The equations, and how Berlok's variables map onto ours

Linearizing our equations about uniform gas (comoving density 1 + delta, proper peculiar
velocity v, comoving pressure p0 (1 + gamma delta) for an adiabatic ideal gas) with
perturbations proportional to exp(i k x), k comoving:

```
d delta / dt   = -(i k / a) v
d (a v) / dt   = -i k [ c_s(a)^2 - 4 pi G rhobar / (a k^2) ] delta
c_s(a)^2       = c_s(1)^2 a^(-3 (gamma - 1))
```

(the comoving pressure of uniform gas falls as a^(3 - 3 gamma), which the expansion-drag test
on uniform gas already checks). Berlok's scipy right-hand side (`compressible_scipy.py`), written in a
with da/dt = a H, is

```
d delta / da    = -i k / (a^2 adot) * (a du)
d (a du) / da   = -i k / adot * ( Vs^2 / a^(3 (gamma - 1)) + (Va^2 - Vg^2) / a ) * delta
```

which is the same system with

| Berlok | Shaderous |
|---|---|
| delta rho_c / rho_c | delta (the comoving density's relative perturbation) |
| du | v, the proper peculiar velocity |
| Vs | the proper sound speed at a = 1 |
| Vg^2 | 4 pi G rhobar / k^2, with rhobar = 1 (comoving) |
| Va^2 | B_c^2 / (4 pi rhobar), with B_c = a^2 B (stage 4) |
| a from a_i to 1 | a from aStart to aEnd (aEnd need not be 1) |

Two consequences worth knowing before choosing parameters:

1. **Omega_g is not free.** Omega_g = k Vg / H0, and Vg^2 = 4 pi G / k^2 with G = 3 Omega_m /
   (8 pi), so Omega_g^2 = 3 Omega_m / 2 whatever the box or the wavenumber. In Einstein-de
   Sitter, Omega_g = sqrt(3/2) = 1.22: gravity is either on at that strength or off. The
   pressure, through Omega_s = k Vs / H0, is what sets which side of the Jeans scale a wave is.
2. **Normalization.** Berlok's A_u is du / Vs at a_i, with Vs the sound speed at a = 1, not at
   a_i (`f0 = [A_rho, ai*A_u*Vs]`). Our initial conditions take the real part of
   delta = A_rho exp(i k x) and v = Vs A_u exp(i k x) at aStart.

## 3. Stage 1: a float64 reference

`src/reference/comoving_waves.ts`: integrate the complex 2-variable system of section 2 in
float64 with classical RK4 in ln a (fixed steps, fine enough for 1e-9; the system is not stiff
for the parameters we use), for any Omega_m, Omega_L (`hubble` from `src/cosmo.ts`), any
gamma, with optional terms for Va (stage 4) and a velocity damping rate (stage 5). Return
delta(a) and v(a) as complex numbers.

Why an ODE and not the paper's Bessel-function solutions: the general-gamma solution uses
Bessel functions of possibly imaginary order, and Berlok's own test file notes that mpmath's
were unreliable there for gamma other than 1 and 5/3. The ODE is a few lines, works for
LambdaCDM too (the paper's closed forms are Einstein-de Sitter only), and stage 3 uses the
closed form to check it.

Unit tests (`tests/unit/comoving_waves.test.ts`, vitest, no GPU):

- Against the gamma = 4/3 closed form of section 5, oscillating and growing (kappa real and
  imaginary) and at kappa = 0, to 1e-8.
- Against `cosmo-wave`'s WKB phase where WKB holds (Omega_s large, no gravity).
- Against the growing mode delta proportional to a for cold gas with gravity (Omega_s = 0).
- A handful of values copied from running Berlok's Python (analytic for gamma = 5/3, scipy for
  gamma = 1.2) into a small JSON fixture, so the mapping of section 2 is checked against his
  code and not only against our own derivation. Generate it once with his repo (BSD-style
  license, check) in a scratch virtualenv; commit only the numbers and the script that made
  them under `tools/`.

## 4. Stage 2: standing waves with pressure and gravity

New tests in `src/validation/cosmo.ts`, Einstein-de Sitter, aStart = 0.01 to a = 1 (Berlok's
a_i = 1/128 is close; the factor of 100 in a gives the a^(-1/4) and a^(-3/4) decays room to
show: 3.2 and 32), periodic gravity, a plane wave along x with A_rho = 2e-3 (float32 resolves
1e-7 of the mean, so 1e-4 of the wave), A_u chosen with it as Berlok does (1 and 0.2 in his
units scaled down together).

For each of a few (gamma, Omega_s) pairs:

- gamma = 5/3, Omega_s well above the Jeans value (oscillating, several turns);
- gamma = 5/3, Omega_s below it (growing, but slowed by pressure: a test `cosmo-growth`
  cannot make);
- gamma = 4/3, both sides likewise.

Measure at some 40 stops in a (as `cosmo-wave` does with `through`): project the density and
velocity along a line in x onto cos kx and sin kx, giving the complex delta and v; compare with
stage 1. Pass on the largest error over the stops relative to the largest amplitude, separately
for delta and v, real and imaginary parts (Berlok's tests hold Re and Im apart, which checks
amplitude and phase together).

Tolerance: to be set from a resolution study, not guessed. Run n = 16, 32, 64 (cells per
wavelength = n); the error should fall about fourfold per doubling (second-order scheme) until
it meets the float32 floor. Pick the test's n where the error is a few times below 1% and set
the bound at about twice the measured error, as the other cosmo tests do. Record the study's
numbers in the "As built" note. If the error does not converge, that is a finding about the
solver (the drag split, the gravity kick, the a in the gradients), not a reason to loosen the
bound.

Cost: 64^3 for some 40 stops and a few hundred steps; seconds on a desktop GPU.

## 5. Stage 3: the gamma = 4/3 closed form, its critical case, traveling waves

For gamma = 4/3 the sound speed squared falls as 1/a like the other two terms, and the system
has an elementary solution (Berlok's `compressible_analytic.py`):

```
sigma^2 = Omega_s^2 + Omega_A^2 - Omega_g^2,   kappa = sqrt(sigma^2 - 1/16)
psi     = kappa ln(a / a_i)
delta   = (a/a_i)^(-1/4) [ A_rho cos psi + (A_rho - 4 i Omega_s sqrt(a_i) A_u) / (4 kappa) sin psi ]
du/Vs   = (a/a_i)^(-3/4) [ A_u cos psi - ( A_u / (4 kappa)
          + i A_rho (1 + 16 kappa^2) / (16 sqrt(a_i) kappa Omega_s) ) sin psi ]
```

(transcribed from the repository; stage 1's unit tests confirm it against the ODE before any GPU
test uses it). Imaginary kappa is growth, real kappa oscillation with a period uniform in ln a.

- **The critical case kappa = 0** (sigma = 1/4): the solution turns logarithmic, (a/a_i)^(-1/4)
  [A + B ln(a/a_i)]. With gravity on, sigma^2 = Omega_s^2 - 3/2, so kappa = 0 at Omega_s = 5/4;
  without, at Omega_s = 1/4. Both are cheap GPU tests of a case where neither oscillation nor
  growth dominates, and the place a formula with 1/kappa breaks (the reference must branch,
  as Berlok's does).
- **Traveling waves** (Berlok's `animations/traveling_compressible_gamma_four_thirds.py`): the
  closed form with a_i = exp(-2 pi n / kappa) makes exactly n periods by a = 1. This could
  replace `cosmo-wave`'s WKB integral with an exact phase and amplitude, for gamma = 4/3 and
  with gravity. Keep `cosmo-wave` as it is (gamma = 5/3, its own regime) and add this beside it.

## 6. Stage 4: comoving MHD (when comoving MHD is built)

Not a plan for comoving MHD itself, which needs its own (the induction equation's 1/a, the
storage of B_c = a^2 B so that the background field is constant, the constrained transport's
EMF in proper units, the magnetic energy's drag). These are the tests that plan should adopt:

- **Alfvén wave**, standing, B along x, perturbation in y (Berlok's `alfven_analytic.py`):
  delta B_c / B_c and du / Va follow the same form as section 5 with Omega_A in place of
  sigma, exact for any gamma (the Alfvén wave does not feel the pressure): kappa =
  sqrt(Omega_A^2 - 1/16), amplitudes (a/a_i)^(-1/4) and (a/a_i)^(-3/4), critical at
  Omega_A = 1/4. The first test to run: if B is stored with the wrong power of a, the decay
  exponents are wrong at once.
- **Magnetosonic wave**, B along y, k along x: the system of section 2 with Va nonzero
  (Omega_A, Omega_s and Omega_g together), against stage 1 for gamma = 5/3 and the closed form
  for gamma = 4/3.
- **Traveling Alfvén wave** (`animations/traveling_alfven_wave.py`), the analogue of
  section 5's.

Gravity with MHD: `cosmologyUnsupported` allows periodic gravity, and the magnetosonic test is
the only one here that exercises magnetic pressure against self-gravity in an expanding box.

## 7. Stage 5: the viscous Alfvén wave (when comoving viscosity is built)

Berlok's appendix C (`alfven_scipy_with_viscosity.py`) adds a damping term to the velocity
equation, -Gamma a^(-5/2) (a du) per unit a over adot / H0, that is a damping rate Gamma
H0 a^(-3/2) = Gamma H(a) in Einstein-de Sitter. With a proper kinematic viscosity nu and comoving
k, the rate is nu k^2 / a^2, so his closed form (an extra factor (a/a_i)^(-Gamma/2), critical
point moved to Omega_A = (1 + 2 Gamma)/4) holds for nu proportional to a^2 H, i.e. a^(1/2) in
Einstein-de Sitter. This was read off the code, not the paper; confirm against appendix C.

Two ways to use it, whichever matches how comoving viscosity ends up defined:

- If the code takes a constant proper nu (the natural setting for a user), compare with stage
  1's ODE given the damping rate nu k^2 / a^2 at each a. The closed form does not apply.
- If a test can set nu to follow a^2 H, compare with the closed form as well.

The same damping term applies to the compressible waves (shear viscosity damps them at 4/3 the
rate of a pure shear mode, plus any bulk viscosity), so stage 1's ODE covers a viscous sound
wave too; Berlok gives no closed form for that one.

## 8. Not covered

- Isothermal gas (Berlok's gamma = 1 tests): our isothermal option is an equation of state
  behind the comoving wrappers; whether its sound speed is meant to be constant in proper or
  comoving terms is a definition to settle first. Skip unless it comes up.
- Blocks: the cosmology tests above run on the uniform grid. One stage 2 case repeated on
  blocks with one timestep for all (as the expansion-drag test does for uniform gas) is cheap and
  worth adding; refinement across the wave is not needed for a linear test.
- LambdaCDM closed forms do not exist; stage 1 handles LambdaCDM, and one stage 2 case should
  use it (the background is already tested; this tests the wave in it).

## 9. Order of work

1. Stage 1 with its unit tests, including the fixture from Berlok's Python. This settles the
   mapping of section 2.
2. Stage 2's resolution study for one oscillating case; set the tolerance; add the cases.
3. Stage 3's critical cases and the traveling wave.
4. Update `docs/numerics.md` section 22's "Tests" paragraph with the measured errors.
5. Stages 4 and 5 move into the comoving MHD and comoving viscosity plans when those are
   written.

## 10. References

- T. Berlok, *Hydromagnetic waves in an expanding universe: cosmological MHD code tests using
  analytic solutions*, MNRAS 515, 3492 (2022), doi:10.1093/mnras/stac1882. Code:
  <https://github.com/tberlok/comoving_mhd_waves>.
- G. L. Bryan et al., *A piecewise parabolic method for cosmological hydrodynamics*, Comput.
  Phys. Commun. 89, 149 (1995), doi:10.1016/0010-4655(94)00191-4 (the form of the equations we
  use).
