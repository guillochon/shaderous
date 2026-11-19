# Plan: an equation of state for partially ionized hydrogen and helium, with H2

Status: built 2026-11-18, steps 1 to 4 and 6. Step 5 (calibrated pressure ionization) is not
built; a stand-in is in its place. What was built, and where it differs from what follows, is
in section 8; `tools/eos/NOTES.md` has the model's choices and the table's accuracy, and
`docs/numerics.md` section 9 the description. The rest of this file is the plan as proposed
on 2026-11-16, with constants then from memory.

## 1. Goal

An equation of state for gas of hydrogen and helium in chemical equilibrium among H2, H, H+,
He, He+, He++ and electrons, defined by one free energy and tabulated the way the white dwarf
one is ([src/eos/helmholtz.ts](../src/eos/helmholtz.ts)): the free energy and its derivatives
on a grid in log density and log temperature, bicubic Hermite between nodes, pressure and
energy as first derivatives, temperature by Newton's method from an inverse table.

What it opens:

| Problem | Why it needs this |
|---|---|
| Protostellar collapse | The second collapse starts when H2 dissociates near 2000 K and gamma falls below 4/3 |
| Common envelopes, stellar mergers | Recombination energy of H and He is a large part of the envelope's binding energy |
| Disruptions and collisions of Sun-like stars | Realistic envelopes in place of polytropes of an ideal gas |
| Interstellar and circumstellar gas | Molecular, atomic and ionized phases with one equation of state |

### Non-goals for v1

- Metals, dust, and non-equilibrium chemistry (H2 formation times, ionization fronts).
- A composition that varies from cell to cell. One table is for one (X, Y); section 6.
- Replacing `eos/ionizing.wgsl`, which stays as the cheap analytic hydrogen-only model.

## 2. The free energy

An ideal mixture, F = sum over species of N_s k T [ln(n_s lambda_s^3 / z_s) - 1], with
internal partition functions z_s carrying the energies:

| Species | Internal state |
|---|---|
| H2 | Rotation (theta_rot about 85 K, ortho and para summed separately), vibration (theta_vib about 6100 K), ground state as the zero of energy |
| H | Dissociation energy 4.48 eV per molecule above H2; ground state only |
| H+ | 13.60 eV above H |
| He, He+, He++ | 24.59 eV and 54.42 eV |
| Electrons | Ideal, non-degenerate in v1 |

Equilibrium is the minimum of F over the species numbers at fixed nuclei, which gives the
dissociation and Saha equations. With the molecule's ground state as zero, every energy is
positive.

Radiation (a T^4) is added analytically outside the table, as in the white dwarf module.

This is the model of Tomida et al. (2013) and Vaytet et al. (2013) for protostellar collapse.

### Decisions the model needs

| Question | Proposal | Why |
|---|---|---|
| Ortho to para ratio | Fixed 3:1 | The usual choice for collapse; the equilibrium ratio is an option of the table generator |
| Pressure ionization | A density-dependent lowering of the ionization and dissociation energies, inside the free energy | Bare Saha equations recombine the gas at stellar-interior densities, which is wrong. Putting the lowering in F keeps p and e consistent. Its form is calibrated against a published table along a solar profile |
| Electron degeneracy | Not in v1 | Matters above about 100 g/cm^3 at 1e7 K: the Sun's core is marginal. Stated as a limit of validity |
| Coulomb terms | Not in v1 | A Debye-Hueckel term can be added to F later with no change to the method |

Pressure ionization is the item most likely to take longer than planned. Without it the
equation of state is good for collapse, envelopes and the interstellar medium, and wrong for
stellar interiors; a first version could ship with that limit stated and a density above
which the gas is taken as fully ionized.

## 3. Building the table

A script in `tools/eos` (beside `make_table.py`):

1. For each (rho, T), solve the equilibrium for the species numbers in double precision. The
   system reduces to one unknown, the electron density, with the hydrogen and helium
   fractions in closed form given it.
2. Evaluate F at the solution. Its first derivatives in ln rho and ln T are p and the
   entropy, which need no derivative of the solution (it is a minimum). The cross derivative
   the Hermite interpolation wants comes from automatic differentiation through the solve, or
   from differencing the exact first derivatives on a finer grid.
3. Store a scaled free energy that stays of order one, as h does for the electrons.
4. Build the inverse tables, log T from (rho, e) and from (rho, p / rho).

Range and size: log rho from -20 to 3, log T from 1 to 8, at 0.02 dex, is 1151 by 351 nodes,
6.5 MB as an rgba32float texture. Whether 0.02 dex resolves the ionization and dissociation
fronts at the lowest densities, where they are sharpest, is checked in step 4; the white
dwarf table needed that spacing for the pair onset.

## 4. In the code

- **A table equation of state, not a second copy.** The lookup, the Hermite basis, the
  inverse-table guess and the Newton solve in `helmholtz.wgsl` are generic. Move them to a
  shared include with the layout as constants, and leave in each module only what is its own
  (for the white dwarf: the zero-temperature electrons, pairs, ions, radiation, Coulomb
  terms). The float64 mirror and `tools/eos/helm.py` are split the same way.
- **`EosKind` `'hhe'`**, label "Hydrogen and helium (H2, ionization)". Constants: cgs
  conversions as the white dwarf module has, and the radiation constant.
- **One texture binding.** Binding 22 holds whichever table the run's equation of state
  uses.
- **Composition.** v1 ships two tables, solar (X = 0.71, Y = 0.27, with the metals' mass
  counted as helium) and pure hydrogen, chosen by `physics.composition`. A setup with another
  composition is refused with a message naming the two.
- **Temperature from energy.** The energy rises with T everywhere, but nearly flat through
  dissociation and ionization, where Newton in ln T needs the inverse table most.
- **Sound speed.** From second derivatives of the interpolant, as now. Across a front they
  are less accurate than the first; the validation test sets the tolerance.

## 5. Tests

| Test | Checks |
|---|---|
| Limits | Cold: p = rho k T / (2 m_u) per H2 with gamma 7/5 at 300 K and 5/3 at 30 K. Hot: fully ionized ideal gas, mu = 0.62 for solar |
| Hydrogen alone | Ionization fraction against `eos/ionizing.wgsl`'s Saha solution |
| First law | p and e from the table against finite differences of the generator's F; the Maxwell relation |
| Published values | Gamma_1 along the dissociation and ionization valleys against Tomida et al.'s figures; a solar envelope profile against a stellar-evolution table |
| GPU against mirror | Added to the existing equation-of-state validation test |
| Collapse | A new preset: a 1 solar mass cloud with a barotropic-free energy equation, forming a first core, then a second at dissociation. Qualitative: the central temperature against density follows the published track |

## 6. Later

- **Varying X and Y.** The free energy of an ideal mixture is not a product of a hydrogen
  part and a helium part (they share the electrons), so a table per composition does not
  interpolate exactly. Options: a third table axis in X (a 3D texture, some tens of MB), or
  solving the equilibrium in the shader from analytic partition functions (no table, one root
  find in the electron density inside each temperature iteration; slower per call).
- **Degenerate electrons and Coulomb terms**, joining this to the white dwarf table at high
  density for one equation of state from molecular clouds to stellar cores.

## 7. Steps and size

1. Check the constants, partition functions and the choice of pressure-ionization model
   against the sources. Half a day.
2. The generator and its own tests (limits, first law). Two days.
3. The shared table module, with the white dwarf module moved onto it and its tests
   unchanged. One day.
4. The new module, mirror, option and validation test. Two days.
5. Pressure ionization and its calibration. Two to four days; can ship after the rest.
6. The collapse preset, docs. One to two days.

About a week and a half. Step 3 is shared with [aneos-eos.md](aneos-eos.md); whichever is
built first does it.

## 8. As built (2026-11-18)

| Step | Built | Differs from the plan |
|---|---|---|
| 1. Sources | Constants checked against Tomida et al. (2013), appendix: theta_rot 85.32 K, theta_vib 5984.48 K, D = 4.478 eV | Their statistical weights for H2 are not the textbook ones (4.6 times too many); the textbook ones are used, so dissociation is about 6% cooler than theirs |
| 2. Generator | `src/eos/hhe_model.ts` (the free energy), `tools/eos/make_hhe_table.mjs` | Node, not Python: the tests import the same model, and nothing needs installing |
| 3. Shared table module | `eos/table.wgsl`, `src/eos/table.ts`, `tools/eos/table.py`; the white dwarf module is on it, its tests and its GPU numbers unchanged | `make_table.py` was edited to match and not run (its input grid is not in the repository) |
| 4. Module, mirror, option | `eos/hhe.wgsl`, `src/eos/hhe.ts`, `EosKind` `'hhe'`, `physics.eosParams.hheRad`, cases in the `eos` validation test | The grid is 0.1 dex in rho and 0.01 in T, not 0.02 both ways: the same 5.2 MB, five times closer on the fronts. The table holds differences of h, which single precision needs. Solar is the code's X = 0.7381 |
| 5. Pressure ionization | Not built | A stand-in in the free energy (each bound electron costs k T rho / 0.33 g/cm^3) takes dense hot gas as ionized. Not calibrated |
| 6. Collapse preset | `second-collapse`: a first core past its greatest mass, through dissociation to a second core, in (r, theta); validation test `sph-second-collapse` | Not a 1 solar mass cloud from 1e-18 g/cm^3. See below |

**The collapse preset.** The plan asked for a solar-mass cloud forming a first core and then a
second, with an energy equation and no barotrope. Two things stand in the way, neither in the
equation of state. A cloud stays at 10 K until 1e-13 g/cm^3 because it radiates, so the run
needs radiation transport, which is Cartesian only and not yet allowed with this equation of
state; and from a cloud of 1e17 cm to a second core of 1e11 cm is six decades of radius, which
the (r, theta) grid's timestep (set by its innermost cells) cannot span in one run. What is
built is the part this equation of state decides: a first core on the published adiabat (10 K
at 1e-13 g/cm^3), 0.036 solar masses, its center at 2100 K, which collapses when H2
dissociates and forms a second core at 1e-3 to 1e-2 g/cm^3 and 20,000 K. The published track
of central temperature against density is checked twice: as the model's adiabat (a unit test:
2060 K at 1e-8 g/cm^3) and in the run (the center stays within 0.1% of the adiabat over four
decades of density; the test asks for 1% and takes about three minutes).

**More presets (2026-11-18).** Variants of three setups with this equation of state, each
gas alone (`hheRad` 0): `tidal-disruption-hhe`, `common-envelope` (which was the example setup; it opens with
this equation of state, and `common-envelope-ideal` is its ideal-gas variant) and `thermal-instability-hhe` (in
physical units: 31 pc, a hydrogen atom per cm^3, 1000 K). Checked on the GPU at 128^3 with two
levels for two time units (the stars hold as the ideal-gas ones do, no floored cells) and the
instability at 256^2 to t = 6. The giant with a point-mass core runs as the ideal-gas one does (greatest density within
0.2% over two time units); its gas holds 0.24 G M^2 / R more energy than the ideal gas's, the
ionization energy. Not run through the encounter or the spiral-in.

**Still open.**
- Step 5, and with it stellar interiors cooler than a few 1e5 K above 0.1 g/cm^3.
- The comparison of a solar envelope profile against a stellar-evolution table (section 5): no
  such table was at hand. The tests check that a Sun-like interior is ionized and ideal to 1%.
- Radiation transport with this equation of state (`RAD_EOS`), which the cloud-to-first-core
  run needs.
- The radiation term is on by default, as the plan has it. For thin gas that is wrong (section
  "Radiation" of the notes); whether the default should follow the density is undecided.
- Total energy in the preset's run is not conserved once the second core forms (it drifts by
  about the first core's binding energy): spherical runs do not use the flux form of
  gravity's work.
