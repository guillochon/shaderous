# Plan: an equation of state for partially ionized hydrogen and helium, with H2

Status: proposed 2026-11-16, not built. The physical constants and literature below are from
memory and are to be checked against the sources in step 1.

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
