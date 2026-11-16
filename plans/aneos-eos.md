# Plan: ANEOS materials for planetary impacts

Status: proposed 2026-11-16, not built. What is said below about the published tables (who
made them, their format and coverage) is from memory and is step 1's job to confirm, with
their licenses.

## 1. Goal

Run rock and iron bodies: giant impacts, planetesimal collisions, tidal disruption of a
planet. The equation of state comes from ANEOS tables, read into the same table machinery as
the white dwarf one ([src/eos/helmholtz.ts](../src/eos/helmholtz.ts)): a free energy and its
derivatives on a grid in log density and log temperature, bicubic Hermite between nodes,
pressure and energy as first derivatives.

The showpiece is a moon-forming impact, which is also a good teaching problem: gravity,
shocks, vaporization and a debris disk in one run.

## 2. What ANEOS is

An analytic equation of state from Sandia (Thompson & Lauson 1972) built as a Helmholtz free
energy F(rho, T): a cold compression curve, a thermal term for the nuclei that goes from a
Debye solid to an ideal gas, and an electronic term for ionization. It includes melting and
vaporization, with the liquid-vapor dome treated by a Maxwell construction. M-ANEOS (Melosh
2007) adds molecules to the vapor.

Because it is a free energy, pressure, energy and entropy are consistent, which is why it
fits here.

### Import, do not reimplement

ANEOS is a large Fortran code with dozens of parameters per material. Tables computed from it
are published for the materials that matter:

| Material | Source to confirm |
|---|---|
| Forsterite (mantle) | Stewart et al. 2019/2020, with the parameter set and SESAME-style tables |
| Iron alloy (core) | Stewart 2020 |
| Water ice | Several; choose in step 1 |

Those tables give pressure, energy, entropy and sound speed on a non-uniform grid of density
and temperature. From them F = E - T S, with F's density derivative from p and its
temperature derivative from S, which is exactly what a Hermite table of F needs.

## 3. Building a table

A script in `tools/eos` per material:

1. Read the published table; convert to cgs.
2. Resample onto a uniform grid in log rho and log T (cubic in the smooth regions; linear
   across the dome boundary, where second derivatives jump).
3. Store a scaled F with its first derivatives and cross derivative.
4. Build inverse tables for Newton's starting point, log T from (rho, e) and (rho, p / rho).
5. Report the resampling error against the source at held-out points, phase by phase.

Range: roughly 1e-6 to 1e2 g/cm^3 and 100 to 1e6 K. At 0.02 dex that is 400 by 200 nodes,
well under a megabyte per material.

## 4. What is hard

| Problem | Detail | Proposal |
|---|---|---|
| Two materials in one cell | A grid code mixes core and mantle at their interface and wherever they are stirred | Section 5 |
| The vapor dome | Inside it the isotherms are flat in pressure, the sound speed is very low, and second derivatives jump at its edge | Hermite interpolation holds the first derivatives continuous, which is what conservation needs. The sound speed is floored at a small fraction of the cold value, and the validation test measures the error near the edge |
| Tension | A cold solid stretched below its reference density has negative pressure. A fluid code has no strength model to go with it | Pressure floored at a small positive value, as impact codes without strength do; stated in the docs. The cold curve is checked to rise with density above the floor |
| A surface at finite density | Rock at zero pressure is 3.2 g/cm^3, so a planet's surface is a jump to near vacuum | Ambient gas is the same material's low-density vapor from the table. The settling step relaxes the surface as it does a star's |
| Low temperatures in single precision | Thermal energy at 300 K is small beside the cold curve at depth | As in the white dwarf module: the cold curve is subtracted and kept apart, and the solve works on the thermal part |
| No analytic limit to test against | Unlike the electron gas | Published Hugoniots and the 1 bar boiling point are the checks (section 7) |

## 5. Two materials

Three levels, each a version by itself:

1. **One material per run.** Bodies of forsterite alone. Enough to test the machinery,
   vaporization and the dome, and to run undifferentiated bodies.
2. **Two materials, sharp.** A mass fraction of iron is carried as a species, which the code
   already does. In a mixed cell each material takes the volume that brings both to the same
   pressure at the same temperature. That is two unknowns (the temperature and the split of
   volume) in place of one, so a two-dimensional Newton iteration, started from the cell's
   last answer. It costs perhaps three times a single-material call. Most cells are pure and
   take the single-material path.
3. **Three or more** (ice, an atmosphere): the same closure with more unknowns. Not planned.

The closure of level 2 is the standard one (pressure and temperature equilibrium) and is
consistent with a free energy of the mixture, F = sum of x_i F_i at each material's own
partial density.

Level 2 is the larger half of the work and carries the risk: the iteration must converge at
every face of every cell, in single precision, including where one material is a trace.

## 6. In the code

- **The shared table module**: the lookup, Hermite basis, inverse-table guess and Newton
  solve moved out of `helmholtz.wgsl` into an include, with each equation of state supplying
  its layout and its analytic terms. The float64 mirror is split the same way.
- **`EosKind` `'aneos'`** with a material choice (`physics.material`: forsterite, iron, or
  both for level 2). Tables are fetched when chosen, not shipped in the first page load.
- **Units**: a planetary unit system (Earth masses, Earth radii, hours), beside the white
  dwarf and stellar ones.
- **A planet part** for the setup builder: mass, core mass fraction, a surface temperature
  or an entropy; integrated in hydrostatic balance along an isentrope in each layer, then
  settled against the grid's own potential as stars are.
- **Diagnostics**: vapor fraction and entropy as view fields, since they are what impact
  papers plot; bound, orbiting and escaping mass, which the debris tools already separate
  for tidal disruptions.

## 7. Tests

| Test | Checks |
|---|---|
| Resampling | Table against source at held-out points, by phase |
| First law | p and e against finite differences of the tabulated F |
| Hugoniot | Shock pressure against particle velocity for forsterite and iron, against the published curves; in 1D with the code's own shock tube |
| Boiling | The 1 bar boiling temperature of forsterite from the table's dome |
| Planet | An Earth-mass, 30% iron body: radius within a few percent of 1 Earth radius, and it holds still for ten dynamical times |
| GPU against mirror | Added to the equation-of-state validation test, with points inside the dome |
| Two materials | A mixed cell's pressure equals each material's at its partial density; a contact between iron and rock stays in pressure balance as it advects |

## 8. Steps and size

1. Confirm sources, formats and licenses; choose the ice table or drop ice. Half a day.
2. The table builder for forsterite, with its error report. Two days.
3. The shared table module (shared with [hydrogen-helium-eos.md](hydrogen-helium-eos.md);
   whichever is built first does it). One day.
4. The module, mirror and option for one material; validation test; the Hugoniot test. Two
   to three days.
5. Planetary units, the planet part, a single-material collision preset. Two days.
6. Two materials: the closure, its tests, a differentiated planet. One to two weeks.
7. The moon-forming impact preset, vapor and entropy fields, docs and a teaching activity.
   Two days.

Levels 1 (steps 1 to 5) is about a week and a half and stands alone. Level 2 doubles it.
