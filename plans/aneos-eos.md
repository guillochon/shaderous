# Plan: ANEOS materials for planetary impacts

Status (2026-11-18): built, for rock and iron: the tables, the equation of state on the GPU with
its float64 mirror, two materials in a cell, planets, the `planet` and `moon-forming-impact`
presets and the analysis of what an impact leaves. `docs/numerics.md` section 25 is the
description as built. Departures from the text below:

- **The table holds ANEOS's pressure and energy, not its free energy** (sections 1, 2 and 4
  say free energy). ANEOS's p and s are not the derivatives of the F it reports in the fluid
  with ionization: they differ by 1 to 13% at 2e4 K in the iron alloy. A Hermite table of F
  with p and s as slopes was wrong by 3% at the median there. Each material has two bicubic
  Hermite tables with limited slopes, asinh(p / 1e9) and ln(e - E_c), and a table is 2.2 MB,
  not "well under a megabyte".
- **Two materials are tabulated, not solved in the shader** (section 6 planned a 2D Newton
  iteration per mixed cell). The closure is solved in the page when a pair is first asked for
  (1.4 s), into tables at mass fractions 1/4, 1/2 and 3/4, and a cell blends the two on either
  side of its own. The shader reads a mixture exactly as it reads one material.
- **The shared table module (section 7, step 3) was not built.** The ANEOS module shares the
  Hermite basis with the white dwarf one by copy, a dozen lines; the two tables differ in
  everything else.
- **Planets are not settled** (section 7 said settled as stars are): their layers are not one
  barotrope. They start on the integrated structure and ring by 3.6% in central density at 16
  cells to the radius.
- **Vapor fraction and entropy view fields: not built.** Nor a teaching activity.
- **Water: the table is built and tested against ANEOS** (`water.bin`), and a planet can be
  made of it, but no icy preset or IAPWS-95 test exists yet.
- Tests as built: `tests/unit/aneos.test.ts`, `planet.test.ts`, `impact.test.ts`; GPU
  `aneos-eos`. The Hugoniot test against laboratory points (section 8) is not built: the table
  is tested against ANEOS itself, which the decks' own notebooks compare with the laboratory.

Sources and table formats of section 3 were checked against the repositories and their Zenodo
records on 2026-11-18, with a search for newer releases; section 10 marks the references that
are still from memory.

## 1. Goal

Run rock and iron bodies: giant impacts, planetesimal collisions, tidal disruption of a
planet. The equation of state comes from ANEOS tables, read into the same table machinery as
the white dwarf one ([src/eos/helmholtz.ts](../src/eos/helmholtz.ts)): a free energy and its
derivatives on a grid in log density and log temperature, bicubic Hermite between nodes,
pressure and energy as first derivatives.

The showpiece is a moon-forming impact ([Canup 2004][Ca04]; [Ćuk & Stewart 2012][CS12];
[Lock et al. 2018][Lo18]), which is also a good teaching problem: gravity, shocks, vaporization
and a debris disk in one run. Its icy counterpart is the collision that made Pluto's Charon
(Canup 2005).

| Level | Result | Size |
|---|---|---|
| 1 | One material per run: rock, iron or ice bodies, their collisions, the vapor dome | About a week and a half |
| 2 | Two materials: a differentiated planet (iron core and rock mantle, or rock core and ice mantle) and the moon-forming impact | The same again |

### Non-goals

- Material strength, porosity and fracture. The bodies are fluids, which is right above a few
  hundred kilometers and wrong for small planetesimals; the docs must say so.
- Three materials in one cell, atmospheres of hydrogen and helium.
- Reimplementing ANEOS in the browser. Tables are built offline.

## 2. What ANEOS is

An analytic equation of state from Sandia ([Thompson & Lauson 1972][TL72]; [Thompson
1990][Th90]) built as a Helmholtz free energy F(rho, T) in three parts: a cold compression
curve, a thermal term for the nuclei that goes from a Debye solid to an ideal gas, and an
electronic term for ionization. It includes melting and vaporization, with the liquid-vapor
dome treated by a Maxwell construction. Later changes, which together are M-ANEOS:

| Change | Reference |
|---|---|
| Molecules in the vapor; a Mie-type potential for the expanded solid | [Melosh 2007][Me07] |
| A solid-solid transition and melting in one material | [Collins & Melosh 2014][CM14] |
| An adjustable limit to the heat capacity of the liquid (the Dulong-Petit value was hard-coded) | [Stewart et al. 2020][St20] |

Because it is a free energy, pressure, energy and entropy are consistent, which is why it
fits here.

## 3. Sources

Checked 2026-11-18. All are open source.

| What | Where | Version | Released |
|---|---|---|---|
| M-ANEOS, the Fortran code | [isale-code/M-ANEOS][MAN], [doi:10.5281/zenodo.3525030][MANz] | v1.0 | 2019-11 |
| pyaneos, a Python port of M-ANEOS | [ImpactsWiki/pyaneos][PyA], [doi:10.5281/zenodo.20736271][PyAz] | v1.0 | 2026-06 |
| Pyrolite (mantle) | [ststewart/aneos-pyrolite-2022][Py], [doi:10.5281/zenodo.6968539][Pyz] | SLVTv0.2G1 | 2022-08 |
| Forsterite (mantle) | [ststewart/aneos-forsterite-2019][Fo], [doi:10.5281/zenodo.3478631][Foz] | SLVTv1.0G1 | 2019-10 |
| Iron alloy Fe85Si15 (core) | [ststewart/aneos-Fe85Si15-2020][FeSi], [doi:10.5281/zenodo.3866550][FeSiz] | SLVTv0.2G1 | 2020-05 |
| Iron (core) | [ststewart/aneos-iron-2020][Fe], [doi:10.5281/zenodo.3866507][Fez] | SLVTv0.2G1 | 2020-05 |
| Water and ice | [ststewart/aneos-water][Wa], [doi:10.5281/zenodo.14226693][Waz] | SLVTv0.3G1 | 2024-11 |
| Fused silica | [ststewart/aneos-fused-silica][Si], [doi:10.5281/zenodo.14226685][Siz] | SLVTv0.2G1 | 2024-11 |
| Quartz, serpentine, dunite inputs | `input/` of M-ANEOS | | 2019 |

### What is current

The search for newer releases (GitHub and Zenodo, 2026-11-18) found:

- **Forsterite and both irons have not been revised.** Their repositories' only change since
  release is a notebook added in 2023; the tables are those of 2019 and 2020.
- **Pyrolite has replaced forsterite as the mantle** in the same group's work. The most recent
  giant-impact study from it ([Postema, Lock & Stewart 2026][Po26]) uses the pyrolite table
  for mantles and Fe85Si15 for cores, both at SLVTv0.2G1, and calls them its updated
  equations of state. The pyrolite record is marked "in development"; its reference is an
  abstract (Stewart et al. 2022, LPSC 53, 1535), and the paper its notebooks promise (Stewart,
  "The thermodynamics of giant impacts", in preparation) has not appeared.
- **Water is new** (2024), in the same form as the others. It was used while in development
  by Kovačević et al. (2022).
- **pyaneos is new** (2026): M-ANEOS in Python, bit-identical to the Fortran on its test
  suite. It adds a `NOTENSION` keyword that tabulates the sublimation region in place of the
  stretched solid, melting curves of negative slope (water's), and chains of two or three
  solid phases. Its own example decks (`sio2-2026`, `h2o-2026`) are marked as test fixtures,
  not for science: they show what the next released models are likely to hold.

So the defaults are **pyrolite and Fe85Si15** for a rocky planet and **water** for ice, with
forsterite and pure iron kept as choices because the published comparisons are made with them.

### The files

Each material's repository (for pyrolite, the Zenodo archive) holds the same set:

| File | Contents |
|---|---|
| `ANEOS.INPUT`, `ANEOS.OUTPUT` | The parameter set, and the code's own report: phase boundaries, critical and triple points, the 1 bar melting and boiling points, a Hugoniot |
| `NEW-SESAME-STD.TXT` | SESAME-style 201 and 301 records on a grid of density and temperature: pressure, specific internal energy and **the Helmholtz free energy** (g/cm^3, K, GPa, MJ/kg, MJ/kg) |
| `NEW-SESAME-EXT.TXT` | On the same grid: specific entropy, sound speed, heat capacity and ANEOS's phase flag KPA (MJ/K/kg, cm/s, MJ/K/kg) |
| `NEW-SESAME-STD-NOTENSION.TXT` | The same with the negative pressures of the stretched solid replaced |
| `NEW-SESAME-MDQ.TXT` | A quality rating per node (see below) |
| `NEW-GADGET-*.TXT` | Tables on density and entropy for SPH codes; not needed here |
| `tablegrid.txt` | The grid M-ANEOS was told to tabulate on |
| `*aneos*.ipynb`, `eostable.py`, `data/` | The documentation notebook, a Python reader for the tables, and the laboratory data the model was fitted to |

The grids are not uniform: they are dense around the melting curve and the dome. The
forsterite table's 301 record is 1,921,126 words long, which fits 860 by 744 nodes; the file
is 31 MB of text. The water table's is 1,998,028 words.

### The models' limits

From the notebooks; the tests of section 8 must respect them.

- ANEOS cannot fit the triple point, the 1 bar boiling point and the critical point at once.
  The vapor side of the dome is rated "use with caution" in the quality table.
- The 1 bar boiling point of forsterite has not been measured. A test of it is a test against
  `ANEOS.OUTPUT`, not against the laboratory.
- The iron model's reference density is 8.06 g/cm^3, not the true 7.874, so that the Hugoniot
  at high pressure is right.
- The 2019 to 2024 `NOTENSION` tables fill the tension region with the values of the adjacent
  node, which the author notes is bad for interpolation.
- Melting across a coarse grid gives isotherms that are not flat, and so energy errors: the
  iron table needs a very dense grid in density to hold its melting curve.
- **The water model has one solid, and it is not ice Ih.** Its reference solid is 1.25 g/cm^3
  at 150 K, denser than ice Ih's 0.92, "because of limitations in the melt curve model". The
  liquid, the vapor, the dome (fitted to the triple point's pressure and the critical
  temperature) and ionization are what an impact needs and what the model is for.

### Ice

Ice is in v1, from the water table: icy moons and dwarf planets, ice mantles on rock cores,
comets striking a planet. What the one-solid model costs is the cold outer layers: a body
below 273 K is about a third too dense near its surface, so its radius for a given mass is
too small, and the low-pressure phases (Ih, II, III, V, VI) and their transitions are absent.
The docs say so beside the material's name, and the planet test of section 8 states the
radius error it accepts.

Two ways to do better, neither needed to start:

| Alternative | What it gives | Why not first |
|---|---|---|
| A water deck with ice Ih and a high-pressure ice, in pyaneos (as its `h2o-2026` examples sketch) | The right surface density and melting curve, in the same free-energy form | No calibrated deck is released; watch the `aneos-water` repository |
| AQUA ([Haldemann et al. 2020][Ha20]), or the five-phase table of Senft & Stewart (2008) | Real ice phases from several fitted equations of state | Tables of p, not one free energy: the first law holds only to the tables' accuracy. AQUA's entropy was revised in 2026 |

**Precedent.** The SWIFT code reads these same tables in their SESAME-style form for giant
impacts ([Kegerreis et al. 2019][Ke19]; [Ruiz-Bonilla et al. 2021][RB21]), with initial
conditions from integrating each layer along an adiabat, as section 7 plans.

## 4. Building a table

Two routes to a table uniform in log density and log temperature:

| Route | How | For | Against |
|---|---|---|---|
| A. Generate | Run pyaneos with our grid as `tablegrid.txt` and the published `ANEOS.INPUT` | Values at our nodes, free energy included, no resampling across phase boundaries; any resolution; Python only, as the other table builders are; `NOTENSION` done properly | A deck written for the 2019 Fortran must give the same table in pyaneos |
| B. Resample | Read the published tables with `eostable.py`'s format; interpolate onto our grid | Nothing to run | Interpolating across the dome's edge and the melting curve, where second derivatives jump |

Route A, with the published table as its check: our generated values against the published
ones at the published nodes. pyaneos is tested as bit-identical to the Fortran, so a
difference there is a difference of version or of grid, to be understood before going on. If a
deck does not run in pyaneos, the Fortran M-ANEOS is the fallback for that material.

A script in `tools/eos` per material:

1. Run pyaneos on the grid, and on one twice as dense (the notebooks measure their own
   interpolation error this way).
2. Convert to cgs. Form the table's entries: F; its density derivative p / rho^2; its
   temperature derivative -S; the cross derivative from differences of p along temperature on
   the denser grid.
3. Store a scaled F with the cold curve subtracted (section 5), its first derivatives and
   cross derivative.
4. Build inverse tables for Newton's starting point, log T from (rho, e) and (rho, p / rho).
5. Report: the Hermite table against the denser run at the nodes it does not hold, phase by
   phase (the KPA flag says which), and against the published table.

Range: roughly 1e-6 to 1e2 g/cm^3 and 100 to 1e6 K (icy bodies are
colder; water's lower bound follows its published table's). At 0.02 dex that is 400 by 200 nodes,
well under a megabyte per material. Whether 0.02 dex holds the melting curve of iron is what
step 5's report decides; if not, the density axis is refined or melting is smoothed over a
node, stated in the docs.

## 5. What is hard

| Problem | Detail | Proposal |
|---|---|---|
| Two materials in one cell | A grid code mixes core and mantle at their interface and wherever they are stirred | Section 6 |
| The vapor dome | Inside it the isotherms are flat in pressure, the sound speed is very low, and second derivatives jump at its edge | Hermite interpolation holds the first derivatives continuous, which is what conservation needs. The sound speed is floored at a small fraction of the cold value, and the validation test measures the error near the edge |
| Tension | A cold solid stretched below its reference density has negative pressure. A fluid code has no strength model to go with it | Tables are generated with pyaneos's `NOTENSION`, which puts the solid and its vapor in equilibrium there, so the pressure is small and positive without a floor of ours. The cold curve is checked to rise with density |
| A surface at finite density | Rock at zero pressure is 3.2 g/cm^3 (the water model's solid 1.25), so a planet's surface is a jump to near vacuum | Ambient gas is the same material's low-density vapor from the table. The settling step relaxes the surface as it does a star's |
| Low temperatures in single precision | Thermal energy at 300 K is small beside the cold curve at depth | As in the white dwarf module: the cold curve is subtracted and kept apart, and the solve works on the thermal part |
| No analytic limit to test against | Unlike the electron gas | The laboratory Hugoniots in each repository's `data/` and the values of `ANEOS.OUTPUT` are the checks (section 8) |
| Resolution | Collision outcomes with these equations of state converge slowly with resolution ([Meier et al. 2021][Me21]) | The preset's claims are qualitative; the docs give the resolution of the published runs beside ours |

## 6. Two materials

Three levels, each a version by itself:

1. **One material per run.** Bodies of rock alone, iron alone or ice alone. Enough to test
   the machinery, vaporization and the dome, and to run undifferentiated bodies.
2. **Two materials, sharp.** Any pair: iron and rock, or rock and ice. A mass fraction of
   the second is carried as a species, which the code already does. In a mixed cell each material takes the volume that brings both to the same
   pressure at the same temperature. That is two unknowns (the temperature and the split of
   volume) in place of one, so a two-dimensional Newton iteration, started from the cell's
   last answer. It costs perhaps three times a single-material call. Most cells are pure and
   take the single-material path.
3. **Three in one cell** (iron, rock and ice where a body's layers are stirred together): the
   same closure with one more unknown. Not planned. A three-layer body can still be run at
   level 2 if a cell holding all three is closed over its two most abundant, with the third's
   mass given to the nearer of them in density; whether that is good enough is for a test to
   say.

The closure of level 2 is the standard one (pressure and temperature equilibrium; [Miller &
Puckett 1996][MP96] set out the alternatives) and is consistent with a free energy of the
mixture, F = sum of x_i F_i at each material's own partial density.

Level 2 is the larger half of the work and carries the risk: the iteration must converge at
every face of every cell, in single precision, including where one material is a trace.

## 7. In the code

- **The shared table module**: the lookup, Hermite basis, inverse-table guess and Newton
  solve moved out of `helmholtz.wgsl` into an include, with each equation of state supplying
  its layout and its analytic terms. The float64 mirror is split the same way. Shared with
  [hydrogen-helium-eos.md](hydrogen-helium-eos.md); whichever is built first does it.
- **`EosKind` `'aneos'`** with a material choice (`physics.material`: pyrolite, forsterite,
  Fe85Si15, iron, water, or two of them for level 2). Tables are fetched when chosen, not shipped
  in the first page load.
- **Units**: a planetary unit system (Earth masses, Earth radii, hours), beside the white
  dwarf and stellar ones.
- **A planet part** for the setup document: mass, the materials of its layers and their mass
  fractions, a surface temperature
  or an entropy; integrated in hydrostatic balance along an isentrope in each layer, then
  settled against the grid's own potential as stars are.
- **Diagnostics**: vapor fraction and entropy as view fields, since they are what impact
  papers plot; bound, orbiting and escaping mass, which the debris tools already separate
  for tidal disruptions.
- **Credit**: the docs and the page's material menu cite, as the M-ANEOS README asks, the
  material's Zenodo record with [Thompson & Lauson 1972][TL72], [Melosh 2007][Me07] and the
  M-ANEOS record.

## 8. Tests

| Test | Checks |
|---|---|
| Build | Our run of M-ANEOS against the published table at its nodes |
| Table | The Hermite table against the denser run at held-out nodes, by phase |
| First law | p and e against finite differences of the tabulated F |
| Water | The saturation curve and critical point against IAPWS-95 ([Wagner & Pruß 2002][WP02]), with the error the notebook reports as the bar; the Hugoniot of liquid water from the repository's data |
| Hugoniot | Shock pressure against particle velocity: pyrolite against the points in its notebook; forsterite against the gas-gun points of Mosenfelder et al. 2007 and the Z-machine points of [Root et al. 2018][Ro18]; iron against the shock-wave database points; both from the repositories' `data/`. In 1D with the code's own shock tube |
| Phase points | The critical point and the 1 bar melting and boiling points read from the table's dome, against `ANEOS.OUTPUT` |
| Planet | An Earth-mass body with 30% iron: radius within a few percent of 1 Earth radius, and it holds still for ten dynamical times. A body of Pluto's mass, a third ice over rock: it holds still, and its radius is below Pluto's by the amount the dense solid predicts |
| GPU against mirror | Added to the equation-of-state validation test, with points inside the dome |
| Two materials | A mixed cell's pressure equals each material's at its partial density; a contact between iron and rock, and one between rock and ice, stays in pressure balance as it advects |

## 9. Steps and size

1. Install pyaneos and reproduce the published Fe85Si15 table from its deck; then the
   pyrolite and water decks. Half a day.
2. The table builder, with its error report, for Fe85Si15, pyrolite and water (forsterite and
   pure iron are then a deck each). Two to three days.
3. The shared table module. One day. Done 2026-11-18 with the hydrogen and helium equation
   of state: `eos/table.wgsl`, `src/eos/table.ts`.
4. The module, mirror and option for one material; validation test; the Hugoniot and water
   tests. Two to three days.
5. Planetary units, the planet part, single-material collision presets (rock, ice). Two days.
6. Two materials: the closure, its tests, a differentiated planet of each kind. One to two
   weeks.
7. The moon-forming impact preset and an icy one, vapor and entropy fields, docs and a
   teaching activity. Two to three days.

Level 1 (steps 1 to 5) is about a week and a half and stands alone. Level 2 doubles it.

## 10. References

Checked 2026-11-18 (the record or the paper was opened, or the DOI resolved to this title):

- [Canup 2004][Ca04], Icarus 168, 433: simulations of a late lunar-forming impact.
- [Collins & Melosh 2014][CM14], 45th Lunar and Planetary Science Conference, abstract 2664:
  improvements to ANEOS for multiple phase transitions.
- [Ćuk & Stewart 2012][CS12], Science 338, 1047: the Moon from a fast-spinning Earth.
- [Haldemann et al. 2020][Ha20], A&A 643, A105: AQUA, a collection of H2O equations of state.
- [Kegerreis et al. 2019][Ke19], MNRAS, doi:10.1093/mnras/stz1606: giant impacts with SWIFT,
  spherical initial conditions.
- [Lock et al. 2018][Lo18], JGR Planets, doi:10.1002/2017JE005333: the Moon from a synestia.
- [Meier, Reinhardt & Stadel 2021][Me21], MNRAS, doi:10.1093/mnras/stab1441: convergence of
  collision simulations with resolution and equation of state.
- [Melosh 2007][Me07], Meteoritics & Planetary Science 42, 2079: a hydrocode equation of state
  for SiO2 (molecules in the vapor).
- [Miller & Puckett 1996][MP96], J. Comput. Phys. 128, 134: a Godunov method for multiple
  condensed phases.
- [Postema, Lock & Stewart 2026][Po26], arXiv:2606.23168: thermal and rotational effects of
  giant impacts; the pyrolite and Fe85Si15 tables in use.
- pyaneos v1.0, 2026: [doi:10.5281/zenodo.20736271][PyAz].
- [Root et al. 2018][Ro18], GRL 45, 3865: the principal Hugoniot of forsterite to 950 GPa.
- [Ruiz-Bonilla et al. 2021][RB21], MNRAS, doi:10.1093/mnras/staa3385: pre-impact spin in the
  moon-forming collision; layered initial conditions.
- Stewart 2020, the iron and Fe85Si15 models: [doi:10.5281/zenodo.3866507][Fez],
  [doi:10.5281/zenodo.3866550][FeSiz].
- Stewart et al. 2019, the forsterite model: [doi:10.5281/zenodo.3478631][Foz].
- Stewart 2022, the pyrolite model: [doi:10.5281/zenodo.6968539][Pyz]. Its reference is
  Stewart et al. 2022, 53rd Lunar and Planetary Science Conference, abstract 1535.
- Stewart 2024, the water model: [doi:10.5281/zenodo.14226693][Waz]; Stewart & Amodeo 2024,
  fused silica: [doi:10.5281/zenodo.14226685][Siz].
- [Stewart et al. 2020][St20], AIP Conference Proceedings 2272 (SCCM19),
  doi:10.1063/12.0000946, arXiv:1910.04687: the shock physics of giant impacts, key
  requirements for the equations of state. Article 080003.
- Thompson, Lauson, Melosh, Collins & Stewart 2019, M-ANEOS v1.0:
  [doi:10.5281/zenodo.3525030][MANz].
- [Thompson 1990][Th90], Sandia report SAND89-2951 (doi:10.2172/6939284): ANEOS input manual.
- [Thompson & Lauson 1972][TL72], Sandia National Laboratories: Improvements in the Chart D
  radiation-hydrodynamic code III, revised analytic equations of state. A copy is in the
  M-ANEOS repository's `docs/`. (Report number SC-RR-71 0714 is from memory.)

- [Wagner & Pruß 2002][WP02], J. Phys. Chem. Ref. Data 31, 387: the IAPWS-95 formulation for
  water (as cited in the water notebook).
- Kovačević et al. 2022, Scientific Reports 12, 13055: miscibility of rock and ice (as cited
  in the water notebook; used the water model in development).

From memory, to check before they are cited in `docs/numerics.md`:

- Canup 2005, Science 307, 546: a giant impact origin of Pluto and Charon.
- Senft & Stewart 2008, Meteoritics & Planetary Science 43, 1993: a five-phase equation of
  state for H2O in impact calculations.
- Mosenfelder, Asimow & Ahrens 2007, J. Geophys. Res. 112, B06208: shock data for forsterite
  and enstatite (named in the forsterite notebook as its gas-gun source).

[Ca04]: https://doi.org/10.1016/j.icarus.2003.09.028
[CM14]: https://github.com/isale-code/M-ANEOS/blob/master/docs/CollinsANEOSLPSC2014.pdf
[CS12]: https://doi.org/10.1126/science.1225542
[Fe]: https://github.com/ststewart/aneos-iron-2020
[Fez]: https://doi.org/10.5281/zenodo.3866507
[FeSi]: https://github.com/ststewart/aneos-Fe85Si15-2020
[FeSiz]: https://doi.org/10.5281/zenodo.3866550
[Fo]: https://github.com/ststewart/aneos-forsterite-2019
[Foz]: https://doi.org/10.5281/zenodo.3478631
[Ha20]: https://doi.org/10.1051/0004-6361/202038367
[Ke19]: https://doi.org/10.1093/mnras/stz1606
[Lo18]: https://doi.org/10.1002/2017JE005333
[MAN]: https://github.com/isale-code/M-ANEOS
[MANz]: https://doi.org/10.5281/zenodo.3525030
[Me07]: https://doi.org/10.1111/j.1945-5100.2007.tb01009.x
[Me21]: https://doi.org/10.1093/mnras/stab1441
[MP96]: https://doi.org/10.1006/jcph.1996.0200
[Po26]: https://arxiv.org/abs/2606.23168
[Py]: https://github.com/ststewart/aneos-pyrolite-2022
[Pyz]: https://doi.org/10.5281/zenodo.6968539
[PyA]: https://github.com/ImpactsWiki/pyaneos
[PyAz]: https://doi.org/10.5281/zenodo.20736271
[Ro18]: https://doi.org/10.1029/2017GL076931
[RB21]: https://doi.org/10.1093/mnras/staa3385
[St20]: https://arxiv.org/abs/1910.04687
[Th90]: https://doi.org/10.2172/6939284
[TL72]: https://github.com/isale-code/M-ANEOS/tree/master/docs
[Si]: https://github.com/ststewart/aneos-fused-silica
[Siz]: https://doi.org/10.5281/zenodo.14226685
[Wa]: https://github.com/ststewart/aneos-water
[Waz]: https://doi.org/10.5281/zenodo.14226693
[WP02]: https://doi.org/10.1063/1.1461829
