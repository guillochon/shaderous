# Plan: stars from MESA profiles

Status: proposed 2026-11-21. Nothing built.

## 1. Goal

A star part whose structure is read from a MESA profile (`profileN.data`) instead of solved as
a polytrope: its density, pressure, temperature, composition and, where MESA has it, rotation
as functions of radius, mapped onto the grid, settled into the code's own balance, and run with
the equation of state that fits its gas.

Every star in Shaderous today is a polytrope, a cold white dwarf or an ANEOS planet. The
problems below ask about things a polytrope leaves out: a core that is denser than any n = 3
star, an envelope whose entropy profile is not one power law, and layers of different
composition.

| Problem | Presets today | What a MESA star changes |
|---|---|---|
| Tidal disruption | `tidal-disruption*` (n = 5/3 or n = 3, ideal or hhe gas) | The fallback rate dM/dt and when a star is fully disrupted both depend on how concentrated it is; a main-sequence star becomes more concentrated with age. Law-Smith, Guillochon and Ramirez-Ruiz (2019, 2020) did this with MESA stars in FLASH. The debris diagnostic (`bodies.debris`) already measures dM/dE. Composition carried in the debris (He and CNO lines) is a result in its own right |
| Common envelope | `common-envelope`, `common-envelope-ideal` (n = 1.5 envelope around a cored point mass) | A real giant's envelope is not n = 1.5: its density falls off differently, and its binding energy, which decides whether the envelope is ejected, follows from that. The core is replaced by a point mass as today, but matched to the profile (section 4) |
| Roche lobe overflow and stellar collisions | `roche-lobe-overflow-stars`, the collisions the shaderous-sim skill builds | The donor's response to losing mass, and how the two stars mix, both depend on their density profiles; composition shows where each star's gas ends up |
| White dwarfs with a composition profile | `white-dwarf`, `sub-chandra`, `wd-collision*` | A MESA white dwarf has a C/O profile (oxygen-rich in the middle) and a helium layer of the mass the evolution leaves, instead of uniform C/O with a helium shell painted on (`helium-shell-ignition`). Warm white dwarfs, not only cold ones |
| A magnetized or radiating star | `polytrope-dipole`, `radiation-star` | The pressure profile the field is built from (`StarProfile`), and the temperature the radiation starts at, from a real star |

### Non-goals for v1

- MESA `.mod` files (the restart format), GYRE files and FGONG. Profiles only: they are what
  people already have, and what `profile_columns.list` controls.
- Reproducing MESA's equation of state. The run's equation of state is the code's own; section
  5 says what is kept when the two disagree.
- Resolving a giant's degenerate core. It is a point mass, as now (section 4).
- Non-spherical MESA output (MESA is 1D). Rotation is taken as shellular.
- Isotopes the code has no species for (section 6 says where they go).

## 2. What a profile file is

A MESA profile is text: a header of global values (line 2 names, line 3 values: `model_number`,
`star_age`, `star_mass`, `initial_z`, ...), then the column names on line 6 and one row per zone,
from the surface inward. Columns depend on the user's `profile_columns.list`.

| Need | Columns, in order of preference | If missing |
|---|---|---|
| Radius | `radius_cm`, `radius` (Rsun), `logR_cm`, `logR` (log Rsun) | Error |
| Density | `rho`, `logRho` | Error |
| Pressure | `pressure`, `logP` | From rho and T with the run's equation of state, with a warning |
| Temperature | `temperature`, `logT` | From rho and p with the run's equation of state |
| Enclosed mass | `mass` (Msun), `mass_grams` | Integrated from rho; compared with the header's `star_mass` when both exist |
| Composition | isotope columns (`h1`, `he3`, `he4`, `c12`, `n14`, `o16`, `ne20`, `mg24`, ...), else `x_mass_fraction_H`, `y_mass_fraction_He`, `z_mass_fraction_metals` | The run's composition everywhere |
| Rotation | `omega` (rad/s) | None |

Default MESA profiles have all of radius, `logRho`, `logP`, `logT`, `mass`, `x_mass_fraction_H`
and the isotopes of the network in use, so a plain run of MESA's `1M_pre_ms_to_wd` or
`make_co_core` test suite works with no change to `profile_columns.list`.

## 3. How a profile gets into a setup

### The document carries the profile

A setup document stays self-contained: the star part carries the profile it was built from,
already parsed and trimmed, not a path. A document is allowed 2 MB (`codec.ts` MAX_DOCUMENT); a
profile of 2000 zones by 8 columns written as JSON is about 250 kB, and section 3.2 trims it to
a few hundred.

```json
{ "kind": "star", "id": "star", "type": "profile",
  "profile": {
    "source": "MESA r24.08.1, 1M_pre_ms_to_wd, model 845 (age 4.6 Gyr)",
    "units": "cgs",
    "r":   [ ... ],        // cm, increasing (the file's order reversed)
    "rho": [ ... ],        // g/cm^3
    "p":   [ ... ],        // dyn/cm^2
    "T":   [ ... ],        // K, optional
    "X":   { "h1": [ ... ], "he4": [ ... ], "c12": [ ... ], "n14": [ ... ], "o16": [ ... ] },
    "omega": [ ... ]       // rad/s, optional
  },
  "core": { "radius": 0.05, "soft": 0.02 },
  "pos": [0, 0, 0] }
```

`mass` and `radius` of the part become readouts of the profile, not inputs: the profile's
radius and mass with the setup's units give them. A `scale` option (`{ "mass": 1, "radius": 1 }`
in code units) is allowed for the cases where the units are fixed by something else (two stars
of different MESA models in one run): the profile is then scaled homologously, rho by M/R^3, p
by G M^2 / R^4, and a note says so.

### Reading and trimming

`src/problems/mesa.ts` (no browser APIs, so Node and the page share it):

1. `parseMesaProfile(text)`: header and columns to typed arrays; errors that name the missing
   column; order reversed to increasing radius.
2. `trimProfile(prof, { zones, rCut })`: resample to at most `zones` points (default 400),
   uniform in a mixture of log p and enclosed mass so both the center and the surface keep
   points; drop zones outside `rCut` (default: the photosphere, MESA's outer boundary). Linear
   in log rho, log p, log T and in X between points.
3. `checkProfile(prof)`: monotone r, rho and p; X summing to 1 within 1e-6 (renormalized, with a
   note, otherwise); hydrostatic balance dp/dr = -G m rho / r^2 to the accuracy MESA keeps it,
   reported as the largest relative error.

### Where people get profiles

- **Import in the setup dialog** (`ui/setup/dialog.ts`): a star part of type `profile` has a
  "Load MESA profile" button (an `<input type="file">`, as `#import-file` for .athdf). The file
  is parsed and trimmed in the page and written into the document. Nothing leaves the browser.
- **The shaderous MCP tools and the headless runner**: the `load` tool and `tools/headless`
  take a `profile` path for a star part and read the file from disk (Node), so a script can set
  up a run on its own MESA output.
- **Shipped profiles**, for the presets of section 7: `src/presets/profiles/*.json`, already
  trimmed, referenced from a preset as `"profile": { "ref": "sun-zams" }` and resolved when the
  preset is read (as `extends` is). The MESA inlists that made them live in `tools/mesa/`, with
  the MESA version, so they can be made again. They are run once, on a Linux machine with the
  MESA SDK (a Runpod CPU pod is enough; see the shaderous-on-runpod memory for the pod
  routine); the build does not need MESA.
- **Links**: a `?setup=` link with an embedded profile is long. A preset with a shipped profile
  links by its `ref`; a document with an imported profile is saved to a file or the browser's
  library, and a link to it says it cannot carry the profile (as an .athdf import cannot).

## 4. The core: what to do with what the grid cannot resolve

A 1 Msun main-sequence star is resolved by a few hundred cells across. A red giant is not: its
degenerate helium core is 0.03 Rsun in a 50 Rsun envelope, five orders of magnitude denser. The
usual answer, and the one the cored polytrope already gives (problems/cored_polytrope.ts), is a
softened point mass for the core and a smooth envelope inside the cut that matches the profile
outside it. For a profile:

- **`core.radius`** (code units, or `"auto"`: the radius where the profile's density scale
  height is three cells of the finest level the star is refined to): the profile is kept as it
  is outside it.
- Inside it, the gas is replaced, after Ohlmann et al. (2017, A&A 599, A5): a softened point
  mass of `core.soft` and gas of a modified Lane-Emden polytrope in the
  softened potential, whose density, pressure and their gradients match the profile at the cut.
  This is the equation `coredPolytrope` already integrates, with boundary conditions at the cut
  instead of at the surface: integrate inward from the cut, shoot on the point mass and the
  polytrope's central density so that the enclosed mass at the cut is the profile's and theta
  is regular at the center. Ohlmann et al. pick n by the gradient's match; take n free in
  [1.5, 5] and solve for it the same way.
- The default `core.soft` needs choosing for this code's softening. Ohlmann et al. use a cubic
  spline softening, which is exactly Newtonian beyond a few softening lengths, and tie the cut
  radius to it (look up their ratio before building); the point masses here are
  Plummer-softened (gravity/bh.wgsl), which is never exactly Newtonian, so a Plummer
  length of about a third of the cut radius is the starting guess, to be checked by the
  red-giant test of section 9 (how far the point mass drifts and how much the envelope rings).
- The point mass gets the profile's mass inside the cut less the replaced gas's. A star without
  `core` keeps its whole profile (a main-sequence star, a white dwarf); a check warns when the
  center is resolved by fewer than four cells at the finest level.

## 5. Equation of state: what to keep when MESA's disagrees

MESA's equation of state (FreeEOS, OPAL, HELM, Skye, blended) is not the run's. At a given rho
and the profile's T, the run's pressure is not the profile's p, by a few percent in a hydrogen
envelope with the `hhe` table, by much more with an ideal gas. A star built on one of the two
cannot have both.

**Keep rho and p; let T follow.** Hydrostatic balance is in rho and p, and it is what the run
must start in: rho and p from the profile, the specific energy from the run's equation of state
at (rho, p, X), T whatever that makes it. This is what FLASH users of MESA stars do. The
check reports the largest |T_run / T_MESA - 1| over the star, mass weighted and at its worst,
and the gamma_1 of the two, so a mismatch that matters (gamma_1 far apart in an envelope,
which moves the star toward or away from convective instability) shows.

Which equation of state fits which star:

| Star | Equation of state | Composition carried |
|---|---|---|
| Main-sequence star, giant envelope | `hhe` (H, He, partial ionization, H2) | Fixed: the table is one (X, Y). The profile's surface X picks the table (solar, or hydrogen); a star whose X varies (a main-sequence star past its middle age, every giant core) gets passive species for its composition (section 6), but its equation of state does not follow them. The check reports the mass in gas whose X is more than 0.1 from the table's |
| Same, simpler | `ideal` with the mixture's per-cell mu, or `gas_radiation` | Varying: nuclide species set mu cell by cell (species/mixture.ts) |
| Stellar interior, massive star, white dwarf | `helmholtz` | Varying: Abar and Zbar from the species |

The `hhe` limit is the largest one. Its plan (plans/hydrogen-helium-eos.md, section 6) has
tables per composition; interpolating between two tables in X is the step that would lift it,
and is left to that plan.

## 6. Composition and species

- Isotopes the profile has and the code has (`SPECIES` in units/composition.ts: h1, he3, he4,
  c12, n14, o16, ne20, mg24, si28, ... ni56) become the run's species, in that order, unless the
  setup names its species (an `aprox13` run keeps its own thirteen, and the profile's isotopes
  are mapped into them).
- Isotopes the code does not have are folded into the nearest one it has by mass number, so the
  mass fractions still sum to one and Abar and Zbar move little: h2 into h1, li7 and be7 into
  he4, c13 into c12, n15 into n14, o17 and o18 into o16, f19 into ne20, na23 into ne20 or mg24,
  and every heavier one into the next alpha nucleus at or below it. The fold is listed in the
  check's notes.
- A setup can ask for fewer species (`profile.species: ["h1", "he4", "metals"]`): metals is then
  a tracer, not a nuclide, and the run's composition is a mixture again only if all are nuclides.
- Species fractions are painted by density, not by radius (section 7), so that a settled star's
  composition goes where its gas went.

## 7. Building the star in code

A new star type, `'profile'`, beside `'polytrope'`, `'white-dwarf'` and `'planet'` (STAR_TYPES in
setup/spec.ts), built in `setup/parts/bodies.ts`'s `buildStar`:

- **`rho(r)`**: the profile, interpolated in log rho over r (with the core's gas inside the cut).
- **`pressure(rho)` and the barotrope**: the profile's own p(rho). Density falls monotonically
  outward in every star this is meant for, so p is a function of rho along the profile, and the
  enthalpy h(rho) = integral of dp / rho along it is a table. A star is then settled exactly as
  a polytrope is (problems/settle.ts): h(rho) + phi = C. Settling moves gas: a cell whose
  density changes takes the pressure, temperature and composition of the profile at its new
  density, not at its radius, which keeps the star's p(rho), its entropy and its layers of
  composition consistent with each other through settling.
  - The barotrope's `gpu` kind does not exist for a table: amr/settle.wgsl knows polytropes and
    degenerate gas. A third kind, `table`, a 1D texture (or storage buffer) of (h, rho) with
    log spacing, read by linear interpolation. Until it exists, a profile star on an adaptive
    mesh is settled on the CPU, as a cored star is now, with the memory limit the comment in
    settle.ts names.
  - A star whose density is not monotone (a density inversion, which MESA does make at the
    edge of some convective zones and in some giant envelopes) is made monotone by taking the
    running minimum outward, with a note giving how much mass moved.
- **`profile()`** (StarProfile, for fields inside it): the profile's pressure.
- **Painting** (setup/compile.ts initOf): rho from settling, p from the barotrope, X from the
  profile at that density, and velocity from the star's motion plus omega(r) x r if the profile
  has rotation and the part asks for it (`rotate: true`). A rotating star is not settled for its
  rotation (only a rotating frame flattens stars now); MESA's rotation is slow enough in the
  stars above that this is a small error, and the check says how small (omega^2 r / g at the
  equator, at its largest).
- **Temperature**, where an option needs it (a hot spot, radiation's starting energy, burning):
  from the run's equation of state at the cell's rho, p and X (section 5), not the profile's T.
- **The atmosphere**: MESA's outer boundary is the photosphere, where the density is many
  orders above any ambient gas the code can carry. The star ends at its profile's last point;
  the ambient gas (`background`, `relativeTo` the star) is set from that point as it is from a
  polytrope's cut, and a polytropic tail (n = 3, matched in rho and p) can be added down to the
  ambient density with `tail: true`, so the edge is not a cliff.

## 8. Presets

Each in an existing family, as a variant beside the polytrope one, so the two can be compared
with the same knobs:

| Family | New variant | Profile shipped |
|---|---|---|
| Tidal disruption | "MESA Sun (ZAMS / 4.6 Gyr / TAMS)", the age a featured choice | 1 Msun at three ages, `hhe` gas, h1, he4, c12, n14, o16 as species |
| Tidal disruption | "MESA 3 Msun, middle age" | 3 Msun at half its main-sequence life |
| Common envelope | "MESA red giant" | 1 Msun at 50 Rsun on the red giant branch (or the 12 Msun red supergiant of Lau et al. 2022 for a heavier one), core cut at 5% of the radius |
| Sub-Chandrasekhar | "MESA C/O white dwarf with helium layer" | A 1 Msun C/O dwarf with its helium layer, `helmholtz`, aprox13 species |
| Roche lobe overflow | "MESA donor" | The 1 Msun middle-aged star as donor |
| A star at rest | "MESA Sun at rest" | The 4.6 Gyr Sun alone, the test of section 9 as a preset |

Presets stay small enough for the page: a shipped profile of 400 zones is about 40 kB of JSON,
loaded only with its preset.

## 9. Tests

- **Parsing** (vitest): a real MESA profile (a small one, trimmed, under `src/problems/__fixtures__`),
  every column variant of section 2 (log and linear, Rsun and cm), a missing required column,
  isotopes the code lacks, non-monotone density.
- **Round trip**: a polytrope written out as a MESA-shaped profile and read back as a `profile`
  star starts within a part in 1e4 of the polytrope part's state, cell for cell, and settles
  the same.
- **A star at rest** (GPU suite, on a static build per the memory on GPU tests): the 4.6 Gyr Sun
  on a 128^3 mesh with two levels, `hhe` gas, run for 5 dynamical times. Central density within
  1%, the largest velocity below a tenth of the surface's sound speed, mass in species
  conserved to round-off. The same with the cored red giant, where the measure is the point
  mass's drift and the envelope's oscillation.
- **Tidal disruption against Law-Smith et al.**: the critical beta for full disruption of the
  ZAMS and TAMS Sun, at modest resolution, within their quoted band (~0.9 and ~2.0); dM/dt's
  shape against their figures. This is the one test that says the import does what it is for.
- **Random setup testing** (the random-test skill): the `profile` type and its options added to
  the option table so seeds reach them, with a shipped profile as the source.

## 10. Steps

1. `problems/mesa.ts`: parse, trim, check (section 3), with its vitest tests. No setup changes.
2. The `profile` star type in spec.ts, options.ts (and the schema), bodies.ts and compile.ts,
   without a core and with CPU settling; species from the profile (section 6); the round-trip
   test.
3. The `table` barotrope in amr/settle.wgsl, so profile stars settle on the GPU.
4. Core replacement (section 4), reusing cored_polytrope.ts's integrator.
5. Import: the dialog's button, the headless runner's and MCP `load`'s `profile` path, and
   shipped profiles with `ref`.
6. Make the profiles with MESA (`tools/mesa/`: inlists, a script that trims and writes the
   JSON), and the presets of section 8.
7. The star-at-rest and tidal-disruption tests; `docs/setups.md` (the star part) and
   `docs/numerics.md` (settling a tabulated star, core replacement, which state is kept).

Steps 1 and 2 alone make a usable feature for main-sequence stars and white dwarfs on a uniform
or small mesh; the giants need step 4.

## 11. Questions to settle before building

| Question | Proposal |
|---|---|
| Keep p or T when the equations of state disagree | p (section 5); keeping T would put the star out of balance by the mismatch |
| Paint composition by density or by radius after settling | Density (section 7) |
| Ship MESA-made profiles, or only the import | Both; the presets need shipped ones, and the shipped ones need `tools/mesa/` to be reproducible |
| Default number of zones after trimming | 400; enough for the steepest gradient (the edge of a giant's core region, after the cut) at the finest resolution the page reaches |
| `hhe` with a composition that varies | Not in this plan; flagged by the check, and left to the hydrogen and helium plan |
| MESA version the shipped profiles come from | The current release when step 6 is done, recorded in each profile's `source` |
