# Setup files

A setup is everything that describes a run before it starts: the grid, the physics, the
boundaries, the initial conditions and the objects in the box. It is one JSON document. Every
problem of the preset menu is such a file (`src/presets/<id>.json`), and a setup you make in
the Setup dialog is the same kind of file. Either can be saved, loaded on another machine,
shared as a link, and passed to the headless runner and the MCP tools.

A file holds numbers and names, never code or formulas, so a file from a stranger is safe to
open. Reading one never trusts it: keys this build does not know are dropped and listed, values
out of range are held to it and listed, and a value of the wrong type is an error.

The source of truth for what a file may hold is the option table, `src/setup/options.ts`. The
dialog's forms, the conflict checks, the defaults and the ranges are all read from it.

## Shape of the document

```jsonc
{
  "format": "shaderous-setup",       // required
  "version": 1,                      // required: counts changes to the format
  "id": "binary",                    // presets: the name links and tests use
  "name": "Binary accretion",
  "group": "Point masses", "family": "binary", "familyName": "Binary accretion", "variant": "Standard",
  "blurb": "Two point masses orbiting inside a gas disk: …",
  "extends": "tidal-disruption",     // optional: give only the differences from a preset
  "domain": { … }, "physics": { … }, "mesh": { … }, "background": { … },
  "parts": [ … ], "run": { … }, "view": { … }, "featured": [ … ]
}
```

Every key but `format` and `version` is optional. What is absent takes the default of the
option table, so a file lists only what makes it that setup. Positions and sizes are in code
units of the setup's own box, which is the cube `[-box/2, box/2]³` centered on the origin.

### `domain`

| Key | Meaning |
|---|---|
| `ndim`, `n`, `amr` | The grid the setup opens on: 2 or 3 dimensions, cells per side (the finest level's on an adaptive mesh), levels of refinement. Absent: the grid already chosen is kept |
| `box` | Side of the box (default 1) |
| `supports2d` | The setup also makes sense on a 2D (x, y) grid |
| `bc` | `{ "lo": [x, y, z], "hi": [x, y, z] }`: 0 periodic, 1 outflow, 2 reflecting, 3 outflow only. Periodic faces come in pairs |
| `bc2d` | The same for a 2D run, where it differs |
| `units` | What one code unit is (below). Without it the setup is scale free |

Units are a length and a mass (the time unit then makes G = 1), or a length, a density and a
velocity:

```json
"units": { "name": "solar", "length": { "unit": "Rsun", "value": 1 }, "mass": { "unit": "Msun", "value": 1 } }
```

Units: `m`, `cm`, `km`, `Rsun`, `au`, `pc`; `kg`, `g`, `Msun`; `kg/m3`, `g/cm3`; `m/s`, `km/s`,
`cm/s`. A value may be `per` a code value, or per a part's quantity: `"per": "star.radius"`
says the star with id `star` has that many solar radii, whatever its radius in code units
(`<id>.mass` and `<id>.radius`).

### `physics`

`eos` (`ideal`, `barotropic`, `gas_radiation`, `degenerate`, `ionizing`, `helmholtz`) and
`eosParams` (`gamma`, `K`, …); `gravity` (`off`, `constant`, `periodic`, `isolated`) and `G` (a
number, or `"units"` for what the units make it); `gext`, a uniform acceleration as a vector or
`{ "down": g }` (along -y in 2D, -z in 3D); `cooling` and `coolParams`; `viscosity`,
`conductivity`; `riemann`, `limiter`, `cfl`; `floors` (`rho`, `p`); `fluff` (`rho`, `u`, `v`,
`va`: the limits on tenuous gas); `frame` (`inertial`, `gas`, `masses`); `jeansFloor`,
`jeansCells`, `vcycles`, `gravEnergyFlux`; `composition` (`solar`, `solarNeutral`, `hydrogen`,
`carbonOxygen`, or `{ "name", "X": { species: fraction }, "ionization" }`); `species` (names,
or `"aprox13"`), `speciesGamma`; `burn` (`{ "network": "aprox13", "inShocks", "limiter" }`).

### `mesh`, `run`, `view`

`mesh`: `refineHi`, `refineLo`, `refineRho`, `refineDense`, `refineMasses`, `timeBins`,
`jeansRefine`. `run`: `tEnd` (`null` runs forever), `checkpointDt`, `tDyn`, `startPaused`.
`view`: the view the setup opens with (`field`, `log`, `mode`, `axis`, `pitch`).

### `background`

The gas that fills the box where no part puts anything else: `rho`, `p`, `v`, `s` (the dye),
`X` (mass fractions by species name). With stars it is the gas around them, and `cut` is the
density below which a cell is ambient gas, not star; `relativeTo` names a star whose central
density `rho` and `cut` are fractions of. Left out, `p` is the pressure of the stars' outermost
layer, and a polytrope's `cut` is where the ambient gas would be as hot as the star's center.

## Parts

`parts` is a list. Each part has a `kind` and may have an `id`, which other parts and the units
refer to. The initial state is painted in layers: a flow that fills the box or the background,
then disks, then stars, then regions, obstacles and inflows in the order of the list.

| Kind | Fields |
|---|---|
| `mass` | A point mass: `gm` (G M), `pos`, `vel`, `soft` (softening, at least two cells), `sink` (radius; absent: the softening, 0: none), `sinkRate`, `massless`, `fixed` |
| `orbit` | Not an object: places two bodies, `of: [idA, idB]`. `type: "binary"`: two point masses about their center of mass at pericenter (`q`, `separation`, `ecc`, `incl`, `retrograde`). `"flyby"`: a point mass on a parabola about a star that stays at the origin (`q`, `beta` = tidal radius over pericenter, `start` in tidal radii, `debris`). `"collision"`: two stars on a parabola (`impact` in sums of their radii, `starts`) |
| `star` | `type: "polytrope"` (`n`, `radius`, and `mass` or `rhoc`) or `"white-dwarf"` (`mass`); `pos`, `vel`, `dye`, `X`; `field` (`type`: `dipole`, `random`, `gm17-dipole`, `gm17-random`; `beta`; `seed`). Stars are settled into the grid's own hydrostatic balance before the run |
| `disk` | `about: [ids]`, `profile` (`thin`: a surface density with a cavity and a Gaussian vertical profile; `balanced`: a midplane density in vertical balance with the softened potential), `rIn`, `rOut`, `sigma0`, `slope`, `cs`, `floor`, `damp` (`from`, `rate`: the damping zone), `loops` (`beta`, `span`, `count`: field loops), `stresses` |
| `region` | A `shape` with a state that replaces what is under it: `rho`, `p` (or `energy`, a total spread over a sphere's cells), `v`, `s`, `X`; `supersample` for a smooth edge |
| `obstacle`, `inflow` | A `shape` whose cells are solid, or held at their initial state |
| `field` | A uniform magnetic field: `b` (a vector) or `up` (a strength along the axis drawn upward) |
| `flow` | A built-in initial condition by `name`, with its parameters: what shapes cannot say (a shear layer with its seeded modes, a ZND detonation, an Alfvén wave). The flows are in `src/setup/parts/flows/` |

Shapes: `{ "type": "sphere", "center", "radius" }` (or `radiusCells`, in finest cells; a
cylinder in 2D), `{ "type": "box", "min", "max" }` (a bound may be `null`: open on that side),
`{ "type": "half", "normal", "at" }` (the side the normal points away from).

## What cannot be combined

Every option declares what it requires and what it excludes, as data in the option table. The
dialog greys out an option whose conditions are not met and names the reason in its tooltip. A
file that breaks a rule still opens; the dialog's issues list says which. The rules today:

| Option | Requires, excludes |
|---|---|
| Self-gravity (periodic or isolated) | 3D |
| Self-gravity, periodic | every face periodic; no star |
| Self-gravity, isolated | no periodic face |
| Star | 3D, isolated self-gravity, no periodic face |
| White dwarf | the degenerate or Helmholtz equation of state; no polytropic star |
| Polytropic star | an equation of state with an energy; no white dwarf |
| Point mass, disk | no periodic face |
| Disk | a point mass to orbit |
| Damping zone | one a run |
| Frame follows the gas or the masses | a point mass |
| Cooling, conduction, a region's total energy | an equation of state with an energy |
| Network burning | its 13 species, physical units, an equation of state with an energy |
| Refinement thresholds, local timesteps, refine around masses | an adaptive mesh |
| Jeans floor, V-cycles, energy-conserving work | self-gravity |
| A flow that fills the box | no other such flow; the background is not used |

Checks on values run with them and show in the same list: bodies that overlap or leave the
box, a softening under two cells, a disk heavier than a tenth of what it orbits, a polytrope
whose gas has gamma below 1 + 1/n, the grid against the GPU memory budget.

## Featured options

A setup may list the options that matter most to it:

```json
"featured": [
  "physics.viscosity",
  { "path": "parts.2.ecc", "id": "ecc", "label": "Eccentricity", "min": 0, "max": 0.8, "step": 0.05 },
  { "path": ["parts.0.soft", "parts.1.soft"], "id": "soft", "label": "Softening and sink radius" }
]
```

Each entry is a path into the document (or several that take one value), optionally with how
the side bar presents it. They are shown under the preset menus and at the top of the dialog.
An entry's `id` is also the name links know it by (`p.ecc`), so links made before presets were
files still open.

## Variants

A file may name a parent and give only the differences:

```json
{ "format": "shaderous-setup", "version": 1, "id": "sedov-rad", "extends": "sedov",
  "name": "Radiation-dominated blast", "variant": "Radiation dominated",
  "physics": { "eos": "gas_radiation" } }
```

Objects merge key by key; lists (`parts`, `featured`) replace the parent's whole. A file saved
from the dialog or downloaded from its preset list is always the resolved, whole document.

## Saving, loading and sharing

- **File**: `name.shaderous.json`, from Save file in the dialog. Open file reads one, and so
  does dropping it on the page.
- **Library**: named setups kept in this browser (`localStorage`), listed under the presets.
  Files are the durable copy.
- **Link**: a link names a preset and what differs from it, option by option. A setup that
  differs in its parts, or has no preset, travels whole in the link's `setup=` (minified,
  deflated, base64url). Past about 8000 characters Copy link offers the file instead.
- **State files**: an `.athdf` written by Export state carries the whole setup, so it restarts
  as itself whatever the preset files have become since.
- **Scripting**: `window.shaderous.load()`, the headless runner (`--setup file`) and the MCP
  `load` tool take a preset id with overrides, or a setup document. See [api.md](api.md) and
  [headless.md](headless.md).

## Versions

`version` counts changes to the format. The reader keeps one migration per step and applies
them in order, so an old file opens in a new build. A file from a newer build opens with a
warning and a list of what was left out.
