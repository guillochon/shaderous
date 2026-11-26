# Setup files

A setup is everything that describes a run before it starts: the grid, the physics, the
boundaries, the initial conditions and the objects in the box. It is one JSON document. Every
problem of the preset menu is such a file (`src/presets/<id>.json`), and a setup you make in
the Setup dialog is the same kind of file. Either can be saved, loaded on another machine,
shared as a link, and passed to the headless runner and the MCP tools.

A file holds numbers, names and, for a disk's density, formulas of a small fixed language that
is parsed and never run as code, so a file from a stranger is safe to open. Reading one never trusts it: keys this build does not know are dropped and listed, values
out of range are held to it and listed, and a value of the wrong type is an error.

The source of truth for what a file may hold is the option table, `src/setup/options.ts`. The
dialog's forms, the conflict checks, the defaults and the ranges are all read from it.

The format is also published as a JSON Schema, made from the same table:
[shaderous-setup.schema.json](https://guillochon.github.io/shaderous/shaderous-setup.schema.json)
(`public/shaderous-setup.schema.json` in the repository). A file that names it in a first key
`"$schema"` gets its keys completed and its values checked in an editor that reads JSON Schema
(VS Code does). Files saved from the dialog name it. The key is not part of the setup: reading
skips it. The schema is stricter than reading in one way: a key this build does not know is an
error in the editor, where reading drops it with a note.

## Shape of the document

```jsonc
{
  "format": "shaderous-setup",       // required
  "version": 1,                      // required: counts changes to the format
  "id": "binary",                    // presets: the name links and tests use
  "name": "Binary accretion",
  "group": "Accretion disks", "family": "binary", "familyName": "Binary accretion", "variant": "Standard",
  "blurb": "Two point masses orbiting inside a gas disk: …",
  "extends": "tidal-disruption",     // optional: give only the differences from a preset
  "domain": { … }, "physics": { … }, "mesh": { … }, "background": { … },
  "parts": [ … ], "run": { … }, "view": { … }, "featured": [ … ]
}
```

The sections below say what each part of the document is for. Every key, with what it may
hold, its default and what it does, is listed under [Every key](#every-key), which is written
from the option table.

Every key but `format` and `version` is optional. What is absent takes the default of the
option table, so a file lists only what makes it that setup. Positions and sizes are in code
units of the setup's own box, which is the cube `[-box/2, box/2]³` centered on the origin.

### `domain`

| Key | Meaning |
|---|---|
| `ndim`, `n`, `amr` | The grid the setup opens on: 2 or 3 dimensions, cells per side (the finest level's on an adaptive mesh), levels of refinement. Absent: the grid already chosen is kept |
| `box` | Side of the box (default 1) |
| `supports2d` | The setup also makes sense on a 2D (x, y) grid |
| `geometry`, `rMin`, `logR` | `cartesian` (default), or a 2D run in `cylindrical` (R, z) or `spherical` (r, theta) coordinates; a spherical run's inner radius, and whether its cells are spaced evenly in ln r |
| `symmetric`, `symmetricField`, `mirror` | The mirrors the setup is symmetric under (`"x"`, `"xy"`, …), whether its magnetic field is too, and the mirrors a run uses to compute only the part above them |
| `bc` | `{ "lo": [x, y, z], "hi": [x, y, z] }`: 0 periodic, 1 outflow, 2 reflecting, 3 outflow only, 4 line-tied wall (no slip, and it holds the magnetic field lines that cross it), 5 moving wall (a line-tied wall sliding along itself at the velocity `wallVelocity` gives its face). Periodic faces come in pairs |
| `wallVelocity` | `{ "xlo": [vx, vy, vz], "xhi": ..., "ylo": ..., "yhi": ..., "zlo": ..., "zhi": ... }`: the velocity of each face that is a moving wall, along it |
| `bc2d` | The same for a 2D run, where it differs |
| `units` | What one code unit is (below). Without it the setup is scale free |

Units are a length and a mass (the time unit then makes G = 1), or a length, a density and a
velocity:

```json
"units": { "name": "solar", "length": { "unit": "Rsun", "value": 1 }, "mass": { "unit": "Msun", "value": 1 } }
```

Units: `m`, `cm`, `km`, `Rearth`, `Rsun`, `au`, `pc`; `kg`, `g`, `Mearth`, `Msun`; `kg/m3`,
`g/cm3`; `m/s`, `km/s`, `cm/s`. A value may be `per` a code value, or per a part's quantity:
`"per": "star.radius"` says the star with id `star` has that many solar radii, whatever its radius in code units
(`<id>.mass` and `<id>.radius`).

### `physics`

`eos` and `eosParams` (`gamma`, `K`, the liquid's `liqC` and `liqRho`, the hydrogen and helium
gas's `hheRad`, the ANEOS material `aneosA`, …); `gravity` and `G` (a number, or `"units"` for
what the units make it); `gext`, a uniform acceleration as a vector or `{ "down": g }` (along -y
in 2D, -z in 3D); `cooling` and `coolParams`; `radiation`, `opacity` and `opacityParams`, `radC`
(the speed of light), `radChat` (its reduced value), `radLocalC`, `radSolver`, `radSweeps`;
`viscosity`, `conductivity`, `resistivity`; `driving` (turbulence driving); `riemann`,
`limiter`, `cfl`; `pressure` (`explicit` or `implicit`) with `pressureCycles`, `pressureTol`
and `acousticCfl`; `floors` (`rho`, `p`); `fluff` (`rho`, `u`, `v`, `va`: the limits on tenuous
gas); `startDamping` (`rate`, `time`: every cell's velocity damped at `rate` per unit time at the
start, falling smoothly to zero at `time`, so that settled stars do not ring); `frame`, `rotation`, `shear` (a shearing box), `cosmology` (comoving coordinates);
`jeansFloor`, `jeansCells`, `vcycles`, `gravEnergyFlux`; `composition` (a name, or
`{ "name", "X": { species: fraction }, "ionization" }`); `species` (names, or `"aprox13"`; with
ANEOS, material names such as `pyrolite` and `fe85si15`), `speciesGamma`; `burn`
(`{ "network": "aprox13", "inShocks", "limiter", "switchAt" }`, or `{ "fuel", "ash", "q", "ea", "k" }`);
`toolScale`. The choices each takes are under [Every key](#every-key).

**Watching for burning.** The network's thirteen species cost a run their memory long before
anything burns: a helium donor and a carbon-oxygen accretor need only he4, c12 and o16 until
the stream heats the accreted helium. A setup with `"species": ["he4", "c12", "o16"]` and the
network's burning does not burn. In place of the network's pass a watch (burn/watch.ts) works
out, in each cell hot and dense enough to burn, the energy the network's reactions would
release over the step: its eight reactions that these three species take part in, each rate the
term the network takes as an input, within 30% of what the network releases over a short step.
Once that passes `switchAt` (default 0.1) of a cell's internal energy, the run stops (the
runner's `stopped` is `burn`) and writes its last outputs, a checkpoint among them. It carries
on from there with all thirteen species: `--from FILE --switch-burn` sets the species to
`aprox13`, the other ten starting at zero, and the network burns. The chain of jobs
(tools/headless/slurm/dwd.sbatch) does this by itself. With more species each block is larger
and the budget holds fewer: the checkpoint's mesh is rebuilt in the order of the refinement
zones' tiers, so that what gives way is the lowest tier's.

The equations of state (`eos`) are `ideal`, `barotropic`, `liquid` (weakly compressible,
p = c₀² (ρ - ρ₀)), `gas_radiation`, `degenerate`, `ionizing`, `helmholtz` (white dwarf
matter), `hhe` (hydrogen and helium with H₂, its dissociation and the ionization of both, from
tables for the solar compositions and pure hydrogen; its hydrogen fraction comes from
`composition`) and `aneos` (rock, iron alloy and water, which needs physical units). With
`aneos`, a run whose two species are materials holds both, the second's mass fraction in each
cell setting the mix; a planet with a core makes them so.

### `mesh`, `run`, `view`

`mesh`: `refineHi`, `refineLo`, `refineRho`, `refineDense`, `refineMasses`, `timeBins`,
`jeansRefine`, `zones` (below). `run`: `tEnd` (`null` runs forever), `checkpointDt`, `tDyn`, `startPaused`.
`view`: the view the setup opens with (`field`, `log`, `mode`, `axis`, `pitch`).

`mesh.zones` (adaptive mesh) are parts of the box kept at the finest level, each with a tier
that says who gives up blocks first when the block budget runs out: tier 1 is kept longest, and
gas in no zone is given up before any zone's. A block in a zone is refined whatever its gas and
never merged with its siblings; blocks are refined best tier first, and room is made from the
lowest tier up, never from a tier above the block that needs it (within a tier, as without
zones, only from blocks flagged well below it). A point is in the first zone in the list that
holds it, so a zone listed later yields to those before it where they overlap. Each zone is
`{ "shape": "sphere", "at", "r", "rIn", "radii" }`, a sphere or shell about `at`, or
`{ "shape": "cone", "from": { "at", "r", "radii" }, "to": { ... } }`, the cone tangent to two
such spheres, bridging them; with `tier`, an optional `name`, and `below` or `above`
(`{ "y": 0 }`: only the side of the plane y = 0 below or above it, in the box's coordinates).
`at` is `"L1"`, a circular binary's inner Lagrange point, or a star's id; `r` and `rIn` are in
code units, or in the star's radii with `radii`. The zones stay where the setup places the
bodies: in the frame that follows a binary that is where they are. The run's status line gives
each tier's share of blocks at the finest level. When the starting mesh cannot hold every zone
at the finest level, the lowest tier's zones are left out of it first, then the next tier's,
and a level is left out (as a self-gravitating run does rather than half fill one) only once no
zone is left; the run then keeps them all, and its tiers decide who gives up blocks. A zone that
gives way gives way whole, dense gas included: a star in a low tier can lose its finest level
altogether to the zones above it. In the Setup dialog the zones are on the Mesh page, one card
each, moved up and down as parts are; the preview shades them by tier where the box's middle
planes cut them, and a click on one opens its card. **Add binary zones** puts in the set below
for the setup's circular binary of two stars, its sphere about L1 a quarter of the donor's
radius. For the helium donor of a double white dwarf, seen from above with the donor at -x:

```json
"zones": [
  { "name": "L1", "shape": "sphere", "at": "L1", "r": 0.25, "tier": 1 },
  { "name": "shell, stream side", "shape": "sphere", "at": "accretor", "rIn": 1, "r": 2, "radii": true, "below": { "y": 0 }, "tier": 2 },
  { "name": "shell, far side", "shape": "sphere", "at": "accretor", "rIn": 1, "r": 2, "radii": true, "above": { "y": 0 }, "tier": 3 },
  { "name": "accretor", "shape": "sphere", "at": "accretor", "r": 1, "radii": true, "tier": 4 },
  { "name": "donor", "shape": "sphere", "at": "donor", "r": 1, "radii": true, "tier": 5 },
  { "name": "stream", "shape": "cone", "from": { "at": "L1", "r": 0.25 }, "to": { "at": "accretor", "r": 2, "radii": true }, "below": { "y": 0.25 }, "tier": 1 },
  { "name": "neck", "shape": "cone", "from": { "at": "L1", "r": 0.25 }, "to": { "at": "donor", "r": 1, "radii": true }, "tier": 1 }
]
```

The two cones come last, so the spheres take precedence where they overlap: the stream's cone
is the first tier between L1 and the accretor's shell, and the neck's between the donor's
surface and L1.

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
| `mass` | A point mass: `gm` (G M), `pos`, `vel`, `soft` (softening, at least two cells), `sink` (radius; absent: the softening, 0: none), `sinkRate`, `massless`, `fixed`, `onOrbit` (held on its orbit: below) |
| `orbit` | Not an object: places two bodies, `of: [idA, idB]`. `type: "binary"`: two bodies, point masses or stars, about their center of mass at pericenter (`q`, `separation` or `fill`, `ecc`, `incl`, `retrograde`, `corotating`: below). `"flyby"`: a point mass on a parabola about a star that stays at the origin (`q`, `beta` = tidal radius over pericenter, `start` in tidal radii, `debris`). `"collision"`: two stars or planets on a parabola (`impact` in sums of their radii, `starts`) |
| `star` | `type: "polytrope"` (`n`, `radius`, and `mass` or `rhoc`), `"white-dwarf"` (`mass`) or `"planet"` (`mass`, `material`, and for a core `coreMaterial` and `coreFraction`; `surfaceT`, the kelvin its layers' adiabats start from: a planet needs the `aneos` equation of state and physical units, and its materials are the run's species, the mantle's first); `core` (`mass`, `soft`: a giant, with that much of a polytrope's mass a point mass at its center, which joins the run's point masses as `<id>.core`); `pos`, `vel`, `dye`, `X`; `field` (`type`: `dipole`, `random`, `gm17-dipole`, `gm17-random`; `beta`; `seed`). Stars are settled into the grid's own hydrostatic balance before the run. Self-gravity (isolated) and stars need 3D or an axisymmetric (R, z) run, where a star sits on the axis |
| `disk` | `about: [ids]` (point masses, stars or both; the disk is centered on their center of mass and moves with it), `profile` (`thin`: a surface density with a cavity and a Gaussian vertical profile; `balanced`: a midplane density in vertical balance with the softened potential), `rIn`, `rOut`, `sigma0`, `slope`, `cs`, `floor`, `radialFn`, `verticalFn`, `verticalIn` (a density profile of your own: below), `damp` (`from`, `rate`: the damping zone), `loops` (`beta`, `span`, `count`: field loops), `stresses` |
| `region` | A `shape` with a state that replaces what is under it: `rho`, `p` (or `energy`, a total spread over a sphere's cells), `v`, `s`, `X`; with radiation transport `erad` and `frad`, the radiation's energy and flux; `supersample` for a smooth edge |
| `obstacle`, `inflow` | A `shape` whose cells are solid, or held at their initial state |
| `field` | A uniform magnetic field: `b` (a vector) or `up` (a strength along the axis drawn upward) |
| `tunnel` | A wind tunnel: `speed`, `axis`, `mach`, `rho`, `sides`, `floor`, `sponge`, `inlet`, and the coefficients' reference `area` and `length` (below). At most one |
| `flow` | A built-in initial condition by `name`, with its parameters: what shapes cannot say (a shear layer with its seeded modes, a ZND detonation, an Alfvén wave). Every flow has parameters (densities, speeds, amplitudes, wavelengths, seeds), listed with their ranges in the dialog and by `describe().setup.flows`; left out, each takes the value its preset has always had. The flows are in `src/setup/parts/flows/` |

Shapes: `{ "type": "sphere", "center", "radius" }` (or `radiusCells`, in finest cells; a
cylinder in 2D), `{ "type": "box", "min", "max" }` (a bound may be `null`: open on that side;
`rotate`, degrees about its center about x, then y, then z),
`{ "type": "half", "normal", "at" }` (the side the normal points away from), and
`{ "type": "mesh", "file", ... }`, a body from a mesh file (below).

### Bodies from mesh files

A shape of type `mesh` is the inside of a triangle mesh: an obstacle in the shape of a car, a
wing, a rabbit (or a region of gas in that shape).

```json
{ "kind": "obstacle", "id": "car",
  "shape": { "type": "mesh", "file": "car.stl", "fit": 0.3, "center": [-0.5, 0, 0], "rotate": [0, 0, 10], "up": "z" } }
```

| Key | Meaning |
|---|---|
| `file` | The file, by its name beside the setup document (the runner, the MCP server's `dir`), a file dropped on the Setup dialog, or a URL. STL (binary or ASCII), OBJ, PLY (ASCII or binary), glTF or GLB (uncompressed). For 2D runs also an airfoil's coordinates (`.dat`, Selig or Lednicer order) or an SVG path |
| `node` | A glTF node or an OBJ group, with what is below it; the whole file if absent |
| `fit` or `scale` | The longest side as a fraction of the box, or a factor on the file's lengths. STL has no units: it needs one of the two (`fit` 0.25 is taken otherwise, with an error from the check) |
| `center` | Where the middle of the mesh's bounding box goes |
| `rotate` | Degrees about x, then y, then z, as for a box |
| `up` | Which of the file's axes is up: `y` for glTF and most 3D programs, `z` (the default) for most CAD. It becomes the box's z, or y in 2D |

In 2D the body is the mesh's cross-section through the plane z = 0, after it is placed. STEP and
IGES files hold exact surfaces, not triangles, and are not read: export STL or glTF from the
CAD program (FreeCAD: File > Export, Mesh formats; Blender: File > Export > glTF or STL;
Fusion and Onshape: export as STL or OBJ).

The check reports a missing file, a part thinner than two finest cells (a wall thinner than that
is not a wall), and a mesh so broken that what is inside is a guess. Holes, flipped triangles
and pieces that pass through each other are taken care of
([numerics.md](numerics.md#26-bodies-in-a-stream)). On the page, the Setup dialog's
"Open file…" button and dropping files on the dialog add mesh files: each is kept in the
browser, and one the setup does not name yet becomes an obstacle. A link to such a setup
carries the file's name, and finds the file again only in a browser that has it (or by URL).

### A wind tunnel

A `tunnel` part makes the box a wind tunnel, and replaces what the document says of the box's
faces, the gas and the background:

```json
{ "kind": "tunnel", "speed": 1, "axis": "x", "mach": 0.1, "sides": "slip", "floor": "none", "sponge": 0.15 }
```

- The stream flows along `axis` toward + at `speed`. The cells within `inlet` cells (of the coarsest level, on blocks)
  (default 2) of the upstream face are held at the stream's state; the downstream face is open,
  with a sponge over its last `sponge` of the box's length (default 0.1; 0 for none): a damping
  zone that pulls the gas back to the stream's state, its rate rising as the square of the
  depth to 4 U / (sponge x box) at the face, so that a wake leaves without reflecting.
- `sides` (and the roof): `slip` walls (the default), `open` faces, or `periodic`. `floor`, the
  lower face along the box's up axis (z in 3D, y in 2D): `none` (as the sides), `slip`, `fixed`
  (no slip) or `moving` (a wall moving with the stream: a car on a road).
- The gas: below Mach 0.3 (`mach`, default 0.1) the liquid with implicit pressure, of density
  `rho` and sound speed speed / mach; above, an ideal gas (gamma 1.4, explicit) at that Mach
  number. The whole box starts at the stream's speed.
- Viscosity is the document's own (`physics.viscosity`): the Reynolds number is the speed
  times the body's size over it.
- The diagnostics' force coefficients are over the stream's dynamic pressure and a reference
  area: `area`, or the first obstacle's shadow along the stream (in 2D a length). Drag is
  along the stream, lift along the box's up axis. The check warns when the first obstacle
  blocks more than 5% of the tunnel's cross-section.

A rotor (3D): `rotor` is an angular velocity about the stream's axis through the box's center.
The run is then in the frame that turns with it, so that a rotor centered there is at rest and
its torque and thrust come steady: the stream, the inlet and the sponge's target are
U - Omega x r, and the cells outside the cylinder the box holds are held at that state. Only
what is symmetric about the axis is steady in that frame (no tower, no ground, no yaw).
A mesh rotor goes on the axis with its shape's `"anchor": "origin"` (the file's origin, its hub,
at `center`; the default centers its bounding box, which for three blades is off the axis). The
diagnostics then add the power coefficient C_p: Omega times the torque about the axis over
rho U^3 pi R^2 / 2, with R the first obstacle's reach from the axis (positive when the wind
drives the rotor; the Betz limit is 0.593).

The `wind-turbine` preset is such a rotor: `builtin:turbine`, three blades and a spherical
spinner made in code (scale 0.8, so R = 0.8), on the uniform 128³ grid of a box of 3 (cells of
R/34), turning at Omega = 1.25, a tip-speed ratio of 1. The blades are wide (chord 0.4 R at the
root to 0.25 R at the tip, 14 to 8.5 cells), 20% thick and never under 0.06 R (2 cells), and
pitched for that ratio with a 10 degree angle of attack. Measured (2026-11-20, in the page):
torque +0.035 about the axis, thrust 0.29, C_p 0.044, the same at t = 15 and 30; 7.4 ms a step
(about 135 steps a second). It is a slow, inefficient rotor on purpose: a blade two or three
cells thick turns into a staircase, which shows a pitch of 20 to 40 degrees but not the 3 to 10
degrees of a fast rotor's outer blade. Pitched for a tip-speed ratio of 4, the same rotor was
nearly flat to the grid, felt drag only and was braked by the wind (C_p -0.3 to -2.9) at any speed. On
the adaptive mesh it fails too: the coarsest level sees a sub-cell blade in pieces, so the
surface that asks for refinement is never all there (the preset is on the uniform grid for this).
Over the rotor's speed (headless, at t = 2): C_p 0.038 at Omega = 0.6, 0.048 at 0.9, about 0 at
1.6 and -0.12 at 2.2, where the wind brakes it. The rotor is a hole in the gas (the blades'
fractional cells are plan stage 7). The picture is in the turning frame: the rotor stands
still. The View section's Turn with frame (link `turn=1`) turns the box at the frame's rate, so
the volume view is the one a camera at rest sees, the rotor turning and its wake left behind.

Yaw and angle of attack turn the body (its shape's `rotate`), not the stream. The `wind-tunnel`
preset puts a built-in car-like body (`"file": "builtin:ahmed"`, with the Ahmed body's
proportions; `builtin:ahmed-35` slants at 35 degrees) in a tunnel.

### Obstacles that move

An obstacle may move (a sphere or a box, on the uniform grid, for now):

```json
{ "kind": "obstacle", "id": "paddle", "shape": { "type": "box", "min": [-0.05, 0.1, -0.2], "max": [0.05, 0.8, 0.2] },
  "motion": { "spin": { "axis": [1, 0, 0], "about": [0, 0, 0], "rate": "2 * min(t, 1)" } } }
```

| Motion | Keys |
|---|---|
| `spin` | `axis`, `about` (a point on it), `rate` in radians per unit time |
| `slide` | `velocity` |
| `swing` | `axis`, `about`, `amplitude` in degrees, `period`, `phase` in degrees |
| `path` | `pos` and `rotate` (degrees about x, then y, then z, about the point `about`), each a formula of t |

Rates, velocities and paths may be numbers or formulas of `t` (a rotor spun up over a turn).
`parent` names another obstacle whose motion this one's is made on top of: its pivots move with
the parent (a blade that pitches on a hub that turns). The body's walls move: the gas is pushed
and dragged, and the body does work on it (the energy is not conserved, as with a moving wall
of the box). Its cells change from gas to solid and back as it goes, a cell at a time; mass is
conserved as they do ([numerics.md](numerics.md#26-bodies-in-a-stream)).

### Obstacles on an adaptive mesh

On blocks, an obstacle refines the mesh from its own shape: every block within `refineBand`
finest cells (4 by default) of its surface is kept at `refineLevel` (the finest by default),
from the start and for the whole run, and a block made there by refinement has its solid cells
painted from the shape at its own level. A part thinner than a coarse cell (a blade, a plate),
which the coarse levels see in pieces or not at all, is so whole at the finest level. The band
is wider than the reach of the ghost cells, so the flow next to the body never sees a coarser
level's idea of it. `"refine": "swept"` refines everywhere the surface passes as the obstacle
moves (over the run, or one period of its motions when the run has no end): what a rotor's
blades sweep. `"refine": "off"` leaves the obstacle to the flow's own refinement, as before.

### Partly filled cells

With `"physics": { "cutCells": true }` (link key `cut`) an obstacle cuts the cells its surface
crosses: each cell holds gas in a share of its volume and each face is open in a share of its
area, taken from the obstacle's own shape, so a body keeps its true outline below a cell (a
thin blade, its pitch) where it is otherwise a staircase of whole cells. The wall pushes on the
gas with the cell's pressure, its shear is taken from the velocity a cell and a half out from
it, and a cell holding little gas is merged with its neighbors after each stage of the step.
For the liquid with implicit pressure (the tunnel below Mach 0.3), on the uniform grid, with
obstacles at rest, so far. A cell holding under 5% gas counts as solid.

### Forces on obstacles

Every `obstacle` is a body, and the diagnostics report the force the gas exerts on each (as
`bodies`: the force, its pressure and viscous parts, and the torque about the body's center),
in the page's panel, the runner's `diag.jsonl` and the MCP `diagnostics` tool. The first four
obstacles are reported apart, in the order of the list, and any after them together with the
fourth. The force is the momentum the gas lost to the body's walls over the last step
([numerics.md](numerics.md#26-bodies-in-a-stream)). Not in axisymmetric or spherical runs.

### A point mass held on its orbit

`"onOrbit": true` on a point mass holds it on its orbit: the other point masses move it and the
gas does not. Neither the gas's pull nor what its sink takes changes its motion, and it does not
grow, so a binary with both masses held keeps the orbit it was given, as in most studies of
circumbinary disks. Its sink still takes gas, and what it took is still counted in the
diagnostics. It can be switched while a run goes on: the Binary accretion preset has it as
**Hold the orbit** beside its other fields.

### Stars in a binary

A `binary` orbit may hold a star and a point mass, or two stars, in the x-y plane with their
center of mass at the box center:

```json
{ "kind": "star", "id": "donor", "type": "polytrope", "n": 1.5, "mass": 1, "radius": 1 },
{ "kind": "mass", "id": "accretor", "gm": 0.7, "soft": 0.06, "sink": 0.06 },
{ "kind": "orbit", "of": ["donor", "accretor"], "type": "binary", "fill": 1.03, "corotating": true }
```

- The masses are the parts' own; `q` sets a point mass's against the star's.
- `fill` takes the place of `separation`: the first body, a star, is that many times the size
  of its Roche lobe, by Eggleton's (1983) formula. At 1 it just fills it; above 1 it pours gas
  through the inner Lagrange point L1 onto its companion from the start (Roche lobe overflow).
  The checks warn of a star that overflows when `fill` does not say so.
- On a circular orbit the stars are tidally locked: they are settled in the Roche potential (the
  grid's own, the point masses' and the centrifugal term of the frame turning with the orbit),
  so that each starts stretched toward its companion in hydrostatic balance, and each reaches
  no farther than its L1 point. In an inertial frame their gas then turns with the orbit.
- On a circular orbit the angular velocity is found as the stars settle. Two extended stars,
  each stretched toward the other, pull harder than two points of their masses: started at the
  point-mass Omega they fall toward each other and swing in and out on an eccentric orbit (two
  white dwarfs of 0.6 and 0.9 solar masses at `fill` 0.935 came 2.7% closer at pericenter). So
  the orbit turns at the Omega that balances the pull of the settled stars on each other, a
  percent or so above the point-mass one: the frame turning with it, or the stars' velocities in
  an inertial frame (not when a point mass is part of the orbit there, whose velocity is not the
  settling's to change). A snapshot keeps it. The separation, and so `fill`, are as placed.
- `"corotating": true` runs in the frame turning with the orbit (as `physics.rotation` would,
  at the orbit's angular velocity): every body starts at rest, and a binary in balance stays
  still. Not with a disk, whose gas is set up moving in an inertial frame.
- `"track": true` with it makes the frame follow the orbit. A corotating frame turns at the
  orbit's starting rate, so a binary whose orbit changes (mass transfer, drag, gas lost through
  the faces) drifts round the grid. Following it, the frame turns faster or slower to keep the
  line between the two bodies where it started, and the gas and the point masses feel the Euler
  force of its changing turn. The line it keeps is the axis of the binary's quadrupole in the
  orbital plane: that of the stars' gas (denser than where they end, `background.cut`, so that
  a stream, a disk or an outflow does not turn it) and the point masses, about their center of
  mass. No gas needs to be told apart as one star's or the other's, so any two bodies can be
  followed: two stars, a star and a point mass, two point masses. The frame's angular velocity is
  in the diagnostics (`frame`), and a snapshot keeps it.
- `"release": true` with `track` lets the frame go after a merger. Once one star is torn apart
  over the other, the mass has no axis left to keep, and a frame still keeping it turns at
  random. Released, the frame watches how much the mass still looks like a binary,
  e = sqrt(A² + B²) / (I_xx + I_yy) of the same gas and masses (1 for two points, 0 for a body
  symmetric about z; 0.85 to 0.94 for a pair before it merges, 0.2 to 0.4 for the remnant after),
  smoothed over half an orbit and never let rise. As it falls from 0.7 to 0.3 the frame hands
  over from keeping the axis to slowing down by its starting rate per orbit, so it stops within
  an orbit, and once it has stopped the run
  carries on in an inertial frame: the energy and momentum in the diagnostics are then the true
  ones. A snapshot taken before the release, loaded with `release` set, is released from there.
  The diagnostics' `frame.release` says how far it has got.
- An eccentric binary with a star places it at pericenter as two point masses are placed, and
  settles each star in its own potential, as a star falling freely.

The Roche lobe overflow preset is a Sun-like star at `fill` 1.03 beside a 3 solar mass black
hole. Whether the transfer is stable is the setup's to choose: a convective (n = 1.5) donor
heavier than about 0.63 of its accretor swells as it loses mass while its lobe shrinks, and the
transfer runs away; at a mass ratio of 1.4 the pair merged within two orbits. Its variant
**Two stars** (`roche-lobe-overflow-stars`) puts two resolved stars in a followed frame: a
Sun-like star filling its lobe beside a 0.6 solar mass star of 0.6 solar radii.

### Disk profiles of your own

A disk's density in radius and in height can be a formula in place of the built-in laws:

```json
{ "kind": "disk", "about": ["star"], "profile": "balanced", "rIn": 1, "rOut": 4, "sigma0": 1e-3, "cs": 0.1,
  "radialFn": "sigma0 * (r / rIn)^-1 * exp(-(r / rOut)^2) * step(r - rIn)",
  "verticalFn": "1 / cosh(z)^2" }
```

- `radialFn` is the surface density (`thin`) or the midplane density (`balanced`) as a function
  of the radius `r`. It may also name `rIn`, `rOut`, `sigma0`, `cs` and `H` (the scale height
  cs / Ω at `r`). Where it is not a positive number there is no disk. The rotation of a
  `balanced` disk follows from it: the pressure term uses its logarithmic slope.
- `verticalFn` is the density relative to the midplane as a function of the height `z` above
  it, the same on both sides. `z` is in scale heights of each radius, so the profile flares
  with the disk, or in code lengths with `"verticalIn": "length"`. A `thin` disk keeps its
  surface density whatever the shape. A disk with a vertical profile of its own is in vertical
  balance only if the profile is, and settles as the run starts.

A formula is arithmetic (`+ - * /`, powers as `x^y`, parentheses), numbers, the names above,
`pi` and `e`, and these functions: `exp`, `log` (or `ln`), `log10`, `sqrt`, `abs`, `sin`, `cos`,
`tan`, `atan`, `sinh`, `cosh`, `tanh`, `step(x)` (1 where x is positive: a sharp edge),
`pow(x, y)`, `min(a, b)`, `max(a, b)`, `clamp(x, lo, hi)`. It is parsed, never run as code, so a
file with formulas is as safe to open as any other; anything else in it is an error that names
the place. The damping zone's target is generated from the same formula.

A file may give either profile as a table of points instead: `"radial": [[1, 1e-3], [2, 2.5e-4],
[3, 0]]` (radius, density; a power law between points, no disk outside them) and
`"vertical": [[0, 1], [2, 0]]` (height, relative density; straight lines between points). A
formula wins over a table.

About a star, keep the disk's inner radius outside 1.3 of the star's radius: the star is
settled within that, and gas there that is not the star's is left as it is. Where a setup names
a background density (or has stars, with the gas around them), that gas shows wherever the
disk is thinner than it.

## Every key

These tables are written from the option table by `npm run gen`, so they list what this build
reads: the key's path in the document, its name in the Setup dialog, what it may hold with its
default, and what it does. A number out of its range is held to it when the file is read.

### General options

<!-- gen:setup-options -->
**The document**

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `name` | Name | text | What the setup is called in the preset list and in its file |
| `blurb` | Description | text | A sentence or two shown under the preset menu |

**`domain`**

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `domain.ndim` | Dimensions | `3`, `2` | Two or three dimensions. A 2D run is one layer of cells in x and y |
| `domain.n` | Resolution | a whole number, 16 to 16384 | Cells across the box; on an adaptive mesh, at its finest level |
| `domain.amr` | Refinement | a whole number, 0 to 5 | Adaptive mesh: levels of refinement below the coarsest grid. 0 puts every cell at the full resolution |
| `domain.box` | Box size | a number, 1e-12 to 1e+30. Default `1` | Side of the box, in code units: the domain is a cube of this side centered on the origin |
| `domain.geometry` | Geometry | `cartesian`, `cylindrical`, `spherical`. Default `cartesian` | Cartesian, or a 2D body of revolution about the y axis, in (R, z): x is then R from the axis (the lower x face) to the box side, and cells are rings; or in (r, theta): x is then r from the inner radius to the box side, y theta from pole to pole |
| `domain.rMin` | Inner radius | a number, 1e-12 to 1e+30. Default `0.01` | A spherical (r, theta) run: the radius of its inner face (above zero); the box side is then its outer radius |
| `domain.logR` | Logarithmic in r | `true` or `false`. Default `false` | A spherical (r, theta) run: cells spaced evenly in ln r, so that they stay about as long as they are wide, rather than evenly in r |
| `domain.symmetric` | Symmetric under | `none`, `x`, `y`, `z`, `xy`, `xz`, `yz`, `xyz`. Default `none` | The mirrors (planes through the origin, named by the axis normal to each) the setup is symmetric under: a run may then compute only the part above them |
| `domain.symmetricField` | Field symmetric too | `true` or `false`. Default `false` | The magnetic field is symmetric under those mirrors as well (a vector potential odd in its normal component and even in the others) |
| `domain.mirror` | Mirror symmetry | `none`, `x`, `y`, `z`, `xy`, `xz`, `yz`, `xyz`. Default `none` | Compute only the part of a symmetric setup above its planes of symmetry, each mirrored axis’s lower face a plane of symmetry; the totals are those of the whole. With every axis mirrored (an octant in 3D, a quadrant in 2D) each runs from 0 to the box side: twice the resolution for the same cells. With only some (a half, or a quadrant in 3D) those run from 0 to half the box with half the cells: the same cube as without mirrors, at half or a quarter of the cost |
| `domain.supports2d` | Makes sense in 2D | `true` or `false`. Default `false` | The setup can also run on a 2D (x, y) grid. Without it the setup is three-dimensional only |
| `domain.units` | Physical units | an object, which may be left out, in one of 2 shapes: length and mass (`mass`); or length, density and velocity (`density`, `velocity`) | What one code unit is, for a setup that stands for something real: a length and a mass (the time unit then makes G = 1), or a length, a density and a velocity. Without them the setup is scale free and shown in code units |
| `domain.units.name` | Name | text. Default `custom` | A name for the unit system |
| `domain.units.length.value` | Length | a number, 1e-300 to 1e+300. Default `1` | This many of the unit beside it are one code unit of length |
| `domain.units.length.unit` | Length unit | `m`, `cm`, `km`, `Rearth`, `Rsun`, `au`, `pc`. Default `Rsun` | The unit of the length |
| `domain.units.mass.value` | Mass | a number, 1e-300 to 1e+300. Default `1` | This many of the unit beside it are one code unit of mass |
| `domain.units.mass.unit` | Mass unit | `kg`, `g`, `Mearth`, `Msun`. Default `Msun` | The unit of the mass |
| `domain.units.density.value` | Density | a number, 1e-300 to 1e+300. Default `1` | This many of the unit beside it are one code unit of density |
| `domain.units.density.unit` | Density unit | `kg/m3`, `g/cm3`. Default `g/cm3` | The unit of the density |
| `domain.units.velocity.value` | Velocity | a number, 1e-300 to 1e+300. Default `1` | This many of the unit beside it are one code unit of velocity |
| `domain.units.velocity.unit` | Velocity unit | `m/s`, `km/s`, `cm/s`. Default `km/s` | The unit of the velocity |
| `domain.bc` | Faces of the box | `{ "lo": [x, y, z], "hi": [x, y, z] }`, each 0 periodic, 1 outflow, 2 reflecting, 3 outflow only, 4 line-tied wall, 5 moving wall | What the gas meets at each face of the box. Periodic faces come in pairs |
| `domain.wallVelocity.xlo` | Wall velocity, lower x face | `[x, y, z]`. Default `[0,0,0]` | Where the lower x face is a moving wall: its velocity along itself (the part normal to it is ignored) |
| `domain.wallVelocity.xhi` | Wall velocity, upper x face | `[x, y, z]`. Default `[0,0,0]` | Where the upper x face is a moving wall: its velocity along itself (the part normal to it is ignored) |
| `domain.wallVelocity.ylo` | Wall velocity, lower y face | `[x, y, z]`. Default `[0,0,0]` | Where the lower y face is a moving wall: its velocity along itself (the part normal to it is ignored) |
| `domain.wallVelocity.yhi` | Wall velocity, upper y face | `[x, y, z]`. Default `[0,0,0]` | Where the upper y face is a moving wall: its velocity along itself (the part normal to it is ignored) |
| `domain.wallVelocity.zlo` | Wall velocity, lower z face | `[x, y, z]`. Default `[0,0,0]` | Where the lower z face is a moving wall: its velocity along itself (the part normal to it is ignored) |
| `domain.wallVelocity.zhi` | Wall velocity, upper z face | `[x, y, z]`. Default `[0,0,0]` | Where the upper z face is a moving wall: its velocity along itself (the part normal to it is ignored) |

**`physics`**

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `physics.toolScale` | Scale of the editing tools | an object, which may be left out | Least density and speed the editing tools (paint, push) measure their amounts against. They take the densest and fastest gas of the run, which leaves them nothing to go by in a box that holds next to nothing |
| `physics.toolScale.rho` | Density | a number, at least 0. Default `1` | The tools’ density scale is at least this |
| `physics.toolScale.v` | Speed | a number, at least 0. Default `1` | The tools’ speed scale is at least this |
| `physics.eos` | Equation of state | `ideal`, `barotropic`, `liquid`, `gas_radiation`, `degenerate`, `ionizing`, `helmholtz`, `hhe`, `aneos`. Default `ideal` | How pressure follows from density and internal energy. Changing it carries each cell’s pressure over, and the run continues |
| `physics.eosParams.aneosA` | Material | `0`, `1`, `2`, `3`. Default `0` | The material of the ANEOS equation of state: rock, iron alloy or water, each from its published tables. A run whose two species are materials (a planet with a core) takes them instead |
| `physics.eosParams.gamma` | gamma | a number, 1 to 100. Default `1.6666666666666667` | Adiabatic index γ: p = (γ - 1) ρ e for an ideal gas, p = K ρ^γ for a barotropic one |
| `physics.eosParams.K` | K (p = K rho^gamma) | a number, at least 0 | Barotropic constant K: p = K ρ^γ. With γ = 1 it is the square of the sound speed |
| `physics.eosParams.liqC` | Sound speed | a number, at least 1e-30. Default `10` | The liquid’s sound speed c₀, a constant: p = c₀² (ρ - ρ₀). It is chosen, not physical: ten times the fastest flow keeps the density within 1% of ρ₀ (a Mach number of 0.1), and the step shrinks in proportion as it grows |
| `physics.eosParams.liqRho` | Reference density | a number, at least 1e-30. Default `1` | The density ρ₀ at which the liquid’s pressure is zero: p = c₀² (ρ - ρ₀). The liquid should start near it |
| `physics.eosParams.arad` | Radiation constant a | a number, at least 0 | Radiation pressure a T⁴ / 3 is added to the gas pressure |
| `physics.eosParams.degRho` | Degeneracy density | a number, at least 1e-30 | Density at which the electrons’ Fermi momentum reaches mₑc. For a setup with physical units it is set from the composition (about 2×10⁶ g/cm³ for C/O); in code units |
| `physics.eosParams.degP` | Degeneracy pressure scale | a number, at least 1e-30 | The scale π mₑ⁴ c⁵ / 3h³ of the degenerate electron pressure. Set from physical constants for a setup with physical units; in code units |
| `physics.eosParams.helmCoulomb` | Coulomb corrections | a number, 0 to 1. Default `1` | The ions’ Coulomb energy, as a multiplier: 1 for all of it, 0 for none. It lowers the cold pressure by a percent or two in dense white dwarf matter (more in the envelope) and adds to the heat capacity of cold, strongly coupled ions |
| `physics.eosParams.hheRad` | Radiation term | a number, 0 to 1. Default `1` | The radiation’s pressure a T⁴ / 3 and energy a T⁴ in the equation of state, as a multiplier: 1 for gas in equilibrium with its radiation (a star’s interior), 0 for gas alone (a cloud or thin gas, which its radiation leaves). Radiation transport does not run with this equation of state |
| `physics.eosParams.ionT` | Ionization temperature | a number, at least 1e-30 | The ionization energy of hydrogen, 13.6 eV, over k, in code temperature. Set from physical constants for a setup with physical units |
| `physics.eosParams.ionRho` | Saha density | a number, at least 1e-30 | Density scale of the Saha equation in code density: denser gas needs a higher temperature to ionize. Set from physical constants for a setup with physical units |
| `physics.composition` | Composition | `solar`, `solarNeutral`, `hydrogen`, `carbonOxygen`. Default `solar` | What the gas is made of, which sets the kelvin scale and the physical constants of the equation of state: solar (fully ionized), solarNeutral, hydrogen or carbonOxygen. The hydrogen and helium equation of state has tables for the solar ones and for hydrogen |
| `physics.species` | Species | a list of names | Species whose mass fractions the gas carries, by name (he4, c12, ...), separated by commas; or aprox13 for the burning network’s thirteen. None by default |
| `physics.speciesGamma` | Gamma by species | `{ species: gamma }` | Adiabatic index of each species the gas carries, by name, where the species set the equation of state (a mixture of ideal gases): a species not listed takes the gas’s gamma |
| `physics.burn` | Burning | an object, which may be left out, in one of 2 shapes: network (aprox13) (`network`, `limiter`, `switchAt`); or fuel to ash (arrhenius) (`fuel`, `ash`, `q`, `ea`, `k`) | Burning of the gas, wherever it is hot and dense enough: nuclear, by the trained network of the 13-isotope alpha chain, or a single fuel to ash reaction at an Arrhenius rate |
| `physics.burn.network` | Network | `aprox13`. Default `aprox13` | The burning network |
| `physics.burn.fuel` | Fuel | text. Default `fuel` | The species that burns: one the gas carries (Species) |
| `physics.burn.ash` | Ash | text. Default `ash` | The species it burns to: one the gas carries (Species) |
| `physics.burn.q` | Heat release q | a number, at least 0. Default `50` | Energy released per unit mass of fuel burned, in code units |
| `physics.burn.ea` | Activation temperature Ea | a number, at least 0. Default `25` | The rate goes as exp(-Ea / T), T the code temperature p / ρ |
| `physics.burn.k` | Rate constant k | a number, at least 0. Default `1` | The rate per unit time of fuel far above its activation temperature: df/dt = -k f exp(-Ea / T) |
| `physics.burn.inShocks` | Burn inside shocks | `true` or `false`. Default `false` | Let cells the shock detector flags burn. Off, as in most supernova codes: a captured shock is a few cells of numerical smearing. Fuel to ash burning burns in them unless told not to |
| `physics.burn.limiter` | Burning limiter f | a number, 0 to 1. Default `0` | No cell takes in burning energy faster than f times its internal energy per sound crossing of the cell (Kushnir et al. 2013). 0 switches it off |
| `physics.burn.switchAt` | Switch at | a number, 0 to 10. Default `0.1` | With the species he4, c12 and o16 alone: nothing burns, and the run stops for the switch to all thirteen (a checkpoint, restarted with them) once the energy the network’s reactions would release in a cell over a step passes this share of the cell’s internal energy |
| `physics.floors.rho` | Density floor | a number, at least 0. Default `1e-10` | No cell is thinner than this |
| `physics.floors.p` | Pressure floor | a number, at least 0. Default `1e-14` | No cell has less pressure than this |
| `physics.fluff.rho` | Tenuous below density | a number, at least 0. Default `0` | Gas thinner than this has its heat and speed capped (the next two fields), so a few hot or fast cells of ambient gas cannot set the timestep for the whole grid. Not conservative, but such gas carries next to no mass. 0 is off |
| `physics.fluff.u` | Max specific energy | a number, at least 0. Default `0` | Largest internal energy per unit mass tenuous gas may have. 0 is no cap |
| `physics.fluff.v` | Max speed | a number, at least 0. Default `0` | Largest speed tenuous gas may have. 0 is no cap |
| `physics.fluff.va` | Max Alfvén speed | a number, at least 0. Default `0` | With a magnetic field: largest Alfvén speed \|B\| / √ρ anywhere. Where the field is stronger than that for the gas it threads, mass is added at rest until it is not. Not conservative. 0 is no cap |
| `physics.cooling` | Cooling function | `off`, `power`, `ism`. Default `off` | Optically thin radiative cooling (and heating) of the gas |
| `physics.coolParams.rate` | Coefficient Λ₀ | a number, at least 0. Default `1` | Power law: the cooling rate per unit volume is ρ² Λ₀ T^α |
| `physics.coolParams.alpha` | Exponent of T | a number, -4 to 4. Default `0.5` | Power law: the exponent α of T in the cooling rate (1/2 is bremsstrahlung) |
| `physics.coolParams.heat` | Heating rate | a number, at least 0. Default `0` | Heating per unit mass (power law), or the heating rate at unit density (interstellar) |
| `physics.coolParams.tUnit` | Kelvin per unit T | a number, at least 0.000001. Default `1000` | Interstellar: kelvin per code unit of temperature (code density is atoms per cm³) |
| `physics.coolParams.tFloor` | Temperature floor | a number, at least 0. Default `0` | Gas at or below this temperature does not cool |
| `physics.radiation` | Radiation transport | `off`, `m1`. Default `off` | Gray radiation transport with the M1 closure: the radiation’s energy and flux, carried with the gas, streaming and diffusing between cells, and trading heat and momentum with the gas through its opacity (docs/numerics.md section 24). Cartesian coordinates, on the uniform grid or the adaptive mesh (all blocks then take one timestep), without a magnetic field |
| `physics.opacity` | Opacity | `constant`, `electron`, `kramers`. Default `constant` | How opaque the gas is to the radiation, per unit mass: a constant, electron scattering, or Kramers’ free-free absorption with electron scattering. With physical units the scattering and Kramers constants come from the composition |
| `physics.opacityParams.absorption` | Absorption κ_a | a number, at least 0. Default `1` | Absorption opacity per unit mass (the Planck mean), which exchanges heat between gas and radiation. With electron scattering, a small true absorption |
| `physics.opacityParams.scattering` | Scattering κ_s | a number, at least 0. Default `0` | Scattering opacity per unit mass: it drags on the radiation’s flux without heating the gas (electron scattering is 0.2 (1 + X) cm²/g) |
| `physics.opacityParams.kramers` | Kramers K₀ | a number, at least 0. Default `0` | Kramers: the free-free absorption is K₀ ρ T^-3.5 per unit mass, in code units |
| `physics.opacityParams.tFloor` | Kramers least T | a number, at least 0. Default `0` | Kramers: the opacity is taken at no lower temperature than this (it grows without bound as T falls) |
| `physics.radC` | Speed of light c | a number, at least 0 | The speed of light in code units. With physical units it is theirs; a scale-free setup sets its own |
| `physics.radChat` | Reduced speed of light | a number, at least 0 | The speed light crosses transparent gas at in the run (at most c): its step is about dx / chat. Ten times the fastest gas is enough; opaque cells take the true speed whatever this is |
| `physics.radLocalC` | Speed of light per cell | `true` or `false`. Default `true` | Opaque cells take the true speed of light and only transparent ones the reduced one, so the radiation in a star has its true heat capacity. Off, one reduced speed everywhere |
| `physics.radSolver` | Radiation flux | `hll`, `glf`. Default `hll` | The flux between cells: HLL with the M1 closure’s wave speeds, or global Lax-Friedrichs (the speed of light: more diffusive, more isotropic around sources) |
| `physics.radSweeps` | Radiation sweeps per step | a whole number, 1 to 16. Default `8` | Radiation sweeps per hydro step (1 to 16). The hydro step is at most this many of the sweep’s own steps, so more sweeps let it be longer |
| `physics.viscosity` | Viscosity ν | a number, 0 to 1e+30. Default `0` | Kinematic viscosity: the stress is ρ ν times the shear. Obstacles hold the gas at their walls. 0 is off |
| `physics.conductivity` | Conductivity κ | a number, 0 to 1e+30. Default `0` | Thermal conduction: the heat flux is -κ ρ grad T, so the temperature of an ideal gas diffuses at (γ - 1) κ. 0 is off |
| `physics.resistivity` | Resistivity η | a number, 0 to 1e+30. Default `0` | With a magnetic field: magnetic diffusivity, so the field diffuses as η ∇²B (the electric field gains η curl B) and the current it dissipates, η J², heats the gas. 0 is off (ideal MHD, with only the scheme’s own diffusion) |
| `physics.driving.accel` | Driving acceleration | a number, 0 to 1e+30. Default `0` | Turbulence driving: the root-mean-square acceleration of a random force that stirs the gas at large scales (a band of wavelengths, correlated over a time τ). The gas reaches speeds of about (accel × box / band)^½ where it balances the dissipation. 0 is off |
| `physics.driving.tau` | Driving correlation time τ | a number, 0.000001 to 1e+30. Default `0.5` | How long the random force keeps its pattern (its Ornstein-Uhlenbeck correlation time) |
| `physics.driving.kMin` | Driving band, longest | a number, 1 to 16. Default `1` | The longest wavelengths driven, as wavelengths per box side |
| `physics.driving.kMax` | Driving band, shortest | a number, 1 to 16. Default `2` | The shortest wavelengths driven, as wavelengths per box side (at most 64 wavevectors are kept, the longest first) |
| `physics.driving.zeta` | Solenoidal share ζ | a number, 0 to 1. Default `1` | What the force stirs: 1 swirls only (solenoidal, no compression), 0 compresses only, 0.5 the natural mix |
| `physics.driving.seed` | Driving seed | a whole number, 0 to 1000000000. Default `1` | Which random force: each seed gives another, and the same seed the same run |
| `physics.gravity` | Gravity | `off`, `constant`, `periodic`, `isolated`. Default `off` | Off, a uniform field, or the self-gravity of the gas. Point masses pull either way |
| `physics.G` | G | a number, at least 0. Default `1` | Gravitational constant in code units, or the strength of the uniform field. A setup with physical units may leave it to them |
| `physics.gext` | Uniform acceleration | `[x, y, z]`. Default `[0,0,0]` | A uniform acceleration on the gas, besides the Constant field |
| `physics.vcycles` | V-cycles | a whole number, 1 to 8 | Multigrid V-cycles per step for self-gravity. More converge the potential further, at more cost. On the adaptive mesh this is the most per step |
| `physics.jeansFloor` | Jeans floor | `true` or `false`. Default `false` | Pressure floor that keeps the Jeans length resolved (Truelove) |
| `physics.jeansCells` | Cells per Jeans length | a whole number, 4 to 32. Default `4` | The Jeans floor raises the pressure so the Jeans length spans at least this many cells |
| `physics.gravEnergyFlux` | Conserve energy | `true` or `false`. Default `true` | Flux-form gravitational work: conserves total energy |
| `physics.rotation.omega` | Rotating frame Ω | `[x, y, z]`. Default `[0,0,0]` | Angular velocity of a frame turning about an axis through the box center: the gas feels the Coriolis force (and the centrifugal force, below). Velocities are those in the turning frame. In 2D only the z part turns the plane. Zero: an inertial frame |
| `physics.rotation.centrifugal` | Centrifugal force | `true` or `false`. Default `true` | With a rotating frame: the centrifugal force as well as the Coriolis force. Off, the Coriolis force alone (an f-plane, as in a shearing box) |
| `physics.shear` | Shearing box | an object, which may be left out | The local model of a disk: a frame turning at Ω about z with the tidal pull of the shear, x radial, y along the orbit. Its x faces are periodic with the shift of the background flow v_y = -q Ω x, which the initial state should hold. Hydrodynamics on a uniform grid, periodic in y (and z unless stratified) |
| `physics.shear.omega` | Ω | a number, 0 to 1000000. Default `1` | Angular velocity of the frame |
| `physics.shear.q` | Shear q | a number, 0 to 2. Default `1.5` | -d ln Ω / d ln R: 1.5 for a Keplerian disk, 1 for a flat rotation curve |
| `physics.shear.stratified` | Stratified | `true` or `false`. Default `false` | The vertical tidal pull, -Ω² z, toward the midplane |
| `physics.cosmology` | Comoving coordinates | an object, which may be left out | An expanding universe: positions comoving, densities comoving (mean 1), velocities peculiar, times in 1/H0 from the starting redshift. G follows from the matter density. A periodic box of gas, with or without self-gravity |
| `physics.cosmology.omegaM` | Matter density Ω_m | a number, 0 to 10. Default `1` | Matter density today, in units of the critical density |
| `physics.cosmology.omegaL` | Cosmological constant Ω_Λ | a number, 0 to 10. Default `0` | Its density today, in units of the critical density (the curvature makes up the rest) |
| `physics.cosmology.zStart` | Starting redshift | a number, 0 to 10000. Default `99` | Redshift at t = 0 |
| `physics.cosmology.zEnd` | Ending redshift | a number, 0 to 10000. Default `0` | Redshift the run stops at (it sets the end time) |
| `physics.cosmology.boxMpc` | Box size (Mpc/h) | a number, 0 to 100000 | The box’s comoving side in Mpc/h. With it the run has physical units, and the readouts show proper densities, pressures and lengths at the current scale factor. Left out, the run is scale free |
| `physics.cosmology.h` | Hubble constant h | a number, 0.1 to 2. Default `0.7` | H0 in units of 100 km/s/Mpc, for the physical units a box size gives |
| `physics.frame` | Frame | `inertial`, `gas`, `masses`. Default `inertial` | An inertial frame, or one that accelerates with the gas’s center of mass or the point masses’, so that what the run follows stays in the box |
| `physics.riemann` | Riemann solver | `hllc`, `hllclm`, `hll`. Default `hllc` | How the flux between neighboring cells is worked out. HLLC resolves contact waves; its low-Mach version cuts dissipation in slow flow; HLL is the most diffusive and the most robust |
| `physics.limiter` | Limiter | `koren`, `mc`, `vanleer`, `minmod`. Default `koren` | How steeply the reconstruction may vary within a cell: Koren is sharpest, Minmod most diffusive and most robust |
| `physics.cfl` | CFL number | a number, 0.05 to 1. Default `0.8` | Timestep as a fraction of the time a signal takes to cross a cell. Smaller is more stable and slower |
| `physics.startDamping.rate` | Start-up damping rate | a number, at least 0. Default `0` | For a while after the start, every cell’s velocity is damped at this rate (per unit time), so that whatever settled stars are still out of balance with on the grid is taken away while it is small instead of ringing through the run. The rate falls smoothly to zero over the start-up damping time. 0 is off |
| `physics.startDamping.time` | Start-up damping time | a number, at least 0. Default `0` | Time from the start over which the start-up damping falls smoothly to zero (as a half cosine). 0 is off |
| `physics.pressure` | Pressure | `explicit`, `implicit`. Default `explicit` | How the pressure acts. Explicit: in the fluxes between cells, with a timestep that follows the sound speed. Implicit: advection is explicit and the pressure is solved for each step, so the timestep follows the flow speed, however fast the sound: for the liquid and the ideal gas, in Cartesian coordinates, on the uniform grid or the adaptive mesh |
| `physics.storedPrims` | Stored primitives | `auto`, `on`, `off`. Default `auto` | Uniform grid: convert each cell from its conserved state to density, velocity and pressure once a step and keep the result, instead of converting it again at every one of the some twenty loads of the cell in a step. The results are the same. It costs 32 bytes of GPU memory per cell, and saves most where the equation of state solves for its temperature or reads a table (Helmholtz or ANEOS: steps about three times faster on an A100). Automatic: on for every equation of state but the ideal gas, the barotropic gas and the liquid |
| `physics.cutCells` | Partly filled cells | `true` or `false`. Default `false` | Obstacles cut the cells their surfaces cross: each cell holds gas in a share of its volume and each face is open in a share of its area, so a body keeps its true shape below a cell (a thin blade, its pitch), where the plain grid makes it a staircase of whole cells. For the liquid with implicit pressure, on the uniform grid, with obstacles at rest |
| `physics.pressureCycles` | Pressure V-cycles | a whole number, 1 to 16. Default `3` | Implicit pressure: multigrid V-cycles per step (warm-started). More converge the pressure further, at more cost |
| `physics.pressureTol` | Pressure tolerance | a number, 1e-8 to 1. Default `0.0001` | Implicit pressure: the residual of the pressure equation after a step’s V-cycles, relative to its largest right-hand side, above which the step is counted as unconverged |
| `physics.acousticCfl` | Acoustic steps per step | a number, 1 to 1000000. Default `1000` | Implicit pressure: the most steps of the explicit scheme that one step may span. It is what bounds the step of a liquid at rest; sound that a step spans many times over is damped |

**`mesh`**

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `mesh.refineHi` | Refine above | a number, 0.05 to 1. Default `0.8` | Adaptive mesh: a block is split where the normalized second derivative of density or pressure exceeds this anywhere in it. The indicator is near 1 at a shock or contact and small where the gas varies smoothly. Lower refines more |
| `mesh.refineLo` | Merge below | a number, 0 to 1. Default `0.2` | Adaptive mesh: four sibling blocks are merged where the indicator is below this in all of them. Kept below the refinement threshold |
| `mesh.refineRho` | Refine above density | a number, at least 0 | Adaptive mesh: gas thinner than this does not ask for refinement, so ripples in the ambient gas far from a star leave its blocks coarse. 0 is off |
| `mesh.refineDense` | Always refine above density | a number, at least 0. Default `0` | Adaptive mesh: gas denser than this is always at the finest level, as the smooth interior of a star must be. 0 is off |
| `mesh.refineMasses` | Refine around point masses | `true` or `false`. Default `false` | Adaptive mesh: keep the gas around every point mass at the finest level, so that softening and sinks can be as small as the finest cells allow |
| `mesh.timeBins` | Local timesteps | `0`, `1`, `2`, `3`. Default `0` | Adaptive mesh: blocks of slow gas take longer steps than the fastest block, where that saves work |
| `mesh.jeansRefine` | Refine the Jeans length | a whole number, 0 to 64. Default `16` | With self-gravity, refine where the Jeans length is under this many cells. 0 is off |

**`background`**

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `background.rho` | Ambient density | a number, at least 1e-300. Default `1` | Density of the gas that fills the box where no part puts anything else. With stars, the gas around them |
| `background.p` | Ambient pressure | a number, at least 0 | Pressure of that gas. With stars and left empty, the pressure of the stars’ outermost layer |
| `background.v` | Ambient velocity | `[x, y, z]`. Default `[0,0,0]` | Velocity of that gas |
| `background.s` | Ambient dye | a number, 0 to 1. Default `0` | Passive scalar of that gas: a dye carried with it |
| `background.X` | Ambient composition | `{ species: fraction }` | Mass fractions of that gas, by species name |
| `background.cut` | Stars end at density | a number, at least 0 | With stars: a cell thinner than this is ambient gas, not star. Left empty, a polytrope ends where the ambient gas would be as hot as its center |

**`run`**

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `run.tEnd` | End time | a number, at least 0, or `null` | The run pauses when it reaches this time. Empty runs on without end |
| `run.checkpointDt` | Checkpoint every | a number, at least 0 | Time between checkpoints, when the page is asked to save them: the value the Time panel starts with |
| `run.tDyn` | Dynamical time | a number, at least 0 | The time over which the flow visibly changes (a star’s sqrt(R³ / G M), a binary’s period): a movie takes a frame every 600th of it by default |
| `run.startPaused` | Start paused | `true` or `false`. Default `false` | The run waits, paused, for Run |

**`view`**

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `view.mode` | Opening view | `volume`, `slice`, `projection`. Default `volume` | How the gas is drawn when the setup opens in 3D |
| `view.field` | Opening field | a whole number, 0 to 36. Default `0` | The quantity shown when the setup opens (the View panel’s Field, by its number) |
| `view.log` | Logarithmic colors | `true` or `false`. Default `true` | The color scale the setup opens with |
| `view.axis` | Slice normal | `2`, `1`, `0`. Default `2` | The axis the opening slice is perpendicular to (0 x, 1 y, 2 z) |
| `view.pitch` | Camera elevation | a number, -1.5 to 1.5. Default `0.45` | Elevation of the volume view’s camera when the setup opens, in radians |
<!-- /gen:setup-options -->

### Parts

A part's fields are given relative to the part (`soft` is `parts[i].soft`).

<!-- gen:setup-parts -->
**`mass`** (Point mass). A point mass: softened gravity on the gas, with a sink that takes gas in. Point masses pull on one another and feel the gas. At most 32

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `gm` | G M | a number, at least 0. Default `1` | Its mass times G, in code units |
| `pos` | Position | `[x, y, z]`. Default `[0,0,0]` | Where it starts, relative to the box center |
| `vel` | Velocity | `[x, y, z]`. Default `[0,0,0]` | How it moves at the start |
| `soft` | Softening | a number, at least 0 | Plummer softening length of its gravity: at least two cells, and two cells if left empty. Changes take effect without a restart |
| `sink` | Sink radius | a number, at least 0 | Radius within which it takes gas in: the softening length if left empty, 0 for none. Changes take effect without a restart |
| `sinkRate` | Sink rate | a number, at least 0. Default `1` | Removal rate at the sink’s center, in units of sqrt(G M / radius³) |
| `onOrbit` | Hold on its orbit | `true` or `false`. Default `false` | The other point masses move it; the gas does not: neither the gas’s pull nor what its sink takes changes its motion, and it does not grow. A binary with both masses held keeps the orbit it was given, as in most studies of circumbinary disks. What the sink takes is still counted. Changes take effect without a restart |
| `massless` | Massless | `true` or `false`. Default `false` | A particle without a mass of its own: it pulls on nothing and moves as the gas and the masses pull it |
| `fixed` | Held in place | `true` or `false`. Default `false` | Nothing moves it, whatever pulls on it or it takes in |

**`orbit`** (Orbit). Not an object: puts two bodies on a Kepler orbit

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `of` | Bodies | the ids of two of `mass` or `star` parts | The two bodies it places: point masses or stars (binary), a star and a point mass (flyby), or two stars (collision) |
| `type` | Kind of orbit | `binary`, `flyby`, `collision`. Default `binary` | Binary: two bodies, point masses or stars, about their center of mass, at pericenter; on a circular orbit its stars turn with it and start in the balance of the Roche potential. Flyby: a point mass on a parabola about a star that stays at the origin (a tidal disruption in the star’s frame). Collision: two stars on a parabola about their center of mass |
| `q` | Mass ratio | a number, 0.000001 to 1000000000 | Mass of the second body over the first. It sets their masses: in a binary of two point masses the total stays, in a binary with a star or a flyby the star’s |
| `separation` | Separation | a number, at least 0.000001. Default `1` | Semi-major axis of the binary’s orbit |
| `ecc` | Eccentricity | a number, 0 to 0.95. Default `0` | Eccentricity of the binary’s orbit |
| `incl` | Inclination (°) | a number, 0 to 180. Default `0` | Tilt of the binary’s orbit about the line through pericenter, in degrees (3D): 180 is retrograde |
| `fill` | Roche lobe fill | a number, 0.05 to 2 | In place of the separation: the first body, a star, is this many times the size of its Roche lobe (Eggleton 1983), which sets how far apart the two are. 1 just fills it; above 1 it pours gas through L1 onto its companion from the start |
| `corotating` | Corotating frame | `true` or `false`. Default `false` | A circular binary: the run is in the frame turning with the orbit, so the bodies start at rest and a binary in balance stays still |
| `track` | Follow the orbit | `true` or `false`. Default `false` | With the corotating frame: the frame turns faster or slower as the orbit changes (mass transfer, drag, gas lost through the faces), keeping the line between the two bodies where it started, and the gas and the point masses feel the Euler force of its changing turn. It follows the axis of the stars’ and point masses’ quadrupole, so any two bodies can be followed |
| `release` | Release after a merger | `true` or `false`. Default `false` | With the frame following the orbit: once the mass no longer looks like a binary (the axis of its quadrupole fades, as when one star is torn apart over the other), the frame stops following the axis and comes to rest over about an orbit, and the run carries on in an inertial frame, where the energy and momentum it keeps are the true ones |
| `retrograde` | Retrograde | `true` or `false`. Default `false` | In 2D: the binary orbits clockwise |
| `beta` | Impact parameter β | a number, 0.05 to 20. Default `1` | Tidal radius over pericenter distance: above 1 the mass passes inside the star’s tidal radius |
| `start` | Starts at (tidal radii) | a number, 1 to 20. Default `3` | Separation at the start, in tidal radii (never inside 1.5 pericenters) |
| `debris` | Debris energies | `true` or `false`. Default `false` | Bin the star’s gas by its orbital energy about the mass, for the rate at which bound debris returns |
| `impact` | Impact parameter | a number, 0 to 5. Default `0` | Pericenter distance over the sum of the two radii: 0 is head-on, 1 just grazes |
| `starts` | Starts at (sums of radii) | a number, 1.05 to 20. Default `4` | Separation at the start, in sums of the two radii (closer if they would not fit in the box) |

**`star`** (Star). A polytrope, a cold white dwarf or a planet of ANEOS materials, settled into the grid’s own hydrostatic balance before the run. At most 4

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `type` | Kind of star | `polytrope`, `white-dwarf`, `planet`. Default `polytrope` | A polytrope of index n, a cold white dwarf held up by degenerate electrons, or a planet of an ANEOS material, with or without a core of another (physical units) |
| `n` | Polytropic index n | a number, 0.5 to 4.5. Default `1.5` | The star’s structure, p = K ρ^(1 + 1/n): the larger n, the more centrally concentrated. Stable against convection only if the gas’s adiabatic index is at least 1 + 1/n |
| `mass` | Mass | a number, at least 1e-12. Default `1` | Its mass, in code units (solar masses for a white dwarf in its usual units) |
| `radius` | Radius | a number, at least 1e-12. Default `0.2` | A polytrope’s radius, in code units. A white dwarf’s follows from its mass |
| `material` | Material | `pyrolite`, `fe85si15`, `water`, `forsterite`. Default `pyrolite` | What a planet is made of; with a core, its mantle. A run holds two materials at most, and they are its species: the mantle’s first |
| `coreMaterial` | Core material | `pyrolite`, `fe85si15`, `water`, `forsterite` | A planet’s core, if it has one: the inner part of its mass, of this material |
| `coreFraction` | Core mass fraction | a number, 0.01 to 0.99. Default `0.3` | The core’s share of a planet’s mass (the Earth’s is 0.32) |
| `surfaceT` | Surface temperature (K) | a number, 50 to 20000. Default `2000` | A planet’s temperature at its surface, in kelvin. Each of its layers lies along its adiabat from there: 2000 K is a body still hot from its accretion, as the published giant-impact calculations start |
| `rhoc` | Central density | a number, at least 1e-300 | Instead of the mass: a polytrope’s central density |
| `core` | Point-mass core | an object, which may be left out | A giant: part of the star’s mass is a point mass at its center, and the rest a polytropic envelope in balance with its own gravity and the core’s softened pull. The core moves with the star and then as the gas and the other masses pull it. By mass and radius only |
| `core.mass` | Core mass | a number, at least 1e-12. Default `0.4` | The core’s share of the star’s mass, in code units: less than the star’s |
| `core.soft` | Core softening | a number, at least 1e-12 | Plummer softening length of the core’s gravity: a twentieth of the star’s radius if left empty. At least two finest cells, and the envelope’s steep inner part should span several |
| `pos` | Position | `[x, y, z]`. Default `[0,0,0]` | Where it starts, relative to the box center |
| `vel` | Velocity | `[x, y, z]`. Default `[0,0,0]` | How it moves at the start |
| `dye` | Dye | a number, 0 to 1 | Passive scalar of its gas, which tells the stars apart: spread over (0, 1] by their order if left empty |
| `X` | Composition | `{ species: fraction }` | Mass fractions of its gas, by species name (the ambient gas’s if left empty) |
| `field` | Magnetic field inside | an object, which may be left out | A field confined to the star: a dipole along z or a tangled Kolmogorov field, with plasma beta at the center; or the same as Guillochon & McCourt (2017) set them, by their β_M |
| `field.type` | Shape | `dipole`, `random`, `gm17-dipole`, `gm17-random`. Default `dipole` | Dipole or random, by the plasma beta at the center; or Guillochon & McCourt’s, by β_M |
| `field.beta` | Plasma beta | a number, 1 to 1000000000000. Default `1000` | Gas pressure over magnetic pressure B² / 2 at the center (or β_M): the larger, the weaker the field |
| `field.seed` | Random seed | a whole number, 1 to 9999. Default `1` | Which random field: each seed gives another |

**`disk`** (Disk). A gas disk in the x-y plane about one or more bodies

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `about` | Orbits | ids of `mass` or `star` parts | The point masses or stars it orbits: about several, their total mass. The disk is centered on their center of mass and moves with it. About a star, keep the inner radius outside 1.3 of the star’s radius: the star is settled within that |
| `profile` | Profile | `thin`, `balanced`. Default `balanced` | Thin: a surface density with a cavity and a Gaussian vertical profile, on circular orbits (the binary’s disk). Balanced: a midplane density in vertical balance with the softened potential, rotating as its pressure gradient asks (the magnetized disk); 3D only |
| `rIn` | Inner radius | a number, at least 0.000001. Default `0.3` | Where the disk is tapered away toward the center (the cavity’s radius) |
| `rOut` | Outer radius | a number, at least 0.000001. Default `1` | Where the disk is tapered away outward |
| `sigma0` | Density scale | a number, at least 1e-300. Default `0.001` | Thin: the surface density. Balanced: the midplane density at r = 1 |
| `slope` | Density slope | a number, -4 to 2. Default `-1.5` | Balanced: the midplane density goes as r to this power (-3/2 is a flat surface density) |
| `radialFn` | Radial profile (formula) | text | A density profile of your own in radius, as a formula of r: the surface density (thin) or the midplane density (balanced). It replaces the density scale’s law, the slope and the tapers. Example: sigma0 * (r / rIn)^-1 * exp(-(r / rOut)^2) * step(r - rIn). Names: r, rIn, rOut, sigma0, cs, H (the scale height cs / Ω at r), pi, e. Functions: exp, log, ln, log10, sqrt, abs, sin, cos, tan, atan, sinh, cosh, tanh, step, pow, min, max, clamp; powers as x^y. Where it is not a positive number there is no disk. A file may give a table of points instead (radial) |
| `verticalFn` | Vertical profile (formula) | text | A density profile of your own in height, as a formula of z, the height above the midplane (the same on both sides), relative to the midplane: it replaces the Gaussian (thin) or the balance with the potential (balanced). Examples: exp(-z^2 / 2), 1 / cosh(z)^2, max(1 - z / 3, 0). Such a disk is in vertical balance only if the profile is, and settles as the run starts. 3D only. A file may give a table of points instead (vertical) |
| `verticalIn` | Heights in | `H`, `length`. Default `H` | The z of the vertical profile: in scale heights cs / Ω of each radius (the profile then flares with the disk), or in code lengths |
| `cs` | Sound speed | a number, at least 0.000001. Default `0.1` | Sound speed of the disk’s gas: its scale height is cs / Ω |
| `floor` | Gas outside the disk | a number, at least 0. Default `1e-8` | Thin: a fraction of the density scale. Balanced: the density of the ambient gas |
| `damp` | Damping zone | an object, which may be left out | Beyond a radius the gas is relaxed toward the disk as it started, more strongly toward the box’s edge, so the outer disk is a reservoir and the boundaries see only it |
| `damp.from` | Damping from r | a number, 0 to 1e+30. Default `1` | Radius from the box center where the damping zone starts |
| `damp.rate` | Damping rate | a number, 0 to 1e+30. Default `3` | How many times per local orbital period the gas at the box edge is relaxed toward its initial state. 0 is off |
| `loops` | Field loops | an object, which may be left out | Poloidal loops of magnetic field inside the disk, with no net flux: the seed of the magnetorotational instability |
| `loops.beta` | Plasma beta at the midplane | a number, 1 to 1000000000. Default `50` | Gas pressure over magnetic pressure where two loops meet: the larger, the weaker the field |
| `loops.span` | Span of a loop | a number, 1.05 to 4. Default `1.35` | Each loop spans this factor in radius |
| `loops.count` | Loops | a whole number, 1 to 16. Default `4` | How many loops, outward from the inner radius |
| `stresses` | Disk diagnostics | `true` or `false`. Default `false` | Report the disk’s stresses and MRI quality factors about the first point mass |

**`region`** (Gas region). A shape with a state (density, pressure, velocity, dye, composition) that replaces what is under it. A value left empty keeps what is there

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `shape.type` | Shape | `sphere`, `box`, `half`, `mesh`. Default `sphere` | A sphere (a cylinder in 2D), a box, the half of space on one side of a plane, or a mesh file (STL, OBJ, PLY, glTF; its cross-section through its center in 2D) |
| `shape.center` | Center | `[x, y, z]`. Default `[0,0,0]` | Center of the sphere, or where the middle of the mesh’s bounding box goes |
| `shape.file` | File | text. Default `` | The mesh file: its name beside the setup document (the runner, the MCP server), a file dropped on the Setup dialog, or a URL. STL, OBJ, PLY, glTF or GLB. STEP and IGES are not read: export STL or glTF from the CAD program |
| `shape.anchor` | Placed by | `box`, `origin`. Default `box` | What the center places: the middle of the mesh’s bounding box, or the file’s own origin (a rotor’s hub, which a rotor’s frame needs on its axis) |
| `shape.node` | Part of the file | text | A glTF node or an OBJ group, with the parts below it. Empty: the whole file |
| `shape.fit` | Fit | a number, 0 to 1 | The mesh’s longest side, as a fraction of the box. STL has no units: give this or the scale |
| `shape.scale` | Scale | a number, at least 0 | A factor on the file’s lengths, instead of fitting it to the box (glTF’s are meters) |
| `shape.up` | Up axis | `x`, `y`, `z`. Default `z` | Which of the file’s axes is up: y for glTF and most 3D programs, z for most CAD. It becomes the box’s z (y in 2D) |
| `shape.radius` | Radius | a number, at least 0. Default `0.1` | Radius of the sphere |
| `shape.radiusCells` | Radius in cells | a number, 0.5 to 64 | Instead of the radius: this many finest cells, whatever the resolution (a point explosion) |
| `shape.min` | Lower corner | `[x, y, z]`. Default `[-0.1,-0.1,-0.1]` | Corner of the box nearest -x, -y, -z |
| `shape.max` | Upper corner | `[x, y, z]`. Default `[0.1,0.1,0.1]` | Corner of the box nearest +x, +y, +z |
| `shape.rotate` | Rotation | `[x, y, z]`. Default `[0,0,0]` | Degrees the box or mesh is turned about its center: about x, then y, then z. In 2D only the turn about z counts for a box |
| `shape.normal` | Normal | `[x, y, z]`. Default `[1,0,0]` | The plane’s normal: the shape is the side the normal points away from |
| `shape.at` | Plane at | a number. Default `0` | Distance of the plane from the box center, along its normal |
| `rho` | Density | a number, at least 1e-300 | Density inside the shape |
| `p` | Pressure | a number, at least 0 | Pressure inside the shape |
| `energy` | Total energy | a number, at least 0 | Instead of a pressure: this much energy in all, spread evenly over the sphere’s cells (a point explosion; per unit length in 2D) |
| `v` | Velocity | `[x, y, z]` | Velocity inside the shape |
| `s` | Dye | a number, 0 to 1 | Passive scalar inside the shape |
| `X` | Composition | `{ species: fraction }` | Mass fractions inside the shape, by species name |
| `erad` | Radiation energy | a number, at least 0 | With radiation transport: the radiation’s energy density inside the shape. Empty: that in equilibrium with the gas, a T⁴ |
| `frad` | Radiation flux | `[x, y, z]` | With radiation transport: the radiation’s flux inside the shape, at most c times its energy (a beam). Empty: the flux of diffusion. With an inflow over the same shape, a lamp that shines for the whole run |
| `supersample` | Smooth edge (samples) | a whole number, 2 to 8 | A sphere’s edge cells take the share of their volume inside it, from this many samples a side. Empty: a sharp edge |

**`obstacle`** (Obstacle). A solid body the gas reflects off

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `shape.type` | Shape | `sphere`, `box`, `half`, `mesh`. Default `sphere` | A sphere (a cylinder in 2D), a box, the half of space on one side of a plane, or a mesh file (STL, OBJ, PLY, glTF; its cross-section through its center in 2D) |
| `shape.center` | Center | `[x, y, z]`. Default `[0,0,0]` | Center of the sphere, or where the middle of the mesh’s bounding box goes |
| `shape.file` | File | text. Default `` | The mesh file: its name beside the setup document (the runner, the MCP server), a file dropped on the Setup dialog, or a URL. STL, OBJ, PLY, glTF or GLB. STEP and IGES are not read: export STL or glTF from the CAD program |
| `shape.anchor` | Placed by | `box`, `origin`. Default `box` | What the center places: the middle of the mesh’s bounding box, or the file’s own origin (a rotor’s hub, which a rotor’s frame needs on its axis) |
| `shape.node` | Part of the file | text | A glTF node or an OBJ group, with the parts below it. Empty: the whole file |
| `shape.fit` | Fit | a number, 0 to 1 | The mesh’s longest side, as a fraction of the box. STL has no units: give this or the scale |
| `shape.scale` | Scale | a number, at least 0 | A factor on the file’s lengths, instead of fitting it to the box (glTF’s are meters) |
| `shape.up` | Up axis | `x`, `y`, `z`. Default `z` | Which of the file’s axes is up: y for glTF and most 3D programs, z for most CAD. It becomes the box’s z (y in 2D) |
| `shape.radius` | Radius | a number, at least 0. Default `0.1` | Radius of the sphere |
| `shape.radiusCells` | Radius in cells | a number, 0.5 to 64 | Instead of the radius: this many finest cells, whatever the resolution (a point explosion) |
| `shape.min` | Lower corner | `[x, y, z]`. Default `[-0.1,-0.1,-0.1]` | Corner of the box nearest -x, -y, -z |
| `shape.max` | Upper corner | `[x, y, z]`. Default `[0.1,0.1,0.1]` | Corner of the box nearest +x, +y, +z |
| `shape.rotate` | Rotation | `[x, y, z]`. Default `[0,0,0]` | Degrees the box or mesh is turned about its center: about x, then y, then z. In 2D only the turn about z counts for a box |
| `shape.normal` | Normal | `[x, y, z]`. Default `[1,0,0]` | The plane’s normal: the shape is the side the normal points away from |
| `shape.at` | Plane at | a number. Default `0` | Distance of the plane from the box center, along its normal |
| `refine` | Refines | `surface`, `swept`, `off`. Default `surface` | On an adaptive mesh: Surface refines every block within a band of the obstacle’s surface, decided from its shape (so a part thinner than a coarse cell is whole at the finest level); Swept, everywhere its surface passes as it moves (a rotor’s disk; for a body at rest the same as Surface); Off leaves it to the flow’s own refinement |
| `refineBand` | Refined band (cells) | a number, 0 to 64. Default `4` | How far from the surface the mesh is kept at the obstacle’s level, in finest cells |
| `refineLevel` | Refined to level | a whole number, 0 to 5 | The level its band is kept at, 0 the coarsest. Empty: the finest |

**`tunnel`** (Wind tunnel). A stream through the box from an inlet to an outlet with a sponge, and the box’s faces, its gas and its background set to match: put obstacles in it, and the diagnostics give their drag and lift coefficients. Below Mach 0.3 the gas is the liquid with implicit pressure; above, an ideal gas. At most 1

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `speed` | Speed | a number, at least 0. Default `1` | The stream’s speed |
| `axis` | Along | `x`, `y`, `z`. Default `x` | The axis the stream flows along, toward +. Yaw and angle of attack turn the body, not the stream |
| `mach` | Mach number | a number, 0.01 to 5. Default `0.1` | The stream’s speed over the gas’s sound speed. Below 0.3 the gas is the liquid with implicit pressure (incompressible, in effect); above, an ideal gas with the explicit scheme (shocks) |
| `rho` | Density | a number, at least 0. Default `1` | The gas’s density |
| `sides` | Sides and roof | `slip`, `open`, `periodic`. Default `slip` | Slip: walls the gas slides along. Open: faces it may leave by. Periodic: the box repeats sideways |
| `floor` | Floor | `none`, `slip`, `fixed`, `moving`. Default `none` | The lower face along the box’s up axis (z in 3D, y in 2D): as the sides, a slip wall, a no-slip wall, or a wall moving with the stream (a car on a road) |
| `sponge` | Sponge | a number, 0 to 0.5. Default `0.1` | The outlet’s damping zone, as a fraction of the box’s length: it pulls the gas back to the stream so that the wake leaves without reflecting. 0 for none |
| `inlet` | Inlet cells | a whole number, 1 to 16. Default `2` | Cells held at the stream’s state at the upstream face: finest cells on a uniform grid, the coarsest level’s on blocks |
| `rotor` | Rotor frame | a number. Default `0` | An angular velocity about the stream’s axis through the box’s center: the run is in the frame that turns at it, so that a rotor centered there is at rest (the stream is U - Omega x r, the corners of the box held at it). Torque and thrust then come steady. 0: none. 3D only |
| `area` | Reference area | a number, at least 0 | The area the force coefficients are over (a length in 2D). Empty: the first obstacle’s shadow along the stream |
| `length` | Reference length | a number, at least 0 | The length the moment coefficients are over. Empty: the square root of the area (the area itself in 2D) |

**`inflow`** (Inflow). Cells held at their initial state for the whole run: an inlet

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `shape.type` | Shape | `sphere`, `box`, `half`, `mesh`. Default `sphere` | A sphere (a cylinder in 2D), a box, the half of space on one side of a plane, or a mesh file (STL, OBJ, PLY, glTF; its cross-section through its center in 2D) |
| `shape.center` | Center | `[x, y, z]`. Default `[0,0,0]` | Center of the sphere, or where the middle of the mesh’s bounding box goes |
| `shape.file` | File | text. Default `` | The mesh file: its name beside the setup document (the runner, the MCP server), a file dropped on the Setup dialog, or a URL. STL, OBJ, PLY, glTF or GLB. STEP and IGES are not read: export STL or glTF from the CAD program |
| `shape.anchor` | Placed by | `box`, `origin`. Default `box` | What the center places: the middle of the mesh’s bounding box, or the file’s own origin (a rotor’s hub, which a rotor’s frame needs on its axis) |
| `shape.node` | Part of the file | text | A glTF node or an OBJ group, with the parts below it. Empty: the whole file |
| `shape.fit` | Fit | a number, 0 to 1 | The mesh’s longest side, as a fraction of the box. STL has no units: give this or the scale |
| `shape.scale` | Scale | a number, at least 0 | A factor on the file’s lengths, instead of fitting it to the box (glTF’s are meters) |
| `shape.up` | Up axis | `x`, `y`, `z`. Default `z` | Which of the file’s axes is up: y for glTF and most 3D programs, z for most CAD. It becomes the box’s z (y in 2D) |
| `shape.radius` | Radius | a number, at least 0. Default `0.1` | Radius of the sphere |
| `shape.radiusCells` | Radius in cells | a number, 0.5 to 64 | Instead of the radius: this many finest cells, whatever the resolution (a point explosion) |
| `shape.min` | Lower corner | `[x, y, z]`. Default `[-0.1,-0.1,-0.1]` | Corner of the box nearest -x, -y, -z |
| `shape.max` | Upper corner | `[x, y, z]`. Default `[0.1,0.1,0.1]` | Corner of the box nearest +x, +y, +z |
| `shape.rotate` | Rotation | `[x, y, z]`. Default `[0,0,0]` | Degrees the box or mesh is turned about its center: about x, then y, then z. In 2D only the turn about z counts for a box |
| `shape.normal` | Normal | `[x, y, z]`. Default `[1,0,0]` | The plane’s normal: the shape is the side the normal points away from |
| `shape.at` | Plane at | a number. Default `0` | Distance of the plane from the box center, along its normal |

**`field`** (Uniform magnetic field). A uniform field through the whole box, in units where the magnetic pressure is B² / 2

| Key | In the dialog | Values | Meaning |
| --- | --- | --- | --- |
| `b` | Field | `[x, y, z]` | The field’s vector |
| `up` | Strength, upward | a number | Instead of a vector: a strength along the axis drawn upward (y in 2D, z in 3D) |

**`flow`** (Flow). A named, built-in initial condition with its parameters, for what shapes cannot say

Its fields are its `name` and that flow's parameters, listed below.
<!-- /gen:setup-parts -->

### Flows

<!-- gen:setup-flows -->
| `name` | What it is | Parameters, with their defaults |
| --- | --- | --- |
| `advection` | Advected Gaussian blob. A Gaussian density blob at the center, carried by a uniform flow at a uniform pressure. The passive scalar follows the blob. On a periodic box it tests conservation. (fills the box) | `rho0` 1.5 (Density around the blob); `amp` 1 (Blob height); `width` 0.1 (Blob width); `vx` 1 (Velocity x); `vy` 0.5 (Velocity y); `vz` 0.25 (Velocity z); `p` 1 (Pressure) |
| `alfven-collision` | Alfven-wave collision. Two plane Alfven waves travelling in opposite directions along a guide field B0 (along z), in uniform gas of density 1: one varies along x and is polarized along y, the other varies along y and is polarized along x (Howes & Nielson 2013). Needs a 3D periodic box. (fills the box) | `b0` 1 (Guide field); `amp` 0.25 (Wave amplitude); `beta` 10 (Plasma beta); `kperp` 1 (Waves across the field); `kpar` 1 (Waves along the field) |
| `alfven-packets` | Elsasser packets. Two Alfven-wave packets on a guide field B0 (along z), each with a Gaussian profile along the field: a z+ packet above the center heading down and a z- packet below it heading up, in uniform gas of density 1 (Verniero et al. 2018). They meet at the center at t = box / (4 B0). Needs a 3D periodic box. (fills the box) | `b0` 1 (Guide field); `amp` 0.25 (Wave amplitude); `beta` 10 (Plasma beta); `kperp` 1 (Waves across the field); `kpar` 0 (Wavelengths in a packet); `width` 0.08 (Packet width) |
| `co-ignition` | Carbon–oxygen fuel with an ignition layer. Carbon and oxygen at rest at one density, with a hot layer at the low-x edge of the box that sets it alight. Needs physical units, an equation of state with a temperature (Helmholtz) and the species of the aprox13 network. (fills the box) | `logrho` 7 (Density (log₁₀ g/cm³)); `carbon` 0.5 (Carbon fraction); `tfuel` 0.5 (Fuel temperature (10⁸ K)); `tign` 4 (Ignition temperature (10⁹ K)); `width` 30 (Ignition layer (km)) |
| `noh` | Noh implosion. Cold gas of uniform density falling at one speed toward a point, a line or a plane (Noh 1987). A shock leaves the center at a third of that speed (for gamma = 5/3), and the gas behind it is at rest, compressed 64 times in spherical symmetry (16 in cylindrical, 4 in planar). The exact solution is known everywhere, which makes it the standard test of wall heating at the center (fills the box) | `symmetry` 3 (Symmetry); `speed` 1 (Speed); `rho` 1 (Density); `p` 0.000001 (Pressure) |
| `uniform-collapse` | Uniform collapse. Gas of uniform density and temperature falling inward at a speed proportional to the distance, u = -r / t_c, so all of it reaches the center at once at t = t_c (ExactPack’s noh2). It stays uniform as it is compressed along its adiabat, with no shock: the density grows as (1 - t / t_c)^-k, for k = 3, 2 or 1 by the symmetry (fills the box) | `symmetry` 3 (Symmetry); `tCollapse` 1 (Collapse time); `rho` 1 (Density); `e` 1 (Specific internal energy) |
| `znd-detonation` | ZND detonation. The steady (ZND) detonation of the single-step Arrhenius model, moving toward +x into fuel of density 1 and pressure 1 (for gamma = 1.2; heat release 50), with its burning. Lengths are half-reaction lengths. Needs the species fuel and ash. (fills the box) | `ea` 25 (Activation temperature Ea); `overdrive` 1 (Overdrive (D / D_CJ)²); `shockburn` true (Burn inside shocks); `frontFrame` false (Frame of the front); `pocket` false (Pocket of unburned gas) |
| `evrard` | Evrard sphere. A cold gas sphere whose density falls as 1/r, rho = M / (2 pi R^2 r), in thin gas at the same specific internal energy (for gamma = 5/3). With G = M = R = 1 it collapses in about a unit of time. (fills the box) | `mass` 1 (Mass); `radius` 1 (Radius); `u` 0.05 (Specific internal energy); `rhoAmb` 0.0001 (Density outside) |
| `first-core` | First core (protostellar). A first hydrostatic core of protostellar collapse at the middle of the box: molecular gas on one adiabat, in balance with its own gravity, in thin gas at its surface pressure. Past a central density of 3e-9 g/cm^3 its center is above 2000 K, where H2 dissociates, and it collapses. Needs physical units and an equation of state with a temperature (hydrogen and helium). (fills the box) | `logrhoc` -7.75 (Central density (log₁₀ g/cm³)); `tad` 10 (Adiabat (K at 10⁻¹³ g/cm³)); `deficit` 0.05 (Pressure deficit); `logamb` -13 (Density outside (log₁₀ g/cm³)) |
| `helium-shell-ignition` | Helium shell with a hot spot. Turns the outermost mass of the first star into helium over a carbon-oxygen core and heats a spot at the base of that shell, on the +z axis. Needs the 13 species of the burning network, the tabulated equation of state and physical units. | `mhe` 0.05 (Helium shell (M☉)); `tign` 3 (Ignition temperature (10⁹ K)); `spot` 300 (Hot spot radius (km)) |
| `isothermal-collapse` | Turbulent velocity field. Uniform gas in a unit periodic box, stirred by a few long-wavelength velocity modes with random directions and phases. With the preset’s isothermal gas (sound speed 0.1) the default speed is Mach 5. (fills the box) | `rho` 1 (Density); `p` 0.01 (Pressure); `vrms` 0.5 (Rms speed); `modes` 12 (Velocity modes); `seed` 12345 (Random seed) |
| `jeans` | Jeans-unstable wave. A small adiabatic density wave along x about density 1 and sound speed 1. With one wavelength across a unit box and G = 2 pi it grows at the rate 2 pi; a shorter wave grows more slowly, or oscillates as sound once k^2 passes 4 pi G. (fills the box) | `amp` 0.001 (Amplitude); `modes` 1 (Wavelengths across) |
| `kelvin-helmholtz` | Kelvin-Helmholtz shear layer. A dense slab across the middle of the box (in y), sliding through lighter gas at one pressure, with one seeded mode at its two edges. With helium, the slab is helium in hydrogen (species he4 and h1, which then set the equation of state), denser by the ratio of their mean molecular weights. (fills the box) | `helium` false (Helium slab in hydrogen); `rhoIn` 2 (Slab density); `rhoOut` 1 (Density outside); `shear` 1 (Shear speed); `p` 2.5 (Pressure); `width` 0.5 (Slab thickness); `amp` 0.1 (Kick amplitude); `modes` 2 (Wavelengths across); `sigma` 0.035 (Kick width); `zmod` 0.2 (Variation along z) |
| `gresho` | Gresho vortex. A steady vortex (Gresho & Chan 1990): the gas turns rigidly out to a fifth of the box, slows to rest by two fifths, and the pressure balances the centrifugal force. Nothing should change; what does is the solver’s own dissipation. In 3D it is a column along z. (fills the box) | `mach` 0.1 (Mach number) |
| `taylor-green` | Taylor-Green vortex. An array of counter-rotating vortices. In 2D with viscosity it keeps its shape and decays as exp(-2 ν k² t), an exact solution of the incompressible equations; in 3D the same start, varying along z, breaks down into turbulence. (fills the box) | `mach` 0.1 (Mach number); `modes` 1 (Wavelengths across) |
| `shear-layers` | Double shear layer. Two shear layers in a doubly periodic box, the band between them moving at +1 and the rest at -1, with a small sideways kick (Bell, Colella & Glaz 1989): each layer rolls up into one vortex. Under-resolved, a scheme grows spurious vortices along the thin braids between them. (fills the box) | `mach` 0.1 (Mach number); `sharp` 30 (Layer sharpness); `kick` 0.05 (Kick amplitude) |
| `orszag-tang` | Orszag-Tang vortex. The Orszag-Tang vortex on a periodic unit box (for gamma = 5/3): uniform gas of density 25 / 36 pi and pressure 5 / 12 pi (sound speed 1) with a velocity vortex and a field of twice its wavenumber in x. (fills the box) | `v0` 1 (Velocity amplitude); `b0` 1 (Field amplitude) |
| `mhd-rotor` | Rotor. A dense disk spinning rigidly in gas of density 1 at rest, at one pressure, with a ring around it where density and speed taper (Balsara & Spicer 1999). The field is a part of its own. (fills the box) | `rho` 10 (Disk density); `v0` 2 (Rim speed); `r0` 0.1 (Disk radius); `r1` 0.115 (Outer radius of the ring); `p` 1 (Pressure) |
| `brio-wu` | Brio-Wu shock tube. The MHD shock tube of Brio & Wu (1988) along x (for gamma = 2): gas at rest in two states either side of a membrane at the center, a field B_x throughout and B_y reversing across the membrane. (fills the box) | `rhoL` 1 (Density, left); `pL` 1 (Pressure, left); `rhoR` 0.125 (Density, right); `pR` 0.1 (Pressure, right); `bx` 0.75 (Field along the tube); `by` 1 (Field across the tube) |
| `field-loop` | Field loop advection. A weak loop of magnetic field, A_z = A0 (R - r) inside radius R, in gas of density 1 and pressure 1 carried by a uniform flow: at (2, 1) on a periodic unit box it is back where it started at t = 1 (Gardiner & Stone 2005). (fills the box) | `dense` false (Dense loop); `rhoIn` 2 (Density of a dense loop); `radius` 0.3 (Loop radius); `a0` 0.001 (Field strength); `vx` 2 (Velocity x); `vy` 1 (Velocity y) |
| `alfven-wave` | Circularly polarized Alfven wave. A circularly polarized Alfven wave along the diagonal of the x-y plane of a periodic unit box: density 1, a field of 1 along the wave, and a perpendicular field and velocity that turn about it (Toth 2000). It returns to its start after each period, 1 / sqrt(2). (fills the box) | `amp` 0.1 (Amplitude); `p` 0.1 (Pressure) |
| `seed-field` | Weak in-plane field. A weak tangled magnetic field in the x-y plane (random modes of the periodic box, from a vector potential A_z), for the flow to stretch and wind up. It adds to the other parts’ fields and leaves the gas as they paint it. | `amp` 0.01 (Field strength); `kMax` 3 (Largest wavenumber); `seed` 1 (Seed); `in3d` false (Also in 3D) |
| `rayleigh-taylor` | Rayleigh-Taylor layers. Heavy gas over light gas in a unit box, in hydrostatic balance with the uniform downward acceleration of the setup (Gravity: Uniform acceleration), with one seeded mode that vanishes at the walls. Up is y in 2D and z in 3D. (fills the box) | `rhoTop` 2 (Density above); `rhoBottom` 1 (Density below); `p0` 2.5 (Pressure at the interface); `amp` 0.05 (Kick amplitude); `modes` 2 (Wavelengths across) |
| `harris-sheet` | Harris current sheet. A current sheet along y = 0: B_x = B0 tanh(y / a), with the gas pressure higher in the sheet so that p + B² / 2 is uniform, density 1, an optional uniform guide field B_z, and a perturbation A_z = psi B0 L cos(k x) cos(pi y / L) that seeds the tearing mode (Harris 1962; Furth, Killeen & Rosenbluth 1963). Periodic along x, walls along y. (fills the box) | `b0` 1 (Field strength); `width` 0.05 (Sheet half-width); `beta` 1 (Plasma beta outside); `guide` 0 (Guide field); `psi` 0.0001 (Perturbation); `modes` 1 (Wavelengths across) |
| `island-coalescence` | Island coalescence. A chain of magnetic islands along y = 0, the Fadeev equilibrium A_z = -lambda B0 ln(cosh(y / lambda) + eps cos(x / lambda)), with the gas pressure p = p_out + B0² (1 - eps²) / (2 (cosh(y / lambda) + eps cos(x / lambda))²) that holds it, density 1, and a flow along x that pushes pairs of islands together (Fadeev et al. 1965; Finn & Kaw 1977). Periodic along x, walls along y. (fills the box) | `b0` 1 (Field strength); `eps` 0.2 (Island size); `islands` 2 (Islands across); `beta` 0.4 (Plasma beta outside); `kick` 0.1 (Push) |
| `x-point` | Stressed X-point. An X-type magnetic null at the center, A_z = (B0 / 2L) (y² - alpha² x²) with L the box side, so B = (B0 / L) (y, alpha² x), in gas of density 1 and uniform pressure. With alpha above 1 it carries a current its own force cannot hold, and collapses into a current sheet. Needs walls that hold the field lines crossing them: line-tied on all four sides. (fills the box) | `b0` 1 (Field strength); `alpha` 1.2 (Stress); `p` 0.05 (Pressure) |
| `shearing-wave` | Shearing wave. The background flow of a shearing sheet, v_y = -q Omega x, with a small density wave cos(kx x + ky y) on it (kx and ky in wavelengths across the box). A leading wave (kx < 0) is swung to trailing by the shear and amplified on the way. Needs physics.shear. (fills the box) | `amplitude` 0.01 (Amplitude); `kx` -2 (Wavenumber along x); `ky` 1 (Wavenumber along y); `pressure` 0.01 (Pressure) |
| `shearing-noise` | Shearing box. The background flow of a shearing box, v_y = -q Omega x, in gas of uniform density 1 and pressure, with small random velocities on it (a fraction of the sound speed) to seed instabilities such as the magnetorotational. Needs physics.shear. (fills the box) | `noise` 0.001 (Velocity noise); `pressure` 0.6 (Pressure) |
| `shock-bubble` | Shock hitting a helium bubble. A planar shock in air (density 1, pressure 1, gamma 1.4) running toward +x into a bubble of helium at the same pressure and temperature, after Haas & Sturtevant (1987). Needs the species he4 and air, in that order, each with its own gamma. (fills the box) | `mach` 1.22 (Shock Mach number); `xhe` 0.72 (Helium in the bubble); `radius` 0.15 (Bubble radius) |
| `thermal-instability` | Gas in unstable thermal balance. Uniform gas at the temperature where interstellar cooling balances heating on the unstable branch, with small random density perturbations at one pressure. Density in atoms per cm^3, temperature in units of 1000 K: needs the interstellar cooling, in those units. (fills the box) | `n0` 2 (Mean density (cm⁻³)); `amp` 0.02 (Perturbation); `kMax` 8 (Smallest wavelength); `seed` 20021 (Random seed) |
| `zeldovich` | Zel’dovich pancake. One plane wave along x on its growing mode in an Einstein-de Sitter universe (physics.cosmology with Omega_m = 1): the gas converges into a sheet, a caustic, at the given redshift, and then shocks. Cold gas: its specific internal energy at the start is given. (fills the box) | `zCaustic` 1 (Caustic redshift); `temperature` 1e-10 (Specific internal energy) |
<!-- /gen:setup-flows -->

## What cannot be combined

Every option declares what it requires and what it excludes, as data in the option table. The
dialog greys out an option whose conditions are not met and names the reason in its tooltip. A
file that breaks a rule still opens; the dialog's issues list says which. The rules today:

| Option | Requires, excludes |
|---|---|
| A 2D run | a setup that makes sense in 2D; no self-gravity and no star, unless the run is axisymmetric (R, z) (or, for self-gravity, spherical) |
| Axisymmetric or spherical geometry | 2D |
| Mirror symmetry | a setup symmetric under those mirrors, with its field (if any) symmetric too; a z mirror needs 3D |
| Self-gravity, periodic | 3D; every face periodic; no star |
| Self-gravity, isolated | 3D, or an axisymmetric or spherical run; no periodic face |
| Uniform gravity or acceleration | not a spherical run |
| Star | 3D or an axisymmetric run, isolated self-gravity, no periodic face |
| White dwarf | the degenerate or Helmholtz equation of state; no polytropic star |
| Polytropic star | an equation of state with an energy; no white dwarf |
| Planet | the ANEOS equation of state |
| ANEOS equation of state | physical units |
| Radiation transport | an equation of state with an energy, not Helmholtz or ANEOS; no magnetic field; Cartesian geometry |
| A region's radiation energy or flux | radiation transport |
| Resistivity, Alfvén speed cap | a magnetic field |
| Burning, fuel to ash | species that are not all nuclides |
| Point mass, disk | no periodic face |
| Disk | a point mass or a star to orbit |
| Damping zone | one a run |
| Frame follows the gas or the masses | a point mass |
| Cooling, conduction, a region's total energy | an equation of state with an energy |
| Network burning | its 13 species, physical units, an equation of state with an energy |
| Refinement thresholds, local timesteps, refine around masses | an adaptive mesh |
| Jeans floor, V-cycles, energy-conserving work | self-gravity |
| A flow that fills the box | no other such flow; the background is not used |

Checks on values run with them and show in the same list: bodies that overlap or leave the
box, a softening under two cells, a disk heavier than a tenth of what it orbits, a polytrope
whose gas has gamma below 1 + 1/n, a star's point-mass core heavier than the star or softened
over less than two cells, the grid against the GPU memory budget.

The same list shows, as an error, what the simulators refuse to run (`refusal` in `src/sim.ts`),
which is not in the option table. Among it: the liquid equation of state takes no magnetic field,
self-gravity or comoving coordinates; implicit pressure is for the liquid and the ideal gas, in
Cartesian geometry, without radiation, cooling, damping zones, burning, species that set the
equation of state, rotating or accelerating frames, shearing boxes, mirrors, per-block
timesteps, or (for the liquid) point masses, and its point masses take in no gas; radiation transport takes the ideal gas, gas + radiation, degenerate or
ionizing equation of state, not implicit pressure, comoving coordinates or a shearing box; the
hydrogen and helium equation of state needs a composition it has a table for (solar or
hydrogen); obstacles and inflows take no magnetic field.

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
