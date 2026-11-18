# Frames, symmetric domains and coordinate systems: implementation plan

Status: mostly built (2026-11-17, C and D finished 2026-11-17); see the table below. Written 2026-11-15 from a read of the
code at commit `efc14ac`. File and line references are to that commit. What was built is
described in [docs/numerics.md §22](../docs/numerics.md#22-rotating-frames-symmetry-and-coordinates)
and tested in `src/validation/{frames,symmetry,cosmo,shear,cyl}.ts`.

| Step | State |
|---|---|
| G1 domain origin | done |
| G2 frame slot, A rotating frame (A1-A5, A7) | done; A6 (rotating polytropes) not done |
| B1-B4, B7, B8 mirror symmetry | done (view shows the images; field lines stay in the computed part) |
| B5 mirror parity for fields | done |
| B6 rectangular grids | **not done**: half and 3D-quadrant runs work but gain nothing on a cubic grid |
| G3 clock, C1-C5 comoving coordinates | done (well-balanced reconstruction off in comoving runs: see §22) |
| C6 | done: every equation of state but the tabulated stellar one, through comoving wrappers; cooling (`cosmo-eos`) |
| C7 refusals | done |
| C8 | done: `physics.cosmology` with `boxMpc`/`h` units (proper values in the readouts), the pancake preset, a and z on the status line, `cosmo-wave`; the Layzer-Irvine check (optional) not done |
| D1-D3 shearing box (hydro, uniform grid) | done; the boundary's shift had the wrong sign until `shear-wave` found it (fixed). The background drifts by float32 rounding, ~1e-4 of the shear speed an orbit (orbital advection, D5, would remove it) |
| D4 MHD shearing box, D6 `shear-wave`/`shear-flux`/`shear-mri` | done (`mhd/shear.wgsl`), with the `mri-box` preset; resistivity and viscosity with a field refused; `mri-box` does not yet have mri.wgsl's disk diagnostics in the local frame |
| E1-E3, E7 (R, z) hydro, diagnostics, files | done (display by revolution not done) |
| E4 (R, z) on blocks | done, without per-block timesteps |
| E5 step 1 masses on the axis, step 2 self-gravity on the uniform grid | done |
| E5 step 3 self-gravity on blocks, settling in (R, z) | **not done** |
| E6 MHD and viscosity in (R, z) | **not done** (refused) |
| E8 tests | rest, conserve, sedov, sink (against 3D), blocks, sphere, evrard, column; noh, bondi, torus and mhd not done |
| E9 spherical | its own plan: [spherical-plan.md](spherical-plan.md) |

This plan covers five additions, in the order they should be built:

| Part | What | Size | Needs |
|---|---|---|---|
| [A](#part-a-rotating-frame) | Rotating Cartesian frame | small | G2 |
| [B](#part-b-mirror-symmetry-octant-quadrant-half) | Mirror symmetry: octant, quadrant, half | medium | G1 |
| [C](#part-c-comoving-coordinates) | Comoving (expanding) coordinates | large | G2, G3 |
| [D](#part-d-shearing-box) | Shearing box | large | A, G3 |
| [E](#part-e-axisymmetric-grids) | Axisymmetric (R, z), then (r, θ) | large | G1, B, G4 |

G1 to G4 are the shared groundwork of the [first section](#shared-groundwork).

**Contents**

1. [What the code assumes today](#what-the-code-assumes-today)
2. [Shared groundwork](#shared-groundwork)
3. [Part A: rotating frame](#part-a-rotating-frame)
4. [Part B: mirror symmetry](#part-b-mirror-symmetry-octant-quadrant-half)
5. [Part C: comoving coordinates](#part-c-comoving-coordinates)
6. [Part D: shearing box](#part-d-shearing-box)
7. [Part E: axisymmetric grids](#part-e-axisymmetric-grids)
8. [Order of work](#order-of-work)
9. [Open questions](#open-questions)

---

## What the code assumes today

- **One cell size.** `g_dx` is a scalar ([common.wgsl:124](../src/hydro/common.wgsl)), `P.dx`
  on the uniform grid and `P.dx / 2^level` on blocks ([grid.wgsl:33](../src/amr/grid.wgsl)).
  The update multiplies net fluxes by `g_dt / g_dx` ([advance.wgsl:11](../src/hydro/advance.wgsl));
  36 expressions in 16 shaders divide by the cell size.
- **A cube, centered on the origin.** `P.n` is the cell count along x and y, `P.dims.z` is `n`
  or 1, and block grids have `P.amr.y` root blocks per side on every axis. A cell's position is
  `(c + 0.5) * dx - 0.5 * P.box`, written out separately in 10 shaders and about 15 places in
  TypeScript (listed under [G1](#g1-one-definition-of-where-a-cell-is)).
- **One cell volume.** `P.geom.x` ([common.wgsl:9](../src/hydro/common.wgsl)); restriction and
  prolongation weight children equally (`0.125` or `0.25`, [cons.wgsl:33](../src/amr/cons.wgsl),
  [regrid.wgsl:74](../src/amr/regrid.wgsl), [species_regrid.wgsl:51](../src/amr/species_regrid.wgsl)).
- **Boundaries per face**, as an index mapping with a velocity flip
  (`bc_map`, [common.wgsl:167](../src/hydro/common.wgsl)); on blocks, neighbor codes -2 outflow,
  -3 reflecting, -4 outflow-only ([grid.wgsl:67](../src/amr/grid.wgsl)).
- **One gravity boundary for all faces.** `P.grav_bc` is 0 periodic or 1 isolated
  ([stencil.wgsl:16](../src/gravity/stencil.wgsl), [accel.wgsl:11](../src/gravity/accel.wgsl),
  [prolong.wgsl:12](../src/gravity/prolong.wgsl), [gfield.wgsl:28](../src/gravity/gfield.wgsl);
  on blocks any non-neighbor that is not coarser is treated as isolated,
  [amr/gravity/stencil.wgsl:25](../src/amr/gravity/stencil.wgsl)).
- **No self-gravity in 2D.** `configFor` switches it off ([sim.ts:300](../src/sim.ts)); both
  multigrid solvers are 3D only.
- **External forces already have one home.** [gravity/ext.wgsl](../src/gravity/ext.wgsl) gives
  the point masses' pull and the accelerating frame's as `ext_accel`, `ext_dphi` and `ext_phi`.
  All 30 uses (the kicks, the predictors, the timestep, the well-balanced reconstruction, the
  display) go through these three functions, and this part of the kick always takes its energy
  in kick form. Parts A and D add their potentials here.
- **The second half kick is its own pass**, run only when there is gravity
  (`anyGravity`, [sim.ts:1453](../src/sim.ts)).
- **The clock is 16 bytes on the GPU** (`dt, t, tc, steps`,
  [common.wgsl:50](../src/hydro/common.wgsl)), with the size written as a literal `16` in nine
  places (`sim.ts` 714, 859, 862, 1611, 1750; `amr_sim.ts` 408, 409, 2347; `ui/api.ts` 294).

## Shared groundwork

Each item is a refactor that leaves every existing run bit-identical. Each lands as its own
change, with the validation suite (`npm run test:gpu`) as the check.

### G1. One definition of where a cell is

Add `org: vec4<f32>` to `Params`: the position of the domain's lower corner, today
`-0.5 * box` on every axis. `(c + 0.5) * dx + P.org.xyz` is the same arithmetic on the same
values as today, so results do not change.

- **Shaders.** Replace `- 0.5 * P.box` with `+ P.org.xyz` in
  [hydro/update.wgsl:86](../src/hydro/update.wgsl), [hydro/diagnostics.wgsl:36](../src/hydro/diagnostics.wgsl),
  [hydro/damping.wgsl:22](../src/hydro/damping.wgsl), [gravity/bh.wgsl:37](../src/gravity/bh.wgsl),
  [gravity/multipole.wgsl:8](../src/gravity/multipole.wgsl), [gravity/stencil.wgsl:21](../src/gravity/stencil.wgsl),
  [amr/grid.wgsl:55](../src/amr/grid.wgsl), [amr/gravity/stencil.wgsl:30](../src/amr/gravity/stencil.wgsl),
  [render/field.wgsl:98](../src/render/field.wgsl) and [mhd/mri.wgsl:40](../src/mhd/mri.wgsl).
  The two damping zones written in presets ([binary.ts:52](../src/problems/binary.ts),
  [mri_disk.ts:105](../src/problems/mri_disk.ts)) take the box half-width as a literal and
  need the same care.
- **TypeScript.** One helper, `domainLo(cfg): [number, number, number]`, replaces
  `box / 2` in `sim.ts` (1160, 1246, 1546), `amr/amr_sim.ts` (1191, 2176, 2484),
  `amr/init_cpu.ts` (79, 188), `mhd/mhd.ts` (87), `mhd/amr_mhd.ts` (582), `io/capture.ts` (11)
  and `reference/mhd.ts` (209).
- **Params.** `PARAM_BYTES` grows from 272 to 288 ([sim.ts:624](../src/sim.ts)); `packParams`
  writes `org` at float index 68.

### G2. A `frame` slot for velocity-dependent forces

A new shader slot, `<frame>`, next to `eos`, `riemann` and `diffusion` in `hydroSlots`
([sim.ts:355](../src/sim.ts)). Its stub, `gravity/frame_off.wgsl`, declares
`const ROTATING = false;` and empty functions, so runs without a frame compile exactly as now.
Its hooks sit at the two half kicks of every update. Part A fills it with the Coriolis turn,
Part D reuses that, and Part C fills it with the expansion drag.

### G3. A clock that can carry more than the time

Replace the literal `16` with `CLOCK_BYTES` and grow `Clock` to 40 bytes:
`dt, t, tc, steps, a, ah, a1, hub, shear, shear_c`. `a`, `ah` and `a1` are Part C's scale
factor at the start, middle and end of the step and `hub` its expansion rate; `shear` and
`shear_c` are Part D's boundary offset and its Kahan compensation. They are one or zero
otherwise and nothing reads them. Restart files gain the
new fields (`readRestart`, [sim.ts:713](../src/sim.ts); `RestartState`).

### G4. A `geom` slot for areas and volumes

Only Part E needs this, so it is specified [there](#e1-the-geom-slot). It is listed here
because Parts B and E share the boundary-kind mask it introduces (`P.sym`).

---

## Part A: rotating frame

**Goal.** Run in a frame that turns at a constant angular velocity $`\boldsymbol\Omega`$ about
an axis through the origin: corotating binaries, rotating stars at rest, and the Coriolis force
the shearing box needs.

**Equations.** The gas feels

```math
\mathbf{a} = -2\,\boldsymbol\Omega\times\mathbf{v} - \nabla\Phi_c,
\qquad \Phi_c = -\tfrac12\,|\boldsymbol\Omega\times\mathbf{x}|^2 .
```

The induction equation is unchanged in a rotating frame, so MHD needs only the momentum terms.

### A1. Configuration

- `Problem.rotation?: (params, ndim) => { omega: [number, number, number]; centrifugal?: boolean }`
  in [types.ts](../src/problems/types.ts), next to `frame`. In 2D only `omega[2]` may be
  non-zero. `centrifugal: false` gives Coriolis alone (an f-plane), which Part D and one test use.
- `Params` gains `rot: vec4<f32>` ($`\boldsymbol\Omega`$, and 1 in `w` for the centrifugal
  term), written by `packParams`.
- `hydroSlots` maps `frame` to `gravity/frame_rot.wgsl` when the preset has a rotation.

### A2. Centrifugal potential

It goes into [gravity/ext.wgsl](../src/gravity/ext.wgsl), so every user gets it:

- `ext_on()` becomes `P.bh.z != 0.0 || P.rot.w != 0.0`.
- `ext_accel` adds $`-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf{x})`$.
- `ext_dphi(x, d, ahead)` adds the exact difference
  $`-(\boldsymbol\Omega\times\mathbf{x})\cdot(\boldsymbol\Omega\times\mathbf{d}) - \tfrac12|\boldsymbol\Omega\times\mathbf{d}|^2`$,
  formed directly, as the point masses' part is, not as a difference of two large potentials.
- `ext_phi` adds $`\Phi_c`$, so the potential view shows the effective potential.

Consequences that follow without further code: the well-balanced reconstruction holds a gas at
rest in the rotating frame; the timestep's gravity limit sees the centrifugal pull
([rates.wgsl:39](../src/hydro/rates.wgsl)); its work is in kick form, like the frame's.

Two gates need widening, because both are keyed to gravity today:

- `anyGravity` ([sim.ts:1410](../src/sim.ts), and its counterpart in `amr_sim.ts`) must be true
  with a rotation, so the correction pass runs.
- The clock applies the gravity limit only if `P.grav_on != 0u || PER_DX`
  ([clock.wgsl:29](../src/hydro/clock.wgsl)).

`BH` is bound by [ext_bind.wgsl](../src/gravity/ext_bind.wgsl) in every pass that includes it;
confirm the buffer exists in a run with no masses.

### A3. Coriolis force as an exact turn

`gravity/frame_rot.wgsl` provides

```wgsl
fn frame_turn(m: vec3<f32>, dt: f32) -> vec3<f32>   // m rotated by -2 |Omega| dt about Omega
```

by Rodrigues' formula. A turn leaves $`|\mathbf{m}|`$ unchanged, so the Coriolis force adds no
kinetic energy and `E` needs no source. The step becomes

```
turn(dt/2)  kick(dt/2)  fluxes  kick(dt/2)  turn(dt/2)
```

which is symmetric in time. Where each piece goes:

| Pass | Change |
|---|---|
| [hydro/update.wgsl:185](../src/hydro/update.wgsl) | turn `u1.yzw` by `dt/2` before `advance`; `ekick` uses the turned momentum |
| [gravity/correct.wgsl:59](../src/gravity/correct.wgsl) | turn `mom` by `dt/2` after the kick |
| [amr/update.wgsl:396](../src/amr/update.wgsl), [amr/correct.wgsl:66](../src/amr/correct.wgsl) | the same, with `g_dt` (per-block timesteps included; the `g_deposit` branch applies no kick and no turn) |
| [mhd/update.wgsl:268](../src/mhd/update.wgsl) | stage 1: turn by `dt/2` (its whole half-step source); stage 2: as the hydro update |
| [mhd/amr_update.wgsl:455](../src/mhd/amr_update.wgsl) | as the uniform MHD update |
| [hydro/predictor.wgsl:87](../src/hydro/predictor.wgsl) | add $`-2\boldsymbol\Omega\times\mathbf{v}`$ to `g` in the half-step velocity (first order in the predictor is enough) |

The predictor is shared by the uniform grid ([predict.wgsl:46](../src/hydro/predict.wgsl)) and
blocks ([amr/predict.wgsl:165](../src/amr/predict.wgsl)), so the one edit covers both.

### A4. Point masses

- [bh_drift.wgsl](../src/gravity/bh_drift.wgsl): half kick, turn by `dt/2`, drift.
- [bh_kick.wgsl](../src/gravity/bh_kick.wgsl): add the centrifugal acceleration to `acc`; turn
  by `dt/2`, then the closing half kick.
- The accelerating frames (`P.bh.y`) keep working: the rotation axis passes through the origin,
  where both frames hold their center of mass.

### A5. Timestep, diagnostics, output

- **Timestep.** [clock.wgsl](../src/hydro/clock.wgsl): `dt <= 0.1 / |Omega|`, about 63 steps a
  turn. The turn is exact, so this limits the splitting error, not stability.
- **Energy.** [hydro/diagnostics.wgsl:35](../src/hydro/diagnostics.wgsl) and
  [amr/diagnostics.wgsl:59](../src/amr/diagnostics.wgsl) add $`\rho\,\Phi_c`$ to `egrav`, so
  `eTot` is the Jacobi integral, which is what the frame conserves.
- **Debris energies** ([debris.wgsl:47](../src/debris/debris.wgsl)) and the **disk
  diagnostics** ([mri.wgsl:59](../src/mhd/mri.wgsl)) use velocities relative to the frame. Both
  need $`\boldsymbol\Omega\times\mathbf{x}`$ added back.
- **Initial conditions.** `Problem.init` returns velocities in the frame of the run. A helper in
  `problems/`, `toRotating(v, x, omega)`, converts an inertial setup.
- **Files and links.** The rotation comes from the preset's parameters, which snapshots and
  links already carry. The API's `describe` reports it. Velocities in `.athdf` files are frame
  velocities; `docs/api.md` says so.

### A6. Later, optional

Rigidly rotating stars: add $`\Phi_c`$ to the potential the settling balances against
([settle.ts](../src/problems/settle.ts), [amr/settle.wgsl](../src/amr/settle.wgsl)), giving
rotating polytropes at rest in the frame.

### A7. Tests

New entries in [validation/tests.ts](../src/validation/tests.ts):

| Test | Passes when |
|---|---|
| `rot-off` | a preset with `omega = 0` gives the same bits as without the slot |
| `rot-inertial` | uniform gas with velocity $`v_0`$, periodic box, Coriolis only: the velocity turns once in $`\pi/\Omega`$, kinetic energy constant to round-off, phase within $`10^{-4}`$ |
| `rot-balance` | isothermal gas with $`\rho\propto\exp(\Omega^2R^2/2c_s^2)`$ in a reflecting box stays at Mach $`<10^{-4}`$ for a sound crossing (uniform grid and blocks) |
| `rot-blob` | a pressure-matched blob moving in a straight line in the inertial frame follows $`\mathsf{R}(-\Omega t)(\mathbf{x}_0 + \mathbf{v}_0 t)`$ to half a cell over a quarter turn |
| `rot-binary` | two point masses on a circular orbit, frame turning at the orbital rate: each stays within $`10^{-3}`$ of the separation for two orbits |
| `rot-mhd` | `rot-inertial` with a uniform field along $`\boldsymbol\Omega`$: field unchanged to round-off |

Unit test: `frame_turn`'s TypeScript mirror against a rotation matrix.

---

## Part B: mirror symmetry (octant, quadrant, half)

**Goal.** Compute one part of a mirror-symmetric problem: the lower face of each mirrored axis
is the plane of symmetry, through the origin.

### B0. A limit to settle first

The grid is a cube with cubic cells. Mirroring one axis moves that axis's domain from
$`[-L/2, L/2]`$ to $`[0, L]`$ and leaves the others alone. So:

| Symmetry | Domain | Gain on a cubic grid |
|---|---|---|
| Octant (3D) | $`[0, L]^3`$ | twice the resolution, or an eighth of the cells |
| Quadrant (2D) | $`[0, L]^2`$ | twice the resolution, or a quarter of the cells |
| Quadrant (3D), half (2D or 3D) | mixed | **none**: the unmirrored axes still span the whole object |

A star of radius $`R`$ in a half domain still needs $`L \ge 2R`$ for the unmirrored axes, which
is the full box. Half and 3D-quadrant runs only pay with a rectangular grid ($`n_x \ne n_y \ne n_z`$,
cells still cubic). That is step [B6](#b6-rectangular-grids-for-half-and-quadrant-runs), and it
is the larger half of this part. This corrects the earlier ranking, which treated all three
cases alike.

### B1. Configuration

- `SimConfig.symmetry: [boolean, boolean, boolean]`, default all false.
- `Problem.symmetry?: { axes: [boolean, boolean, boolean]; mhd?: boolean }`: which mirrors the
  setup is actually symmetric under. The interface offers only these, and a link or API call
  that asks for another is ignored, as other out-of-range values are.
- With a mirror on axis `a`: `domainLo[a] = 0` (G1), `bcLo[a] = REFLECT`.
- `Params` gains `sym: vec4<u32>`: per axis, 0 none, 1 mirror plane, 2 axis of symmetry
  (Part E). The hydro boundary stays `REFLECT`, so every existing reflecting path applies. The
  mask marks where a reflecting face is a plane of symmetry and not a wall.
- Link key `sym` (`x`, `xy`, `xyz`) in [ui/link.ts](../src/ui/link.ts), written near line 123
  and read near 289. A selector in the grid panel of [ui/app.ts](../src/ui/app.ts).

### B2. Hydro

Nothing changes in the update. Two things to verify and one to change:

- **Zero mass flux through the plane.** With mirrored states the Riemann problem is symmetric
  and the contact speed should be exactly zero. A test checks the mass flux is zero to the bit;
  if a solver leaves a rounding residue, the face flux is set to zero where `P.sym` says so.
- **Wall or plane in the well-balanced reconstruction.** At a reflecting wall, `dphi_step`
  continues the potential in a straight line across the face
  ([gfield.wgsl:59](../src/gravity/gfield.wgsl)). At a plane of symmetry the potential is even,
  and the face must be treated as an interior face between a cell and its image: `face = seen`.
  The override at lines 59 to 69 is skipped where `P.sym[axis] == 1`. The same in
  [amr/update.wgsl:95](../src/amr/update.wgsl) and [amr/predict.wgsl:68](../src/amr/predict.wgsl).
- Sources, obstacles, brushes and the damping zone work in the computed domain as they are.

### B3. Self-gravity

A mirror face takes a zero-gradient condition; the other faces stay isolated.

**Uniform grid.**

| File | Change at a mirror face |
|---|---|
| [stencil.wgsl:10](../src/gravity/stencil.wgsl) `nb_one` | return the cell's own value and no diagonal term (ghost = image), on every level |
| [prolong.wgsl:10](../src/gravity/prolong.wgsl) `coarse` | sign +1 for the mirrored axis, where the isolated error takes -1 |
| [accel.wgsl:6](../src/gravity/accel.wgsl) `grav_axis` | lower neighbor is the cell itself, weight stays 0.5 |
| [gfield.wgsl:26](../src/gravity/gfield.wgsl) `phi_self` | the mirrored cell's value, not the linear extrapolation |

**Blocks.**

| File | Change at a mirror face (neighbor code -3 with `P.sym`) |
|---|---|
| [amr/gravity/stencil.wgsl:25](../src/amr/gravity/stencil.wgsl) `g_nb` | the cell's own value, no diagonal term, on every sublevel |
| [amr/gravity/ghost.wgsl:79](../src/amr/gravity/ghost.wgsl) | ghosts are mirror images (`depth 1 -> p0`, `depth 2 -> p1`), not extrapolated |
| [amr/gravity/ghost.wgsl:19](../src/amr/gravity/ghost.wgsl) `coarse_phi` | a coarse cell beyond the plane is its image, not the fallback |
| `amr/gravity/prolong.wgsl` `coarse` | sign as on the uniform grid |

**Multipole boundary.** The reductions keep summing over the grid
([multipole.wgsl](../src/gravity/multipole.wgsl), [amr/multipole.wgsl](../src/amr/multipole.wgsl)).
`phi_boundary` ([multipole_eval.wgsl](../src/gravity/multipole_eval.wgsl), shared by both grids)
unfolds them: with $`k`$ mirrored axes, mass and second moments times $`2^k`$, the center of mass
zero along mirrored axes, and every off-diagonal moment that involves a mirrored axis zero.
About ten lines in one file.

### B4. Point masses, frames, stars

- **Point masses** must lie on every mirror plane (position and velocity zero along mirrored
  axes); a preset that places one elsewhere is refused with a message. A mass on the plane is
  its own image, so its pull on the gas is unchanged.
- **Their reduction** ([bh_cell.wgsl](../src/gravity/bh_cell.wgsl), read in
  [bh_kick.wgsl](../src/gravity/bh_kick.wgsl)): gas mass, accreted mass and the unmirrored
  components of pull and momentum times $`2^k`$; mirrored components zero. Done once where
  `bh_kick` reads `RED`, so the per-cell code is untouched.
- **Accelerating frames:** `abar` is zero along mirrored axes by the same rule.
- **Settling** ([settle.ts](../src/problems/settle.ts) `balancedDensity`,
  [amr/settle.wgsl](../src/amr/settle.wgsl)): the mass the grid must hold is the star's over
  $`2^k`$. A star not on the planes has images, which must join the "other stars" whose pull is
  removed ([settle.ts:223](../src/problems/settle.ts)); first version: stars on the planes only.

### B5. Magnetic fields

A reflecting face today flips the normal field (`bcell(m.c) * m.flip`,
[mhd/common.wgsl:57](../src/mhd/common.wgsl)): the condition at a conducting wall. A magnetic
field is a pseudovector, so under a mirror it is the **tangential** components that change sign
and the normal one that is kept. A plane of symmetry therefore needs the opposite parity, in
every place the field is mirrored:

- cell-centered: [mhd/common.wgsl:57](../src/mhd/common.wgsl), [flux.wgsl:85](../src/mhd/flux.wgsl),
  [emf.wgsl:52](../src/mhd/emf.wgsl), and on blocks [amr_common.wgsl](../src/mhd/amr_common.wgsl)
  (85, 107, 159, 212, 331) and [amr_update.wgsl:107](../src/mhd/amr_update.wgsl);
- faces: `face_texel` ([mhd/common.wgsl:30](../src/mhd/common.wgsl)) treats a tangential face
  as even across a boundary; across a mirror it is odd, so `bface` needs a sign as well as a
  texel;
- the normal face on the plane itself is a real, evolving face, updated from the edge fields on
  the plane; check the upwinding in [emf.wgsl](../src/mhd/emf.wgsl) there.

Initial fields: a vector potential is an ordinary vector, so it must be odd in its normal
component and even in its tangential ones. `Problem.symmetry.mhd` says the preset's field
qualifies. The stars' dipole along z ([star_fields.ts](../src/mhd/star_fields.ts)) is symmetric
under the z mirror only, so magnetized stars are a half-domain case and wait for B6.

MHD symmetry is the last step of this part and can ship after the rest.

### B6. Rectangular grids, for half and quadrant runs

Cells stay cubic and `g_dx` stays a scalar; only the counts differ per axis. This is much less
than non-cubic cells, and the multigrid smoother is unaffected. What changes:

- `SimConfig.n` becomes the count along the longest axis, with a per-axis divisor of 1 or 2;
  `P.dims.xyz` already carries three counts.
- `P.n` is read in 5 places in 3 shaders (`bc_map`, `gfield.wgsl`, `accel.wgsl`) and must
  become `P.dims[axis]`.
- The uniform multigrid's level size `L.n` (6 uses) becomes a vector; the coarsest level is
  $`2\times2\times4`$ or $`2\times4\times4`$, with a few more sweeps.
- Block grids: `P.amr.y` root blocks per side (15 uses) becomes three counts, in
  [tree.ts](../src/amr/tree.ts) and the shaders that wrap or index by it.
- Texture allocation, dispatch sizes, the renderer's box, the API's box fractions and
  `.athdf` root sizes ([capture.ts:27](../src/io/capture.ts)) take three counts.

### B7. Diagnostics, display, files

- **Totals** are reported for the whole configuration. One helper, applied where the reductions
  are unpacked into `Diagnostics` ([sim.ts:677](../src/sim.ts) and `amr_sim.ts`): extensive
  quantities times $`2^k`$; momentum and center of mass zero along mirrored axes. The same for
  `MagDiagnostics` and the debris census. `Diagnostics` gains `symmetry`.
- **Display.** A "Show mirror images" option, on by default: the volume samplers
  ([volume_uniform.wgsl](../src/render/volume_uniform.wgsl),
  [volume_atlas.wgsl](../src/render/volume_atlas.wgsl)), the slice
  ([slice.wgsl](../src/render/slice.wgsl)) and the depth pick ([pick.wgsl](../src/render/pick.wgsl))
  take the absolute value of the sample position along mirrored axes, and the camera's box
  doubles along them. Block outlines and field lines stay in the computed domain.
- **Files.** Snapshots hold the computed domain with its true bounds (`lo`, `hi` from
  `domainLo`), so yt reads an octant as an octant. The settings in the `Shaderous` attribute
  carry `symmetry`, so a restart continues on the same mesh.

### B8. Presets and tests

Presets that declare a symmetry in the first version: `sedov`, `sedov-rad`, `evrard`, the
single polytropes and white dwarf, the uniform sphere, the isothermal collapse. A blast's
deposit region ([sedov.ts:28](../src/problems/sedov.ts)) is set by position, so it needs no
change: the origin is on a cell corner in both layouts.

| Test | Passes when |
|---|---|
| `sym-refactor` | G1 alone: every existing test unchanged |
| `sym-flux` | mass flux through a mirror face is zero to the bit |
| `sym-sedov` | octant at $`n`$, $`L/2`$ against the full box at $`2n`$, $`L`$: densities agree to $`10^{-5}`$ (2D quadrant too) |
| `sym-sphere` | potential of a uniform sphere in an octant within 0.3% of the analytic one, and within $`10^{-4}`$ of the full run |
| `sym-ellipsoid` | the same for a triaxial body, which exercises the unfolded quadrupole |
| `sym-polytrope` | a settled star in an octant keeps its central density as well as the full run |
| `sym-evrard` | energy conservation as in the full run; on blocks with per-block timesteps too |
| `sym-mass` | a point mass with a sink at the origin: accreted mass equal to the full run's to $`10^{-5}`$ |
| `sym-mhd` | a field from $`\mathbf{A} = (x y^2 f(r), 0, 0)`$, symmetric under all three mirrors: octant against full run, $`\nabla\cdot B`$ at round-off |

Unit tests: the moment unfolding, the diagnostics unfolding, the link key, `domainLo`.

---

## Part C: comoving coordinates

**Goal.** Gas in an expanding background, for cosmological test problems, with any equation of
state and with cooling.

**Scope warning.** Shaderous has no collisionless particles, so this is a gas-only universe.
It gives the Zel'dovich pancake, linear growth and adiabatic expansion; a realistic structure
run needs dark matter, which is a separate and larger project.

### C1. Formulation: comoving positions, proper peculiar velocities

The form Enzo uses ([Bryan et al. 1995](https://doi.org/10.1016/0010-4655(94)00191-4),
[2014](https://doi.org/10.1088/0067-0049/211/2/19)). With scale factor $`a(t)`$ and $`t`$ the
proper cosmic time:

- $`\mathbf{x}`$ is the comoving position, so the grid and the box do not change;
- $`\rho = a^3\rho_\text{proper}`$ and $`p = a^3p_\text{proper}`$ are comoving densities, so
  the total mass on the grid is constant;
- $`\mathbf{v}`$ is the proper peculiar velocity, and the specific internal energy $`e`$ is the
  proper one, so the temperature and sound speed are physical as they stand.

```math
\begin{aligned}
\partial_t\rho + \tfrac1a\nabla\cdot(\rho\mathbf{v}) &= 0,\\
\partial_t(\rho\mathbf{v}) + \tfrac1a\nabla\cdot(\rho\mathbf{v}\mathbf{v} + p\,\mathsf{I}) &= -\tfrac{\dot a}{a}\rho\mathbf{v} - \tfrac{\rho}{a}\nabla\Phi,\\
\partial_tE + \tfrac1a\nabla\cdot\left[(E + p)\mathbf{v}\right] &= -\tfrac{\dot a}{a}\left(\rho v^2 + 3p\right) - \tfrac{\rho}{a}\mathbf{v}\cdot\nabla\Phi,\\
\partial_t(\rho e) + \tfrac1a\nabla\cdot(\rho e\,\mathbf{v}) &= -\tfrac{p}{a}\nabla\cdot\mathbf{v} - 3\tfrac{\dot a}{a}p,\\
\nabla^2\Phi &= \tfrac{4\pi G}{a}\left(\rho - \bar\rho\right).
\end{aligned}
```

Three things differ from the equations the code solves now, and each has one home:

| Difference | Where it goes |
|---|---|
| every gradient carries $`1/a`$ | the cell size the passes use becomes $`a\,\Delta x`$ (C3) |
| the expansion drag, $`-\tfrac{\dot a}{a}\rho\mathbf{v}`$ and $`-3\tfrac{\dot a}{a}p`$ | the hooks of the `frame` slot (C4) |
| the Poisson source carries $`1/a`$ | the two source passes (C5) |

These equations must be checked against the papers before coding; each test in C8 names what
it exercises.

**Code units.** Time in units of $`1/H_0`$, the mean comoving density 1, and so
$`G = 3\Omega_m/8\pi`$. The initial conditions must have unit mean density. The clock starts at
zero at $`a_\text{start}`$.

### C2. Evolving the scale factor on the GPU

The timestep never visits the CPU, so $`a`$ is advanced in
[clock.wgsl](../src/hydro/clock.wgsl):

```math
\frac{da}{dt} = a\,H(a), \qquad H(a) = \sqrt{\Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_\Lambda}
```

- One RK4 step per timestep, writing `clock.a`, `clock.ah` and `clock.a1` (the start, middle
  and end of the step) and `clock.hub` ($`H`$ at the middle), G3.
- A new limit: $`\Delta t \le 0.01/H(a)`$, so $`a`$ changes by at most 1% a step.
- The end time and output times are given as redshifts and converted on the CPU by integrating
  the same equation in float64.
- `Params` gains `cosmo: vec4<f32>` ($`\Omega_m`$, $`\Omega_k`$, $`\Omega_\Lambda`$, on).

### C3. The factor 1/a on every gradient

Every pass sets `g_dx` first ([common.wgsl:122](../src/hydro/common.wgsl)). With cosmology it
is set to the proper cell size, `clock.ah * P.dx` (and the block's own on a mesh). One scalar
then covers:

- the flux update, `g_dt / g_dx` ([advance.wgsl:11](../src/hydro/advance.wgsl)), and the
  predictor ([predictor.wgsl:76](../src/hydro/predictor.wgsl));
- the timestep: $`C\,a\Delta x/(|v| + c_s)`$ and $`C\sqrt{a\Delta x/|g|}`$;
- the gradients of viscosity and conduction ([diffusion_core.wgsl:82](../src/hydro/diffusion_core.wgsl));
- the flux-form gravitational work, which rides on the same `dt/dx`.

The well-balanced reconstruction needs nothing: it works with potential differences, and
$`\tfrac1a\nabla p = -\tfrac{\rho}{a}\nabla\Phi`$ has the same $`a`$ on both sides.

Both cells of a face use the same $`a`$, so their fluxes still agree to the bit.

What needs work is the places that read `P.dx` directly and not `g_dx`. Of the 36 cell-size
expressions in 16 shaders, the ones to change are
[species.wgsl:73](../src/hydro/species.wgsl), [species_predict.wgsl:19](../src/hydro/species_predict.wgsl),
[clock.wgsl:27](../src/hydro/clock.wgsl), [accel.wgsl:18](../src/gravity/accel.wgsl) (so the
stored field is the proper acceleration, $`-\nabla\Phi/a`$), the block field pass
([amr/gravity/field.wgsl](../src/amr/gravity/field.wgsl)), and the six `dtdx` sites each of
[mhd/update.wgsl](../src/mhd/update.wgsl) and [mhd/amr_update.wgsl](../src/mhd/amr_update.wgsl).
The first step of this part is to route all of them through one helper, `cell_dx()`, as a
refactor that changes no bits. Cell **positions** keep the comoving `P.dx`.

### C4. The expansion drag

Both drag terms are solved exactly over a time interval, so they are applied as an operator
split around the step, half before and half after (second order). They go where Part A's turn
goes, through the same `frame` slot hooks (G2), with the cosmological module in the slot:

- **Momentum:** $`\mathbf{v}\propto 1/a`$, so the momentum is multiplied by
  $`a(t)/a(t + \Delta t/2)`$: by `clock.a / clock.ah` in the update
  ([update.wgsl:185](../src/hydro/update.wgsl)) and `clock.ah / clock.a1` in the correction
  ([correct.wgsl:59](../src/gravity/correct.wgsl)), and likewise on blocks and in the MHD
  updates (the table of [A3](#a3-coriolis-force-as-an-exact-turn)).
- **Internal energy:** $`d(\rho e)/d\ln a = -3p`$.
  - Ideal gas: $`\rho e`$ is multiplied by $`(a/a')^{3(\gamma - 1)}`$, exactly.
  - Any other equation of state: two midpoint stages with `eos_pressure`.
- **Total energy:** `E` is changed by exactly what the kinetic and internal energies changed
  by, so the dual-energy pair stays synchronized and `EOS_KEEP_TOTAL` gases keep their rule.
- The correction pass must run (`anyGravity`, [sim.ts:1410](../src/sim.ts)), as in A2.

Unlike Part A's turn, the drag depends on when a block steps, which matters for C7.

### C5. Gravity

- **Poisson source.** [gravity/rhs.wgsl:13](../src/gravity/rhs.wgsl) and
  [amr/gravity/rhs.wgsl:15](../src/amr/gravity/rhs.wgsl) divide by `clock.a1` (the solve is for
  the updated state). Both bind the clock for the first time.
- **The field** is stored as the proper acceleration (C3), so the kicks, the predictor and the
  timestep use it unchanged.
- **Jeans length.** The floor ([pfloor.wgsl:6](../src/hydro/pfloor.wgsl)), the refinement
  criterion ([flag.wgsl](../src/amr/flag.wgsl)) and the diagnostic compare a proper Jeans
  length, $`c_s\sqrt{\pi a^3/G\rho}`$, with the proper cell size; each takes its factor of $`a`$.

### C6. Equations of state and cooling

The specific energy is proper, but the density and pressure the state holds are comoving, so a
call to the equation of state must convert: with $`\rho_p = \rho/a^3`$,

```math
p = a^3\,p_\text{eos}\!\left(\rho_p,\ \rho_p e\right), \qquad
c_s = c_{s,\text{eos}}(\rho_p, p/a^3), \qquad T = T_\text{eos}(\rho_p, \rho_p e).
```

- **Ideal gas:** $`p = (\gamma - 1)\rho e`$ is the same in both, so nothing changes. This is the
  first step, and it needs no wrapper.
- **Other equations of state:** the four functions of the interface
  ([numerics.md §9](../docs/numerics.md#9-equations-of-state)) are called through wrappers that
  divide and multiply by $`a^3`$, from a per-invocation `g_a3` set beside `g_dx`. Most calls
  pass through [prim.wgsl](../src/hydro/prim.wgsl); every other direct call (the predictor, the
  diagnostics, the display, the floors, cooling) is found by a search for `eos_` and moved to
  the wrappers. Without cosmology the wrappers are the plain calls.
- **Cooling** ([hydro/cooling.wgsl](../src/hydro/cooling.wgsl)) gets the proper density and the
  temperature from the wrappers, and its rate per proper volume is multiplied by $`a^3`$.
- **Composition** moments are mass fractions and need nothing.

### C7. What is refused in the first version

Guards in `configFor`, with a message each, as `mhdUnsupported` does for MHD:

- point masses, sinks, isolated gravity and non-periodic boundaries;
- **per-block timesteps.** A slower block takes a faster neighbor's fluxes, recomputed, and
  each would divide by its own $`a`$. The later fix is one $`a`$ for a whole cycle of substeps,
  with the 1% limit applied to the cycle;
- **magnetic fields.** The comoving field and its expansion source have to be derived and
  checked first; with the right power of $`a`$ the source is a uniform scaling of every face,
  which keeps the divergence at zero.

### C8. Units, display, presets, tests

- **Units.** `Problem.cosmology = { omegaM, omegaL, aStart, h, boxMpc }`. A cosmological unit
  system in `src/units/` converts for display with the current $`a`$: proper density
  $`\rho/a^3`$ and proper lengths $`a\,x`$; velocities and temperatures are already physical.
  The status line and `Diagnostics` show $`a`$ and the redshift.
- **Diagnostics.** The total energy is not conserved in an expanding box. The Layzer-Irvine
  integral could be added later as the conserved check.
- **Presets** in a new "Cosmology" group: the Zel'dovich pancake, and a small-amplitude
  plane wave for linear growth (a variant of [jeans.ts](../src/problems/jeans.ts)).

| Test | Passes when | Checks |
|---|---|---|
| `cosmo-off` | a non-cosmological run gives the same bits, after `cell_dx()` and after the wrappers | C3, C6 compile out |
| `cosmo-a` | $`a(t)`$ on the GPU within $`10^{-5}`$ of float64 from $`a = 0.01`$ to 1, for Einstein-de Sitter and a flat $`\Lambda`$ model | C2 |
| `cosmo-drift` | uniform gas with a uniform velocity, periodic: $`v \propto 1/a`$ and comoving $`p \propto a^{3 - 3\gamma}`$ to $`10^{-5}`$ | C4 |
| `cosmo-wave` | a sound wave's comoving wavelength is fixed while its period follows $`a/c_s`$ | C3 |
| `cosmo-growth` | a 1% plane wave grows as $`a`$ in Einstein-de Sitter, within 1% over a factor 10 | C5 |
| `cosmo-pancake` | Zel'dovich pancake: density and velocity against the exact solution before the caustic; on blocks too | everything together |
| `cosmo-eos` | uniform gas + radiation expanding: $`T(a)`$ against a float64 integration of $`de = -p\,d(1/\rho_p)`$ to $`10^{-4}`$ | C6 |
| `cosmo-cool` | uniform gas with power-law cooling: $`T(a)`$ against the float64 solution | C6 |

---

## Part D: shearing box

**Goal.** The local model of a differentially rotating disk: a Cartesian box corotating at
$`\Omega`$ with shear parameter $`q = -d\ln\Omega/d\ln R`$ (3/2 for Keplerian), $`x`$ radial,
$`y`$ along the orbit, $`z`$ vertical. This is the standard setting for MRI turbulence.

**Scope of the first version.** Uniform grid, hydro and MHD, no self-gravity, cubic box. Blocks
are refused: across the shearing boundary a block's neighbor changes with time, which the
tree's static neighbor tables cannot express.

### D1. Forces

Part A's machinery with a different potential:

```math
\mathbf{a} = -2\,\boldsymbol\Omega\times\mathbf{v} - \nabla\Phi_t,
\qquad \Phi_t = -q\,\Omega^2x^2 \;\left(+\tfrac12\Omega^2z^2 \text{ if stratified}\right).
```

`Problem.shear = { omega, q, stratified? }` selects a `frame` slot that turns as Part A's does
and puts $`\Phi_t`$ into `ext_accel`, `ext_dphi` and `ext_phi` in place of the centrifugal
potential. The background flow is $`v_y = -q\Omega x`$.

### D2. The shearing boundary for cell loads

Along x the box is periodic with a shift that grows in time:

```math
f(x + L, y, z) = f(x,\ y - q\Omega L\,t,\ z), \qquad v_y \to v_y - q\Omega L .
```

- **The offset** $`\Delta y = q\Omega L\,t \bmod L`$ is accumulated in the clock pass with
  Kahan compensation (`clock.shear`, G3): formed from $`t`$ directly it would lose precision in
  float32 over a long run.
- **A new boundary kind**, `SHEAR = 4`, for both x faces. In `bc_axis` it wraps like periodic;
  `bc_map` ([common.wgsl:167](../src/hydro/common.wgsl)) then shifts the y index by the whole
  cells of the offset. That part is an exact index remap.
- **The fractional part** needs interpolation along y. A ghost cell's state is a conservative
  linear reconstruction (the MC slope the mesh already uses for prolongation) of two source
  cells. This is new: until now a ghost was always one cell. `load_q`, `load_pred` and
  `load_vel` ([update.wgsl:41](../src/hydro/update.wgsl), [recon.wgsl](../src/hydro/recon.wgsl))
  grow a shear branch, used only by the shell pipeline; the interior pipeline is unchanged.
- **Velocity.** `bc_velocity` adds $`\mp q\Omega L`$ to $`v_y`$ of a ghost beyond a shear face,
  and the energy is rebuilt for it.
- **Half-step states** (`PA`, `PB`) are loaded with the offset at the half step.

### D3. Conservation at the boundary

The two cells that share a shearing face compute its flux from different ghosts, so their
results differ, and mass would not be conserved. The fix uses the idea the mesh already uses at
refinement boundaries: a cell **recomputes** the fluxes of the cells on the other side.

For a cell on the lower x face, with $`s`$ the whole-cell shift and $`\epsilon`$ the fraction:

```math
F_\text{lo}(j) = \tfrac12\left[F_\text{own}(j) + (1 - \epsilon)\,F_\text{other}(j + s) + \epsilon\,F_\text{other}(j + s + 1)\right],
```

and the mirror expression on the upper face. Summed over $`j`$, both faces give the same total,
so mass, x momentum and z momentum are conserved to round-off (the sum, not each face, since
the weights differ). The y momentum and energy fluxes are transformed by the velocity shift
before averaging. Cost: three flux evaluations per boundary face, on two planes of cells. This
goes in `axis_fluxes` and `flux_between` ([flux.wgsl:96](../src/hydro/flux.wgsl)); the tiled
update ([tile.wgsl](../src/hydro/tile.wgsl)) must skip its cache on those two planes.

The species pass ([species.wgsl](../src/hydro/species.wgsl)) reads the recorded mass fluxes,
which are then already the averaged ones.

### D4. Magnetic fields

The MHD passes compute each face once into storage buffers, so the correction is simpler than
in the hydro update: one small pass after the flux pass and one after the edge-field pass
rewrite the two x-boundary planes.

- **Fluxes:** the average of D3, applied to the stored boundary faces.
- **Edge electric fields:** $`E_y`$ and $`E_z`$ on the x-boundary edges are averaged with their
  remapped partners in the same way, as Athena's shearing box does, so the net vertical flux
  through the box is conserved. $`E_z`$ takes the $`-q\Omega L\,B_x`$ term of the frame change.
- **Face mapping:** `face_texel` ([mhd/common.wgsl:30](../src/mhd/common.wgsl)) needs the shear
  shift for tangential indices, and the interpolation of D2 for ghost fields.
- Slabs run along z and the remap is along y, so each slab still holds what it needs.

### D5. What comes later

- **Orbital advection** (FARGO): without it the timestep is set by the shear speed
  $`q\Omega L/2`$ at the box edge, not the sound speed.
- **A non-cubic box.** MRI boxes are usually $`H \times \pi H \times H`$ or taller. This needs
  B6 at least, and non-cubic cells for the usual aspect ratios.
- **Self-gravity** with shear-periodic boundaries.

### D6. Tests

| Test | Passes when |
|---|---|
| `shear-steady` | the background flow $`v_y = -q\Omega x`$ is steady to round-off for 10 orbits |
| `shear-mass` | a blob crossing the x boundary: mass conserved to round-off, its shape continuous |
| `shear-epicycle` | a uniform velocity kick oscillates at $`\kappa = \sqrt{2(2-q)}\,\Omega`$; the epicyclic energy is constant to $`10^{-4}`$ |
| `shear-wave` | the compressive shearing wave of Johnson & Gammie (2005) against its analytic amplitude |
| `shear-flux` | MHD: net $`B_z`$ flux through the box constant to round-off, $`\nabla\cdot B`$ at round-off |
| `shear-mri` | MHD: the fastest channel mode grows at $`0.75\,\Omega`$ within 5% (Keplerian) |

Presets: `shearing-wave` and `mri-box`, the second with the disk diagnostics of
[mri.wgsl](../src/mhd/mri.wgsl) adapted to the local frame.

---

## Part E: axisymmetric grids

**Goal.** 2D runs that are bodies of revolution: first cylindrical $`(R, z)`$, then spherical
$`(r, \theta)`$.

### E0. How it maps onto the 2D grid

A 2D run is one periodic layer of the 3D scheme, and it already carries the third velocity
component. That layer becomes the azimuth:

| Code axis | Meaning in $`(R, z)`$ |
|---|---|
| x | $`R`$, from 0 (the axis) to $`L`$ |
| y | $`z`$ |
| z (the layer) | $`\phi`$; `v_z` is $`v_\phi`$ and `B_z` is $`B_\phi`$ |

So swirl and toroidal fields come at no extra storage. The constant gravity of 2D runs already
points along $`-y`$, which is $`-z`$ here. `Problem.geometry?: 'cartesian' | 'cylindrical'`
selects it, for 2D only.

### E1. The geom slot

`<geom>` supplies, with the Cartesian stub returning constants:

```wgsl
fn geom_face(c: vec3<i32>, axis: i32, upper: bool) -> f32   // face area over (cell volume / dx)
fn geom_vol(c: vec3<i32>) -> f32                            // cell volume
fn geom_predict(q: Q, c: vec3<i32>) -> Q                    // geometric terms of the half step
fn geom_source(...)                                         // geometric momentum source
```

With the axis at the lower corner (G1, `org.x = 0`), cell $`i`$ has $`R_c = (i + \tfrac12)\Delta`$,
and its radial faces weigh

```math
\frac{R_-}{R_c} = \frac{i}{i + \tfrac12}, \qquad \frac{R_+}{R_c} = \frac{i + 1}{i + \tfrac12},
```

pure index ratios with no cancellation in float32. The z faces weigh 1. The volume is
$`2\pi R_c\Delta^2`$, so masses are true masses.

On blocks the same ratios are formed from the block's global cell index and level.

### E2. The update

```math
\partial_t\mathbf{U} + \frac{1}{R}\partial_R\left(R\,\mathbf{F}_R\right) + \partial_z\mathbf{F}_z = \mathbf{S}.
```

- **Fluxes.** `net_inflow` and `net_inflow_heated` ([flux.wgsl:206](../src/hydro/flux.wgsl))
  weigh each face's flux by `geom_face`. Both grids build their net there, so
  [advance.wgsl](../src/hydro/advance.wgsl) is unchanged. The face's flux itself is still the
  same bits from both cells; conservation of $`\sum R_c\,U`$ then holds to round-off (not to the
  bit: $`(R_-/R_c)\,R_c`$ rounds).
- **The $`p\,\nabla\cdot\mathbf{v}`$ term** uses the same weights on the face normal velocities
  (`net.vn`), which gives the cylindrical divergence.
- **Angular momentum.** The $`\phi`$ momentum flux through radial faces is weighted by
  $`(R_\pm/R_c)^2`$ and has no source: $`R\rho v_\phi`$ is then conserved to round-off.
- **Radial source:** $`(p + \rho v_\phi^2)/R`$. For the pressure part, the value that keeps the
  scheme well balanced is the **mean of the cell's two radial face pressures**: with it, the
  cylindrical balance reduces exactly to the Cartesian one the reconstruction already holds.
  That requires the Riemann solvers to return the face pressure, a new field of `Flux`
  ([flux.wgsl:22](../src/hydro/flux.wgsl); `hll.wgsl`, `hllc.wgsl`, `hllclm.wgsl`).
- **Predictor** ([predictor.wgsl:85](../src/hydro/predictor.wgsl)): `geom_predict` adds
  $`-\rho v_R/R`$ to the density, $`-\rho c_s^2 v_R/R`$ to the pressure, $`+v_\phi^2/R`$ to
  $`v_R`$ and $`-v_Rv_\phi/R`$ to $`v_\phi`$, each times $`\Delta t/2`$.
- **Species** ([species.wgsl:73](../src/hydro/species.wgsl), [amr/species.wgsl](../src/amr/species.wgsl)):
  the same face weights on the recorded mass fluxes.
- **Reconstruction** stays as it is: second order in $`R`$ near the axis, where the Cartesian
  parabola is no longer exact. Curvilinear weights (Mignone 2014) are a later refinement.

### E3. The axis

The lower x face is the axis: `P.sym.x = 2`. It mirrors like a reflecting face, with two
differences from a mirror plane: $`v_\phi`$ changes sign as well as $`v_R`$, and the field has
$`B_R`$ and $`B_\phi`$ odd and $`B_z`$ even (regularity, not the pseudovector rule of B5). Its
face has zero area, so its flux weight is exactly zero and nothing crosses it. The potential
treatment is that of a mirror plane (B2).

### E4. Blocks

- **Restriction and prolongation** must be volume weighted: [cons.wgsl:33](../src/amr/cons.wgsl),
  [regrid.wgsl:74](../src/amr/regrid.wgsl), [species_regrid.wgsl:51](../src/amr/species_regrid.wgsl),
  and the prolongation slopes at [cons.wgsl:87](../src/amr/cons.wgsl), so that
  $`\sum R_c\,U`$ over the children equals the parent's.
- **Refinement boundaries.** The coarse cell recomputes the fine fluxes and sums them; each
  fine flux takes its own area. On a radial face both fine faces sit at the same $`R`$; on a z
  face their areas sum to the coarse face's exactly.

### E5. Gravity

Three steps, each usable alone:

1. **Uniform field and point masses on the axis:** nothing to do. `cell_pos` returns
   $`(R, z, 0)`$ ([bh.wgsl:36](../src/gravity/bh.wgsl)), and the distance to a mass on the axis
   is then right. A preset that puts a mass off the axis is refused. This already covers jets,
   blasts, bubbles, accretion onto a central object and tori.
2. **Self-gravity on the uniform grid.** A 2D multigrid: the levels become one layer deep and
   the stencil ([gravity/stencil.wgsl](../src/gravity/stencil.wgsl)) drops its z neighbors and
   weighs the radial ones by $`R_\pm/R_c`$, with the diagonal their sum. Cells stay square, so
   red-black smoothing is unaffected. The multipole boundary needs no new evaluator: fill the
   Cartesian moments by symmetry ($`S_{xx} = S_{yy} = \tfrac12\sum mR^2`$,
   $`S_{zz} = \sum mz^2`$, center of mass on the axis) and `phi_boundary` is right as it stands.
3. **Self-gravity on blocks.** The block solver says "3D only"
   ([amr/gravity/common.wgsl:15](../src/amr/gravity/common.wgsl)); a 2D variant of each pass.
   This is the largest single item of Part E and may be left out of a first release.

Settling stars in $`(R, z)`$ follows from step 2 with volume-weighted sums.

### E6. Magnetic fields, viscosity

- **Constrained transport** with areas and edge lengths: the $`b_R`$ face has area
  $`\propto R_\text{face}`$, the $`b_z`$ face $`\propto R_c`$, and the $`E_\phi`$ edge length
  $`\propto R`$. Then

  ```math
  \partial_t b_R = -\partial_z E_\phi,\qquad
  \partial_t b_z = \frac{1}{R}\partial_R(R\,E_\phi),\qquad
  \partial_t b_\phi = \partial_z E_R - \partial_R E_z,
  ```

  which keeps $`\tfrac1R\partial_R(Rb_R) + \partial_z b_z`$ at zero. The cell-centered $`B_R`$
  is the area-weighted mean of its faces ([bcell.wgsl](../src/mhd/bcell.wgsl)). The radial
  momentum source gains $`(B^2/2 - B_\phi^2)/R`$. Touches the six `dtdx` sites of
  [mhd/update.wgsl](../src/mhd/update.wgsl), [emf.wgsl](../src/mhd/emf.wgsl) and their block
  counterparts.
- **Conduction** works through the face weights.
- **Viscosity** needs the curvilinear stress ($`\tau_{\phi\phi}`$, $`\tau_{R\phi}`$ and the
  $`-\tau_{\phi\phi}/R`$ source) in [diffusion_core.wgsl](../src/hydro/diffusion_core.wgsl).
  Refused in the first version.

### E7. Diagnostics, display, files

- **Volumes.** `P.geom.x` becomes `geom_vol(c)` in the reductions:
  [hydro/diagnostics.wgsl:14](../src/hydro/diagnostics.wgsl), [amr/diagnostics.wgsl](../src/amr/diagnostics.wgsl),
  the point masses' reduction, the multipole, the debris census and the magnetic diagnostics.
- **Display.** The slice view shows the $`(R, z)`$ half-plane, with the mirror option of B7 to
  show both sides of the axis. A volume view by revolution, sampling the 2D field at
  $`(\sqrt{x^2 + y^2}, z)`$, is optional.
- **Files.** Athena++'s cylindrical order is $`(R, \phi, z)`$. A snapshot is written with
  `Coordinates = cylindrical`, root size $`(n_R, 1, n_z)`$, and the second and third velocity
  and field components exchanged, which is a relabeling of the arrays
  ([athdf.ts:84](../src/io/athdf.ts)). The reader accepts the same layout
  ([athdf.ts:173](../src/io/athdf.ts)).

### E8. Tests for (R, z)

| Test | Passes when |
|---|---|
| `cyl-rest` | uniform gas at rest stays at rest to round-off (the pressure source cancels) |
| `cyl-conserve` | a ring with swirl, advected: mass, energy, z momentum and angular momentum to round-off |
| `cyl-sedov` | a point blast is spherical: shock radius along $`R`$ and $`z`$ within 2% of the 3D similarity solution |
| `cyl-noh` | the spherical Noh implosion: post-shock density within 5% away from the axis |
| `cyl-column` | a stratified column under uniform gravity stays at Mach $`<10^{-4}`$ |
| `cyl-bondi` | accretion onto a point mass on the axis: the Bondi rate within 5% |
| `cyl-torus` | a pressure-supported torus around a point mass stays within 1% for 5 orbits |
| `cyl-3d` | a jet or blast against the same problem in 3D Cartesian |
| `cyl-sphere` | E5 step 2: potential of a uniform sphere within 0.3% |
| `cyl-blocks` | `cyl-sedov` and `cyl-conserve` on two levels |
| `cyl-mhd` | a poloidal field loop advected along $`z`$: $`\nabla\cdot B`$ at round-off, energy as in the Cartesian test |

### E9. Spherical (r, θ)

To be planned in detail once $`(R, z)`$ is in. What is already clear:

- It needs **non-uniform spacing**: logarithmic in $`r`$ in practice, so the cell size differs
  by axis and by index. `g_dx` can no longer be one scalar, which Parts A to E avoid. This is
  the main cost and the reason it comes last.
- Areas and volumes: $`A_r \propto r^2\,\Delta(\cos\theta)`$,
  $`A_\theta \propto r_c\,\Delta r\,\sin\theta_\text{face}`$,
  $`V \propto \tfrac13\Delta(r^3)\,\Delta(\cos\theta)`$. On a logarithmic grid these differences
  cancel in float32 and need closed forms.
- Sources in two directions: $`(2p + \rho(v_\theta^2 + v_\phi^2))/r`$ radially, and
  $`(\cot\theta\,(p + \rho v_\phi^2) - \rho v_rv_\theta)/r`$ in $`\theta`$.
- Both poles are axes (E3); the inner radial boundary sits at $`r_\text{min} > 0`$.
- The timestep is set by $`r_\text{min}\Delta\theta`$, so per-block timesteps matter.
- Gravity: a monopole from enclosed mass is the cheap first step.

---

## Order of work

1. **G1** (domain origin). Alone, with the whole suite as its test.
2. **G2 + Part A**, rotating frame. Small and independent of everything else.
3. **Part B, steps B1 to B4, B7, B8**: octant and 2D quadrant, hydro and gravity.
4. **G3 + Part C**, comoving coordinates: `cell_dx()`, the scale factor, the drag and gravity
   for the ideal gas (C2 to C5); then the equation-of-state wrappers and cooling (C6). It uses
   the `frame` hooks of step 2 and is independent of B, so it can swap with step 3.
5. **Part B, step B5**: mirror parity for magnetic fields.
6. **Part B, step B6**: rectangular grids. Unlocks half and 3D-quadrant runs, and helps D.
7. **Part D**, shearing box: forces, then hydro boundary, then MHD.
8. **Part E** $`(R, z)`$: E1 to E3 (hydro on the uniform grid), E5 step 1, E7; then E4
   (blocks), E6 (MHD), E5 steps 2 and 3.
9. **Part E9**, spherical, after its own plan.

Every step ends with: unit tests (`npm test`) and the GPU suite passing, a section in
[docs/numerics.md](../docs/numerics.md), the new keys in [docs/api.md](../docs/api.md), and the
paper's method section where the feature is one it describes.

Rules that hold throughout:

- A run that does not use a feature compiles without it and gives the same bits as before. Each
  part has an `-off` test for this.
- Anything unsupported in combination is refused in `configFor` with a message, never run
  silently wrong.
- Both cells of a face still compute the same flux bits. Geometry enters as a weight on the
  flux, never inside it.

## Open questions

1. **Symmetry and the box size.** Should `box` stay the side of the computed domain (as
   planned), or the side of the full configuration, with the computed side derived? The first
   keeps `dx = box / n`; the second keeps presets' numbers unchanged.
2. **Wall or plane.** The plan marks symmetry planes with a separate mask (`P.sym`) and leaves
   the boundary code at `REFLECT`. The alternative is a fifth boundary code, which is more
   visible in the interface but touches every place that tests for a reflecting face.
3. **Face pressure from the Riemann solvers** (E2). It is the clean way to a well-balanced
   radial source, but it widens `Flux`, which the update is sensitive to (texture-fetch bound,
   so probably free; to be measured). The fallback is the cell's half-step pressure, exact for
   uniform pressure only.
4. **Comoving magnetic fields and per-block timesteps** (C7) are refused in the first version.
   Both are tractable; which one matters first decides what follows C6.
5. **Shearing box on blocks.** Refused in this plan. If adaptive MRI boxes are wanted, the tree
   needs neighbor relations that depend on time, which is its own design.
6. **Rectangular grids** (B6) are scheduled after the octant. If half-domain runs (tidal
   disruptions about the orbital plane, magnetized stars about the equator) are the main use,
   B6 should move ahead of B3.
