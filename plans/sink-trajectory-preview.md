# Plan: trajectory preview for a point mass being placed

Status: built 2026-11-15. What changed while building:

- **Fitting the lumps** (section 3.3). Dropping the lumps that hold next to nothing was wrong:
  the thin gas around a polytrope, dropped, shifted the pull on a mass beside it by 1%. All of
  the gas is kept, and `fitLumps()` cuts the box as a tree instead: the piece with the most
  mass is cut in two along each axis until there are 768 pieces, so a star keeps the
  resolution it was read at and thin gas is summed coarsely. The uniform grid is read at 32
  lumps a side in 3D and 64 in 2D. Fitting 65536 lumps takes 9 ms, a path through 768 lumps
  with three bodies 4 ms.
- **A lump's softening** is a third of its size, and fades to none two sizes away: softened
  everywhere by half its size, a star's lumps pulled 1% too weakly from 0.4 of the box away.
- **The gas's pull on the other masses** is found every fourth step from the start, not as a
  fallback.
- **The tube targets** stayed in `render/fieldlines.ts` (`tubeTargets()`), shared with
  `render/pathtube.ts`; there is no `tube_targets.ts`. The ray-marcher now also draws a tube
  where the ray misses the box, since a path may leave it.
- **Step cap** 1200, 160 points, tube radius 0.005 of the box.
- **Not done**: the fade of the 2D line along the path, ending a closed orbit after one turn,
  and the `window.shaderous.previewOrbit` hook (the validation test `orbit-preview` calls
  `previewOrbit()` itself). The comparison against a run agrees to 0.23% of the box after a
  quarter of an orbit.

Since then (2026-11-15):

- **32 point masses** (`MAX_MASSES`), from 4. The masses' sums over the gas are reduced in
  groups of four (`gravity/mass_sums.ts`), one reduction per group in use, so a run with up to
  four masses does what it did. The preview finds the gas's pull on the other masses every
  n - 1 steps once there are more than five.
- **A flux of exactly 0** at the left end of the sink tool's slider: a point mass that takes
  no gas. Its marker is the dot alone.
- **Checked against runs of the binary preset**: a light and a heavy mass sent retrograde into
  the cavity, and a light one prograde, each stay within about 2% of the binary's separation
  of the preview over more than a binary period. The looping paths of slow or retrograde
  masses there are the binary's two members passing by.
- **The other masses' paths** over the same span are drawn with the preview: a faint line in
  2D and slices, thinner and dimmer tubes in the volume view (`Orbit.others`).
- **A velocity arrow on every point mass**: half the length of the drag that would give that
  speed, at most 48 px, and none below a few tenths of the tools' reference speed.
- **The sink tool's Soft slider is labeled Edge** (as the source's): it is how a massless
  sink's rate fades toward its edge, not the softening of a point mass's gravity, which is its
  Size.

## 1. Goal

While the sink tool is placing a point mass (a sink with a mass: press, then drag to set its
velocity), show the path it would take from that position with that velocity, redrawn as the
pointer moves.

- **Volume view of a 3D run**: a lit tube with a cone arrowhead at its far end, drawn among
  the gas (gas in front of it hides it), as the field-line tubes are.
- **2D runs and the slice view**: a line with an arrowhead over the map.

A sink without a mass does not move, so it gets no preview.

### Non-goals for v1

- The response of the gas: the preview holds the gas as it is now (section 3.3). No dynamical
  friction, no accretion onto the new mass, no gas flowing on while the path is traced.
- Paths for the star tool, and for the masses already in the run (section 8).
- The preview in recorded movies and through `window.shaderous` (except the test hook of
  section 7).

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Where the path is integrated | On the CPU, in TypeScript | At most 5 bodies and a few hundred steps: far below a millisecond for the masses alone, and it must rerun on every pointer move without waiting on a readback |
| What pulls on the new mass | The other point masses, the gas, and the frame's acceleration: what `gravity/bh_kick.wgsl` applies | The preview should be the orbit the run would give if the gas stood still |
| The other masses | Integrated along with the new one, pulling on each other and feeling it | A heavy newcomer moves a binary; holding them on rails would mislead |
| The gas | A frozen list of lumps (mass at its center of mass, and a size), read back from the GPU a few times a second | The run finds the gas's pull by a reduction over every cell per step, which a preview cannot repeat at each of its steps |
| Integrator | Kick-drift-kick with an adaptive step, the pair softening of `bh_kick.wgsl` | The same scheme and force law as the run |
| 3D drawing | Capsules and a cone in the volume view's tube targets, which the ray-marcher already composites by distance | Depth among the gas for free; the SVG overlay cannot be hidden by gas |
| 2D drawing | An SVG path in `#brush`, projected with `outlineMatrix()` | As the source and sink markers are drawn; no new GPU pass |
| One source for the velocity and the mass | `strokeVelocity()` and `massSpec()` shared by the preview and by `placeMass()` | The path shown must be for exactly the mass that a release places |

## 3. The preview orbit

A new module, `src/gravity/preview.ts`, with no GPU or DOM in it (so vitest covers it):

```ts
interface Body { gm: number; pos: Vec3; vel: Vec3; soft: number }
interface Lump { m: number; pos: Vec3; h: number }           // gas: mass, center of mass, size
interface OrbitInput {
  bodies: Body[];            // the run's masses, then the new one last
  lumps: Lump[];             // may be empty
  G: number;
  frame: 0 | 1 | 2;          // P.bh.y: inertial, with the gas, with the masses
  box: number;
  ndim: 2 | 3;
}
function previewOrbit(input: OrbitInput, limits?: Limits): { points: Float32Array; times: Float32Array; end: 'length' | 'steps' | 'left' | 'time' }
```

`points` is the new mass's path (xyz, relative to the box center, in code units), `times` the
time of each point from now.

### 3.1 Forces

As `gravity/bh_kick.wgsl`, in double precision:

- **Mass on mass**: `-G M_j d / (d² + ε²)^(3/2)` with `ε = max(soft_i, soft_j)`.
- **Gas on a mass**: the sum over lumps of `G m d / (d² + ε²)^(3/2)` with
  `ε² = soft_i² + (h / 2)²`. The run softens each cell's pull by the mass's own length
  (`bh_accel`); the `h / 2` keeps a lump the mass is inside from pulling like a point.
- **Frame** (`Problem.frame`): 1 subtracts the mass-weighted mean of the masses' pull on the
  gas, 2 subtracts the acceleration of the masses' center of mass and zeroes its velocity after
  each kick. Both come from the same sums.
- A 2D run uses the same law with everything in the plane z = 0, as `cell_pos()` does.
- `G`: `cfg.G` if positive, else 1, which is what `placeMass()` will set. The preview must not
  write `cfg`.

A uniform field (`gravity: 'constant'`) does not act on point masses in the run, so not here.

### 3.2 Stepping and where the path ends

- Step: `dt = η min_i sqrt(ε_i / |a_i|)` with η about 0.1, capped so no body moves more than
  1% of the box in a step, and never more than a step cap away from the last.
- The path ends at the first of: 2 box sides of path length; 600 steps; the mass more than
  10% of a box side outside the box (masses do not wrap, even in a periodic box); the time
  left to `cfg.tEnd`.
- A closed orbit simply overlaps itself. Ending it after one turn (back within a tube radius
  of the start, heading the same way) is a refinement to try once it is on screen.
- The points are resampled to at most 192, evenly spaced along the path, so tubes and SVG get
  the same shape whatever the steps were.

### 3.3 The gas, as lumps

`Simulation` gains one method, in the style of `readMasses()` (never waits; null while the
last read is in flight):

```ts
/** The gas as lumps (mass, center of mass, size) for previews of orbits; null while a read is in flight. */
readGasLumps(): Promise<Lump[] | null>;
```

- **Uniform grid** (`src/sim.ts`): a compute pass, `gravity/lumps.wgsl`, writes a K³
  `rgba32float` texture (K = 16 in 3D; K × K with K = 32 in 2D): each texel sums the mass and
  the mass-weighted position of its `(n / K)^ndim` cells, skipping solid and fixed cells as
  `gravity/bh_moment.wgsl` does. Copied to a mappable buffer and read back.
- **Adaptive mesh** (`src/amr/amr_sim.ts`): one lump per leaf block, from a pass over the leaf
  list (`listLeaf`, as the debris census is dispatched): each block's mass and center of mass,
  with `h` the block's side. Finer where the mesh is, which is where the gas is. To check
  when building: cells per block and the largest leaf count; above about 4000 leaves the CPU
  folds the lumps into the uniform K³ grid.
- On the CPU: lumps below 1e-6 of the total mass are dropped (most of a box is near-empty in
  the setups where masses are placed).

Refresh: when the sink tool has a mass and the pointer is over the canvas, at most 4 times a
second; always once on press. Until the first read lands, the preview uses the masses alone.

Cost to hold to: under 4 ms per path. Worst case is 600 steps × 5 bodies × 4096 lumps, some
12 M force terms, which is too many; pruning empty lumps is expected to bring it down by an
order of magnitude. If a measured case is still over, in this order: K = 12; evaluate the gas's
pull on the *other* masses every fourth step; a second, coarser grid for lumps more than three
coarse cells away.

### 3.4 What the preview is not

It is the orbit in the gas as it stands. It will differ from the run where the gas moves on
the orbit's timescale, where the new mass stirs it (wakes, tides on a star it passes), and by
what the sink accretes. The tool's tooltip says "path in the gas as it is now".

## 4. Wiring in `src/ui/app.ts`

1. Factor out of `advanceStroke()`:
   - `strokeVelocity(s: Stroke): { at, vel }`, the drag-to-velocity rule (source, sink and
     star share it today as three copies);
   - `massSpec(pos, vel, q, G, gas): PointMass`, the `addMass()` argument `placeMass()` builds.
2. `gasLumps: { of: Simulation; list: Lump[] } | null`, refreshed as in 3.3 next to the
   `readMasses()` call in the frame loop.
3. `updatePreview()`, once per frame after `advanceStroke()`: if `stroke?.kind === 'sink'`,
   `sinkHasMass()`, `stroke.ready` and not released, build the bodies (`liveMasses`, then
   `massSpec(...)`), call `previewOrbit()`, and keep the result in `orbitPreview`. Otherwise
   clear it. It recomputes only when its inputs changed (pointer, tool sliders, a new
   `liveMasses` or `gasLumps`), so a paused run with a still pointer costs nothing.
4. No room for another mass (`MAX_MASSES`): no preview; the existing notice explains on release.
5. Escape or a canceled drag clears it with the stroke.

## 5. Drawing in 2D and the slice view

In `drawBrush()`, a `<path class="orbit">` added to `#brush`:

- Each point through the `at()` projection of `drawSources()` (lift that helper out so both
  use it), joined as a polyline; a triangle arrowhead at the last point, along the last
  segment, sized as the source markers' arrows.
- Color: the sink orange (`#ff9d5c`), 2 px, round joins; opacity falling from 1 to 0.5 along
  the path (an SVG gradient in `userSpaceOnUse`, or three sub-paths).
- The existing straight drag line stays, thinner and dimmer: it is the velocity handle, and the
  path leaves the mass along it.
- Slice view of a 3D run: the path is projected onto the slice plane. Stretches more than a
  cell layer off the plane are dashed. (The drag gives no out-of-plane velocity, so with
  masses and gas symmetric about the plane it stays in it.)

## 6. Drawing in the volume view

The ray-marcher already takes a tube color and a distance-along-the-ray texture (bindings 9
and 10 of the volume view, `render/raymarch.wgsl`) and draws gas in front of a tube over it.
Today only `FieldLines` fills them.

1. **Share the targets.** Move `ensureTargets()` and the color, distance and depth textures
   out of `render/fieldlines.ts` into a small `render/tube_targets.ts` owned by the renderer.
   `FieldLines.encode()` and the new path pass both draw into them: the first one clears, the
   second loads, and the depth attachment sorts field tubes against the path (so its
   `depthStoreOp` becomes `store` when a second pass follows). The renderer binds the targets
   whenever either drew.
2. **`render/pathtube.ts` and `render/pathtube.wgsl`.** A `PathTube` class holding a storage
   buffer of up to 192 points and a pipeline with two entry-point pairs:
   - segments: instanced camera-facing quads with the ray-capsule test of
     `render/fieldtubes.wgsl` (move `capsule()` and `project()` to an include,
     `render/capsule.wgsl`, used by both);
   - arrowhead: one quad covering a cone at the last point along the last segment's
     direction, with an analytic ray-cone test (capped cone: side and base disk).
   Same lights as the field tubes; flat sink-orange color, dimming along the path by the
   point's index. Radius 0.006 of the box (the field lines' is thinner; check `TUBE_RADIUS`
   and settle by eye), cone 6 radii long and 2.5 wide at its base. The tube stops where the
   cone begins.
3. **Renderer API**, as `guides` is today: `renderer.orbit: { points: Float32Array; version: number } | null`,
   set from `updatePreview()` when `cfg.ndim === 3 && !flatView()`. The version joins the
   `inputs` string in `encode()` so a paused run redraws when the path changes.
4. The SVG path of section 5 is not drawn in the volume view; the SVG drag line is.

## 7. Tests

- **Unit** (`src/gravity/preview.test.ts`, vitest, runs in the Pages deploy):
  - no masses, no lumps: a straight line of the capped length;
  - one heavy body, a light one on a circular orbit: closes to 1% after one period, and energy
    holds to 1e-3;
  - lumps sampling a uniform sphere: the pull outside it is `G M / r²` to a few percent, and
    falls to zero at its center;
  - frame 2: the masses' center stays at rest;
  - the path stops when it leaves the box.
- **GPU** (`src/validation/tests.ts`): `readGasLumps()` on a uniform and an adaptive run
  returns the total gas mass of the diagnostics to 1e-5, and the center of mass.
- **Against the run**: a static polytrope with a light mass (1e-3 of the star) released
  outside it. Run half an orbit; the mass is within 2% of the box of where the preview put it
  at that time. A small test hook, `window.shaderous.previewOrbit(pos, vel, mass)`, returns
  the points and times for this.
- **By eye**, in the page: 2D Kepler disk, the binary, the tidal disruption preset (frame
  with the gas), the volume view with field lines on (tubes of both kinds sort correctly),
  light and dark backgrounds, a phone-sized viewport.

## 8. Order of work

1. `gravity/preview.ts` with its unit tests (masses only, then lumps and frames).
2. The `app.ts` refactor (`strokeVelocity`, `massSpec`) and `updatePreview()`, masses only.
3. The 2D line (section 5). First point at which it can be used.
4. `readGasLumps()` on the uniform grid, then the adaptive mesh; GPU tests; timing of 3.3.
5. Shared tube targets, then `PathTube` (section 6).
6. The comparison against a run, tooltip text, README line on the sink tool.

Each step leaves the page working, so each can be pushed on its own.

## 9. Later, if wanted

- The same preview for the star tool (the star's center moves as a point mass of its mass
  would, until tides matter).
- A preview on hover, before the press, of a mass released at rest.
- Tick marks along the path at even times, or coloring by speed.
- Faint paths for the masses already in the run over the same span.
