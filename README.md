# Shaderous

A browser-based 3D compressible hydrodynamics code for astrophysical test problems. All fluid
and gravity computation runs in WGSL compute shaders on WebGPU. The design is in
[plan.md](plan.md).

**Live:** https://guillochon.github.io/shaderous/ (needs a WebGPU browser). Validation suite:
https://guillochon.github.io/shaderous/validate.html

- **Hydro:** unsplit MUSCL-Hancock, third-order reconstruction limited only where the flow is
  not smooth (or a classic slope limiter), HLLC (or HLL) fluxes,
  positivity fallback, shock-flagged HLL blend, dual-energy formalism, floors.
- **Self-gravity:** multigrid Poisson solve every step, periodic or isolated (multipole) boundaries.
- **Self-gravity or a uniform field:** a constant external acceleration drives the
  Rayleigh-Taylor preset.
- **Tidal disruption:** a polytropic star passed by a point mass on a parabolic orbit, run in the
  star's rest frame. The frame accelerates with the gas's centre of mass, the point mass's orbit
  is integrated alongside the gas, and the boundaries let gas out but not in.
- **2D or 3D:** presets that make sense in the plane (shock tubes, blasts, Kelvin-Helmholtz,
  Rayleigh-Taylor, advection) can run on an n x n grid up to 2048 squared. A 2D run is one periodic
  layer of the same 3D scheme, without self-gravity.
- **Adaptive mesh refinement:** a tree of 16-cell blocks packed into texture atlases, refined
  where density or pressure jump and merged where they do not. Fluxes are matched across
  refinement boundaries, so mass, momentum and energy are conserved to round-off. Every
  preset can use it, including the self-gravitating ones: the potential is solved level by
  level on the same blocks, and gas that approaches its Jeans length is refined.
- **Equations of state:** ideal gas, barotropic (isothermal / polytropic), gas plus radiation
  pressure, chosen when the shaders are assembled.

## Running

Needs Node 20+ and a browser with WebGPU (current Chrome, Edge or Firefox).

```bash
npm install
```

```bash
npm run dev
```

Then open http://localhost:5183. The validation suite is at http://localhost:5183/validate.html
and runs on your own GPU.

## Tests

```bash
npm test
```

runs the unit tests (preprocessor, equations of state, exact Riemann solver, float64
reference scheme) in Node.

```bash
npm run test:gpu
```

drives the installed Chrome through Playwright and runs the validation suite on the real GPU.
Set `PW_CHANNEL=msedge` to use Edge, or `PW_HEADED=1` if headless Chrome finds no adapter.

## Layout

```
src/
  gpu/         device setup and limit checks, shader preprocessor, hierarchical reductions
  hydro/       timestep rates, clock, predictor and update passes; riemann/ and limiters/ modules
  eos/         WGSL equation-of-state modules and their float64 mirrors
  gravity/     multigrid passes, multipole boundary, half-kick corrector
  amr/         block tree, block-atlas passes and the adaptive driver; gravity/ for its Poisson solve
  render/      field extraction, ray-marcher, slice view, transfer function
  problems/    one file per preset
  reference/   float64 CPU implementation of the hydro scheme, exact Riemann solver
  validation/  the in-page validation suite
  ui/          the simulator front end
  sim.ts       owns the textures and encodes a step
tests/         unit/ (Vitest) and gpu/ (Playwright)
```

## Notes on the numerics

- **Gravity coupling** is kick-drift-kick: half a kick with the old potential inside the update
  pass, the other half after the new potential is solved. Applying the whole kick after the
  flux update leaks `rho g^2 dt^2 / 2` of internal energy per step from gas in hydrostatic
  balance, which slowly cools a star.
- **Gravitational energy source** defaults to the flux form (face mass flux times the potential
  difference), which conserves total energy but can overheat low-density cells next to a sharp
  density contrast. The alternative adds exactly the kinetic energy of each kick: gravity never
  touches the internal energy, but total energy is not conserved where the flow is
  under-resolved. Both are selectable in the UI (`gravEnergyFlux` in `SimConfig`).
- **Float32.** Conservation is at round-off level, but round-off is biased when a uniform
  background sits exactly on a power of two (see `problems/advection.ts`).
- **Reconstruction.** The default gives each face of a cell the value of the parabola through
  the cell and its two neighbours, which is third-order accurate on the stencil a slope
  limiter already uses. A three-cell limiter cannot tell a smooth maximum from a jump and
  flattens both, so the limiter (Koren's) is applied only where a quantity changes by more
  than 1% of its natural scale across a cell: the density, the pressure, or the sound speed
  for velocities. A smooth peak then keeps its height (the largest error in the advection
  test is 2.5 to 5 times smaller than with the MC limiter) and hydrostatic states are
  quieter. The price is that kinks and jumps below 1% are not limited: shock-tube errors are
  a few percent higher on axis-aligned tubes. MC, van Leer and minmod remain selectable.
- **A step is four hydro passes:** timestep rates, clock, half-step predictor, update. On the
  uniform grid there are no primitive-variable textures: the predictor and the update convert
  the conserved state as they load it. The flux update is bound by texture fetches (with all
  of its arithmetic removed it still takes 85% as long), so a pass that only rewrites the
  state in another form costs more than repeating the conversion. The rates pass writes
  nothing but the first stage of the timestep reduction, folded in workgroup-shared atomics.
  Storage buffers in place of 3D textures were measured and are slower (about 1.4 times for
  the same stencil). The predictor and the update are each compiled twice from one source
  (`hydro/region_pass.ts`): with the boundary mapping compiled out for cells at least two
  from every face, and in general form for the shell, whose slabs get workgroups two cells
  thick so they launch no more invocations than they have cells. Fluxes on the seam between
  the two agree bit for bit (the flux-determinism test covers it). A 2D run uses 16 x 16 x 1
  workgroups, which more than halves its flux update against 8 x 8 x 4 with three idle
  layers. With gravity it adds the Poisson source, V-cycles, a pass that writes the
  acceleration field, and the second half kick. `sim.profile()` times each pass.
- **Frame pacing** adapts the number of steps per frame to the GPU. Firefox reports finished
  GPU work only every ~100 ms, so there the loop steers by backlog rather than by timing.
  Results agree across browsers to round-off; axis-permuted runs are bit-identical in Chrome
  and differ in the last bit in Firefox.
- **Adaptive mesh.** Ghost cells are never copied within a texture (WebGPU cannot read and
  write one rgba32float texture in a pass). Instead the primitives pass computes every texel,
  ghost or not, from wherever its value lives: a neighbouring block, a boundary image, the
  average of finer cells, or an interpolation from a coarser block. At a refinement boundary
  the coarse cell recomputes the fine fluxes rather than using its own, which conserves
  without storing fluxes; this needs every level to take the same timestep. The mesh is
  rebuilt from the CPU every 8 steps, from a per-block indicator read back without blocking.
- **Gravity on the adaptive mesh.** The potential lives on every block, parents included.
  The density is averaged into the parents, ghost cells of a level come from its neighbours
  or are interpolated from the level above, and a multigrid cycle runs inside the blocks (16,
  8, 4, 2 and 1 cells across). The coupling is one-way, so a fine level never corrects the
  coarse potential beneath it; on a uniform sphere the error stays below 0.3% with three
  levels. A fresh mesh is solved level by level from the coarsest down. During a run every
  level relaxes in the same dispatches, each against the boundary values its coarser level
  had at the start of the cycle, so the cost does not grow with the depth of the tree; the
  solve then takes about as long as on a uniform grid of the finest resolution when half the
  volume is refined, and less when the refined region is smaller.
  Refinement also triggers where the Jeans length falls below `jeansRefine` cells (16 by
  default). The atlas holds at most 2197 blocks in 3D (about 3 GB of GPU memory with
  gravity; it shrinks automatically if the GPU cannot allocate that); if the initial mesh of a
  self-gravitating run cannot fit a whole level within that, the level is left out (the
  status line says so) rather than letting a refinement boundary cut through a star at an
  arbitrary place, which disturbs hydrostatic balance. Periodic gravity assumes periodic hydro boundaries on the adaptive mesh.
- **Barotropic gas** uses HLL, since there is no contact wave for HLLC to resolve.

## Licence

MIT; see [LICENSE](LICENSE).
