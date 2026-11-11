# Plan: WebGPU astrophysical hydrodynamics simulator

## 1. Goal

A browser-based 3D compressible hydro code in which all fluid and gravity computation runs in WGSL compute shaders. Version 1 must handle:

- **Shocks**: a Godunov-type finite-volume scheme that stays stable and positive through strong shocks.
- **Self-gravity**: a Poisson solve every step, with periodic and isolated boundaries.
- **Multiple equations of state**: selectable without touching the hydro solver.

Target: interactive frame rates at 128³ on a mid-range laptop GPU, with 256³ as a stretch goal on discrete GPUs.

### Non-goals for v1

Adaptive mesh refinement, magnetic fields, radiation transport, cosmological expansion, sink particles, multi-GPU, and a WebGL fallback. The architecture should not block the first three.

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Grid | Uniform Cartesian, cell-centered | Simplest thing that maps onto 3D textures |
| Hydro scheme | Unsplit MUSCL-Hancock, piecewise-linear reconstruction, HLLC flux | Second order in one pass per step, so only two copies of the state |
| Flux evaluation | Gather: each cell computes its own six face fluxes | No write races, no flux storage; costs 2x flux evaluations |
| State storage | `rgba32float` 3D textures | Not subject to the 128 MB default storage-buffer binding limit |
| Gravity | Geometric multigrid, warm-started from the previous potential | Works with any boundary condition; 1 to 2 V-cycles per step once warm |
| Equation of state | WGSL module chosen when the shader is assembled | WGSL has no dynamic dispatch |
| Timestep | Computed and consumed on the GPU | Avoids an async readback every step |
| Rendering | Own ray-marcher on the same device | Avoids sharing a device with three.js; three.js can be added later for UI |

## 3. Numerical method

### 3.1 Hydro

Conserved variables: density, momentum (3), total energy, plus an auxiliary internal energy and one passive scalar.

Per step:

1. **Primitives pass.** Convert conserved to primitive variables and call the equation of state once per cell (pressure, sound speed). Write to the primitive textures.
2. **Timestep pass.** Reduce max(|v| + c_s)/dx and the gravity limit sqrt(dx/|g|) over the grid, write dt to a small buffer.
3. **Update pass.** For each cell: limited slopes (monotonized-central limiter) for itself and its six neighbors, Hancock half-step predictor, HLLC flux at six faces, conservative update plus gravity source. Stencil is 25 cells.
4. **Swap** the two state texture sets.

Conservation relies on both cells that share a face computing bit-identical fluxes from identical inputs. This is tested, not assumed (section 7).

### 3.2 Shock robustness

- **Positivity:** if a reconstructed face state has non-positive density or pressure, drop that cell to first order for the step.
- **Carbuncle:** blend HLLC toward HLL in cells flagged by a pressure-jump shock detector.
- **Dual energy:** evolve internal energy alongside total energy and use it where kinetic energy dominates (Bryan et al. 1995 switch), because float32 loses the pressure in high-Mach flow.
- **Floors:** density and pressure floors, with a counter of floored cells exposed as a diagnostic.

### 3.3 Equation of state

Every equation of state implements the same WGSL interface:

```wgsl
fn eos_pressure(rho: f32, eint: f32) -> f32;      // eint = internal energy density
fn eos_sound_speed(rho: f32, p: f32) -> f32;
fn eos_eint_from_p(rho: f32, p: f32) -> f32;
fn eos_temperature(rho: f32, eint: f32) -> f32;    // diagnostics, cooling later
```

HLLC only needs pressure and sound speed on each side, so it is equation-of-state agnostic as long as wave speeds use Davis/Einfeldt estimates rather than gamma-specific ones.

Shipped in v1:

1. **Ideal gas** with adjustable gamma.
2. **Barotropic** (isothermal and polytropic): pressure depends on density only, energy equation ignored.
3. **Gas plus radiation pressure**: Newton iteration for temperature, fixed iteration count.

Designed for but shipped after v1: **tabulated** (for example Helmholtz-style) via a 2D lookup texture in (log density, log internal energy).

### 3.4 Self-gravity

Solve the Poisson equation for the potential, then apply -rho grad(phi) to momentum and -rho v . grad(phi) to energy.

- **Solver:** V-cycle multigrid with red-black Gauss-Seidel smoothing, full-weighting restriction, trilinear prolongation. One `r32float` texture per level.
- **Periodic boundaries:** subtract mean density from the source.
- **Isolated boundaries:** a reduction pass computes mass, center of mass, and quadrupole moments; the boundary values of the potential come from that multipole expansion.
- **Time coupling:** predictor uses the old potential; after the hydro update, solve for the new potential and apply a correction so the source is time-centered.
- **Resolution check:** flag cells where the Jeans length is under 4 cells (Truelove criterion) and show the count. Optional pressure floor to enforce it.

### 3.5 Boundary conditions

Periodic, outflow, and reflecting, selectable per face. Implemented in the index-mapping function used by every texture load, so there are no ghost cells to fill.

### 3.6 Units and precision

Everything is float32. Use code units scaled so density, velocity, and length are of order one, and carry G as a parameter. Sum diagnostics hierarchically to limit round-off.

## 4. GPU architecture

### 4.1 Textures

| Texture set | Format | Contents |
|---|---|---|
| State A1, A2 | rgba32float x2 | (rho, mx, my, mz), (E, e_int, scalar, spare) |
| State B1, B2 | rgba32float x2 | Same, swap target |
| Primitives | rgba32float x2 | (rho, vx, vy, vz), (p, c_s, scalar, shock flag) |
| Potential levels | r32float per level | phi, plus residual and right-hand side per level |

Memory: about 200 MB of state at 128³ and 1.6 GB at 256³, plus roughly 10% for gravity.

### 4.2 WebGPU limits to respect

Defaults per MDN; request more from the adapter where available and re-check at milestone 0.

| Limit | Default | Consequence |
|---|---|---|
| maxStorageTexturesPerShaderStage | 4 | A pass writes at most two state textures; fine |
| maxSampledTexturesPerShaderStage | 16 | Plenty for reads via `textureLoad` |
| maxStorageBuffersPerShaderStage | 8 | Reason not to use one buffer per field |
| maxStorageBufferBindingSize | 128 MB | Reason not to pack state into one buffer |
| maxComputeInvocationsPerWorkgroup | 256 | Use 8 x 8 x 4 workgroups |
| maxTextureDimension3D | 2048 | Not a constraint |

`rgba32float` is written as a write-only storage texture and read with `textureLoad`; `r32float` can be read-write, which the Gauss-Seidel smoother uses.

### 4.3 Frame loop

Each frame encodes N substeps (primitives, timestep, update, gravity) into one command buffer, then the render pass. N adapts to hold the target frame rate. Diagnostics are read back asynchronously every few frames and never block stepping.

### 4.4 Shader assembly

A small preprocessor resolves `#include` and injects the chosen equation of state, Riemann solver, and limiter. Changing any of them rebuilds the affected pipelines. The flux function is one swappable module, which leaves room for experiments such as a learned flux.

## 5. Rendering and UI

- Ray-marched volume of density, temperature, or the passive scalar, log-scaled with an editable transfer function.
- Axis-aligned slice view with a color bar, for checking against reference plots.
- Controls: problem preset, resolution, equation of state and its parameters, gravity on/off and boundary type, CFL number, pause/step/reset.
- Live readouts: time, dt, total mass, momentum, energy (kinetic, internal, gravitational), max Mach number, floored-cell count, under-resolved Jeans count.

## 6. Tech stack and layout

TypeScript, Vite, raw WebGPU, WGSL. Vitest for unit tests; Playwright driving a WebGPU-enabled browser for GPU tests.

```
src/
  gpu/        device setup, limits, texture and pipeline helpers, shader preprocessor
  hydro/      primitives.wgsl, timestep.wgsl, update.wgsl, riemann/, limiters/
  eos/        ideal.wgsl, barotropic.wgsl, gas_radiation.wgsl
  gravity/    multigrid.ts, smooth.wgsl, restrict.wgsl, prolong.wgsl, multipole.wgsl
  render/     raymarch.wgsl, slice.wgsl, transfer function
  problems/   initial conditions, one file per preset
  reference/  float64 CPU implementation of the same scheme
  ui/
tests/
```

## 7. Validation

A float64 CPU implementation of the same algorithm serves as the reference for small grids (32³), so shader bugs can be separated from scheme limitations.

| Test | Exercises | Pass criterion |
|---|---|---|
| Sod shock tube along x, y, z, and the diagonal | Hydro, symmetry | Matches exact solution; axis runs identical under permutation |
| Strong shock tube (pressure ratio 10⁵) | Positivity, dual energy | No floors triggered away from the shock |
| Sedov-Taylor blast | 3D shocks | Shock radius within 2% of the similarity solution; spherical |
| Kelvin-Helmholtz | Contact handling | Qualitative match to reference at same resolution |
| Uniform advection, 10 box crossings | Conservation | Mass and momentum drift at round-off level |
| Uniform-density sphere potential | Poisson solver, isolated boundaries | Potential error under 1% |
| Jeans instability, periodic | Gravity coupling | Growth rate within 5% of linear theory |
| Polytrope in hydrostatic equilibrium | Gravity plus equation of state | Central density stable over 10 dynamical times |
| Evrard collapse | Gravity plus shocks | Energy curves match published reference |
| Each equation of state against its analytic limits | EOS modules | Round-trip p to e_int to p at float32 precision |

## 8. Milestones

All three headline features land in v1; this is build order, not release order. The equation-of-state interface exists from milestone 1 so nothing is retrofitted.

| # | Deliverable | Exit test |
|---|---|---|
| 0 | Device setup, limit checks, texture helpers, slice renderer, preprocessor | Renders an analytic field at 128³ |
| 1 | First-order hydro (HLL), ideal gas behind the EOS interface, GPU timestep, CPU reference | Sod along all axes |
| 2 | MUSCL-Hancock, HLLC, boundary conditions | Sedov, advection conservation |
| 3 | Shock robustness: positivity fallback, dual energy, carbuncle fix, floors | Strong shock tube |
| 4 | Barotropic and gas-plus-radiation equations of state | EOS tests |
| 5 | Multigrid Poisson solver, periodic and isolated | Sphere potential |
| 6 | Gravity coupling in the hydro step | Jeans, polytrope, Evrard |
| 7 | Volume renderer, UI, presets, diagnostics panel | Usable demo |
| 8 | Profiling and tuning; 256³ where the adapter allows | Frame-rate target met |

After v1: tabulated equation of state, cooling, magnetic fields with constrained transport, static mesh refinement, learned-flux experiments.

## 9. Risks and open questions

- **Float32.** The largest technical risk. Dual energy and code units mitigate it; hydrostatic equilibria and long runs will show whether it is enough. No double-precision fallback exists in WGSL.
- **Flux determinism.** If any driver evaluates the same flux differently in two invocations, conservation breaks. The fallback is a stored face-flux pass, at a memory cost.
- **Stencil cost.** The 25-cell gather means about 50 texture loads per cell. If this is the bottleneck, add a slope-storage pass and trade memory for speed.
- **Multipole boundaries.** Quadrupole order may be too crude for strongly non-spherical mass distributions near the box edge. Raising the order is straightforward; a padded-domain solve is the heavier alternative.
- **Browser and device coverage.** WebGPU availability and adapter limits vary; check current support and decide the minimum supported configuration at milestone 0.
- **Open:** whether to port rendering to three.js for camera and UI convenience once the simulation core is stable.
