# Shaderous

A browser-based 3D compressible hydrodynamics code for astrophysical test problems. All fluid
and gravity computation runs in WGSL compute shaders on WebGPU. The design is in
[plan.md](plan.md).

**Live:** https://guillochon.github.io/shaderous/ (needs a WebGPU browser). Validation suite:
https://guillochon.github.io/shaderous/validate.html

- **Hydro:** unsplit MUSCL-Hancock, piecewise-linear reconstruction, HLLC (or HLL) fluxes,
  positivity fallback, shock-flagged HLL blend, dual-energy formalism, floors.
- **Self-gravity:** multigrid Poisson solve every step, periodic or isolated (multipole) boundaries.
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
  hydro/       primitives, timestep, clock and update passes; riemann/ and limiters/ modules
  eos/         WGSL equation-of-state modules and their float64 mirrors
  gravity/     multigrid passes, multipole boundary, half-kick corrector
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
- **A step is five hydro passes:** primitives, timestep reduction, clock, half-step predictor,
  update. With gravity it adds the Poisson source, V-cycles, a pass that writes the
  acceleration field, and the second half kick. `sim.profile()` times each pass.
- **Frame pacing** adapts the number of steps per frame to the GPU. Firefox reports finished
  GPU work only every ~100 ms, so there the loop steers by backlog rather than by timing.
  Results agree across browsers to round-off; axis-permuted runs are bit-identical in Chrome
  and differ in the last bit in Firefox.
- **Barotropic gas** uses HLL, since there is no contact wave for HLLC to resolve.

## Licence

MIT; see [LICENSE](LICENSE).
