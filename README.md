# Shaderous

**Compressible gas dynamics, with shocks, gravity and adaptive mesh refinement, running
entirely on your GPU inside a web browser. For physics research and teaching.**

**▶ Open it: https://guillochon.github.io/shaderous/** · Validation suite:
[validate.html](https://guillochon.github.io/shaderous/validate.html) · Requires a browser
with WebGPU (current Chrome, Edge, Firefox or Safari).

## Quick start

**In the browser (no install).** Open the [live site](https://guillochon.github.io/shaderous/).
It starts with a 2D Kelvin-Helmholtz instability. Choose another problem from the left
panel: shock tubes, blasts, instabilities, a supersonic wind tunnel, and astrophysical flows
such as stars, white dwarf collisions, tidal disruptions and accreting binaries. Drag in the view to rotate it, and use the brushes to stir, heat or add
gas. The address bar always describes the current setup, so copying it (or pressing
**Share**) gives someone else the same run. Some examples:

- [Rayleigh-Taylor instability](https://guillochon.github.io/shaderous/?problem=rayleigh-taylor)
- [A Mach 3 wind tunnel with a step](https://guillochon.github.io/shaderous/?problem=forward-step)
- [Sedov-Taylor blast in 3D](https://guillochon.github.io/shaderous/?problem=sedov&dim=3&n=128)
- [A star torn apart by a black hole, on an adaptive mesh](https://guillochon.github.io/shaderous/?problem=tidal-disruption&dim=3&n=128&amr=2)
- [Thermal instability of the interstellar medium](https://guillochon.github.io/shaderous/?problem=thermal-instability)
- [A binary accreting from a circumbinary disk](https://guillochon.github.io/shaderous/?problem=binary)

**Locally.** Running it yourself needs Node 20+.

```bash
git clone https://github.com/guillochon/shaderous.git
```

```bash
cd shaderous && npm install
```

```bash
npm run dev
```

Then open http://localhost:5183 (the validation suite is at
http://localhost:5183/validate.html). `npm test` runs the unit tests in Node, and
`npm run test:gpu` runs the full validation suite on your GPU through Playwright.

## What it is

Shaderous is a three-dimensional finite-volume code for compressible gas dynamics. Every
fluid and gravity computation runs in WGSL compute shaders through WebGPU, so it runs on GPUs
from any vendor, from a phone to a workstation, with nothing to install. Nothing in the
scheme is specific to astrophysics: it solves the compressible Euler equations with optional
gravity, diffusion, heating and cooling, and fits a fluid dynamics course as well as an
astrophysics one. The problems that ship with it are mostly astrophysical because they test
every part of the code at once: strong shocks, self-gravity, stars that must stay in
hydrostatic balance, accreting point masses, cooling, and matter as far from an ideal gas as
degenerate electrons. It uses the methods of production simulation codes:

- an unsplit MUSCL-Hancock scheme with third-order reconstruction and HLLC fluxes;
- a multigrid Poisson solver with periodic or isolated boundaries;
- block-structured adaptive mesh refinement with per-block timesteps;
- point masses with sinks;
- equations of state from an ideal gas to degenerate electrons and ionizing hydrogen;
- optically thin cooling, viscosity, thermal conduction, and species carried with the gas,
  whose composition sets the equation of state cell by cell.

WebGPU is a graphics interface, and fitting a conservative scheme into it shaped the design.
No face flux is ever stored: the two cells that share a face compute bit-identical fluxes, so
mass, momentum and energy are conserved to round-off even across refinement and time-step
boundaries. Ghost cells are computed as they are loaded, never exchanged. The timestep and the
scheduling of blocks happen on the GPU, so a step never waits for the CPU. Everything is single
precision, as WGSL requires. A well-balanced reconstruction and initial states settled against
the grid's own potential still hold stars and atmospheres in hydrostatic balance.

Shaderous is meant for teaching, for exploring a problem before committing to a production
run, and for sharing a reproducible setup as a link. It is not a replacement for codes such as
Athena++, FLASH or RAMSES: it runs on one GPU, within the memory the browser allows, and has
no magnetic fields or radiation transport. The state can be exported as Athena++-style HDF5
that [yt](https://yt-project.org/) reads directly. A suite of 36 validation tests runs on your
own GPU, in the browser or from the command line.

## Documentation

| | |
|---|---|
| **[Teaching guide](docs/teaching.md)** | Using Shaderous in the classroom: activities, links and discussion questions for non-major undergraduates, physics and astronomy majors, and graduate students. |
| **[Numerics](docs/numerics.md)** | Every part of the solver in detail: equations, discretization, Riemann solvers, reconstruction, gravity, AMR, local timesteps, equations of state, cooling, diffusion, species. With references. |
| **[Architecture](docs/architecture.md)** | How the code maps onto WebGPU: constraints, data layout, shader assembly, gathered fluxes, ghost cells on load, the AMR atlas, GPU scheduling, rendering, memory, I/O and testing. |

The method paper is in progress in [`paper/`](paper/), and the original design plan is in
[plan.md](plan.md).

## License

MIT; see [LICENSE](LICENSE).
