# Plan: thousands of gravitating particles (a hybrid of point masses and mesh particles)

Status: not started.

Related: [spiral-galaxy.md](spiral-galaxy.md) section 7 plans collisionless star particles on
the mesh for a live stellar disk. This plan is the general machinery that stage would stand on,
built first for small counts and the interactive tools; where the two overlap (storage, deposit,
force gather, per-block timesteps) the choices here follow that section.

## 1. Goal

Two populations of particles in one run:

- **Heavy** (what exists): up to 32 point masses (`MAX_MASSES`). Each has its own softening
  length, however small against the grid, a sink that takes gas, and exact diagnostics. They
  cost a pass over the grid per four masses per step (`gravity/mass_sums.ts`).
- **Light** (new): thousands to millions of particles that live on the mesh. They feel gravity
  by reading the field where they are, and the gas feels them through the Poisson solve. Their
  cost per step hardly depends on how many there are.

The light population also takes over the jobs that today spend heavy slots on particles that
pull on nothing: tracers, the particles moving sources ride, and pinned queries.

### Non-goals for v1

- Sinks on light particles (accretion needs each cell to find the particles near it).
- Particle-particle forces sharper than the mesh (section 3.5 says why).
- Light particles that gravitate in 2D runs or with self-gravity off: there they are tracers.
- Star formation, feedback, particle ages: spiral-galaxy.md stage 3 and 4.

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| How light particles pull on the gas | Their mass is deposited on the grid and added to the Poisson source | One solve the run already does; no pass per particle |
| How they are pulled | Gather the field at the particle with the deposit's weights, plus the heavy masses and the frame analytically (`gravity/ext.wgsl`) | Same weights for deposit and gather: no self-force within a level. The heavy masses keep their true softening |
| How heavy masses feel light ones | A direct sum over the light particles in `gravity/bh_kick.wgsl` | 32 × N terms, trivial; the heavy masses read no potential, so the mesh cannot tell them |
| Light on light | The mesh alone (collisionless, softened at about two cells) | A direct pair sum on top would count the long-range force twice (section 3.5) |
| Two stages | Tracers first (gather only), gravitating second (deposit) | The first stage is small, works in every preset, and frees the heavy slots at once |
| Storage | Structure-of-arrays storage buffers, a capacity fixed per run, a live count | As spiral-galaxy.md 7.1; no fixed-size struct array, no 32-bit mask |
| Removal | A dead flag, compacted by a pass when a fraction is dead | Order does not matter to anything but ids, which ride along |
| Which kind a tool places | By what it needs: a sink or a softening below two cells is heavy; everything else is light | The user should not have to choose a population |
| Drawing | GPU point splats; SVG markers only for the few the tools placed | SVG does not scale to thousands |

## 3. The physics

### 3.1 State

Per particle, 32 bytes:

- `pos`: xyz relative to the box center, mass in w (0 for a tracer);
- `vel`: xyz, and a word of flags and id in w (dead, fixed, tool-placed, what rides it).

Capacity 65,536 to start (2 MB), a run setting later (spiral-galaxy.md needs four million, and
then separate bindings for positions and velocities).

### 3.2 Stage A: tracers (gather only)

A compute pass over the particles, once per step, kick-drift-kick as `gravity/bh_drift.wgsl`
and `bh_kick.wgsl` do for the heavy masses:

```
a(x) = g_self(x)                (the field texture, cloud-in-cell; zero without a potential)
     + ext_accel(x)             (heavy masses, softened by the larger of the two lengths; the frame)
     + P.gext                   (a uniform field, if the run has one)
```

- Works in 2D and with self-gravity off: `g_self` is then zero, and the particle feels the
  heavy masses and the frame. This differs from a heavy massless particle, which also feels
  the gas by direct sum; say so in the tool's tooltip.
- Leaving the box: flagged dead (masses do not wrap today either; a periodic box wraps them,
  to decide when building).
- On an adaptive mesh the gather needs position → leaf block. To check first:
  `Simulation.probe()` and the renderer's `displayAtlas()` index (`idx`, a leaf slot per
  finest-level block) already do this on the CPU and in `render/volume_atlas.wgsl`.

### 3.3 Stage B: gravitating (deposit)

1. **Deposit.** Cloud-in-cell scatter of each particle's mass into a grid of `atomic<u32>` in
   fixed point (`debris/` tallies this way: exact and independent of order). The scale is set
   per run from the largest particle mass, so that a cell can hold every particle.
2. **Source.** `gravity/rhs.wgsl` and `amr/gravity/rhs.wgsl` add the deposited density to the
   gas's. A periodic box subtracts the mean of both (`P.rho_mean`), and the isolated
   boundary's multipole reduction (`gravity/multipole.ts`) includes both.
3. **Timing.** Deposit at the particles' new positions, before the solve that follows the
   drift, so the gas and the particles see one potential. With per-block timesteps: deposit
   whenever the potential is solved, particles on the finest substep (spiral-galaxy.md 7.1).
4. **Adaptive mesh.** Deposit into the leaf the particle is in; parents receive it by the
   restriction the gas density uses. A particle beside a refinement boundary feels a small
   self-force, as in every code of this kind. Measure it (section 7) before deciding whether
   to deposit at the coarser level within a cell of a boundary.
5. **Momentum.** The gas feels the particles through the same field they gather from, so the
   pair is conserved to the deposit's rounding on a uniform grid.

Energy diagnostics: `eGrav` gains the particles' potential energy and their kinetic energy
is reported beside the gas's.

### 3.4 Heavy and light together

- Light feels heavy: `ext_accel` (already what the gas uses).
- Heavy feels light: in `bh_kick.wgsl`, each heavy mass adds
  `sum_j G m_j d / (d² + ε²)^(3/2)` over the light particles with mass, `ε` the heavy mass's
  softening or two cells, whichever is larger. A pass over the light particles writes the 32
  sums into the heavy masses' accelerations; no reduction over the grid.
- The frame (`Problem.frame`): 'masses' and 'gas' keep their meanings; light particles are
  carried along as the gas is. To decide: whether their mass counts toward the gas's frame.

### 3.5 Why not direct sums between light particles

On a GPU a few thousand particles can sum their pulls on one another directly (4,096 particles
is 16 million pairs a step). But they already feel each other through the mesh, so the direct
sum would count the long-range force twice. Doing it properly is a P3M split: subtract from
each pair the force the mesh gives it, which needs the mesh's pair force as a function of
separation and position within a cell. That is a project of its own, and only matters for
clusters whose members pass within two cells of one another. Left out; the option, if wanted,
is a short-range correction for particles flagged "collisional" in the same cell or its
neighbors.

A cluster whose two-body encounters matter can instead be made of heavy masses, up to 32.

## 4. The tools

- **Tracer** and **pinned query**: light. The query tool's pin stops using a heavy slot.
- **Moving source**: rides a light particle (`SourceSpec.carrier` becomes a light index;
  `hydro/sources.wgsl` reads the light buffers).
- **Sink**: stays heavy, with or without a mass (its sink needs the per-cell loop the heavy
  masses have). So the 32 slots are for sinks and point masses only.
- **Point mass without a sink** (Flux 0): light if its softening is at least two cells,
  otherwise heavy. The Soft slider shows where that line is.
- **A cluster tool** (new): a ball of N particles (count, total mass, radius; a Plummer
  sphere at rest or in virial equilibrium, drag to set it moving). This is how a user gets
  thousands.
- Clear, Remove and the status line count each population apart.

## 5. Preview, drawing, files

- **Orbit preview** (`gravity/preview.ts`): the gas lumps gain the light particles' mass (the
  lump pass reads the deposit grid), so a previewed orbit feels them. Light particles are not
  integrated in the preview.
- **Drawing**: additive point splats in a pass composed with the gas view (as
  spiral-galaxy.md 7.3); the tools' own particles keep their SVG markers and velocity arrows.
- **Diagnostics**: totals for the light population (count, mass, momentum, energies), not a
  panel block each.
- **Files**: `.athdf` gains particle datasets; snapshots and restarts carry them; the API,
  headless runner and MCP tools report counts.

## 6. Order of work

1. Storage, the tracer pass on the uniform grid (3.2), point drawing, API counts.
2. The gather on an adaptive mesh.
3. Move tracers, pinned queries and source carriers to the light population (section 4).
   First point at which the 32 slots are freed.
4. Deposit and source term on the uniform grid, periodic and isolated (3.3).
5. Heavy feels light (3.4); energy diagnostics.
6. Deposit on an adaptive mesh and with per-block timesteps.
7. The cluster tool; preview lumps; files.

Stages 1 to 3 are useful alone and carry no risk to the gravity solver.

## 7. Tests

- Unit: cloud-in-cell weights sum to 1; the fixed-point deposit returns the mass exactly.
- GPU `tracer-orbit`: a tracer about a heavy mass keeps a circular orbit (as `binary-orbit`),
  uniform and adaptive, 2D and 3D.
- GPU `tracer-in-star`: a tracer released in a polytrope oscillates with the period the
  star's potential gives; compare with a heavy massless particle (they should agree to the
  mesh's softening).
- GPU `particles-two-body`, `particles-plummer`, `particles-momentum`: as spiral-galaxy.md 7.4.
- GPU `particles-heavy-light`: a heavy mass and a light particle of equal mass orbit their
  common center; momentum conserved, the orbit closes.
- GPU `particles-self-force`: one particle at rest beside a refinement boundary stays put to
  a stated tolerance over many steps.
- Timing: step time against particle count (1e3 to 1e6) on the uniform grid.

## 8. Risks and open questions

- **Atomics**: a scatter with `atomic<u32>` storage is new to the step (the debris census does
  it once per readout). Check Firefox and Safari early.
- **Fixed-point range**: one scale must hold both a single light particle and a cell with
  thousands. Two words per cell if one is not enough.
- **Force quality at the cell scale**: cloud-in-cell forces are anisotropic within about two
  cells; a triangular-shaped cloud is the next step up if orbits near a dense clump look wrong.
- **2D**: no Poisson solver, so no gravitating light particles there. A 2D solver would lift
  that and give 2D gas self-gravity too; out of scope here.
- **Which population a point mass is in** can change what a user sees for the same slider
  settings (direct sum against mesh). Keep the rule in section 4 visible in the tooltip.
