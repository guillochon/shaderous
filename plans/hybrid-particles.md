# Plan: particles on the mesh, and a spiral galaxy

Status: not started. Two plans merged 2026-11-18, both written 2026-11-15: thousands of
gravitating particles, and a spiral galaxy, whose last stage was the same machinery.

- **Part 1** (sections 1 to 8): two populations of particles in one run, the point masses
  there are and light particles on the mesh. Built first for small counts and the interactive
  tools.
- **Part 2** (sections G1 to G10): an isolated disk galaxy in four stages. Stages 1 to 3 need
  no particles; stage 4, a live stellar disk, is part 1's light population at four million.

Section G8 gives the order of work for both.

# Part 1: thousands of gravitating particles

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
- Star formation, feedback, particle ages: sections G6 and G7.

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| How light particles pull on the gas | Their mass is deposited on the grid and added to the Poisson source | One solve the run already does; no pass per particle |
| How they are pulled | Gather the field at the particle with the deposit's weights, plus the heavy masses and the frame analytically (`gravity/ext.wgsl`) | Same weights for deposit and gather: no self-force within a level. The heavy masses keep their true softening |
| How heavy masses feel light ones | A direct sum over the light particles in `gravity/bh_kick.wgsl` | 32 × N terms, trivial; the heavy masses read no potential, so the mesh cannot tell them |
| Light on light | The mesh alone (collisionless, softened at about two cells) | A direct pair sum on top would count the long-range force twice (section 3.5) |
| Two stages | Tracers first (gather only), gravitating second (deposit) | The first stage is small, works in every preset, and frees the heavy slots at once |
| Storage | Structure-of-arrays storage buffers, a capacity fixed per run, a live count | No fixed-size struct array, no 32-bit mask; it has to reach millions (G7) |
| Removal | A dead flag, compacted by a pass when a fraction is dead | Order does not matter to anything but ids, which ride along |
| Which kind a tool places | By what it needs: a sink or a softening below two cells is heavy; everything else is light | The user should not have to choose a population |
| Drawing | GPU point splats; SVG markers only for the few the tools placed | SVG does not scale to thousands |

## 3. The physics

### 3.1 State

Per particle, 32 bytes:

- `pos`: xyz relative to the box center, mass in w (0 for a tracer);
- `vel`: xyz, and a word of flags and id in w (dead, fixed, tool-placed, what rides it).

Capacity 65,536 to start (2 MB), a run setting later. The galaxy of G7 needs four million
(10^4 M_sun each for the reference disk), which is 128 MB, the default limit on one buffer
binding: positions and velocities then go in separate buffers, or the device asks for a higher
`maxStorageBufferBindingSize`. Float32 positions resolve 0.005 pc in an 80 kpc box.

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
- Leaving the box: flagged dead, its mass added to a ledger (masses do not wrap today either;
  a periodic box wraps them, to decide when building).
- A regrid does not touch the particles.
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
   whenever the potential is solved (the end of a cycle), with all particles on the finest
   substep and the potential between solves extrapolated as the gas's is. Millions of
   particles cost far less than the mesh, so particle time bins are not needed.
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
- **Drawing**: additive point splats in a pass composed with the gas view, by the off-canvas
  path screenshots and movies already use; the tools' own particles keep their SVG markers and
  velocity arrows.
- **Diagnostics**: totals for the light population (count, mass, momentum, energies), not a
  panel block each.
- **Files**: `.athdf` gains particle datasets; snapshots and restarts carry them; the API,
  headless runner and MCP tools report counts and totals.

## 6. Order of work

1. Storage, the tracer pass on the uniform grid (3.2), point drawing, API counts.
2. The gather on an adaptive mesh.
3. Move tracers, pinned queries and source carriers to the light population (section 4).
   First point at which the 32 slots are freed.
4. Deposit and source term on the uniform grid, periodic and isolated (3.3).
5. Heavy feels light (3.4); energy diagnostics.
6. Deposit on an adaptive mesh and with per-block timesteps.
7. The cluster tool; preview lumps; files.

Steps 1 to 3 are useful alone and carry no risk to the gravity solver. The galaxy's stage 4
(G7) needs steps 4 to 6 and the files of step 7.

## 7. Tests

- Unit: cloud-in-cell weights sum to 1; the fixed-point deposit returns the mass exactly.
- GPU `tracer-orbit`: a tracer about a heavy mass keeps a circular orbit (as `binary-orbit`),
  uniform and adaptive, 2D and 3D.
- GPU `tracer-in-star`: a tracer released in a polytrope oscillates with the period the
  star's potential gives; compare with a heavy massless particle (they should agree to the
  mesh's softening).
- GPU `particles-two-body`: two particles on a Kepler orbit several cells apart; period and
  energy drift.
- GPU `particles-plummer`: a Plummer sphere of particles stays in equilibrium (Lagrangian
  radii over ten crossing times), on the uniform grid and across refinement levels.
- GPU `particles-momentum`: gas and particles together conserve momentum to round-off of the
  deposit (a gas blob and a particle cluster orbiting one another).
- GPU `particles-heavy-light`: a heavy mass and a light particle of equal mass orbit their
  common center; momentum conserved, the orbit closes.
- GPU `particles-self-force`: one particle at rest beside a refinement boundary stays put to
  a stated tolerance over many steps.
- Timing: step time against particle count (1e3 to 4e6) on the uniform grid. Gate C of G8 is
  the deposit's throughput at four million.

## 8. Risks and open questions

- **Atomics**: a scatter with `atomic<u32>` storage is new to the step (the debris census does
  it once per readout). Check Firefox and Safari early.
- **Scatter throughput**: atomic adds from millions of particles into a few dense blocks
  contend. If the deposit is slow, sort particles by cell first (a GPU radix sort is its own
  piece of work) or deposit in several strided passes.
- **Per-block timesteps**: the rule of 3.3 (particles on the finest substep, potential
  extrapolated) must be shown to conserve energy as well as the gas does; if it does not, runs
  with particles fall back to a single timestep.
- **Fixed-point range**: one scale must hold both a single light particle and a cell with
  thousands. Two words per cell if one is not enough.
- **Force quality at the cell scale**: cloud-in-cell forces are anisotropic within about two
  cells; a triangular-shaped cloud is the next step up if orbits near a dense clump look wrong.
- **2D**: no Poisson solver, so no gravitating light particles there. A 2D solver would lift
  that and give 2D gas self-gravity too; out of scope here.
- **Which population a point mass is in** can change what a user sees for the same slider
  settings (direct sum against mesh). Keep the rule in section 4 visible in the tooltip.

# Part 2: a spiral galaxy

## G1. Goal

Simulate an isolated, Milky Way-like disk galaxy whose gas forms spiral arms, in four stages
that each end in a preset, validation tests and a section of `docs/numerics.md`:

| Stage | Result | New machinery | Size |
|---|---|---|---|
| 1 | Gas shocking in an imposed, rotating spiral potential (2D and 3D) | Static galactic potential, rotating spiral term | Small |
| 2 | A 3D self-gravitating gas disk on the adaptive mesh that grows its own flocculent arms | Disk initial conditions, ring diagnostics | Medium |
| 3 | A multiphase interstellar medium regulated by star formation and feedback | Cooling to 10^8 K, star formation, supernova heating | Medium |
| 4 | A live stellar disk: swing-amplified arms, bars, stars formed from gas | Part 1's light particles, at four million | Large |

Stages 1 and 2 need no new kind of data on the GPU. Stage 3 is useful alone but loses the mass
it turns into stars until stage 4 exists. Stage 4 stands on part 1, a new subsystem of about
the weight of the MHD work.

### Non-goals

A live dark halo (the halo stays an analytic potential), cosmological infall, radiation
transport, cosmic rays, chemistry beyond the cooling curve, and galaxy mergers. A magnetized
galaxy is not planned here, but nothing below blocks it: MHD already runs with gravity,
cooling and point masses.

## G2. What exists and what is missing

Already in the code and used as is:

- Isolated-boundary multigrid self-gravity on the uniform grid and on blocks
  (`src/gravity/`, `src/amr/gravity/`), Jeans refinement and the Truelove pressure floor.
- The external-field hook `gravity/ext.wgsl` (`ext_accel`, `ext_dphi`, `ext_phi`), called by
  every pass that needs a pull that is not self-gravity: the updates, predictors, half-kick
  corrections, well-balanced reconstruction, timestep, debris tally and display (13 shaders).
  Its work is taken in kick form, which is what a cold disk orbiting at Mach 20 to 30 needs.
- `refineDense` and `refineRho`, which keep a thin disk at the finest level and the halo
  coarse (as `problems/mri_disk.ts` does), per-block timesteps, damping zones
  (`hydro/damping.wgsl`), fluff limits, dual energy.
- Globally isothermal gas (`eos/barotropic.wgsl` with gamma = 1) and the Koyama & Inutsuka
  two-phase cooling curve (`cooling/ism.wgsl`), with a subcycled integrator.
- 64-bit fixed-point atomic sums (`debris/debris.wgsl`), the precedent for exact, order-free
  scatter that the particles' mass deposit needs (3.3).
- Physical units (`units/units.ts`), projections for a face-on view, movies, `.athdf` output.

Missing:

| Gap | Stage | Notes |
|---|---|---|
| Static potential of a halo, stellar disk and bulge | 1 | Today: at most 32 Plummer-softened point masses and a uniform field |
| Rotating spiral perturbation | 1 | Needs the run time inside `ext.wgsl` |
| Disk in vertical and rotational balance, own gravity included | 2 | Star settling (`problems/settle.ts`) handles gas at rest in a barotrope, not rotating disks |
| Ring diagnostics: Σ(R), v_φ(R), mode amplitudes | 2 | The MRI disk's stress diagnostics are the closest thing |
| Cooling above about 10^4 K | 3 | The current curve has only Lyman-alpha above that |
| Star formation and supernova heating | 3 | Sinks exist only around the point masses |
| Collisionless particles | 4 | Nothing exists: part 1 |

## G3. Units and the reference galaxy

Code units, chosen so the cooling curve needs no conversion and every number is near 1:

- length: 1 kpc;
- density: 1 hydrogen atom per cm^3, which makes the mass unit about 2.5e7 M_sun
  (3.5e7 with helium counted; fix this when the composition is chosen);
- G = 1 (`gravityUnits`), which makes the velocity unit about 10 km/s and the time unit
  about 95 Myr.

Then the warm gas's sound speed is about 0.8, the circular speed about 21, an orbit at
8 kpc takes about 2.35 time units (223 Myr), and code temperature 1 is about 1.3e4 K per unit
mean molecular weight.

Reference model (all of it parameters of the presets):

| Component | Form | Default |
|---|---|---|
| Halo | Logarithmic, Φ = ½ v0² ln(Rc² + R² + z²/q²) | v0 = 220 km/s, Rc = 12 kpc, q = 1 |
| Stellar disk | Miyamoto-Nagai | M = 4e10 M_sun, a = 3 kpc, b = 0.3 kpc |
| Bulge | Hernquist | M = 1e10 M_sun, a = 0.5 kpc |
| Gas disk | Exponential surface density, tapered at both ends | M = 1e10 M_sun, Rd = 6 kpc, out to 15 kpc |
| Spiral (stage 1) | Cox & Gómez (2002), m = 2 or 4 | pitch 15°, Ω_p = 22 km/s/kpc, 5% of the radial force |

Stage 1's 2D preset uses the halo alone with Rc = 1 kpc (a flat curve with a finite central
angular velocity), as the galactic-shock literature does.

## G4. Stage 1: the galactic potential and spiral shocks

### G4.1 The potential module

New `src/gravity/galaxy.wgsl`, included by `gravity/ext.wgsl`, and a float64 mirror
`src/gravity/galaxy.ts` for initial conditions and unit tests (the pattern of
`cooling/cooling.ts`). It adds to each of the three hook functions:

- `gal_accel(x, t)`: analytic acceleration of the three components and the spiral.
- `gal_dphi(x, d, t)`: Φ(x + d) − Φ(x) formed without subtracting two large numbers, as the
  point masses' is. Each component has a cancellation-free form:
  - Miyamoto-Nagai is a Plummer potential in q = R² + (a + √(z² + b²))², so the existing
    G M (q1 − q0) / (r0 r1 (r0 + r1)) applies;
  - Hernquist: G M (r1 − r0) / ((r0 + a)(r1 + a)) with r1 − r0 = (q1 − q0) / (r0 + r1);
  - logarithmic: ½ v0² ln(q1/q0) evaluated as v0² (u + u³/3 + u⁵/5), u = (q1 − q0)/(q1 + q0),
    which is exact to float32 for neighboring cells.
- `gal_phi(x, t)`: for the display and for `.athdf` output.

Parameters go in a new `gal: array<vec4<f32>, 4>` in `Params` (halo, disk, bulge, spiral), with
`PARAM_BYTES` and `packParams` updated, or in a small uniform of their own if `Params` has no
room. `ext_on()` must also be true when a galaxy is present without point masses, and
`grav_on` must be set so the field passes run in 2D with no Poisson solve (the 2D binary
already runs that way with masses only).

`Problem` gets `galaxy?: (params) => GalaxySpec`, and `SimConfig` carries the packed spec.

### G4.2 Time dependence

The spiral's phase is m(φ − Ω_p t) + the logarithmic winding term. Steps never wait for the
CPU, so the phase must come from the GPU clock (`Clock.t`, already Kahan-summed), read by the
passes that call the hook. `ext_dphi` already takes an `ahead` argument for the masses'
half-step position; the spiral uses the same offset. To check first: that every one of the 13
callers has the clock bound, and what time each should use (start of step for the first
half-kick, end of step for the second, half step for the predictor). With per-block timesteps
a block's own time is the bin's, not the global clock's: `amr/bins_*.wgsl` must supply it.

The spiral's amplitude ramps up over one rotation of the pattern so the gas is not shocked by
its sudden appearance.

An axisymmetric potential needs none of this, so G4.1 can land and be tested before G4.2.

### G4.3 Presets

`problems/galaxy.ts`:

- **`galaxy-shocks`** (2D and 3D, `supports2d`): isothermal gas, cs = 8 km/s, uniform or
  exponential surface density, on circular orbits with
  v_φ² = R ∂Φ/∂R + cs² dlnΣ/dlnR. Box 30 kpc, 1024² by default (29 pc cells), outflow
  boundaries with a damping zone beyond 13 kpc, as the binary disk has. In 3D the gas has a
  vertical profile in balance with the potential, ρ = ρ_mid exp(−(Φ(R, z) − Φ(R, 0))/cs²),
  which is exact for an isothermal gas with no self-gravity.
  Parameters: arm number, pitch angle, pattern speed, arm strength, sound speed.
  Expected: shocks along the arms inside corotation, spurs downstream of them, branches at
  the 4:1 resonance.
- **`galaxy-disk`** (no spiral): the same gas in the axisymmetric potential. It should do
  nothing, and is the test of the potential and of the mesh.

### G4.4 Validation

- Unit (vitest): the mirror's acceleration against a numerical gradient of its potential;
  `gal_dphi` against float64 differences; circular speed and epicyclic frequency against
  closed forms.
- GPU `galaxy-field`: the field texture and potential differences against the mirror at
  sampled cells, on the uniform grid and on blocks.
- GPU `galaxy-hold`: `galaxy-disk` for two orbits at 8 kpc. Measures the drift of Σ(R) and of
  the total angular momentum. **This is the gate for stage 2**: a Cartesian grid diffuses
  angular momentum in a cold disk, and the measured drift decides the resolution stage 2
  needs. If it is too high at 4 cells per scale height, options are a higher-order
  reconstruction for these presets or a thicker (warmer) disk.
- GPU `galaxy-shock` (stretch): the azimuthal density profile at one radius against the 1D
  steady solution for tightly wound arms (Roberts 1969; Shu, Milione & Roberts 1973), from a
  small ODE reference in `src/reference/`.

### G4.5 Risks

- Float32 phase error is not a concern (t is compensated), but the potential difference
  between neighboring cells is 1e-3 of the potential: G4.1's difference forms are required,
  not optional.
- The center: with Rc = 1 kpc the inner orbits take 30 Myr and cross few cells. The 2D preset
  may need an inner sink or damping disk if the center becomes noisy.

## G5. Stage 2: a self-gravitating gas disk

### G5.1 Initial conditions

A table in (R, z), built on the CPU in float64 when the preset loads, which `init()`
interpolates:

1. **Vertical structure.** At each radius, integrate cs² dlnρ/dz = −(∂Φ_ext/∂z + g_gas) with
   dg_gas/dz = 4πGρ (the thin-disk form of the gas's own pull), shooting on the midplane
   density until the column equals Σ(R).
2. **Rotation.** v_φ² = R ∂Φ_ext/∂R + R ∂Φ_gas/∂R + (R/ρ) ∂p/∂R, with the gas's radial pull
   from the razor-thin exponential disk (Freeman 1970, Bessel functions). The gas is about a
   tenth of the mass inside any radius, so the error of the thin form is well under 1% of
   v_φ. If `galaxy-hold` shows the disk ringing radially, replace this with a GPU pass that
   sets v_φ from the grid's own solved field after loading, in the spirit of star settling.
3. **Seeds.** Density perturbations of 1% from the deterministic generator the thermal
   instability uses, so every resolution starts from the same field.

The halo gas is tenuous and at rest; for isothermal runs the fluff limits keep it harmless,
as in the MRI disk.

### G5.2 Mesh and budget

- Box 80 kpc, so the isolated boundary (monopole and quadrupole of the gas) is well outside
  the 15 kpc disk; the extra root blocks cost almost nothing.
- Finest cells 39 pc (2048 across the box, five or six levels above a 64³ or 32³ root),
  held by `refineDense` within about two scale heights of the midplane out to 15 kpc: about
  12 million cells. 20 pc cells in the same slab would be about 90 million and do not fit;
  20 pc within 200 pc of the midplane inside 12 kpc is about 23 million and might.
- The step is set by orbital speed on the finest cells, about 0.04 Myr, so 500 Myr is of
  order 10^4 fine steps. Per-block timesteps help little, because the whole disk is at the
  finest level and moves at similar speeds. Measure the cost early; the MRI disk (24 million
  cells with a field) took 14 minutes per orbit on an RTX 4090.
- Jeans refinement stops at the finest level, where the Truelove floor then acts as the
  effective equation of state of the clumps. It, not physics, sets the smallest structure,
  and the plan accepts that until stage 3.

### G5.3 Presets and diagnostics

- **`galaxy-gas`**: isothermal, Toomre Q at 8 kpc as a parameter (through the surface
  density), gas fraction, sound speed, and the stage 1 spiral as an optional forcing.
  Default Q about 1.2: flocculent, sheared arms within two orbits.
- **Ring diagnostics** (`Problem.disk` generalized): a GPU reduction into radial bins of
  mass, angular momentum, radial and azimuthal velocity dispersion, and the cos and sin
  moments of the surface density for m = 1 to 6. From these come Σ(R), the rotation curve,
  Q(R) and the arm amplitudes over time, in the panel, the API and the headless runner.

### G5.4 Validation

- `galaxy-hold` again with self-gravity on and Q = 3: profile and angular momentum drift.
- `galaxy-toomre`: the same disk at Q = 0.7 and Q = 2 after one orbit. The low-Q disk's
  non-axisymmetric power must grow by a set factor and the high-Q disk's must not.
- Potential check: the solved gas potential in the midplane against Freeman's.

## G6. Stage 3: the interstellar medium

### G6.1 Cooling

`cooling/ism_hot.wgsl` (or an extension of `ism.wgsl` behind a flag): Koyama & Inutsuka
below about 10^4.2 K joined to a collisional-ionization curve above (Sutherland & Dopita
1993), the joint used by TIGRESS (Kim & Ostriker 2017). The hot branch is a short piecewise
power-law fit in log T, evaluated in the shader, with a float64 mirror and a unit test
against the tabulated curve. `P.cool.z` is unused by the current module and can carry the
metallicity scale. A fixed mean molecular weight is kept (the ionizing equation of state is
an alternative, at more cost).

The halo becomes hot gas in rough hydrostatic balance, ρ ∝ exp(−Φ/c_hot²) at about 10^6 K.

### G6.2 Star formation and feedback

A pass at the end of the update, in the manner of `hydro/sources.wgsl`:

- **Where:** gas denser than a threshold, colder than about 10^4 K, with converging flow.
- **Rate:** a Schmidt law, ρ̇_* = ε_ff ρ / t_ff with ε_ff about 0.01.
- **Mass:** removed from the cell and added to a ledger, as the sinks keep theirs. Until
  stage 4 it leaves the simulation; over 500 Myr at a few M_sun per year that is about a
  tenth of the gas, which is why this stage is honest only for short runs or with stage 4.
- **Energy:** 10^51 erg per 100 M_sun formed, as heat. Heat spread evenly at 40 pc
  resolution radiates away before it does work, so the heating is stochastic (Dalla Vecchia
  & Schaye 2012): a cell is heated by a fixed 10^7.5 K jump with the probability that makes
  the mean energy right. The random number is a hash of the cell's position and the step
  count, so a run is reproducible and independent of dispatch order.

Hot gas at 500 to 1000 km/s shortens the step severalfold. Most of it is in coarse halo
blocks, which is the case per-block timesteps were built for.

### G6.3 Preset and validation

- **`galaxy-ism`**: `galaxy-gas` with cooling, star formation and feedback.
- Unit: the cooling mirror against the table; the star formation rate of a uniform box
  against the Schmidt law; the mean injected energy against its expectation.
- GPU `sn-remnant`: one heated cell in a uniform cooling medium; shell radius and retained
  momentum against the fits of Kim & Ostriker (2015).
- GPU `galaxy-ks`: the disk's star formation rate surface density against gas surface
  density lies within a factor of a few of the Kennicutt-Schmidt relation, and the disk
  reaches a steady rate instead of collapsing or blowing out.

## G7. Stage 4: live stars

The stars are part 1's light population: storage, deposit, gather, time stepping, drawing and
files are sections 3 and 5, and the particle tests are section 7. What is the galaxy's own:

### G7.1 Coupling to stage 3

Star formation writes new particles through an atomic slot counter instead of the ledger,
restoring mass conservation, and feedback is delayed by the particle's age. The particle's
word of flags and id (3.1) carries its formation time. Stars are drawn colored by age.

### G7.2 Initial conditions

An equilibrium disk and bulge in the static halo:

- built in: Hernquist's (1993) moment method (local Maxwellians from the Jeans equations,
  with the epicyclic approximation and a chosen Q profile), in a worker (`amr/init_worker.ts`
  shows the pattern). Simple, and good enough once the disk relaxes for an orbit;
- loaded: a particle set in GADGET-style HDF5 through the existing HDF5 reader, for models
  made by dedicated tools (AGAMA, GalIC).

The static Miyamoto-Nagai disk and bulge of stages 1 to 3 are replaced by the particles;
the halo stays analytic. Note for the presets: a live disk in a rigid halo forms a bar more
readily than in a live one.

### G7.3 Presets and validation

- **`galaxy`**: live stellar disk and bulge, gas disk, static halo; with and without the
  stage 3 physics. This is the flagship.
- GPU `galaxy-live-hold`: the Q = 2 stellar disk keeps its surface density, scale height
  and dispersion profile for two orbits (measures heating by particle noise).
- GPU `galaxy-swing`: the Q = 1.2 disk grows arms whose amplitude and number are within
  the expected range for swing amplification (the ring diagnostics, extended to particles).

### G7.4 Risks

- **Noise.** 10^4 M_sun particles on 40 pc cells heat the disk and seed arms. `galaxy-live-hold`
  measures it; the remedy is more particles, which memory allows (16 million is 512 MB).
- **Self-force at level boundaries** in the thin disk, where the mesh is refined by height.
  Keeping the stellar disk's scale height inside the finest slab avoids most of it.

## G8. Order of work and gates

1. Stage 1, axisymmetric potential (G4.1), `galaxy-disk`, `galaxy-field`, `galaxy-hold`.
   **Gate A:** angular momentum drift at the planned resolution.
2. Stage 1, spiral term (G4.2), `galaxy-shocks` in 2D then 3D. First public preset and a
   teaching-guide activity.
3. Stage 2 initial conditions and ring diagnostics, `galaxy-gas` on a small mesh, then the
   full mesh on a 24 GB card (a Runpod pod if the desktop is short of memory).
   **Gate B:** cost per orbit and memory per cell, which fix the resolution of stages 3 and 4.
4. Stage 3 cooling, then star formation and feedback, each with its tests.
5. Stage 4: part 1's steps 4 to 7 (section 6) at four million particles, then initial
   conditions (G7.2) and the coupling to stage 3 (G7.1). **Gate C** after part 1's step 4:
   deposit throughput at four million particles.

Part 1's steps 1 to 3 (tracers, and the tools moved off the 32 heavy slots) depend on nothing
here and can be built at any time. Its step 4 does not depend on stage 3 and can start after
gate B if live stars matter more than the interstellar medium.

## G9. Open decisions

| Decision | Options | Leaning |
|---|---|---|
| Halo form | Logarithmic, Hernquist, NFW | Logarithmic: one parameter for a flat curve, exact difference form |
| Where the galaxy parameters live | `Params`, or a uniform of their own | Decide on reading `packParams` and the bind groups of the 13 callers |
| Rotation of the stage 2 disk | Analytic thin-disk form, or set from the grid's solved field | Analytic first; switch if `galaxy-hold` rings |
| Stage 3 before stage 4, or after | Either | Part 1's step 4 in parallel with stage 3 |
| Mean molecular weight with cooling | Fixed, or the ionizing equation of state | Fixed |

## G10. References

- Cox & Gómez 2002, ApJS 142, 261: analytic spiral arm potential.
- Dalla Vecchia & Schaye 2012, MNRAS 426, 140: stochastic thermal feedback.
- Freeman 1970, ApJ 160, 811: potential of an exponential disk.
- Hernquist 1993, ApJS 86, 389: equilibrium disk initial conditions from moments.
- Kim & Ostriker 2002, ApJ 570, 132: spurs from spiral shocks.
- Kim & Ostriker 2015, ApJ 802, 99: supernova remnant momentum at finite resolution.
- Kim & Ostriker 2017, ApJ 846, 133: TIGRESS, cooling and feedback in a galactic disk.
- Koyama & Inutsuka 2002, ApJ 564, L97: the two-phase cooling function (in the code).
- Miyamoto & Nagai 1975, PASJ 27, 533: disk potential.
- Roberts 1969, ApJ 158, 123; Shu, Milione & Roberts 1973, ApJ 183, 819: galactic shocks.
- Sutherland & Dopita 1993, ApJS 88, 253: cooling of hot gas.
- Toomre 1964, ApJ 139, 1217: disk stability.
- Truelove et al. 1997, ApJ 489, L179: the Jeans condition (in the code).
- Wada & Koda 2004, MNRAS 349, 270: instability of spiral shocks.
- Wang et al. 2010, MNRAS 407, 705: equilibrium gas disks by the potential method.
