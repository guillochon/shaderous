# Plan: a spiral galaxy

Status: draft, 2026-11-15. Nothing here is built yet.

## 1. Goal

Simulate an isolated, Milky Way-like disk galaxy whose gas forms spiral arms, in four stages
that each end in a preset, validation tests and a section of `docs/numerics.md`:

| Stage | Result | New machinery | Size |
|---|---|---|---|
| 1 | Gas shocking in an imposed, rotating spiral potential (2D and 3D) | Static galactic potential, rotating spiral term | Small |
| 2 | A 3D self-gravitating gas disk on the adaptive mesh that grows its own flocculent arms | Disk initial conditions, ring diagnostics | Medium |
| 3 | A multiphase interstellar medium regulated by star formation and feedback | Cooling to 10^8 K, star formation, supernova heating | Medium |
| 4 | A live stellar disk: swing-amplified arms, bars, stars formed from gas | Collisionless particles on the mesh | Large |

Stages 1 and 2 need no new kind of data on the GPU. Stage 3 is useful alone but loses the mass
it turns into stars until stage 4 exists. Stage 4 is a new subsystem of about the weight of
the MHD work.

### Non-goals

A live dark halo (the halo stays an analytic potential), cosmological infall, radiation
transport, cosmic rays, chemistry beyond the cooling curve, and galaxy mergers. A magnetized
galaxy is not planned here, but nothing below blocks it: MHD already runs with gravity,
cooling and point masses.

## 2. What exists and what is missing

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
  scatter that stage 4's mass deposit needs.
- Physical units (`units/units.ts`), projections for a face-on view, movies, `.athdf` output.

Missing:

| Gap | Stage | Notes |
|---|---|---|
| Static potential of a halo, stellar disk and bulge | 1 | Today: at most 4 Plummer-softened point masses and a uniform field |
| Rotating spiral perturbation | 1 | Needs the run time inside `ext.wgsl` |
| Disk in vertical and rotational balance, own gravity included | 2 | Star settling (`problems/settle.ts`) handles gas at rest in a barotrope, not rotating disks |
| Ring diagnostics: Σ(R), v_φ(R), mode amplitudes | 2 | The MRI disk's stress diagnostics are the closest thing |
| Cooling above about 10^4 K | 3 | The current curve has only Lyman-alpha above that |
| Star formation and supernova heating | 3 | Sinks exist only around the 4 point masses |
| Collisionless particles | 4 | Nothing exists: storage, deposit, force gather, time stepping, I/O, drawing |

## 3. Units and the reference galaxy

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

## 4. Stage 1: the galactic potential and spiral shocks

### 4.1 The potential module

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

### 4.2 Time dependence

The spiral's phase is m(φ − Ω_p t) + the logarithmic winding term. Steps never wait for the
CPU, so the phase must come from the GPU clock (`Clock.t`, already Kahan-summed), read by the
passes that call the hook. `ext_dphi` already takes an `ahead` argument for the masses'
half-step position; the spiral uses the same offset. To check first: that every one of the 13
callers has the clock bound, and what time each should use (start of step for the first
half-kick, end of step for the second, half step for the predictor). With per-block timesteps
a block's own time is the bin's, not the global clock's: `amr/bins_*.wgsl` must supply it.

The spiral's amplitude ramps up over one rotation of the pattern so the gas is not shocked by
its sudden appearance.

An axisymmetric potential needs none of this, so 4.1 can land and be tested before 4.2.

### 4.3 Presets

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

### 4.4 Validation

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

### 4.5 Risks

- Float32 phase error is not a concern (t is compensated), but the potential difference
  between neighboring cells is 1e-3 of the potential: 4.1's difference forms are required,
  not optional.
- The center: with Rc = 1 kpc the inner orbits take 30 Myr and cross few cells. The 2D preset
  may need an inner sink or damping disk if the center becomes noisy.

## 5. Stage 2: a self-gravitating gas disk

### 5.1 Initial conditions

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

### 5.2 Mesh and budget

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

### 5.3 Presets and diagnostics

- **`galaxy-gas`**: isothermal, Toomre Q at 8 kpc as a parameter (through the surface
  density), gas fraction, sound speed, and the stage 1 spiral as an optional forcing.
  Default Q about 1.2: flocculent, sheared arms within two orbits.
- **Ring diagnostics** (`Problem.disk` generalized): a GPU reduction into radial bins of
  mass, angular momentum, radial and azimuthal velocity dispersion, and the cos and sin
  moments of the surface density for m = 1 to 6. From these come Σ(R), the rotation curve,
  Q(R) and the arm amplitudes over time, in the panel, the API and the headless runner.

### 5.4 Validation

- `galaxy-hold` again with self-gravity on and Q = 3: profile and angular momentum drift.
- `galaxy-toomre`: the same disk at Q = 0.7 and Q = 2 after one orbit. The low-Q disk's
  non-axisymmetric power must grow by a set factor and the high-Q disk's must not.
- Potential check: the solved gas potential in the midplane against Freeman's.

## 6. Stage 3: the interstellar medium

### 6.1 Cooling

`cooling/ism_hot.wgsl` (or an extension of `ism.wgsl` behind a flag): Koyama & Inutsuka
below about 10^4.2 K joined to a collisional-ionization curve above (Sutherland & Dopita
1993), the joint used by TIGRESS (Kim & Ostriker 2017). The hot branch is a short piecewise
power-law fit in log T, evaluated in the shader, with a float64 mirror and a unit test
against the tabulated curve. `P.cool.z` is unused by the current module and can carry the
metallicity scale. A fixed mean molecular weight is kept (the ionizing equation of state is
an alternative, at more cost).

The halo becomes hot gas in rough hydrostatic balance, ρ ∝ exp(−Φ/c_hot²) at about 10^6 K.

### 6.2 Star formation and feedback

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

### 6.3 Preset and validation

- **`galaxy-ism`**: `galaxy-gas` with cooling, star formation and feedback.
- Unit: the cooling mirror against the table; the star formation rate of a uniform box
  against the Schmidt law; the mean injected energy against its expectation.
- GPU `sn-remnant`: one heated cell in a uniform cooling medium; shell radius and retained
  momentum against the fits of Kim & Ostriker (2015).
- GPU `galaxy-ks`: the disk's star formation rate surface density against gas surface
  density lies within a factor of a few of the Kennicutt-Schmidt relation, and the disk
  reaches a steady rate instead of collapsing or blowing out.

## 7. Stage 4: live stars

### 7.1 Design

- **Storage.** Structure-of-arrays storage buffers: position, velocity, mass, and one word
  of id and formation time, 32 bytes a particle. Four million star particles (10^4 M_sun
  each for the reference disk) is 128 MB, the default limit on one buffer binding, so the
  positions and velocities go in separate buffers or the device asks for a higher
  `maxStorageBufferBindingSize`. Float32 positions resolve 0.005 pc in an 80 kpc box.
- **Deposit.** Cloud-in-cell scatter of mass into a fixed-point `atomic<u32>` grid, exact
  and order-free as the debris tally is, then a pass that adds it to the Poisson right-hand
  side (`gravity/rhs.wgsl`, `amr/gravity/rhs.wgsl`). The isolated boundary's multipole
  reduction must include it.
- **Finding a particle's block.** The deposit and the force gather need position → leaf
  slot on the GPU. To check: whether the renderer's lookup or `amr/grid.wgsl` already gives
  this, or whether an index texture of the tree must be added. Parents receive the mass by
  the restriction the gas density already uses.
- **Force.** Gather −∇Φ from the field texture with the same cloud-in-cell weights, so a
  particle does not pull on itself within a level. At refinement boundaries a small
  self-force remains, as in every mesh code of this kind; particles change level there
  without special treatment.
- **Time stepping.** Kick-drift-kick on the gas's step. First version: all particles take
  the finest substep, with the potential between solves extrapolated as the gas's is, and
  mass deposited whenever the potential is solved (the end of a cycle with per-block
  timesteps). Four million particles cost far less than the mesh, so particle time bins
  are not needed.
- **Regrid, boundaries.** Particles are untouched by a regrid. One that leaves the box is
  flagged dead and its mass goes to a ledger.
- **Stage 3 coupling.** Star formation writes new particles through an atomic slot counter
  instead of the ledger, restoring mass conservation, and feedback is delayed by the
  particle's age.

### 7.2 Initial conditions

An equilibrium disk and bulge in the static halo:

- built in: Hernquist's (1993) moment method (local Maxwellians from the Jeans equations,
  with the epicyclic approximation and a chosen Q profile), in a worker (`amr/init_worker.ts`
  shows the pattern). Simple, and good enough once the disk relaxes for an orbit;
- loaded: a particle set in GADGET-style HDF5 through the existing HDF5 reader, for models
  made by dedicated tools (AGAMA, GalIC).

The static Miyamoto-Nagai disk and bulge of stages 1 to 3 are replaced by the particles;
the halo stays analytic. Note for the presets: a live disk in a rigid halo forms a bar more
readily than in a live one.

### 7.3 Drawing and I/O

- Stars drawn as additive point splats in a pass composed with the gas view, by the
  off-canvas path screenshots and movies already use. Colored by age once stage 3 forms them.
- `.athdf` files gain particle datasets; restart reads them back. The API, the headless
  runner and the MCP tools report particle counts and totals.

### 7.4 Presets and validation

- **`galaxy`**: live stellar disk and bulge, gas disk, static halo; with and without the
  stage 3 physics. This is the flagship.
- Unit: cloud-in-cell weights sum to 1; the fixed-point deposit conserves mass exactly.
- GPU `particles-two-body`: two particles on a Kepler orbit several cells apart; period and
  energy drift.
- GPU `particles-plummer`: a Plummer sphere of particles stays in equilibrium (Lagrangian
  radii over ten crossing times), on the uniform grid and across refinement levels.
- GPU `particles-momentum`: gas and particles together conserve momentum to round-off of
  the deposit (a gas blob and a particle cluster orbiting one another).
- GPU `galaxy-live-hold`: the Q = 2 stellar disk keeps its surface density, scale height
  and dispersion profile for two orbits (measures heating by particle noise).
- GPU `galaxy-swing`: the Q = 1.2 disk grows arms whose amplitude and number are within
  the expected range for swing amplification (the ring diagnostics, extended to particles).

### 7.5 Risks

- **Scatter throughput.** Atomic adds from millions of particles into a few dense blocks
  contend. If the deposit is slow, sort particles by cell first (a GPU radix sort is its own
  piece of work) or deposit in several strided passes.
- **Noise.** 10^4 M_sun particles on 40 pc cells heat the disk and seed arms. `galaxy-live-hold`
  measures it; the remedy is more particles, which memory allows (16 million is 512 MB).
- **Self-force at level boundaries** in the thin disk, where the mesh is refined by height.
  Keeping the stellar disk's scale height inside the finest slab avoids most of it.
- **Per-block timesteps.** The first version's rule (particles on the finest substep,
  potential extrapolated) must be shown to conserve energy as well as the gas does; if it
  does not, runs with particles fall back to a single timestep.

## 8. Order of work and gates

1. Stage 1, axisymmetric potential (4.1), `galaxy-disk`, `galaxy-field`, `galaxy-hold`.
   **Gate A:** angular momentum drift at the planned resolution.
2. Stage 1, spiral term (4.2), `galaxy-shocks` in 2D then 3D. First public preset and a
   teaching-guide activity.
3. Stage 2 initial conditions and ring diagnostics, `galaxy-gas` on a small mesh, then the
   full mesh on a 24 GB card (a Runpod pod if the desktop is short of memory).
   **Gate B:** cost per orbit and memory per cell, which fix the resolution of stages 3 and 4.
4. Stage 3 cooling, then star formation and feedback, each with its tests.
5. Stage 4 in this order: particles on the uniform grid with a static mesh and a single
   timestep; blocks; per-block timesteps; initial conditions; drawing and I/O; coupling to
   stage 3. **Gate C** after the first step: deposit throughput at four million particles.

Stage 4's uniform-grid step does not depend on stage 3 and can start after gate B if live
stars matter more than the interstellar medium.

## 9. Open decisions

| Decision | Options | Leaning |
|---|---|---|
| Halo form | Logarithmic, Hernquist, NFW | Logarithmic: one parameter for a flat curve, exact difference form |
| Where the galaxy parameters live | `Params`, or a uniform of their own | Decide on reading `packParams` and the bind groups of the 13 callers |
| Rotation of the stage 2 disk | Analytic thin-disk form, or set from the grid's solved field | Analytic first; switch if `galaxy-hold` rings |
| Stage 3 before stage 4, or after | Either | Stage 4's first step in parallel with stage 3 |
| Mean molecular weight with cooling | Fixed, or the ionizing equation of state | Fixed |

## 10. References

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
