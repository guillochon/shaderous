# Plan: ideal MHD with constrained transport

Status: approved 2026-11-14; phase 1 (uniform grid) in progress.

Goal: ideal magnetohydrodynamics on the uniform grid and the adaptive mesh, with per-block
time bins from the first adaptive-mesh milestone, so that Shaderous can repeat the magnetized
tidal disruptions of [Guillochon & McCourt (2017)][GM17]. The magnetic field is kept
divergence-free to round-off by constrained transport (CT), and can be drawn in the volume
view as lit 3D tubes along field lines.

## 1. What we are reproducing

[Guillochon & McCourt (2017)][GM17] ran FLASH 4.2.2 with the unsplit staggered mesh solver of
[Lee (2013)][Lee13] (CT after [Gardiner & Stone 2008][GS08]):

- An n = 3/2 star of one solar mass and a 10^6 M_sun black hole, started at 10 r_t. Partial
  disruption at beta = 0.7 (a dipole at beta_M = 10^4; random fields at 10^4 and 10^5) and full
  disruption at beta = 1.5 (random field at 10^4, and a hydro control).
- beta_M = 8 pi P / B^2 at the star's center. Dipole axis along the orbital angular momentum.
  The random field has a Kolmogorov spectrum: A ~ k^(-17/6), B = curl A ~ k^(-11/3).
- B was set from a vector potential A with the staggered discretization of the curl, so the
  discrete divergence starts at zero.
- 100 cells across the star initially, up to 10^8 cells by the end.
- Their AMR kept div B at round-off across coarse/fine boundaries, but changing the mesh made
  defects (epsilon |div B| / B up to 5e-3) that the flux-conserving scheme then kept. They
  worked around it by refining everything above 1e-3 of the peak density and rarely
  derefining. We should do better by prolonging and restricting face fields
  divergence-free (section 6).

Results to check against: the surviving core's magnetic energy grows 13 to 20 times with no
dynamo; in the streams, B_perp falls steeply while B_parallel (along the stretching) barely
falls, so the stream's field straightens and its magnetic pressure reaches equipartition with
the gas.

## 2. Choices

### Integrator: van Leer + CT, not USM

USM (Lee 2013) is a corner-transport-upwind scheme: characteristic tracing with MHD
eigenvectors per cell, a transverse corrector, an intermediate Riemann solve to advance the
face-normal fields half a step, then the main Riemann solve. That is the most expensive
per-cell path we could pick on a GPU, and the eigenvector algebra would weigh on compile time,
which has been the bottleneck of recent work.

We use the van Leer predictor-corrector with CT of [Stone & Gardiner (2009)][SG09], as in
Athena++ (VL2):

1. **Half step.** First-order fluxes from the cell states (no reconstruction), a CT update of
   the face fields with first-order edge electric fields, giving U^{n+1/2} and b^{n+1/2}.
2. **Full step.** Limited reconstruction from U^{n+1/2}, Riemann solve, conservative update
   of the cell state from U^n, and a CT update of the face fields from b^n with the stage-2
   edge electric fields.

This fits the code: stage 1 replaces the MUSCL-Hancock predictor pass and writes the same
kind of half-step state (PA/PB plus the field), and stage 2 is the existing update pass with
more variables. The half-step state is also exactly what a faster block reads from a slower
neighbor part way through its step (section 5). Hydro-only runs keep MUSCL-Hancock unchanged;
the MHD integrator is selected only when the field is on.

Lee's upwind-biased electric fields (upwind-MEC) can be added later inside the same
framework if field-loop advection at small angles shows the plain construction is too
diffusive or oscillatory.

### Edge electric fields: Gardiner & Stone (2005)

The edge field is the average of the four face-centered electric fields around the edge
(from the Riemann fluxes, by the duality E_z = -F_y(B_x) = G_x(B_y)) plus gradient terms
upwinded by the sign of the mass flux through the faces, with the cell-centered reference
field E = -v x B. This is the "E_z^contact" construction of [Gardiner & Stone (2005)][GS05],
which is consistent for plane-parallel grid-aligned flow and stable for the field loop.
Stencil per edge: four face fluxes and the four cells around the edge.

### Riemann solver

- **HLLE** first: smallest, robust, enough for the CT machinery and most tests.
- **HLLD** ([Miyoshi & Kusano 2005][MK05]) as the default once its cost is measured. It
  resolves contact and Alfven waves, which matters for the stream's field.
- Next to flagged shocks, fall back to HLLE (as HLLC falls back to HLL now).

Measure both in a spike before committing to HLLD (phase 0).

### Divergence control

CT only. No cleaning variable. The discrete divergence of every cell, from its six face
fields, is preserved to float32 round-off by every operation: updates, coarse/fine matching,
time bins, regridding, initial conditions, the brush.

## 3. Data layout

### Face fields in the atlas

One rgba32float texture per state parity holds the face-centered field, aligned with the cell
atlas: texel (i, j, k) of a block holds (b_x at its lower x face, b_y at its lower y face, b_z
at its lower z face, spare). A block's upper faces are the lower faces of its ghost cells at
index BS, which the padded atlas already has room for. The spare channel can hold the
cell-centered B magnitude for the display, or stay free.

A face on a block boundary exists in both blocks' textures. Both copies must stay
bit-identical: they are updated by the same edge electric fields computed with the same
arithmetic, the way shared faces now get bit-identical fluxes (architecture.md, "Gathered
fluxes and bit-identical faces"). The validation suite checks this (section 9).

### Cell-centered field

B at the cell center is the average of the two face values along each axis. It is not
stored as a conserved variable: the primitives (assemble pass) get a third texture W3 =
(B_x, B_y, B_z, |B|^2 / 2 or spare), and the half-step state a third texture PC.

### Energy

Total energy E includes B^2 / 2. The dual-energy split becomes e_th = E - KE - B^2/2, with B
the cell-centered field after the face update. In a star beta_M ~ 10^4 to 10^6, so B^2/2 is
tiny next to E: the subtraction loses nothing in float32. In the streams beta_M falls toward 1,
where it matters, and the advected internal energy covers it as now.

### Memory

About +4 rgba32float atlas textures (face field x 2 parities, W3, PC): roughly 40 to 50% more
GPU memory per cell than a hydro run with gravity. The 512^3, 4-level TDE presets must be
re-checked against the memory budget (architecture.md, "GPU memory budget").

### Ghost face fields

Ghost faces are computed on load, like ghost cells:

- Same level: copy the neighbor's face fields.
- Coarser neighbor: prolong the coarse face fields to the fine ghost faces divergence-free
  (section 6, the same operator as regridding), so the ghost cells' divergence is zero too.
- Finer neighbor: restrict (area-average the four fine faces covering a coarse face).
- Domain boundaries: outflow copies tangential face fields and sets the boundary normal field
  from the zero-divergence condition in the ghost cell; periodic wraps; a reflecting wall is a
  perfect conductor (normal b = 0 on the wall, tangential mirrored).

## 4. Coarse/fine matching of edge electric fields

The same idea as the existing flux matching, one dimension down: a coarse edge that borders
finer blocks takes the average of the fine edge electric fields along it (two fine edges per
coarse edge in 3D, one in 2D), so the coarse and fine face fields on the shared faces stay
consistent and both sides keep zero divergence ([Balsara 2001][Bal01]; [Toth & Roe 2002][TR02]).

Two ways to get the fine edge fields to the coarse block:

1. **Recompute** (the current architecture): the coarse block recomputes the fine edge fields
   from the fine block's stored states and ghosts with the same arithmetic, as `match_faces`
   now recomputes fine fluxes. Each fine edge needs its four face fluxes, so a coarse/fine
   edge costs about eight Riemann solves.
2. **Edge register**: fine blocks write their boundary edge fields to a buffer that coarse
   blocks read in a later pass.

Start with (1); it keeps "nothing stored" and bit-identical results without ordering
constraints. Fall back to (2) only if profiling shows the recomputation dominates.

**Edges touch up to four blocks**, not two. Every block that shares an edge must use the same
time-integrated edge field. Rule: the edge field is defined by the finest blocks touching it,
averaged to coarser edges, and (section 5) summed over the substeps of the fastest block
touching it.

**Prerequisite: 2:1 balance across edges.** The tree (`src/amr/tree.ts`) enforces 2:1 balance
across faces only. Two blocks that share only an edge can now differ by two levels. CT needs
edge neighbors within one level too (and, for the averaging rule, corners). Extend the balance
rule and the block table with edge neighbors; check how many extra refinements it adds on the
TDE meshes.

## 5. Time bins from the start

Per-block time bins (numerics.md section 18) are part of the adaptive-mesh milestone, not a
later addition. The existing rules carry over, with edges added:

- **Bins.** A block's bin comes from its fast magnetosonic speed |v| + c_f instead of |v| + c_s.
- **Relaxation.** The rules that hold across faces (neighbors at most one bin apart; a block
  next to a faster one steps as fast; never slower than a coarser neighbor) must also hold
  across edges. Extend `bins_relax.wgsl` to edge neighbors.
- **Face fields across a bin boundary.** A face shared by a slow block S and a fast block F:
  F updates its copy every substep. S's copy must arrive at the same value at S's step end.
  So S's CT update on that face uses the electric fields of F's two substeps on the face's
  edges, recomputed from stored states, half at S's start and half at its midpoint, exactly as
  S now takes F's fluxes (`bins_on.wgsl`, the deposit passes).
- **Edges shared by blocks in different bins.** The time-integrated edge field over the
  slowest block's step is the sum over the fastest block's substeps. Every block touching the
  edge applies its share at its own substeps; blocks slower than the fastest take the sum in the
  deposit passes. This is where most of the new bookkeeping is: list, per block and substep,
  the boundary edges whose electric field comes from a faster neighbor.
- **Reading a slower neighbor mid-step.** A faster block reads the slower neighbor's half-step
  state, which with VL2 is the stage-1 state U^{n+1/2}, b^{n+1/2}: exactly what is needed.
  `mid_on.wgsl` extends to the face texture.
- **Invariant to test.** At the end of every cycle, all copies of every shared face agree
  bit for bit, and the divergence of every cell is unchanged from the cycle's start (to
  round-off).

Regridding happens at cycle ends, when all blocks are synchronized, so it does not interact
with the bins.

## 6. Regridding without divergence defects

This is where FLASH's runs picked up defects ([GM17]).

- **Restriction** (merging children): each coarse face takes the area average of the four fine
  faces covering it. Exactly divergence-preserving.
- **Prolongation** (new children), after [Toth & Roe (2002)][TR02]:
  1. Fine faces that lie on the coarse block's faces: if a neighbor at the fine level already
     exists there, **copy its face fields** (never interpolate where fine data exist: this is
     the usual source of defects). Otherwise interpolate the coarse face value linearly
     across the face with limited tangential slopes, keeping the four fine values' average
     equal to the coarse value.
  2. Fine faces inside the coarse cell: Toth & Roe's explicit formulas, which make each fine
     cell's divergence zero given its outer faces.
- **Energy.** The cell-centered field changes on regrid; keep each cell's internal and kinetic
  energy and rebuild E with the new B^2 / 2.
- **Test.** A field loop advected through a mesh that refines and coarsens around it: max |div
  B| dx / |B| stays at round-off through every regrid.

The same prolongation fills ghost faces next to coarser blocks (section 3), so it is written
once and used in both places.

## 7. Physics coupling

- **Gravity, point masses, frames.** Unchanged: they act on momentum and energy only. The
  induction equation is Galilean invariant, so the accelerating frame needs nothing.
- **Sinks.** Remove mass, momentum and energy as now; the field stays (flux cannot be removed
  without breaking CT). Note it in the docs.
- **Well-balanced reconstruction.** Applies to the gas pressure in stage 2 as now. Magnetic
  pressure is not part of the balance (beta_M >> 1 in stars).
- **Species, passive scalar, cooling, viscosity, conduction, damping zones.** Unchanged;
  check each in an MHD run. Viscosity and conduction stay isotropic (no Braginskii).
- **Dual energy, fluff limits, floors.** Subtract B^2/2 where they reconstruct e_th from E.
  The fluff speed cap must keep the field; the Alfven speed in tenuous gas sets the timestep,
  so add an optional cap on |B|^2 / rho there (problem setting, off by default).
- **Brush tools.** Mass and velocity brushes keep the face fields; a "field" brush (add a loop
  or a uniform field through a vector potential) can come later.

## 8. Initial fields

From a vector potential A on cell edges, b = curl A with the discrete curl on each face, so
div b = 0 exactly. To keep coarse and fine consistent on the adaptive mesh, use the line
integral of A along each edge (not a point value): a coarse edge's integral is then the sum of
its two fine edges', and restriction of b reproduces the coarse b exactly.

Configurations for the TDE presets:

- **Dipole** inside the star, axis along the orbital angular momentum: A_phi proportional to
  max(p - p_cut, 0) r sin(theta), which confines the field to the star, normalized to the
  requested beta_M at the center.
- **Random (Kolmogorov)**: a sum of N random Fourier modes of A (random phase, Gaussian
  amplitude, |A_k| ~ k^(-17/6)) between k_min ~ 2 pi / R_star and the grid scale, tapered to the
  star by the same pressure factor. Evaluated per edge on the GPU; no FFT needed on the
  adaptive mesh.
- **Uniform** and **loop** fields for tests.

The field is set after the star is settled (problems/settle.ts), on the settled density, and
does not change the balance at beta_M >= 10^4.

## 9. Validation

Every test runs on the GPU and against an MHD version of the CPU reference solver
(`src/reference/hydro.ts`) where practical.

Uniform grid:

- Brio & Wu shock tube along each axis and the diagonal (HLLE, HLLD).
- Circularly polarized Alfven wave: second-order convergence ([Toth 2000][Tot00]).
- Field loop advection, including a small advection angle ([GS05]).
- Orszag-Tang vortex; MHD rotor; MHD blast wave at low beta ([GS08]).
- div B: max |div b| dx / |b| below 1e-6 in every test.
- Bit-identical shared faces and edges (extends `flux-determinism`).

Adaptive mesh (static mesh first, then moving):

- The same tests with fixed refinement boundaries crossing the structures.
- Field loop through a refining and coarsening mesh (section 6).
- Time bins on: blast wave and field loop with `tbins` on match `tbins` off within tolerance;
  shared face copies agree at cycle ends; div B at round-off.
- Edge 2:1 balance holds after every regrid.

Physics:

- A magnetized polytrope at rest (dipole, beta_M = 10^4): stays still, magnetic energy steady.
- Magnetized TDEs (section 11) against [GM17].

## 10. Visualization: field lines as 3D tubes

### What the user sees

In the volume view, an overlay of lit tubes along magnetic field lines, depth-correct with the
volume rendering, colored by |B|, log beta_M or a chosen field, with the tube radius scaling
with |B| or constant. Controls: on/off, number of lines, seeding mode, color field, radius,
opacity. The slice view gets in-plane field-line segments drawn the same way, clipped to a
slab around the plane.

### Seeding that does not flicker

In ideal MHD, field lines move with the gas. Seeds are therefore advected with the flow (a
passive point set moved each frame with the local velocity) instead of being re-chosen every
frame, so the picture evolves smoothly and tracks the same field lines through the
disruption. Seeding modes:

- **Field strength**: sample seeds with probability proportional to |B|^2 dV (magnetic energy),
  restricted to gas the passive scalar marks (the star), at the start or on request.
- **Uniform** in a box or sphere the user places with the existing tool guides.
- **Reseed** button, and automatic replacement of seeds that leave the box or the star's gas.

### Tracing (compute)

A compute pass per frame (or every few frames, adjustable) traces each seed forward and
backward along the field with RK2 or RK4 and a step of about half the local cell size,
sampling the cell-centered field through the existing atlas lookup (`IDX` texture, as the
volume renderer does), up to a maximum length or until |B| falls below a threshold. It writes
polylines (position, |B|, color value) to a storage buffer with a fixed vertex budget per line
and an indirect draw count. 1,000 lines of 512 points is 0.5 M samples: negligible next to a
hydro step.

### Drawing (render)

- **Tubes as impostors**: each polyline segment is drawn as a camera-facing quad; the fragment
  shader computes the cylinder's normal and depth analytically (ray-cylinder intersection),
  shades it (Lambert plus specular, with the volume view's light), and writes depth. This is
  cheaper and smoother than tessellated cylinders, and joints are covered by sphere impostors
  at the vertices.
- **Depth compositing**: the volume view has no depth buffer today; field lines and block
  outlines are drawn on top. Draw the tubes first into a depth target, then pass that depth to
  the ray-marcher (`render/raymarch.wgsl`) so each ray stops at the tube surface and blends the
  volume in front of it over the tube. The block outlines can use the same depth.
- **Camera**: reuse the projection of `render/lines.wgsl` (the ray-marcher's camera), extended to
  output a proper depth.

### API

View keys for links, `window.shaderous`, the MCP `screenshot` tool and the runner: `tubes`,
`tubeLines`, `tubeSeed`, `tubeColor`, `tubeRadius`. New display fields: B_x, B_y, B_z, |B|,
magnetic energy density, log beta_M, and div B in units of |B| / dx. New slice variables:
`b1`, `b2`, `b3`. Diagnostics: total magnetic energy, max normalized div B, magnetic flux
through each domain face.

The tube renderer needs only a field to trace, so it can be built and tested on analytic fields
(a dipole, a loop) in parallel with phases 1 and 2.

## 11. Magnetized tidal disruption presets

Variants of the gamma = 5/3 presets, matched to [GM17]:

- Partial, beta = 0.7: dipole at beta_M = 10^4; random at 10^4 and 10^5.
- Full, beta = 1.5: random at beta_M = 10^4, with the existing hydro preset as control.

Checks: magnetic energy in the surviving core over time (growth by 13 to 20 times, then a slow
decline); in the streams, B_parallel and B_perp over time and the stream's beta_M approaching
1; div B at round-off throughout. Their equipartition time estimate is about
13 beta_M,6^(2/3) days for a solar-type star.

## 12. Phases

Each phase ends with its tests passing, the docs updated, and the unit tests run before
pushing (the Pages build runs them).

| Phase | Scope | Done when |
| --- | --- | --- |
| 0. Spikes | HLLE and HLLD in the update pass: compile time, registers, step time at 256^3. Tube impostor prototype on an analytic dipole. Edge 2:1 balance cost on TDE meshes. | Numbers recorded; HLLD go/no-go; layout of section 3 confirmed |
| 1. Uniform grid | Face texture, VL2 + CT integrator, GS05 edge fields, HLLE then HLLD, boundaries, dual energy with B, timestep with c_f, display fields, CPU reference. | Section 9 uniform-grid tests pass |
| 2. Adaptive mesh with time bins | Ghost faces on load, coarse/fine edge matching, edge 2:1 balance, edge neighbors in bin relaxation, deposit passes for edge fields, mid-step reads of the face texture. Static mesh. | AMR and time-bin tests pass on a fixed mesh; shared faces bit-identical |
| 3. Regridding | Divergence-free restriction and prolongation (also used for ghost faces), energy rebuild. | Field loop through a moving mesh at round-off div B |
| 4. Initial fields and coupling | Vector-potential initial conditions (dipole, Kolmogorov, uniform, loop) on the adaptive mesh; gravity, point masses, frames, sinks, species checked with B. | Magnetized polytrope at rest stays still |
| 5. Field-line tubes | Advected seeds, tracing pass, impostor tubes with depth, ray-marcher depth input, controls, API keys, slice-view lines. | Tubes on the field loop and the magnetized star, in the app and in headless screenshots |
| 6. Magnetized TDE presets | The section 11 variants, runs, comparison with GM17, docs. | Core amplification and stream straightening reproduced |

Phase 5 can start right after phase 0, since it only needs some field to trace.

## 13. Risks and open questions

- **Compile time and registers.** Eight conserved variables plus the face field, HLLD, and the
  edge-field stencil in one pass. The tiled flux path keeps fluxes in workgroup memory
  (`tf_a`, `tf_b`, `tf_d` in `amr/tile.wgsl`); three more components per face may not fit the
  16 KB guaranteed by WebGPU. Phase 0 measures it; splitting the CT update into its own pass is
  the fallback.
- **Edge bookkeeping with time bins** (section 5) is new: no reference code does CT with
  per-block bins that we know of. Budget most of phase 2 for it.
- **Edge 2:1 balance** adds refinement. Measure on the TDE meshes.
- **Float32.** CT preserves div B to float32 round-off (about 1e-7 relative), not the 1e-16 of
  FLASH's double precision. The tests use 1e-6.
- **Tenuous gas.** The Alfven speed in the ambient medium can set the timestep. The fluff
  limits may need a field cap, and the ambient medium should start unmagnetized.
- **Recompute vs register** for fine edge fields (section 4): decide from phase 2 profiling.

## References

[GM17]: https://scixplorer.org/abs/2017ApJ...834L..19G/abstract "Guillochon & McCourt 2017, ApJL 834, L19"
[Lee13]: https://scixplorer.org/abs/2013JCoPh.243..269L/abstract "Lee 2013, J. Comput. Phys. 243, 269"
[SG09]: https://doi.org/10.1016/j.newast.2008.06.003 "Stone & Gardiner 2009, New Astronomy 14, 139"
[GS05]: https://doi.org/10.1016/j.jcp.2004.11.016 "Gardiner & Stone 2005, J. Comput. Phys. 205, 509"
[GS08]: https://doi.org/10.1016/j.jcp.2007.12.017 "Gardiner & Stone 2008, J. Comput. Phys. 227, 4123"
[MK05]: https://doi.org/10.1016/j.jcp.2005.02.017 "Miyoshi & Kusano 2005, J. Comput. Phys. 208, 315"
[Bal01]: https://doi.org/10.1006/jcph.2001.6917 "Balsara 2001, J. Comput. Phys. 174, 614"
[TR02]: https://doi.org/10.1006/jcph.2002.7120 "Toth & Roe 2002, J. Comput. Phys. 180, 736"
[Tot00]: https://doi.org/10.1006/jcph.2000.6519 "Toth 2000, J. Comput. Phys. 161, 605"

- [Guillochon & McCourt (2017)][GM17]: magnetized tidal disruptions in FLASH.
- [Lee (2013)][Lee13]: the 3D unsplit staggered mesh scheme FLASH uses.
- [Stone & Gardiner (2009)][SG09]: the van Leer + CT integrator planned here.
- [Gardiner & Stone (2005)][GS05], [(2008)][GS08]: CT edge electric fields; 3D CTU + CT and test problems.
- [Miyoshi & Kusano (2005)][MK05]: the HLLD Riemann solver.
- [Balsara (2001)][Bal01], [Toth & Roe (2002)][TR02]: divergence-free prolongation and restriction.
- [Toth (2000)][Tot00]: the circularly polarized Alfven wave and other MHD tests.
