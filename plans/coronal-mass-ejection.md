# Plan: a coronal mass ejection preset

Status: draft, 2026-11-17. Nothing here is built yet.

## 1. Goal

A preset in which a magnetic flux rope sitting in the corona of a star erupts and carries a
bubble of magnetized gas out through the box: a coronal mass ejection (CME). It grew out of
playing with the "Field strength" brush on the surface of the magnetized polytrope, which
already raises loops that look like an active region. The preset should make the same thing
happen on purpose, reproducibly, and for a physical reason.

Four stages, each ending in something that runs, a test and a section of `docs/numerics.md`:

| Stage | Result | New machinery | Size |
|---|---|---|---|
| 1 | A star with a hot, hydrostatic, low-beta corona and a field that reaches out into it | Corona around a star, field outside a star, refinement on the field | Medium |
| 2 | "Coronal mass ejection": a flux rope that erupts by the torus instability | Flux-rope part (analytic vector potential), rope dye, CME diagnostics | Medium |
| 3 | "Active region": the same corona with a bare arcade, erupted by hand | A "Twist field" brush (the pull brush's machinery, rotating) | Small |
| 4 | Looking at it as an astronomer would | Coronagraph view (occulted column density, running difference) | Small |

Stage 2 is the preset the user asked for; stage 1 is what it stands on. Stages 3 and 4 are
worth doing but optional.

### Non-goals

A solar wind (the corona is static and bound), flux emergence from the convection zone,
thermal conduction along the field and radiative losses of the transition region, partial
ionization, energetic particles, and spherical coordinates. The CME leaves through the box's
outflow faces; following it to 1 AU is out of scope.

## 2. What exists and what is missing

Already in the code and used as is:

- Ideal MHD with constrained transport on the uniform grid and on blocks (§21), with
  self-gravity, the first-order flux correction next to near-vacuum (`mhd/face_flux.wgsl`),
  dual energy and floors, optional resistivity (`physics.resistivity`).
- Fields defined as vector potentials whose faces are the line integrals of A
  (`MhdSpec.potential`, `mhd/star_fields.ts`): any potential we can write in closed form goes
  in divergence-free to rounding, on both grids, with coarse faces the mean of fine.
- Star parts settled into the grid's own hydrostatic balance (`problems/settle.ts`, and on
  blocks `amr/settle.wgsl`), for any barotrope h(ρ) + φ = C.
- The Alfvén ceiling (`hydro/fluff.wgsl` `alfven_floor`, `Problem.fluff.va`), which protects
  the timestep from nearly empty magnetized cells.
- Passive scalar (dye) per part, and line-tied and moving walls (`problems/boundaries.ts`).
- The field brushes (`edit/field_brush_core.wgsl`): Field strength, and Pull field through
  the induction equation, divergence-preserving on both grids.
- Field-line tubes in the volume view; projections; movies; `.athdf` with the face field.

Missing:

| Gap | Stage | Notes |
|---|---|---|
| A corona: a hot, thin, hydrostatic envelope around a star | 1 | Today the star sits in a uniform background (ρ = 1e-4) that is not in balance and falls in |
| A field outside a star | 1, 2 | All four star fields are tapered to zero at the star's edge |
| Refinement on the field | 1, 2 | Only density, pressure, Jeans, dense gas, masses and composition refine; a rope in a thin corona refines nothing |
| A flux rope in equilibrium with a strapping field | 2 | New part with an analytic potential |
| Diagnostics of an eruption: apex height, speed, ejected mass and energy | 2 | The dye gives the rope's gas; the rest are reductions |
| Driving by footpoint motions | 3 | A brush, not a surface flow source |
| A coronagraph view | 4 | The projection exists; the occulter and running difference do not |

## 3. The physics we are aiming at

A CME is the loss of equilibrium of a twisted flux rope held down by an overlying ("strapping")
field. The rope's current I pushes it outward (the hoop force); the strapping field's
B_s × I holds it down. If B_s falls off with height h faster than a critical rate,

    n(h) = -d ln B_s / d ln h  >  n_crit ≈ 1.5,

a small upward displacement weakens the restoring force more than the hoop force, and the rope
accelerates away: the torus instability (Kliem & Török 2006). Below the threshold it oscillates
about its equilibrium. The rope's twist also matters: above about 3.5π (Hood & Priest 1981;
Török et al. 2004) it kinks, which can start the rise but rarely ejects it alone.

That gives the preset a single, clean control (where the rope sits relative to the decay of
its strapping field) and the test a sharp pass/fail: a rope below threshold stays, one above
erupts.

The standard numerical setup is the Titov & Démoulin (1999) rope and its modified form
(Titov et al. 2014): a toroidal current ring of major radius R and minor radius a, its center
a depth d below the surface (so only an arch of it is above), held by subsurface sources.
Here:

- **Ring.** The vector potential of a circular current loop is closed form (complete elliptic
  integrals K and E). Inside the minor radius it is smoothed to a uniform current, as Titov et
  al. do. The ring's buried part runs through the dense star, where it does nothing: the
  star's inertia is what line-ties the footpoints, as the photosphere does on the Sun.
- **Axial field (twist).** A field along the ring inside the minor radius gives the rope its
  twist; the toroidal flux sets the number of turns. Default about 2.5π end to end, below
  the kink threshold, so the eruption is the torus instability's alone.
- **Strapping field.** Instead of TD99's pair of magnetic charges (a monopole pair needs a
  Dirac string through the star carrying their flux, singular on the grid), a buried
  horizontal dipole of moment m along the rope's axis, a depth D below the surface under the
  rope. Above it the field is a bipolar arcade straddling the rope, falling off as
  (h + D)^-3 straight up, so n(h) = 3h / (h + D) and the threshold n = 1.5 is at h = D. With
  the rope's apex at height h_a = R − d: h_a < D holds, h_a > D erupts. The dipole's
  potential A = m × (x − x_D) / |x − x_D|^3 has no string. Its singularity is at depth D in the
  star; it is smoothed inside a radius well inside the star, where it does not matter.
- **Equilibrium current.** Shafranov's condition sets I so the ring is in balance at t = 0:
  B_s(apex) = I / (4π R) · (ln(8R/a) − 3/2 + l_i/2), l_i = 1/2 for a uniform current. In
  the code's units B²/2 is the magnetic pressure (Heaviside-Lorentz, μ0 = 1), which is the
  form above. The rope is then close to, not exactly, force-free; it relaxes a little in the
  first Alfvén time, which the below-threshold run shows is harmless.
- **Global field (optional).** A dipole of the star, so the CME has to open the overlying
  field: A_φ = m sinθ / r² outside a radius r0 < R_star, and m r sinθ / r0³ inside (the
  uniformly magnetized sphere, uniform inside, a dipole outside). Its own decay index above
  the surface is 3h / (h + R_star), so it adds strapping that weakens slowly with height.

Below the rope, the stretched legs of the strapping field form a current sheet where they
reconnect (the flare). With no resistivity it reconnects at the grid scale; a small explicit
η (§21, `physics.resistivity`) makes the rate resolution independent and the energy go to
heat instead of the floors. Worth trying both.

The breakout model (Antiochos et al. 1999), a sheared arcade in a quadrupolar field driven by
footpoint shear, is the other classic. It needs sustained surface driving and long runs, so it
is left to the interactive stage 3 rather than a preset of its own.

## 4. Stage 1: a star with a corona

### Corona

An isothermal envelope of sound speed c_c in hydrostatic balance with the star's pull:

    ρ(r) = ρ_b exp( (GM / c_c²) (1/r − 1/R_b) ),   p = ρ c_c²,

with ρ_b the density at its base R_b. Its density tends to a finite value far out, so it is
not a true equilibrium to infinity (that is what drives Parker's wind), but across the box it
is, and the static corona is the point (no wind, section 1). The star's polytrope is
cut where its pressure falls to the corona's base pressure p_b = ρ_b c_c²: there the pressure
is continuous and the density jumps from the star's surface value down to ρ_b, a transition
region, a contact discontinuity held in balance.

Implementation: a corona option on the star part, `corona: { rho, cs }`, which makes the
star's gas a composite barotrope (polytrope above the transition pressure, isothermal below),
so settling (`problems/settle.ts`, `amr/settle.wgsl`) balances star and corona together in the
grid's potential, with no new machinery. The density jump makes ρ(h) multivalued at one h;
settle in pressure there (p as a function of h is single valued), or spread the jump over two
or three cells with a steep but finite segment of the barotrope. Decide by which keeps the
settled transition region still to the test's tolerance. The background part of the setup is
then unused (or the corona's density at the box corner).

The gas keeps the run's γ. An isothermal envelope with γ > 1 is convectively stable; the
star is stable if γ ≥ 1 + 1/n. Default: n = 3 (centrally concentrated, so the surface density
is already low), γ = 5/3.

### Numbers

In units of the star (G = 1, M and R_star):

| Quantity | Default | Why |
|---|---|---|
| R_star | 0.15 of the box | Room for the CME to travel 3 to 4 star radii |
| Star center | (0.5, 0.5, 0.3) | The active region at the north pole faces the box's center and its far face |
| c_c² | 0.2 GM / R_star | Scale height 0.2 R_star at the base: the density falls by about 50 by the box's top. The Sun's is 0.1 |
| ρ_b | 10^-6 ρ_c | Contrast with the star's surface about 10^2 to 10^3 |
| Corona beta (strapping field at the base) | 0.1 | A low-beta corona, as the Sun's |
| Alfvén speed at the base | ≈ 4.5 c_c ≈ 2 v_K | v_A = sqrt(2/β) c_c; the Sun's active regions are similar relative to their escape speed |

Rough cost: the Alfvén speed in the rope, a few times the corona's, sets the timestep; the
eruption takes a few Alfvén times to cross the box. That is a few thousand steps at 128³ with
two or three levels. The Alfvén ceiling should sit above the rope's Alfvén speed, so it only
catches cells that empty, and the run should report how much mass it added.

### Boundaries

Outflow on all faces, so the CME can leave. The corona at the box's faces is held only by
its own gradient across the ghost cells, so it drains slowly; the test measures how much over
the eruption's duration. If too much, a hydrostatic outflow boundary (ghost cells extrapolated
along the isothermal profile in the star's potential, open to outflow but not drawing gas in)
is the fix, and is better than a damping zone, which would also damp the CME.

### Field outside the star

- A new star field type `dipole-global` (the uniformly magnetized sphere above, with r0 = 0.8
  R_star), set by the plasma beta at the corona's base at the pole.
- Potentials from several parts summed (the star's own field, the rope, the strapping dipole).
  Check whether `MhdSpec.potential` already composes parts; add it if not.

### Refinement on the field

The rope lives in thin gas, where χ on density and pressure is small and thin gas never asks
for refinement. Add a criterion on the field: Löhner's χ applied to |B| (or to B²), exempt from
the thin-gas cutoff, and a cap on the current, refine where |J| Δx / |B| > 0.3 (the rope's
edge, the flare current sheet). Kept off unless a preset sets it (`mesh.refineField`).

### Test: `cme-corona`

The star with its corona and the global dipole, no rope, 64³ and on blocks: for five sound
crossings of the corona's scale height, the largest speed in the corona stays below a few
percent of c_c, the density at the base and at 3 R_star within a few percent, the magnetic
energy within 1%. With the field off, the same for the hydro run.

## 5. Stage 2: the flux rope and the preset

### Part

```jsonc
{
  "kind": "fluxrope",
  "id": "rope",
  "on": "star",                // the star it sits on, by id
  "lat": 90, "lon": 0,         // where on it, in degrees (90: the pole, facing +z)
  "pa": 0,                     // the rope axis' angle on the surface, in degrees
  "R": 0.35,                   // major radius, in star radii
  "a": 0.08,                   // minor radius, in star radii
  "d": 0.15,                   // depth of the ring's center below the surface, in star radii
  "twist": 2.5,                // turns of the field end to end, in units of pi
  "strap": { "D": 0.15, "beta": 0.1 }, // strapping dipole: depth, and beta at the base under the rope
  "dye": 1                     // passive scalar of the gas inside the rope
}
```

The apex height is R − d = 0.2 here, above D = 0.15: unstable. The current I comes from
Shafranov's condition, so the user does not set it. The potential is written in the frame of
the active region (z along the local vertical, x along the rope's axis) and rotated into the
box; the part supplies it to the field's line integrals like the star fields. The gas inside
the rope's minor radius takes the dye, and optionally a density a few times the corona's (a
prominence-like load) with the pressure unchanged.

A 2.5D check comes cheaply: the same ring with its axis along y in a 2D box is the Lin &
Forbes (2000) line-current problem, which is a good first test of the vector potential and
the current sheet.

### Preset `coronal-mass-ejection`

Extends `polytrope` (n = 3, γ = 5/3) with the corona, the global dipole (weak, β = 1 at the
base), the rope at the pole, refinement on the field, 128³ with 3 levels by default, the view
on field lines colored by |B| with the density volume behind, and a tilted camera. Run time:
until the front reaches the top face.

Featured controls:

| Control | Path | Effect |
|---|---|---|
| Rope height (apex over strapping depth) | `parts.1.d` via h_a / D | Below 1 it holds, above it erupts: the torus instability's threshold |
| Twist (π) | `parts.1.twist` | Above about 3.5π the rope kinks as it rises, and rotates |
| Overlying field (β) | star field `beta` | A stronger global dipole slows or stops the CME (a failed eruption) |
| Prominence load | `parts.1.rhoRope` | Heavier ropes rise later and slower |
| Corona temperature | `corona.cs` | Scale height, and how much the CME sweeps up |

Variants: "Confined (stable)" with h_a / D = 0.7, a rope that oscillates and stays; "Failed
eruption" with a strong global dipole, which starts and stops; "Kinking" with twist 4.5π.

### Diagnostics

Reported in the diagnostics panel for a run with a rope (`Problem.cme`):

- apex height and its speed: the highest point where dye > 0.5 above the active region,
  along the local vertical;
- ejected mass: dye mass, and all mass, beyond 2 R_star moving outward;
- kinetic energy of the outward-moving gas, and the magnetic energy's change from the start
  (the energy that drives it);
- the decay index of the initial strapping field at the apex (computed on the CPU from the
  potential, shown on the control).

### Tests

- Unit (vitest): the ring's potential against a direct Biot-Savart sum at points around it;
  its curl's divergence (zero by construction, but the rotation into the box is a place to
  get a sign wrong); the net Lorentz force on the rope at t = 0 from the potential by
  finite differences, below 5% of the hoop force alone.
- GPU, `cme-torus`, on both grids at 64³: h_a / D = 0.7 keeps the apex within 10% of its
  height for five Alfvén times; h_a / D = 1.5 lifts it beyond 2 R_star, with a phase of
  exponential rise followed by a nearly constant speed (Török & Kliem 2005). ∇·B at rounding
  throughout; the Alfvén ceiling's added mass below 1% of the ejected mass.
- The preset's own smoke test with the others.

## 6. Stage 3: an active region to erupt by hand

Preset variant "Active region": the corona, the global dipole and the strapping arcade, no rope.
The user builds one with the brushes. Today's Pull field shears footpoints if dragged along the
polarity inversion line, but it drags one side at a time. A "Twist field" brush makes the
classic driver directly: the displacement of a rotation about the brush's axis (the view
direction, or the local vertical when the brush is on a star's surface),

    v = g w (ω × (x − x0)) Δt,

through the same induction-equation step as Pull (donor-cell constrained transport, substeps
of half a finest cell, the grip). Twisting the footpoints of an arcade on either side of the
inversion line in opposite senses builds a twisted rope, and enough twist kinks or erupts it.
It is a new kind in `field_brush_core.wgsl` (the drag replaced by a rotation), with the pull's
tests (energy bookkeeping, shared faces, ∇·B) repeated.

## 7. Stage 4: a coronagraph view

The CME as LASCO sees it: the column density along the line of sight (Thomson scattering is
proportional to n_e with a weight in the angle that can be left out at first), the star and a
ring around it masked by an occulting disk, and the running difference of successive frames,
which shows the classic three-part structure (a bright front, a dark cavity, a bright core
where the prominence load is). The projection view exists; the occulter is a mask in its
fragment shader, and the running difference keeps the previous frame's texture.

## 8. Risks

- **Low beta.** In the rope and under it β reaches 10^-2 or less; the pressure from E − KE −
  B²/2 loses its digits. Dual energy (§7, §21) handles it; watch the floored-cell count.
- **The transition region.** The density jump at the star's surface, with field crossing it,
  is where the first-order flux correction was needed for the tidal disruptions. Expect to
  look at it first if the corona will not hold.
- **The flare current sheet** thins to the grid; it is where the numerical resistivity acts.
  Explicit η gives control, at the price of a smaller diffusive timestep.
- **Draining corona** through the outflow faces (section 4, Boundaries).
- **Rope resolution.** The minor radius should span at least 8 finest cells: 0.08 R_star =
  0.012 of the box is 1.5 cells at 128³, 6 at 512 (two more levels). Refinement on the field
  is not optional.
- **Off-center star and isolated gravity.** The multipole expansion is about the center of
  mass; a star at z = 0.3 is fine for the expansion's order, but check that it does not drift.

## 9. References

- Antiochos, DeVore & Klimchuk 1999, ApJ 510, 485: the breakout model.
- Chen 2011, Living Rev. Solar Phys. 8, 1: review of CME observations and theory.
- Fan & Gibson 2007, ApJ 668, 1232: emergence of a twisted rope into an arcade, torus and kink.
- Hood & Priest 1981, Geophys. Astrophys. Fluid Dyn. 17, 297: the kink threshold.
- Kliem & Török 2006, Phys. Rev. Lett. 96, 255002: the torus instability.
- Lin & Forbes 2000, J. Geophys. Res. 105, 2375: the 2D catastrophe model with a current sheet.
- Titov & Démoulin 1999, A&A 351, 707: the flux rope equilibrium.
- Titov et al. 2014, ApJ 790, 163: its modified form, with closed-form potentials.
- Török & Kliem 2005, ApJ 630, L97: eruption of the TD rope, exponential then linear rise.
- Török, Kliem & Titov 2004, A&A 413, L27: kink of the TD rope.
