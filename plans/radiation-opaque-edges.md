# Notes: the reduced speed of light at opaque edges

Status (2026-11-19): an experiment, finished and set aside. A change to the per-cell reduced
speed of light was built, tested and **dropped**: nothing in the code changed. These notes
record what was found, so that the next attempt starts from it. The dropped change is beside
this file as `radiation-opaque-edges.patch` (it applies to commit `5e3be89`).

Background: `docs/numerics.md` section 24. A cell's reduction factor is
`r = min(1, (chat / c) max(1, tau_c / tau_0))`, with `tau_c = kappa_R rho dx` its own optical
depth and `tau_0 = 2.5 / 3`. Opaque cells run at the true speed of light, transparent ones at
`chat`.

## 1. How this started

The question was whether the rule of He, Krumholz & Wibking (arXiv:2611.21353) is better than
ours. Theirs also picks the factor cell by cell, from the stability of an implicit-explicit
update. It returns to the true speed in cells more than about 3.5 mean free paths thick, where
ours needs `tau_c` of about `0.83 c / chat` (83 at a reduction of 100), but only where the gas
holds most of the heat and the opacity absorbs.

It is not a drop-in. Our face flux scales its dissipation by `min(1, 1 / tau_face)` (Skinner et
al. 2019), and their paper says that with that correction their condition is no longer a
stability bound.

A hybrid (the larger of the two factors, theirs normalized so that transparent cells keep our
Courant number) was tried in the float64 reference, `src/reference/radiation.ts`:

- It was stable wherever ours is: 594 uniform media and 648 layouts of thick beside thin cells.
- In a closed box at `c / chat = 100` it removes the error of the final temperature where the
  ratio of the radiation's heat capacity to the gas's is 0.01 or less and cells are 3.5 to about
  80 optical depths thick (4.2%, 1.5% and 0.39% with ours at 3.5, 10 and 30 optical depths;
  1.3%, 0 and 0 with the hybrid). At a ratio of 0.03 or more it changes nothing, and it never
  raised the factor where absorption was 1% of the opacity or none.

The hybrid was not carried to the GPU.

## 2. What the scans turned up: three weaknesses

All three were found in the float64 reference, with the gas at rest. The first was confirmed on
the GPU.

### 2.1 The per-cell rule at a sharp opaque edge (confirmed on the GPU)

An opaque cell beside a transparent one takes the true speed of light from the rule, but the
light it exchanges with its neighbor streams. The pair is unstable where the opaque cell's
radiation holds a fair share of its heat.

On the GPU: a 64^2 box in 2D, a slab of opaque gas across the middle, thin gas outside,
everything at one temperature with `E = a T^4`, 8 sweeps a step, `c / chat = 100`
(`radC = 1e7`, `radChat = 1e5`), and the gas made too heavy to move (slab density 1e12). The
radiation constant sets the slab's ratio `4 a T^3 / (rho c_v)`. Largest departure of `E` from
its start:

| Slab, optical depths a cell | Thin gas | Heat-capacity ratio | Per-cell rule | Uniform `chat / c` |
|---|---|---|---|---|
| 100 | 0.1 | 1 | past 50% by step 100 | 2e-6 at step 800 |
| 1e4 | 0.1 | 1 | past 50% by step 800 | 7e-5 |
| 1e4 | 0.001 | 1 | past 50% by step 800 | not run |
| 1e4 | 0.1 | 0.01 | past 50% by step 800 | 5e-4 |

In the reference, of 1008 layouts (slabs, alternating cells, pairs, ramps of a factor of 10 a
cell, random opacities; `c / chat` of 10, 100 and 1000; heat-capacity ratios of 1e-4, 0.01 and
1; absorption all or 1% of the opacity; 3000 sweeps), 318 were unstable with the per-cell rule
and 161 with a uniform reduced speed.

No shipped test or preset reaches this. `radiation-shadow` and `radiating-sphere` never raise a
cell above `chat / c`, and the GPU tests with opaque bodies have smoother edges or stay short
of it.

### 2.2 The face flux at a sharp opaque edge (reference only)

The 161 layouts that a uniform reduced speed fails are the transport's own. The dissipation of
a face takes the *mean* of its two cells' optical depths, so the face of an opaque cell on thin
gas is barely damped. With the *least* of the two, the reference has 7 unstable layouts with a
uniform speed.

### 2.3 Nearly transparent gas (reference only)

In uniform gas of 0.001 or 0.01 optical depths a cell, with a heat-capacity ratio of 1 and
`c / chat = 100`, noise of 1e-6 grows to 20 to 40% within 5000 sweeps and stays there, with a
uniform reduced speed and no walls. At 0.1 optical depths a cell it does not. First-order
reconstruction removes it, and so does a Courant number of 0.25 (0.4 does not). This points at
the sweep's single forward Euler stage with linear reconstruction. A second stage, as in He,
Wibking & Krumholz (2024), is the likely remedy. It was not tried.

## 3. The change that was built

Two candidates, alone and together.

- **Neighbor guard.** A cell's `r` from the least optical depth of the cell and its face
  neighbors.
- **Face minimum.** A face's dissipation from the least of its two cells' optical depths.

Unstable layouts of the 1008, in the reference:

| | Unstable |
|---|---|
| Per-cell rule as it is | 318 |
| Neighbor guard (one cell, or two: the same) | 161 |
| Face minimum alone | 247 |
| Both | 9 |
| Uniform reduced speed, flux as it is | 161 |
| Uniform reduced speed, face minimum | 7 |

Both together looked best in the reference and failed on the GPU:

- **Face minimum alone** blew up the slab of section 2.1 (departures of 1e6 and more).
- **Both together** held the slab, but in `rad-shadow`, once the gas moved, the radiation
  energy inside the lit clump reached 700 times the beam's by t = 0.05 and gas left its surface
  faster than `chat / 3` in 1866 cells. With the gas made too heavy to move it did not happen
  (largest `E` 3.7). Not understood. The 1D reference, with the same slab lit by a beam, shows
  nothing of it.
- **Neighbor guard alone** held the slab as a uniform speed does (1.8e-6, 5.2e-5, 7.3e-5,
  4.5e-4 for the four rows of section 2.1) and left `rad-shadow` as it was.

So the neighbor guard alone was the change carried through: it brings the per-cell rule level
with a uniform reduced speed at opaque edges, and no further. It touches eight passes (both
sweeps, both advect passes, the diagnostics and the views; on blocks the neighbors' opacity
comes from the assembled `RK` texture) and adds a validation test, `rad-edge`, for the slab.

With it, 17 of the 18 radiation tests passed unchanged in their criteria. The one that moved
was the supercritical shock, because one more layer of cells runs reduced:

| `chat` | Temperature behind, before | After | Shock position, before | After |
|---|---|---|---|---|
| c / 5 | 3.2% low | 4.1% low | 4.3% | 4.3% |
| c / 2.5 | 1.1% low | 1.5% low | 1.4% | 2.3% |
| c | 0.06% low | 0.06% low | 0.4% | 0.4% |

The full validation suite was not run to the end with it (64 tests in, none failed).

## 4. Why it was dropped

It cures a constructed case and was not shown to help a physical one.

**The two shipped presets** `radiation-shadow` and `radiating-sphere` give results identical to
the last digit with and without it: neither has a cell above `chat / c`.

**An opaque fireball.** `radiating-sphere` with its absorption raised from 50 to 5000: a hot
sphere of gas and radiation a thousand optical depths deep, in thin cold gas, with more heat in
the radiation than in the gas. The reference is the same run at the true speed of light
(`radChat = radC = 100`; the preset reduces it fivefold). Error against it at t = 0.3:

| Run | Rule | Light escaped | Radiation left on the grid |
|---|---|---|---|
| 2D, 256^2, absorption 5000 | as it is | -39% | +20% |
| | neighbor guard | -11% | +21% |
| 2D, 128^2, absorption 5000 | as it is | -34% | -1% |
| | neighbor guard | -6% | 0% |
| 3D, 128^3, absorption 5000 | as it is | -23% | +70% |
| | neighbor guard | +15% | +94% |
| 3D, 64^3, absorption 5000 | as it is | -21% | +12% |
| | neighbor guard | +26% | +41% |
| 2D, 256^2, absorption 50000 | as it is | -73% | 0% |
| | neighbor guard | -76% | 0% |
| 2D, 256^2, absorption 500 | as it is | +20% | +1400% |
| | neighbor guard | +31% | +1431% |

It helps the escaped light in 2D at the base opacity and nowhere else consistently, and the rule
as it is never went unstable: an expanding fireball does not sit in equilibrium long enough.

One result here is not about the change. Both per-cell rules are far better than a uniform
reduced speed inside the fireball, which left 8 to 38 times too much radiation on the grid in
the opaque runs and about 60% too much kinetic energy.

## 5. If this is picked up again

1. **A physical case that sits still.** The slab of section 2.1 is a static opaque body with a
   sharp surface. A planet or a dense clump held in a radiation field, in balance for many
   steps, is the physical run most like it, and the one that would show whether the neighbor
   guard earns its cost. It was not built.
2. **The lit clump with both changes.** Why does the face minimum with the neighbor guard fail
   in `rad-shadow` only when the gas moves? Until that is known the face flux cannot change, and
   the 161 layouts of section 2.2 stay.
3. **A second stage for the sweep** (section 2.3), which is also what would make the sweep
   second order in time.
4. **The hybrid with He, Krumholz & Wibking's rule** (section 1), on the GPU with moving gas,
   where the instabilities their paper warns of could appear.

To apply the dropped change: `git apply plans/radiation-opaque-edges.patch` on `5e3be89`. It
holds the neighbor guard and the `rad-edge` test, without the face minimum; the docs and paper
edits that went with it are not in it.
