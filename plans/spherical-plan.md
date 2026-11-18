# Spherical (r, θ) grids: implementation plan

Status: built on the uniform grid (2026-11-18): S1 as per-cell ratios of the cell's lengths to
the grid's dx (the predictor's slopes, the timestep) with the update's face weights carrying the
rest, compiled in only for spherical runs so every other run is the same bits; S2 and S3 with
uniform or logarithmic spacing in r, both poles, an inner sphere reflecting or open; S4 the
monopole and point masses at the origin; S6 diagnostics with true volumes and `.athdf` written
as `spherical_polar`. Tests `sph-rest`, `sph-conserve`, `sph-sedov`, `sph-polytrope`, `sph-bondi`
pass (docs/numerics.md section 22, "Spherical (r, θ) runs"). Not done: blocks (S1's mesh
refinement, S5), the Mignone slope weights (second order without them), the display mapped to
(R, z) (it draws the native (r, θ) plane), reading such files back, and gravity's work in flux
form (kick form is used: the flux form with the well-balanced reconstruction grew a polytrope's
fundamental mode). Plan as written 2026-11-17 follows; file references are to the commit that
added this file.

## What (R, z) settled, and what carries over

The cylindrical runs (`domain.geometry: "cylindrical"`) put a body of revolution on the 2D
grid with no change to storage or to the Riemann solvers: the layer is the azimuth, geometry
enters only as weights on the face fluxes and as one source term, and both cells of a face
still compute the same flux bits. In (R, z):

- `cyl_weights(i)` in [hydro/common.wgsl](../src/hydro/common.wgsl) gives the radial faces'
  areas over the cell's volume per Δ as index ratios; [hydro/update.wgsl](../src/hydro/update.wgsl)
  weighs the x faces with them (`cyl_face` in [hydro/flux.wgsl](../src/hydro/flux.wgsl), the
  angular momentum by the square) and adds the radial source with the faces' own pressures
  (`face_pressure`).
- The predictor's geometric terms come through `g_cyl_r` ([hydro/predictor.wgsl](../src/hydro/predictor.wgsl)).
- The axis is `P.sym.x = 2`: reflecting, with v_φ odd ([hydro/common.wgsl](../src/hydro/common.wgsl) `bc_map`).
- Volumes for the diagnostics and the point masses' reduction are rings; `.athdf` files are
  written in Athena++'s (R, φ, z) order ([io/athdf.ts](../src/io/athdf.ts) `swapAxes23`).

All of that generalizes to (r, θ), with one exception that drives this plan: **the cells cannot
all be the same size**. A uniform grid in r and θ has cells whose physical size grows as r; a
star or a blast wants logarithmic spacing in r. Every pass that divides by `g_dx` assumes one
size per cell for all axes.

## S1. Non-uniform spacing

Code axes: x is ln r (or r), y is θ, the layer is φ (`v_z` is v_φ).

- **Per-axis cell size.** `g_dx` becomes `g_h: vec3<f32>`, the physical length of the cell
  along each axis: `h_r = r_c Δ(ln r)` (logarithmic) or `Δr`, `h_θ = r_c Δθ`. The flux update
  divides each axis's net flux by its own `g_h[a]` ([advance.wgsl](../src/hydro/advance.wgsl)
  takes a vector `dtdx`), the predictor uses `0.5 dt / g_h[a]` per axis, and the timestep takes
  the smallest of `h_a / (|v_a| + c_s)` per cell, as the unsplit sum it is today
  ([rates.wgsl](../src/hydro/rates.wgsl) folds `rate / h` into the reduction, as the blocks'
  `PER_DX` already does).
- **Cell geometry from indices.** Everything is a function of (i, j) and four constants in
  Params (r_min, Δ ln r, θ_min, Δθ). On a logarithmic grid r_± = r_min e^{(i ± ½)Δ}: formed as
  `r_c * exp(±Δ/2)` with `exp(±Δ/2)` a Params constant, so faces stay exact ratios of their
  neighbors' and the two cells of a face agree on its radius to the bit.
- **Reconstruction.** The Cartesian parabola assumes equal spacing. On a logarithmic grid the
  ratio of neighboring sizes is constant (e^Δ), so the limiter's slopes take one fixed weight per
  axis (Mignone 2014 gives the exact ones); second order without them, as (R, z) is near the axis.
- **Mesh refinement** keeps the logarithmic spacing per level (Δ halved): restriction and
  prolongation are volume weighted, as (R, z) on blocks will need anyway (E4).

## S2. Areas, volumes and their cancellation

```math
A_r \propto r^2\,\Delta(\cos\theta), \qquad A_\theta \propto r_c\,\Delta r\,\sin\theta_\pm, \qquad
V \propto \tfrac13\Delta(r^3)\,\Delta(\cos\theta).
```

The differences cancel in float32 on a fine logarithmic grid (Δ ln r = 1e-3: r³ differs by 0.3%
across a cell). Closed forms that do not:

- `Δ(r³)/3 = r_c³ · sinh(3Δ/2) · (2/3)` for the logarithmic grid (`sinh` from its series below
  0.1, as `rot_sin` in [gravity/rot.wgsl](../src/gravity/rot.wgsl) does for the same reason).
- `Δ(cos θ) = 2 sin θ_c sin(Δθ/2)`, with `sin(Δθ/2)` a Params constant.
- The weights the update needs are again ratios: r_±²/(r_c² · 3 sinh(3Δ/2)/(... )) for the radial
  faces and sin θ_±/sin θ_c · (geometric factor) for the polar ones, each formed from those
  closed forms so that the weights of the two faces of a cell are exact enough that uniform
  pressure is held to rounding (the test below).

## S3. Sources and the poles

- Radial momentum: (2p + ρ(v_θ² + v_φ²))/r; polar: (cot θ (p + ρ v_φ²) − ρ v_r v_θ)/r. The
  pressure in each is the mean of the corresponding faces' pressures (`face_pressure`), the
  construction that made (R, z) well balanced: with it a uniform pressure is in balance exactly
  and a hydrostatic star is held as the Cartesian reconstruction holds it.
- Angular momentum about the axis (r sin θ ρ v_φ) is conserved by weighing the φ momentum flux
  through radial faces by (r_±/r_c)³-ish factors and through polar faces by (sin θ_±/sin θ_c)²:
  the general rule is the face's lever arm over the cell's, times the area weight.
- Both poles are axes: `P.sym.y = 2` at θ = 0 (lower y face) and a new value at θ = π (upper):
  v_θ and v_φ odd, the face area zero (its weight exactly 0, as the (R, z) axis's).
- The inner radial boundary at r_min > 0 is an ordinary face (reflecting, outflow, or a sink
  boundary for accretion).

## S4. Gravity

The cheap first step, and enough for stars and blasts: a monopole from the enclosed mass. A
reduction sums the mass of each radial shell (one atomic per shell row, the grid's x index), a
prefix sum in one invocation gives M(<r), and the field is −G M(<r)/r² along r. Self-gravity
beyond the monopole (a 2D Poisson solve in (r, θ)) is a later step, after the (R, z) multigrid of
coordinate plan E5 step 2, which it would share.

## S5. Timestep and per-block steps

The smallest cells are at r_min near the poles' neighbors (r_min Δθ), and the timestep is set
there. Per-block timesteps (already on blocks) are what make the outer grid cheap; on the
uniform grid, the logarithmic radial spacing keeps the cells roughly square (r Δθ ≈ r Δ ln r)
so the constraint is uniform in r.

## S6. Files, display, units

- `.athdf`: Coordinates `spherical_polar`, (r, θ, φ) with the face arrays non-uniform (x1f from
  the logarithmic grid); yt reads it as such.
- Display: the slice view draws the (r, θ) half-plane mapped to (R, z) = (r sin θ, r cos θ):
  the slice shader samples the field texture at (ln r, θ) of each pixel.

## Tests

| Test | Passes when |
|---|---|
| `sph-rest` | uniform gas at rest stays at rest to rounding, logarithmic grid, both poles |
| `sph-conserve` | a ring with swirl: mass, energy and angular momentum about the axis to rounding |
| `sph-sedov` | a point blast at the origin (r_min tiny, reflecting): shock radius within 1% of the similarity solution at every θ |
| `sph-polytrope` | a polytrope in its monopole field, settled: central density within 1% over 10 dynamical times |
| `sph-bondi` | Bondi accretion through r_min: the rate within 2% |
| `sph-off` | Cartesian and (R, z) runs unchanged bit for bit |

## Order

1. S1 per-axis cell size as a refactor that leaves every run bit-identical (`g_h` = (g_dx, g_dx,
   g_dx) everywhere else), with the whole suite as its test.
2. S2 and S3 on the uniform grid with uniform r spacing first (the closed forms are then exact),
   then logarithmic spacing.
3. S4 monopole gravity; S6 files and display.
4. Blocks, once (R, z) runs on blocks (coordinate plan E4).
