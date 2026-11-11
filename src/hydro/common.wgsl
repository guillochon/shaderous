// Shared parameter block and boundary-condition index mapping.
// Layout must match packParams() in src/sim.ts.

struct Params {
  bc_lo: vec4<u32>,      // per-axis lower-face boundary: 0 periodic, 1 outflow, 2 reflecting, 3 outflow only
  bc_hi: vec4<u32>,      // per-axis upper-face boundary
  eos: vec4<f32>,        // (gamma, K, gas constant R, radiation constant a)
  dims: vec4<u32>,       // cells along x, y, z; z is 1 in 2D runs (one periodic layer)
  geom: vec4<f32>,       // (cell volume, number of dimensions, refinement threshold, cells per
                         //   Jeans length below which a block is refined); 2D is per unit length in z
  gext: vec4<f32>,       // uniform external gravitational acceleration
  bh: vec4<f32>,         // point masses (gravity/bh.wgsl): (how many, 1 if the frame accelerates
                         //   with the gas, 1 if there are any, 1 if there is no Poisson potential)
  fluff: vec4<f32>,      // tenuous-gas limits: (density below which they apply, max specific
                         //   internal energy, max speed, 0); a zero density switches them off
  amr: vec4<u32>,        // block grids: (atlas slots per side of a layer, root blocks per side, max level, layers of slots)
  n: u32,                // cells per side along x and y (level 0 on block grids)
  grav_on: u32,          // the gravity field texture is in use: self-gravity, point masses or both
  grav_bc: u32,          // 0 periodic, 1 isolated
  flags: u32,            // bit 0: flux-form gravitational work; bit 1: the grid holds solid or fixed
                         //   cells; bit 2: there are sources or sinks (hydro/sources.wgsl); bit 3:
                         //   well-balanced pressure reconstruction (hydro/q.wgsl)
  dx: f32,               // cell size (level 0 on block grids)
  cfl: f32,
  G: f32,
  rho_mean: f32,
  rho_floor: f32,
  p_floor: f32,
  eta_dual: f32,         // dual-energy switch: use E - KE when it exceeds eta * E
  t_end: f32,
  dt_max: f32,
  shock_thresh: f32,     // pressure-jump threshold of the shock detector
  jeans_floor: f32,      // 1 enforces the Truelove pressure floor, 0 disables it
  box: f32,              // box side length
  cool: vec4<f32>,       // parameters of the cooling function (cooling/*.wgsl)
  diff: vec4<f32>,       // diffusion (hydro/diffusion.wgsl): (kinematic viscosity, conductivity
                         //   per unit density), then the damping zone's inner radius and rate
                         //   (hydro/damping.wgsl)
}

struct Clock {
  dt: f32,     // timestep of the step being taken
  t: f32,      // time at the start of that step
  tc: f32,     // Kahan compensation for t
  steps: f32,
}

@group(0) @binding(0) var<uniform> P: Params;

// Obstacles. The fourth component of the second state texture marks a cell: 1 is solid (a wall
// the gas reflects off), -1 is fixed (gas held at its state, as an inflow), 0 is ordinary gas.
// Neither kind is ever updated. Passes only look when this says there is something to find.
fn has_solids() -> bool {
  return (P.flags & 2u) != 0u;
}
// Self-gravity: there is a Poisson potential. (The field texture may hold point masses alone.)
fn has_potential() -> bool {
  return P.grav_on != 0u && P.bh.w == 0.0;
}
fn has_sources() -> bool {
  return (P.flags & 4u) != 0u;
}
// Well-balanced pressure reconstruction (hydro/q.wgsl) is on.
fn well_balanced() -> bool {
  return (P.flags & 8u) != 0u;
}
// For it: potential differences across the faces on one side of a cell, per axis, as the
// deviations from balance see them (`seen`: none across a wall or an outflow face) and for
// the increment to the face (`face`: the true difference across a wall).
struct PhiSteps { seen: vec3<f32>, face: vec3<f32> }

// Size of the cell this invocation is working on. Every entry point sets it first: it is P.dx
// on a uniform grid and depends on the block's level on an adaptive one.
var<private> g_dx: f32;
// Length of the step this invocation is taking. Entry points that advance the state set it:
// the clock's dt, or a multiple of it for a block of an adaptive mesh that steps less often.
var<private> g_dt: f32;

// Maps an index along one axis into range. Returns (index, 1 if mirrored by a reflecting face,
// side): side is -1 / +1 for a ghost cell beyond an outflow-only lower / upper face, else 0.
fn bc_axis(i: i32, n: i32, lo: u32, hi: u32) -> vec3<i32> {
  if (i < 0) {
    // The stencil reaches two cells out; the clamp covers a single-layer axis (n = 1) without
    // an integer modulo, which is slow and would be evaluated on every load.
    if (lo == 0u) { return vec3<i32>(max(i + n, 0), 0, 0); }
    if (lo == 1u) { return vec3<i32>(0, 0, 0); }
    if (lo == 3u) { return vec3<i32>(0, 0, -1); }
    return vec3<i32>(-1 - i, 1, 0);
  }
  if (i >= n) {
    if (hi == 0u) { return vec3<i32>(min(i - n, n - 1), 0, 0); }
    if (hi == 1u) { return vec3<i32>(n - 1, 0, 0); }
    if (hi == 3u) { return vec3<i32>(n - 1, 0, 1); }
    return vec3<i32>(2 * n - 1 - i, 1, 0);
  }
  return vec3<i32>(i, 0, 0);
}

struct Mapped {
  c: vec3<i32>,       // in-range cell to load
  flip: vec3<f32>,    // -1 on velocity components mirrored by a reflecting face
  side: vec3<f32>,    // -1 / +1 on axes where this is a ghost beyond an outflow-only face
}

// Set by a pass for a cell whose whole stencil lies inside the grid: its loads skip the
// boundary mapping, which otherwise runs for every one of them.
var<private> g_inner: bool = false;

// True if every cell within `reach` of c along the axes in use is inside the grid.
fn stencil_inside(c: vec3<i32>, reach: i32) -> bool {
  let n = vec3<i32>(P.dims.xyz);
  let ok = (c >= vec3<i32>(reach)) & (c < n - reach);
  return ok.x && ok.y && (ok.z || n.z == 1);
}

// Every state load goes through this, so there are no ghost cells to fill.
fn bc_map(c: vec3<i32>) -> Mapped {
  if (g_inner) { return Mapped(c, vec3<f32>(1.0), vec3<f32>(0.0)); }
  let n = i32(P.n);
  let x = bc_axis(c.x, n, P.bc_lo.x, P.bc_hi.x);
  let y = bc_axis(c.y, n, P.bc_lo.y, P.bc_hi.y);
  let z = bc_axis(c.z, i32(P.dims.z), P.bc_lo.z, P.bc_hi.z);
  return Mapped(
    vec3<i32>(x.x, y.x, z.x),
    vec3<f32>(1.0 - 2.0 * f32(x.y), 1.0 - 2.0 * f32(y.y), 1.0 - 2.0 * f32(z.y)),
    vec3<f32>(f32(x.z), f32(y.z), f32(z.z)),
  );
}

// Velocity of a (possibly ghost) cell: mirrored at reflecting faces, and at outflow-only faces
// the ghost's normal velocity is clamped to point outward, so nothing can flow in.
fn bc_velocity(v: vec3<f32>, m: Mapped) -> vec3<f32> {
  let w = v * m.flip;
  return select(w, max(w * m.side, vec3<f32>(0.0)) * m.side, m.side != vec3<f32>(0.0));
}
