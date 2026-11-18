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
                         //   with the gas or 2 if with the masses, 1 if there are any, 1 if there is no Poisson potential)
  fluff: vec4<f32>,      // tenuous-gas limits: (density below which they apply, max specific
                         //   internal energy, max speed, max Alfven speed); a zero density
                         //   switches the first three off, a zero Alfven speed the last
  amr: vec4<u32>,        // block grids: (atlas slots per side of a layer, root blocks per side, max level, layers of slots)
  n: u32,                // cells per side along x and y (level 0 on block grids)
  grav_on: u32,          // the gravity field texture is in use: self-gravity, point masses or both
  grav_bc: u32,          // 0 periodic, 1 isolated
  flags: u32,            // bit 0: flux-form gravitational work; bit 1: the grid holds solid or fixed
                         //   cells; bit 2: there are sources or sinks (hydro/sources.wgsl); bit 3:
                         //   well-balanced pressure reconstruction (hydro/q.wgsl); bit 4:
                         //   block grids are refined fully around the point masses (amr/flag.wgsl);
                         //   bit 5: ideal MHD (mhd/mhd.ts)
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
  refine: vec4<f32>,     // block grids: (density below which gas does not ask for refinement, density
                         //   above which it is always refined to the finest level (0: none));
                         //   then a movie's soft stop (hydro/clock.wgsl; 1e30 for none), and
                         //   0 always: the zero that opaque() hides from the compiler
  eos2: vec4<f32>,       // more equation-of-state constants, per module (eos/degenerate.wgsl,
                         //   eos/ionizing.wgsl); zero for the others
  ohm: vec4<f32>,        // (magnetic diffusivity eta, 0, 0, 0): mhd/ohm.wgsl
  wallv: array<vec4<f32>, 6>, // each face's wall velocity (moving walls, hydro/boundary.wgsl):
                         //   (x lo, x hi, y lo, y hi, z lo, z hi)
  org: vec4<f32>,        // the domain's lower corner (src/domain.ts domainLo): cell c is centered
                         //   at (c + 0.5) dx + org
  rot: vec4<f32>,        // a rotating frame (gravity/rot.wgsl): its angular velocity, and 1 in w
                         //   for the centrifugal force; all zero for an inertial frame
  sym: vec4<u32>,        // per axis: 0, or 1 where the lower face is a plane of mirror symmetry
                         //   (SimConfig.symmetry; the gas sees it as reflecting)
}

// The state as the frame slot (gravity/frame_off.wgsl) hands it back.
struct FrameState { u1: vec4<f32>, u2: vec4<f32> }

struct Clock {
  dt: f32,     // timestep of the step being taken
  t: f32,      // time at the start of that step
  tc: f32,     // Kahan compensation for t
  steps: f32,
}

@group(0) @binding(0) var<uniform> P: Params;

// x itself, bit for bit, in a form the compiler cannot see through: XOR with a zero it does not
// know is zero (P.refine.w). WGSL lets a compiler reassociate floating-point arithmetic (WGSL
// 15.7.5), and Dawn compiles Metal shaders in relaxed math mode, which does. An expression whose
// exact value is zero, such as Kahan summation's compensation, is then simplified to zero; a
// value passed through here is a fresh unknown and its rounding is kept.
fn opaque(x: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(x) ^ bitcast<u32>(P.refine.w));
}
fn opaque3(x: vec3<f32>) -> vec3<f32> {
  return bitcast<vec3<f32>>(bitcast<vec3<u32>>(x) ^ vec3<u32>(bitcast<u32>(P.refine.w)));
}

// The lower face along `axis` is a plane of mirror symmetry, not a wall: what lies beyond it is
// the mirror image of what lies inside (the potential is even across it).
fn sym_lo(axis: i32) -> bool {
  return P.sym[axis] == 1u;
}

// A position as a fraction of the box from the domain's lower corner (0 to 1 inside it), and
// back. With the corner at -box/2 these are x / box + 0.5 and (f - 0.5) box, bit for bit.
fn box_frac(x: vec3<f32>) -> vec3<f32> {
  return x / P.box - P.org.xyz / P.box;
}
fn frac_pos(f: vec3<f32>) -> vec3<f32> {
  return (f + P.org.xyz / P.box) * P.box;
}

// Kahan summation: add x to s, whose rounding so far is c. Returns the new (s, c). Each rounded
// intermediate goes through opaque(), so that (t - s) - y is computed as written: otherwise it
// is algebraically zero, and on Metal it was (the compensation lost, a slow sink's G M grew by
// a seventh of what it took in).
fn kahan1(s: f32, c: f32, x: f32) -> vec2<f32> {
  let y = opaque(x - c);
  let t = opaque(s + y);
  return vec2<f32>(t, opaque(t - s) - y);
}
struct Kahan3 { s: vec3<f32>, c: vec3<f32> }
fn kahan3(s: vec3<f32>, c: vec3<f32>, x: vec3<f32>) -> Kahan3 {
  let y = opaque3(x - c);
  let t = opaque3(s + y);
  return Kahan3(t, opaque3(t - s) - y);
}

// The composition the equation of state is evaluated for: one everywhere, or per cell
// (species/mixture.ts).
#include <comp>

// Where group g of four species sits in a species atlas whose groups are `stride` texels apart
// (species/species.ts, speciesGroupOffset): stacked along z as far as 2048 texels, the deepest
// a 3D texture is sure to be, and then in further stacks along y, as many texels on. A 3D
// adaptive mesh's atlas is as tall as it is deep, so the stacks do not overlap.
fn species_group(g: i32, stride: i32) -> vec3<i32> {
  let gz = max(1, 2048 / stride);
  return vec3<i32>(0, (g / gz) * stride, (g % gz) * stride);
}

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

#include "hydro/boundary.wgsl"

// Maps an index along one axis into range: the cell (or, across the axis, the face) whose state
// stands for it (hydro/boundary.wgsl).
fn bc_axis(i: i32, n: i32, lo: u32, hi: u32) -> i32 {
  if (i < 0) { return bc_ghost_cell(i, n, lo); }
  if (i >= n) { return bc_ghost_cell(i, n, hi); }
  return i;
}

struct Mapped {
  c: vec3<i32>,       // in-range cell to load
  flip: vec3<f32>,    // signs of the velocity's components (-1 where a wall reverses them)
  side: vec3<f32>,    // -1 / +1 on axes where this is a ghost beyond an outflow-only face
  bflip: vec3<f32>,   // signs of the magnetic field's components
  vadd: vec3<f32>,    // added to the velocity after its signs: twice a moving wall's velocity
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
  var m = Mapped(c, vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(0.0));
  if (g_inner) { return m; }
  let n = vec3<i32>(i32(P.n), i32(P.n), i32(P.dims.z));
  // Most loads are of cells inside the grid: they skip the rest.
  if (all(c >= vec3<i32>(0)) && all(c < n)) { return m; }
  for (var a = 0; a < 3; a++) {
    if (c[a] >= 0 && c[a] < n[a]) { continue; }
    let up = c[a] >= n[a];
    let k = select(P.bc_lo[a], P.bc_hi[a], up);
    m.c[a] = bc_ghost_cell(c[a], n[a], k);
    if (bc_wall(k)) {
      m.flip *= bc_vsign(k, a);
      m.bflip *= bc_bsign(k, a);
      m.vadd += 2.0 * bc_wall_velocity(k, a, up);
    }
    m.side[a] = bc_side(k, up);
  }
  return m;
}

// Velocity of a (possibly ghost) cell: reversed as the walls say (and, beyond a moving wall,
// reflected about the wall's velocity: 2 v_wall - v), and at outflow-only faces
// the ghost's normal velocity is clamped to point outward, so nothing can flow in.
fn bc_velocity(v: vec3<f32>, m: Mapped) -> vec3<f32> {
  let w = v * m.flip + m.vadd;
  return select(w, max(w * m.side, vec3<f32>(0.0)) * m.side, m.side != vec3<f32>(0.0));
}
