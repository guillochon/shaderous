// Shared parameter block and boundary-condition index mapping.
// Layout must match packParams() in src/sim.ts.

struct Params {
  bc_lo: vec4<u32>,      // per-axis lower-face boundary: 0 periodic, 1 outflow, 2 reflecting, 3 outflow only
  bc_hi: vec4<u32>,      // per-axis upper-face boundary
  eos: vec4<f32>,        // (gamma, K, gas constant R, radiation constant a)
  dims: vec4<u32>,       // cells along x, y, z; z is 1 in 2D runs (one periodic layer); w: 1 on a rectangular uniform grid (src/domain.ts gridDims)
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
                         //   (SimConfig.symmetry; the gas sees it as reflecting), 2 where it is
                         //   the axis of an axisymmetric (R, z) run (x only: Problem.geometry)
  cosmo: vec4<f32>,      // comoving coordinates (src/cosmo.ts): (Omega_m, Omega_k, Omega_Lambda, 1);
                         //   all zero otherwise
  shear: vec4<f32>,      // shearing box (Problem.shear): (Omega, q, 1 if stratified, 1); all zero
                         //   otherwise
  sph: vec4<f32>,        // a spherical (r, theta) run (Problem.geometry): (r_min / dx, the step in
                         //   ln r on a logarithmic grid or 0 on a uniform one, the step in theta, 1);
                         //   all zero otherwise
  sph2: vec4<f32>,       // its logarithmic grid's constants, in double precision on the CPU:
                         //   (e^(D/2), 3 / (2 sinh(3D/2)), sinh D, 2 sinh(D/2)), D the step in ln r
}

// The state as the frame slot (gravity/frame_off.wgsl) hands it back.
struct FrameState { u1: vec4<f32>, u2: vec4<f32> }

struct Clock {
  dt: f32,     // timestep of the step being taken
  t: f32,      // time at the start of that step
  tc: f32,     // Kahan compensation for t
  steps: f32,
  a: f32,      // comoving coordinates (src/cosmo.ts): the scale factor at t, at t + dt/2 and at
  ah: f32,     //   t + dt, and the expansion rate at t + dt/2 (hydro/clock.wgsl); 1, 1, 1 and
  a1: f32,     //   0 otherwise
  hub: f32,
  shear: f32,  // shearing box: the offset of the x faces along y, and its Kahan compensation
  shear_c: f32,
}

@group(0) @binding(0) var<uniform> P: Params;

// The equation of state's pressure may be zero or negative (eos/liquid.wgsl): then no pass may
// floor a pressure, divide by one or measure a jump against one. Marked by a negative floor.
fn p_signed() -> bool {
  return P.p_floor < 0.0;
}

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

// Comoving coordinates (src/cosmo.ts): every gradient carries 1/a, which the passes take up by
// working with the proper cell size a dx; positions keep the comoving one.
fn cosmo_on() -> bool {
  return P.cosmo.w != 0.0;
}
fn cell_dx(dx: f32, a: f32) -> f32 {
  return select(dx, dx * a, cosmo_on());
}
// The expansion rate H(a) = da/dt / a, in units of H0.
fn cosmo_hubble(a: f32) -> f32 {
  return sqrt(P.cosmo.x / (a * a * a) + P.cosmo.y / (a * a) + P.cosmo.z);
}
// a^3, the ratio of the state's comoving densities and pressures to the proper ones that the
// equation of state and cooling take. A pass that holds the clock sets g_a3 to the scale
// factor of the states it works on (the middle of the step for the update); elsewhere (the
// timestep's rates, the diagnostics, the display) it is P.cosmo.w, the scale factor the step
// encoder last copied from the clock (src/sim.ts), at most one batch of steps behind. 1 without
// expansion.
var<private> g_a3: f32 = 0.0;
fn cosmo_a3() -> f32 {
  if (g_a3 != 0.0) { return g_a3; }
  if (!cosmo_on()) { return 1.0; }
  let a = P.cosmo.w;
  return a * a * a;
}

// The lower face along `axis` is a plane of mirror symmetry, or the axis of an axisymmetric
// run, not a wall: what lies beyond it is the image of what lies inside (the potential is even
// across it).
fn sym_lo(axis: i32) -> bool {
  return P.sym[axis] != 0u;
}

// An axisymmetric (R, z) run (Problem.geometry 'cylindrical'): code x is R from the axis (the
// lower x face), y is z, and the single layer is the azimuth, v_z being v_phi.
fn cylindrical() -> bool {
  return P.sym.x == 2u;
}
// For the timestep of an (R, z) run, at a cell whose center is rc = R_c / dx from the axis: what
// the radial rate gains (its outer face drains it at (R_c + dx/2) / R_c times what its width
// says: twice next to the axis) and the centrifugal acceleration v_phi^2 / R_c, for the
// acceleration limit (hydro/clock.wgsl). Without them a fast swirl reaching the axis emptied
// the axis cells in one step and went to NaN.
fn cyl_dt_terms(v: vec3<f32>, cs: f32, rc: f32, dx: f32) -> vec2<f32> {
  // (The acceleration over the same weighted length: the gas it starts moving drains the cell
  // through that face, so at most cfl^2 of the cell goes in a step.)
  let wr = 1.0 + 0.5 / rc;
  return vec2<f32>((abs(v.x) + cs) * (wr - 1.0), v.z * v.z / (rc * dx) * wr);
}
// The weights of cell i's radial faces, area over (cell volume / dx): R_- / R_c and R_+ / R_c,
// as pure ratios of indices (no cancellation in float32). The axis face weighs exactly 0.
fn cyl_weights(i: i32) -> vec2<f32> {
  let c = f32(i) + 0.5;
  return vec2<f32>(f32(i) / c, f32(i + 1) / c);
}
// The cell-centered B_R of cell i from its radial faces' fields: their mean weighed by their
// areas, (R_- b_- + R_+ b_+) / (R_- + R_+).
fn cyl_bcell_r(i: i32, lo: f32, hi: f32) -> f32 {
  let w = cyl_weights(i);
  return 0.5 * (w.x * lo + w.y * hi);
}
// An axisymmetric run on blocks: its cells are rings, whose volumes go
// as their radii. A parent's value is the volume-weighted mean of its children's, the child at
// fine radial index i weighing (i + 1/2) / (2 I + 1) against the plain mean, I the parent's
// index: so that the sum of R U over the children is the parent's.
fn cyl_child_weight(parent_i: i32, child_i: i32) -> f32 {
  if (!cylindrical()) { return 1.0; }
  return (f32(child_i) + 0.5) / (2.0 * f32(parent_i) + 1.0);
}
// For the predictor (hydro/predictor.wgsl): the cell's radius in cells (i + 1/2) in an
// axisymmetric run, 0 otherwise.
var<private> g_cyl_r: f32 = 0.0;

// A spherical (r, theta) run (Problem.geometry 'spherical'): code x is r from r_min (uniform or
// logarithmic spacing), y is theta from 0 to pi, the layer is the azimuth (v_z is v_phi). Both
// poles are axes. The grid's dx is a reference length: every weight below is a length over it.
// A constant (the slot <geometry>, src/sim.ts hydroSlots): elsewhere every branch on it is
// compiled out, and Cartesian and (R, z) runs are the same bits as without it.
#include <geometry>
fn spherical() -> bool {
  return SPHERICAL;
}
// The geometry of cell (i, j), as the update needs it, every entry a closed form of the indices
// (docs/numerics.md section 22): face areas over (cell volume / dx), which are differences of
// r^3 and cos(theta) that cancel in float32, are never formed as such.
struct SphCell {
  wr: vec2<f32>,  // the radial faces' weights, area * dx / volume, lower and upper
  wt: vec2<f32>,  // the polar faces' weights (exactly 0 on a pole)
  lr: vec2<f32>,  // lever arms of the radial faces over the cell's: r_-/r_c, r_+/r_c
  lt: vec2<f32>,  // and of the polar faces about the axis: sin(theta_-)/sin(theta_c), ...
  h: vec2<f32>,   // dx over the cell's length along r and along theta
  ir: f32,        // dx times the volume average of 1/r
  cot: f32,       // cot(theta_c)
  r: f32,         // r_c / dx
  st: f32,        // sin(theta_c)
}
// The grid a spherical cell's geometry is taken on: (r_min / dx, the step in ln r or 0, the step
// in theta) of its level, its cells along theta, and its logarithmic constants (P.sph2's). The
// uniform grid's is Params' own (sph_grid); a block's level is sph_level's.
struct SphGrid { r0: f32, d: f32, dt: f32, nt: i32, k: vec4<f32> }
fn sph_grid() -> SphGrid {
  return SphGrid(P.sph.x, P.sph.y, P.sph.z, i32(P.dims.y), P.sph2);
}
// The grid `c` levels coarser than P.sph's (the finest of a block grid, P.n cells per side):
// r_min / dx halved, the steps doubled per level. The logarithmic constants come from
// s = sinh(D/2) by doubling, s' = 2 s sqrt(1 + s^2), which cancels nothing: e^(D/2) = c + s,
// sinh D = 2 s c, sinh(3D/2) = s (3 + 4 s^2), with c = sqrt(1 + s^2). At c = 0, Params' own.
fn sph_level(c: i32) -> SphGrid {
  if (c <= 0) { return SphGrid(P.sph.x, P.sph.y, P.sph.z, i32(P.n), P.sph2); }
  let f = f32(1 << u32(c));
  var s = 0.5 * P.sph2.w;
  for (var m = 0; m < c; m++) { s = 2.0 * s * sqrt(1.0 + s * s); }
  let ch = sqrt(1.0 + s * s);
  let k = vec4<f32>(ch + s, 1.5 / max(s * (3.0 + 4.0 * s * s), 1.0e-30), 2.0 * s * ch, 2.0 * s);
  return SphGrid(P.sph.x / f, P.sph.y * f, P.sph.z * f, i32(P.n) >> u32(c), k);
}
// sin(theta) of the polar face j (0 to n), formed from the nearer pole so that both are exactly 0.
fn sph_sin_face_g(j: i32, q: SphGrid) -> f32 {
  return sin(f32(min(j, q.nt - j)) * q.dt);
}
fn sph_sin_face(j: i32) -> f32 {
  return sph_sin_face_g(j, sph_grid());
}
fn sph_cell_g(i: i32, j: i32, q: SphGrid) -> SphCell {
  var g: SphCell;
  let fi = f32(i) + 0.5;
  if (q.d == 0.0) {
    // Uniform in r: r_c / dx = r_min / dx + i + 1/2, volume / (dx^3 dcos) = r_c^2 + 1/12.
    let rc = q.r0 + fi;
    let d3 = rc * rc + 1.0 / 12.0;
    let rm = rc - 0.5;
    let rp = rc + 0.5;
    g.r = rc;
    g.wr = vec2<f32>(rm * rm, rp * rp) / d3;
    g.lr = vec2<f32>(rm, rp) / rc;
    g.ir = rc / d3;
    g.h.x = 1.0;
  } else {
    // Logarithmic: r_c = r_min e^((i + 1/2) D); r_+- = r_c e^(+-D/2).
    let rc = q.r0 * exp(fi * q.d);
    let e = q.k.x;
    let k = q.k.y / rc;
    g.r = rc;
    g.wr = k * vec2<f32>(1.0 / (e * e), e * e);
    g.lr = vec2<f32>(1.0 / e, e);
    g.ir = k * q.k.z;
    g.h.x = 1.0 / (rc * q.k.w);
  }
  let jm = min(j, q.nt - 1 - j);
  let tc = (f32(j) + 0.5) * q.dt;
  g.st = sin((f32(jm) + 0.5) * q.dt);
  g.cot = cos(tc) / g.st;
  let s = vec2<f32>(sph_sin_face_g(j, q), sph_sin_face_g(j + 1, q));
  g.lt = s / g.st;
  // dcos(theta) = 2 sin(theta_c) sin(dtheta / 2).
  g.wt = g.ir * s / (2.0 * g.st * sin(0.5 * q.dt));
  g.h.y = g.ir / q.dt;
  return g;
}
fn sph_cell(i: i32, j: i32) -> SphCell {
  return sph_cell_g(i, j, sph_grid());
}
// dx over the length the timestep takes along r and along theta: the cell's own (h), or less
// where a face drains it faster than its length says (the face weight, area * dx / volume). A
// cell on a pole has half the volume per area of its polar face that its length suggests, and
// the innermost shell over a small r_min a radial weight up to twice h.x: with h alone, gas
// leaving such a cell emptied it in one step (a swirl near the origin went to vacuum and NaN).
fn sph_rate_h(g: SphCell) -> vec2<f32> {
  return max(g.h, vec2<f32>(max(g.wr.x, g.wr.y), max(g.wt.x, g.wt.y)));
}
// The geometric (centrifugal) acceleration of velocity v = (v_r, v_theta, v_phi), as the
// timestep's acceleration limit takes it (dt <= cfl sqrt(dx / a), hydro/clock.wgsl): along each
// axis over that axis's length, (v_theta^2 + v_phi^2) / r along r and v_phi^2 cot(theta) / r
// along theta, so that dt <= cfl sqrt(L / a) on each. Near a pole cot(theta) / r is large: a
// swirl there starts the gas off the pole within one step, faster than its signal speeds say.
fn sph_accel(g: SphCell, v: vec3<f32>, dx: f32) -> f32 {
  let h = sph_rate_h(g);
  let ir = g.ir / dx;
  return max((v.y * v.y + v.z * v.z) * ir * h.x, v.z * v.z * abs(g.cot) * ir * h.y);
}
// (r_+^3 - r_-^3) / 3 over dx^3 of radial index i, and cos(theta_-) - cos(theta_+) of polar index j.
fn sph_v3(i: i32, q: SphGrid) -> f32 {
  let fi = f32(i) + 0.5;
  if (q.d == 0.0) {
    let rc = q.r0 + fi;
    return rc * rc + 1.0 / 12.0;
  }
  let rc = q.r0 * exp(fi * q.d);
  return rc * rc * rc / q.k.y;
}
fn sph_dcos(j: i32, q: SphGrid) -> f32 {
  let jm = min(j, q.nt - 1 - j);
  return 2.0 * sin((f32(jm) + 0.5) * q.dt) * sin(0.5 * q.dt);
}
// The volume of cell (i, j) of cell size dx, the azimuth's 2 pi included.
fn sph_volume_g(i: i32, j: i32, q: SphGrid, dx: f32) -> f32 {
  return 6.2831853 * sph_v3(i, q) * sph_dcos(j, q) * dx * dx * dx;
}
fn sph_volume(i: i32, j: i32) -> f32 {
  return sph_volume_g(i, j, sph_grid(), P.dx);
}
// For the predictor and the timestep: dx over the cell's own length along each axis (1 on every
// axis of a Cartesian grid, where the cells are cubes of side dx; set per cell in a spherical run).
var<private> g_hr: vec3<f32> = vec3<f32>(1.0);
// For the predictor in a spherical run: (dx times the average of 1/r, cot(theta)) of the cell;
// zero otherwise.
var<private> g_sph_geo: vec2<f32> = vec2<f32>(0.0);

// A position as a fraction of the box from the domain's lower corner (0 to 1 inside it), and
// back. With the corner at -box/2 these are x / box + 0.5 and (f - 0.5) box, bit for bit.
// The uniform grid's cells along each axis: n, or n / 2 along a mirrored axis halved on a
// rectangular grid (src/domain.ts gridDims, P.dims.w). (On blocks, P.n is the finest level's
// count and the boundary mapping takes it along x and y, as it always did.)
fn grid_n() -> vec3<i32> {
  if (P.dims.w == 1u) { return vec3<i32>(P.dims.xyz); }
  return vec3<i32>(i32(P.n), i32(P.n), i32(P.dims.z));
}
// Along such a halved axis the computed domain is half the box, and a fraction is of that half.
fn box_side() -> vec3<f32> {
  let half = vec3<bool>(P.dims.w == 1u) & (vec3<u32>(2u) * P.dims.xyz == vec3<u32>(P.n));
  return select(vec3<f32>(P.box), vec3<f32>(0.5 * P.box), half);
}
fn box_frac(x: vec3<f32>) -> vec3<f32> {
  let s = box_side();
  if (all(s == vec3<f32>(P.box))) { return x / P.box - P.org.xyz / P.box; }
  return (x - P.org.xyz) / s;
}
fn frac_pos(f: vec3<f32>) -> vec3<f32> {
  let s = box_side();
  if (all(s == vec3<f32>(P.box))) { return (f + P.org.xyz / P.box) * P.box; }
  return P.org.xyz + f * s;
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
fn bc_map(c0: vec3<i32>) -> Mapped {
  var m = Mapped(c0, vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(0.0));
  if (g_inner) { return m; }
  let n = grid_n();
  // Most loads are of cells inside the grid: they skip the rest.
  if (all(c0 >= vec3<i32>(0)) && all(c0 < n)) { return m; }
  var c = c0;
  // A shearing box's x faces: periodic, with the cells beyond sheared along y by the boundary's
  // offset (the lower of the two cells a ghost lies between; the loaders blend in the other,
  // shear_frac) and their velocity offset by the jump of the background flow.
  if (shear_on() && (c.x < 0 || c.x >= n.x)) {
    let up = c.x >= n.x;
    let s = i32(floor(f32(c.y) + select(-g_shear, g_shear, up)));
    c.y = ((s % n.y) + n.y) % n.y;
    m.c.y = c.y;
    m.vadd.y += select(1.0, -1.0, up) * shear_jump();
  }
  for (var a = 0; a < 3; a++) {
    if (c[a] >= 0 && c[a] < n[a]) { continue; }
    let up = c[a] >= n[a];
    let k = select(P.bc_lo[a], P.bc_hi[a], up);
    m.c[a] = bc_ghost_cell(c[a], n[a], k);
    if (bc_wall(k)) {
      m.flip *= bc_vsign(k, a);
      // Across the axis of an axisymmetric run the swirl v_phi changes sign too, and so does
      // the azimuthal field B_phi.
      // (Both poles of a spherical run are such axes.)
      if ((!up && P.sym[a] == 2u) || (a == 1 && spherical())) {
        m.flip.z = -m.flip.z;
        m.bflip.z = -m.bflip.z;
      }
      m.bflip *= bc_bsign_at(k, a, up);
      m.vadd += 2.0 * bc_wall_velocity(k, a, up);
    }
    m.side[a] = bc_side(k, up);
  }
  return m;
}

// Shearing box (P.shear, Problem.shear): the offset of its x faces along y, in cells, at the time
// a pass's loads are of: set at entry by the passes that load beyond those faces (0 otherwise).
// g_shear is for the state at the start of the step; load_pred uses g_shear_h, half a step on.
var<private> g_shear: f32 = 0.0;
var<private> g_shear_h: f32 = 0.0;
fn shear_on() -> bool {
  return P.shear.w != 0.0;
}
// The jump of the background flow v_y = -q Omega x across the box, q Omega L.
fn shear_jump() -> f32 {
  return P.shear.y * P.shear.x * P.box;
}
// The ghost c lies beyond an x face of a shearing box, between two sheared cells.
fn shear_ghost(c: vec3<i32>) -> bool {
  return shear_on() && (c.x < 0 || c.x >= i32(P.dims.x));
}
// For such a ghost, the weight of the second of those cells (the one above bc_map's along y).
fn shear_frac(c: vec3<i32>) -> f32 {
  let up = c.x >= i32(P.dims.x);
  let j = f32(c.y) + select(-g_shear, g_shear, up);
  return j - floor(j);
}
// The offset `s` (a length along y, in [0, L)) moved on by the shear over a time dt.
fn shear_half(s: f32, dt: f32) -> f32 {
  let d = s + 0.5 * dt * shear_jump();
  return d - P.box * floor(d / P.box);
}
// That second cell.
fn shear_next(m: vec3<i32>) -> vec3<i32> {
  return vec3<i32>(m.x, (m.y + 1) % i32(P.dims.y), m.z);
}

// Velocity of a (possibly ghost) cell: reversed as the walls say (and, beyond a moving wall,
// reflected about the wall's velocity: 2 v_wall - v), and at outflow-only faces
// the ghost's normal velocity is clamped to point outward, so nothing can flow in.
fn bc_velocity(v: vec3<f32>, m: Mapped) -> vec3<f32> {
  let w = v * m.flip + m.vadd;
  return select(w, max(w * m.side, vec3<f32>(0.0)) * m.side, m.side != vec3<f32>(0.0));
}
