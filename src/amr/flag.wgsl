// Refinement indicator: for every leaf block, the largest normalized second derivative of
// density or pressure over its cells (Lohner 1987, as in FLASH and PLUTO):
//   chi^2 = sum_d (q+ - 2 q + q-)^2 / sum_d (|q+ - q| + |q - q-| + EPS (|q+| + 2|q| + |q-|))^2
// over the axes d. It is near 1 at a discontinuity and falls as dx / L on a smooth profile of
// scale L, so smooth gradients (a disk, a stratified atmosphere) are not refined everywhere as
// a first difference would refine them. The EPS term keeps small ripples from counting.
// With self-gravity a block is also refined where the Jeans length is short: that indicator is
// scaled so it crosses the refinement threshold when the Jeans length falls below the
// requested number of cells.
// Where a preset asks for it (P.flags bit 4), every block that comes near a point mass is
// refined to the finest level, so its softening and sink are resolved as well as they can be.
// Gas thinner than P.refine.x (ambient gas around a star, say) does not ask for refinement by
// the indicator: its ripples, relative to its own tiny density, are not worth resolving. Gas
// denser than P.refine.y, where a preset sets it, is refined to the finest level outright.
// One value per slot, accumulated with an atomic max on the float's bit pattern (which orders
// the same way as the float for non-negative values). Every gas cell writes at least a tiny
// positive value, so a block that reads exactly 0 lies wholly inside an obstacle.
const GAS_MIN = 1e-30;
const EPS = 0.01;
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var W1: texture_3d<f32>;
@group(0) @binding(2) var W2: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> FLAGS: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> BH: Masses;

// The cell lies near a point mass: within twice its softening or sink radius, whichever is
// larger, and at least four of this block's cells.
fn near_mass(slot: i32, l: vec3<i32>) -> bool {
  if ((P.flags & 16u) == 0u) { return false; }
  let x = blk_pos(slot, l);
  for (var i = 0u; i < min(mass_count(), 4u); i++) {
    let m = BH.list[i];
    var d = x - m.pos.xyz;
    if (is2d()) { d.z = 0.0; }
    let r = max(2.0 * max(m.vel.w, m.acc.w), 4.0 * blk_dx(slot));
    if (dot(d, d) < r * r) { return true; }
  }
  return false;
}

// A solid cell with a gas cell beside it.
fn solid_surface(c: vec3<i32>) -> bool {
  var gas = textureLoad(W1, c + vec3<i32>(1, 0, 0), 0).x >= 0.0 || textureLoad(W1, c - vec3<i32>(1, 0, 0), 0).x >= 0.0 ||
            textureLoad(W1, c + vec3<i32>(0, 1, 0), 0).x >= 0.0 || textureLoad(W1, c - vec3<i32>(0, 1, 0), 0).x >= 0.0;
  if (!is2d()) {
    gas = gas || textureLoad(W1, c + vec3<i32>(0, 0, 1), 0).x >= 0.0 || textureLoad(W1, c - vec3<i32>(0, 0, 1), 0).x >= 0.0;
  }
  return gas;
}

// Numerator and denominator of the indicator along one axis, for density (xy) and pressure (zw).
fn second(c: vec3<i32>, e: vec3<i32>, r0: f32, p0: f32) -> vec4<f32> {
  let rp = textureLoad(W1, c + e, 0).x;
  let rm = textureLoad(W1, c - e, 0).x;
  let pp = textureLoad(W2, c + e, 0).x;
  let pm = textureLoad(W2, c - e, 0).x;
  let dr = abs(rp - r0) + abs(r0 - rm) + EPS * (abs(rp) + 2.0 * abs(r0) + abs(rm));
  let dp = abs(pp - p0) + abs(p0 - pm) + EPS * (abs(pp) + 2.0 * abs(p0) + abs(pm));
  return vec4<f32>(rp - 2.0 * r0 + rm, dr, pp - 2.0 * p0 + pm, dp);
}

// The indicator at gas cell c. A negative density marks a solid cell. The surface of an
// obstacle is always refined, so its shape is kept at the finest resolution. Along an axis
// with solid cells on both sides there is no flow to compare.
fn lohner(c: vec3<i32>) -> f32 {
  let r0 = textureLoad(W1, c, 0).x;
  let p0 = textureLoad(W2, c, 0).x;
  var num = vec2<f32>(0.0);
  var den = vec2<f32>(0.0);
  let axes = select(3, 2, is2d());
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let sp = textureLoad(W1, c + e, 0).x < 0.0;
    let sm = textureLoad(W1, c - e, 0).x < 0.0;
    if (sp != sm) { return 1.0; }
    if (sp) { continue; }
    let s = second(c, e, r0, p0);
    num += vec2<f32>(s.x * s.x, s.z * s.z);
    den += vec2<f32>(s.y * s.y, s.w * s.w);
  }
  let chi = sqrt(num / max(den, vec2<f32>(1e-30)));
  return max(chi.x, chi.y);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  // Inside an obstacle there is no flow to resolve: the gas kept there for when it is removed
  // may hold any jumps. Only the surface (seen from the gas cells beside it) asks for refinement.
  if (textureLoad(W1, c, 0).x < 0.0) {
    if (solid_surface(c)) { atomicMax(&FLAGS[u32(inv.slot)], bitcast<u32>(1.0)); }
    return;
  }
  var ind = lohner(c);
  if (textureLoad(W1, c, 0).x < P.refine.x) { ind = 0.0; }
  // Above any threshold the refinement criterion may be given.
  if (near_mass(inv.slot, inv.l)) { ind = 1e30; }
  // Gas above the preset's density for it (the stars of a collision) is kept at the finest level.
  if (P.refine.y > 0.0 && textureLoad(W1, c, 0).x > P.refine.y) { ind = 1e30; }
  if (has_potential() && P.geom.w > 0.0) {
    let rho = abs(textureLoad(W1, c, 0).x);
    let cs = textureLoad(W2, c, 0).y;
    let lj = cs * sqrt(3.14159265 / (P.G * rho));
    ind = max(ind, P.geom.z * P.geom.w * blk_dx(inv.slot) / lj);
  }
  atomicMax(&FLAGS[u32(inv.slot)], bitcast<u32>(max(ind, GAS_MIN)));
}
