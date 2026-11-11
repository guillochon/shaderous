// Refinement indicator: for every leaf block, the largest normalised jump in density or
// pressure between a cell's two neighbours along any axis, |q+ - q-| / (q+ + q-).
// With self-gravity a block is also refined where the Jeans length is short: that indicator is
// scaled so it crosses the refinement threshold when the Jeans length falls below the
// requested number of cells.
// One value per slot, accumulated with an atomic max on the float's bit pattern (which orders
// the same way as the float for non-negative values). Every gas cell writes at least a tiny
// positive value, so a block that reads exactly 0 lies wholly inside an obstacle.
const GAS_MIN = 1e-30;
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var W1: texture_3d<f32>;
@group(0) @binding(2) var W2: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> FLAGS: array<atomic<u32>>;

// A solid cell with a gas cell beside it.
fn solid_surface(c: vec3<i32>) -> bool {
  var gas = textureLoad(W1, c + vec3<i32>(1, 0, 0), 0).x >= 0.0 || textureLoad(W1, c - vec3<i32>(1, 0, 0), 0).x >= 0.0 ||
            textureLoad(W1, c + vec3<i32>(0, 1, 0), 0).x >= 0.0 || textureLoad(W1, c - vec3<i32>(0, 1, 0), 0).x >= 0.0;
  if (!is2d()) {
    gas = gas || textureLoad(W1, c + vec3<i32>(0, 0, 1), 0).x >= 0.0 || textureLoad(W1, c - vec3<i32>(0, 0, 1), 0).x >= 0.0;
  }
  return gas;
}

fn jump(c: vec3<i32>, e: vec3<i32>) -> f32 {
  // A negative density marks a solid cell. The surface of an obstacle is always refined, so
  // its shape is kept at the finest resolution. A gas cell's solid neighbour holds no flow to
  // compare with.
  let sp = textureLoad(W1, c + e, 0).x;
  let sm = textureLoad(W1, c - e, 0).x;
  if ((sp < 0.0) != (sm < 0.0)) { return 1.0; }
  if (sp < 0.0) { return 0.0; }
  let rp = sp;
  let rm = sm;
  let pp = textureLoad(W2, c + e, 0).x;
  let pm = textureLoad(W2, c - e, 0).x;
  return max(abs(rp - rm) / (rp + rm), abs(pp - pm) / (pp + pm));
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
  var ind = max(jump(c, vec3<i32>(1, 0, 0)), jump(c, vec3<i32>(0, 1, 0)));
  if (!is2d()) { ind = max(ind, jump(c, vec3<i32>(0, 0, 1))); }
  if (has_potential() && P.geom.w > 0.0) {
    let rho = abs(textureLoad(W1, c, 0).x);
    let cs = textureLoad(W2, c, 0).y;
    let lj = cs * sqrt(3.14159265 / (P.G * rho));
    ind = max(ind, P.geom.z * P.geom.w * blk_dx(inv.slot) / lj);
  }
  atomicMax(&FLAGS[u32(inv.slot)], bitcast<u32>(max(ind, GAS_MIN)));
}
