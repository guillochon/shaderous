// Hancock predictor on the block grid: half-step states for every block's cells and for the
// first ghost layer.
//
// A ghost's half-step state is the prediction of whatever cell it stands for, evaluated from
// that cell's own neighbourhood. For a same-level neighbour that is exactly the neighbour's
// own prediction, so the two blocks compute identical fluxes at their shared face. A ghost
// next to a coarser block is predicted in place from this block's interpolated ghosts; the
// transverse slopes are dropped where they would need an edge ghost.
#include "hydro/common.wgsl"
#include <limiter>
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;
@group(0) @binding(3) var W2: texture_3d<f32>;
@group(0) @binding(5) var PA: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var PB: texture_storage_3d<rgba32float, write>;

#include "hydro/q.wgsl"

fn load_q(c: vec3<i32>) -> Q {
  let w1 = textureLoad(W1, c, 0);
  let w2 = textureLoad(W2, c, 0);
  return Q(w1, vec4<f32>(w2.x, w2.z, w2.y, 0.0));
}

#include "hydro/predictor.wgsl"

fn predict_cell(c: vec3<i32>, mask: vec3<f32>) -> Q {
  let q = load_q(c);
  var zm = q;
  var zp = q;
  if (!is2d()) {
    zm = load_q(c + vec3<i32>(0, 0, -1));
    zp = load_q(c + vec3<i32>(0, 0, 1));
  }
  return predictor(
    q,
    load_q(c + vec3<i32>(-1, 0, 0)), load_q(c + vec3<i32>(1, 0, 0)),
    load_q(c + vec3<i32>(0, -1, 0)), load_q(c + vec3<i32>(0, 1, 0)),
    zm, zp,
    P.gext.xyz, mask,
  );
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, 1);
  if (!inv.ok) { return; }
  g_dx = blk_dx(inv.slot);
  let s = ghost_src(inv.slot, inv.l);
  if (s.kind == 2) { return; }

  var pr: Q;
  if (s.kind == 0) {
    pr = predict_cell(atl(blk_origin(s.slot), s.l), vec3<f32>(1.0));
    pr.a = vec4<f32>(pr.a.x, src_velocity(pr.a.yzw, s));
  } else {
    // Transverse neighbours of a ghost at the rim of the face would be edge ghosts.
    let rim = (inv.l == vec3<i32>(0)) | (inv.l == vec3<i32>(BS - 1));
    pr = predict_cell(atl(blk_origin(inv.slot), inv.l), select(vec3<f32>(1.0), vec3<f32>(0.0), rim));
  }
  let c = atl(blk_origin(inv.slot), inv.l);
  textureStore(PA, c, pr.a);
  textureStore(PB, c, pr.b);
}
