// Hancock predictor on the block grid: half-step states for every block's cells and for the
// first ghost layer.
//
// A ghost's half-step state is the prediction of whatever cell it stands for, evaluated from
// that cell's own neighborhood. For a same-level neighbor that is exactly the neighbor's
// own prediction, so the two blocks compute identical fluxes at their shared face. A ghost
// next to a coarser block is predicted in place from this block's interpolated ghosts; the
// transverse slopes are dropped where they would need an edge ghost. Every block predicts
// with its own step, its ghosts included (per-block timesteps, amr/bins_on.wgsl).
#include "hydro/common.wgsl"
#include <limiter>
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>
#include "amr/blk_dt.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;
@group(0) @binding(3) var W2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;
@group(0) @binding(5) var PA: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var PB: texture_storage_3d<rgba32float, write>;

#include "hydro/q.wgsl"

fn load_q(c: vec3<i32>) -> Q {
  let w1 = textureLoad(W1, c, 0);
  let w2 = textureLoad(W2, c, 0);
  // A negative density marks a solid cell (amr/assemble.wgsl).
  return Q(vec4<f32>(abs(w1.x), w1.yzw), vec4<f32>(w2.x, w2.z, w2.y, select(0.0, 1.0, w1.x < 0.0)));
}

#include "hydro/predictor.wgsl"

// Acceleration at an interior cell: the uniform field plus self-gravity from the field atlas.
fn grav_at(c: vec3<i32>) -> vec3<f32> {
  var g = P.gext.xyz;
  if (P.grav_on != 0u) { g += textureLoad(G, c, 0).xyz; }
  return g;
}


// For the well-balanced pressure reconstruction (hydro/q.wgsl): the potential difference
// between the adjacent cells l + k e and l + (k + 1) e of block `slot` along an axis, as
// (seen, face), from the field atlas (which holds the potential on two ghost layers) plus the
// uniform field, which is exact. `seen` is what the deviations from balance are measured
// against: beyond a face of the domain the potential is that of the cell the state there is
// taken from, like the state itself. `face` is what takes the pressure from a cell's center
// to its face: the same, except across a reflecting face of the domain, where it is the true
// difference (the atlas's ghost cells there continue the potential in a straight line).
fn dphi_blk(slot: i32, l: vec3<i32>, axis: i32, k: i32) -> vec2<f32> {
  let ra = l[axis] + k;
  let ia = row_src(slot, axis, ra);
  let ib = row_src(slot, axis, ra + 1);
  let o = blk_origin(slot);
  var a = l;
  var b = l;
  a[axis] = ia;
  b[axis] = ib;
  var seen = -P.gext[axis] * f32(ib - ia) * g_dx;
  if (P.grav_on != 0u) { seen += textureLoad(G, atl(o, b), 0).w - textureLoad(G, atl(o, a), 0).w; }
  var face = seen;
  let wall = (ra == -1 && bt(slot, 6u + u32(2 * axis)) == -3) || (ra == BS - 1 && bt(slot, 7u + u32(2 * axis)) == -3);
  if (wall) {
    a[axis] = ra;
    b[axis] = ra + 1;
    face = -P.gext[axis] * g_dx;
    if (P.grav_on != 0u) { face += textureLoad(G, atl(o, b), 0).w - textureLoad(G, atl(o, a), 0).w; }
  }
  return vec2<f32>(seen, face);
}

// Cell l of block `slot` (a ghost, when predicted in place), half a step on.
fn predict_cell(slot: i32, l: vec3<i32>, mask: vec3<f32>, g: vec3<f32>) -> Q {
  let c = atl(blk_origin(slot), l);
  let q = load_q(c);
  var zm = q;
  var zp = q;
  if (!is2d()) {
    zm = load_q(c + vec3<i32>(0, 0, -1));
    zp = load_q(c + vec3<i32>(0, 0, 1));
  }
  var lo = PhiSteps(vec3<f32>(0.0), vec3<f32>(0.0));
  var hi = lo;
  if (well_balanced()) {
    let nax = select(3, 2, is2d());
    for (var axis = 0; axis < nax; axis++) {
      // Not along an axis whose slope is dropped: its neighbors would be edge ghosts.
      if (mask[axis] == 0.0) { continue; }
      let below = dphi_blk(slot, l, axis, -1);
      let above = dphi_blk(slot, l, axis, 0);
      lo.seen[axis] = below.x;
      lo.face[axis] = below.y;
      hi.seen[axis] = above.x;
      hi.face[axis] = above.y;
    }
  }
  return predictor(
    q,
    load_q(c + vec3<i32>(-1, 0, 0)), load_q(c + vec3<i32>(1, 0, 0)),
    load_q(c + vec3<i32>(0, -1, 0)), load_q(c + vec3<i32>(0, 1, 0)),
    zm, zp,
    g, mask, lo, hi,
  );
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, 1);
  if (!inv.ok) { return; }
  g_dx = blk_dx(inv.slot);
  g_dt = blk_dt(inv.slot);
  let s = ghost_src(inv.slot, inv.l);
  if (s.kind == 2) { return; }

  var pr: Q;
  // A neighbor part way through a longer step holds primitives of an earlier time. Its ghost
  // here was filled from its half-step prediction, and is predicted in place, like one next
  // to a coarser block.
  let stale = BINS_ON && s.slot != inv.slot && blk_leaf(s.slot) && !blk_active(s.slot);
  if (s.kind == 0 && !stale) {
    let src = atl(blk_origin(s.slot), s.l);
    pr = predict_cell(s.slot, s.l, vec3<f32>(1.0), grav_at(src));
    pr.a = vec4<f32>(pr.a.x, src_velocity(pr.a.yzw, s));
  } else {
    // Transverse neighbors of a ghost at the rim of the face would be edge ghosts.
    let rim = (inv.l == vec3<i32>(0)) | (inv.l == vec3<i32>(BS - 1));
    // The field atlas holds no acceleration for ghosts; use the adjacent interior cell's.
    let near = atl(blk_origin(inv.slot), clamp(inv.l, vec3<i32>(0), vec3<i32>(BS - 1)));
    pr = predict_cell(inv.slot, inv.l, select(vec3<f32>(1.0), vec3<f32>(0.0), rim), grav_at(near));
  }
  let c = atl(blk_origin(inv.slot), inv.l);
  textureStore(PA, c, pr.a);
  textureStore(PB, c, pr.b);
}
