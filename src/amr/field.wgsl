// Resamples the block grid onto the uniform display volume that the ray-marcher and slice
// view read. Each display voxel takes the value of the leaf cell under its center
// (render/field_value.wgsl), and the pass records the range of what it draws.
//   r = displayed quantity, normalized to [0, 1]
//   g, b, a = 1 on the lower boundary of a block, one display voxel wide, as seen in a z-, x-
//             or y-normal slice
// Derivatives come from the primitive variables, whose ghost cells hold the neighbors across
// block faces at the block's own level (as of the last assembly pass).
#include "hydro/common.wgsl"
#include <eos>
#include "amr/grid.wgsl"
#include "render/field_value.wgsl"

struct View {
  field: u32,      // index into FIELDS (src/problems/types.ts)
  logscale: u32,
  lo: f32,         // range ends, in log10 units when logscale is set
  hi: f32,
  nd: vec4<u32>,   // display volume size
}
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var VOL: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var IDX: texture_3d<u32>;   // leaf slot per deepest-level block
@group(0) @binding(6) var W1: texture_3d<f32>;    // primitives: (density, negative if solid; velocity)
@group(0) @binding(7) var GF: texture_3d<f32>;    // gravity field: acceleration, potential in w
@group(0) @binding(8) var<storage, read_write> RANGE: array<atomic<u32>, 132>;

fn neighbor(c: vec3<i32>) -> vec4<f32> {
  let w = textureLoad(W1, c, 0);
  return vec4<f32>(w.yzw, max(abs(w.x), P.rho_floor));
}

fn gradients(c: vec3<i32>, dx: f32) -> Grad {
  var d = array<vec4<f32>, 3>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
  let axes = select(3, 2, is2d());
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    d[a] = (neighbor(c + e) - neighbor(c - e)) / (2.0 * dx);
  }
  return Grad(vec3<f32>(d[0].x, d[1].x, d[2].x), vec3<f32>(d[0].y, d[1].y, d[2].y), vec3<f32>(d[0].z, d[1].z, d[2].z), vec3<f32>(d[0].w, d[1].w, d[2].w));
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  range_begin(lix);
  if (all(gid < V.nd.xyz)) {
    var frac = (vec3<f32>(gid) + 0.5) / vec3<f32>(V.nd.xyz);
    if (is2d()) { frac.z = 0.0; }
    let m = i32(P.amr.y << P.amr.z);
    let slot = i32(textureLoad(IDX, vec3<i32>(frac * f32(m)), 0).x);
    g_dx = blk_dx(slot);
    let level = bt(slot, 0u);
    let nblk = i32(P.amr.y) * (1 << u32(level));   // blocks across the domain at this level
    let ncell = nblk * BS;
    var l = clamp(vec3<i32>(frac * f32(ncell)) - blk_coord(slot) * BS, vec3<i32>(0), vec3<i32>(BS - 1));
    if (is2d()) { l.z = 0; }
    let c = atl(blk_origin(slot), l);
    let u1 = textureLoad(U1, c, 0);
    let u2 = textureLoad(U2, c, 0);
    var g: Grad;
    if (field_needs_grad(V.field)) { g = gradients(c, g_dx); }
    // The uniform field's potential, measured from the box center, is not in the texture.
    let phi = textureLoad(GF, c, 0).w - dot(P.gext.xyz, blk_pos(slot, l));
    var x = field_value(V.field, u1, u2, g, f32(level), phi);
    let solid = u2.w > 0.5;
    if (!solid) { range_add(x); }
    if (V.logscale != 0u) { x = log(max(x, 1.0e-30)) * 0.4342944819; }
    var t = clamp((x - V.lo) / max(V.hi - V.lo, 1.0e-20), 0.0, 1.0);
    if (solid) { t = -1.0; }   // solid: drawn as a body, not as gas

    // Position within the block in display voxels; the first voxel along an axis is the outline.
    let within = (frac * f32(nblk) - vec3<f32>(blk_coord(slot))) * vec3<f32>(V.nd.xyz) / f32(nblk);
    let edge = vec3<f32>(within < vec3<f32>(1.0));
    textureStore(VOL, vec3<i32>(gid), vec4<f32>(t, max(edge.x, edge.y), max(edge.y, edge.z), max(edge.x, edge.z)));
  }
  range_end(lix);
}
