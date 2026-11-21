// Partly filled cells, second of three passes (hydro/cut_common.wgsl): each gas cell's
// neighborhood mean of the state per unit gas volume (density, momentum with the wall's shear in
// DM, the passive scalar), each member taken with its gas share over its count of neighborhoods.
// A cell under CUT_SMALL full means its neighborhood's; any other, its own state.
#include "hydro/common.wgsl"
#include "hydro/cut_flags.wgsl"
#include "hydro/cut_common.wgsl"

@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var CN: texture_3d<f32>;
@group(0) @binding(5) var DM: texture_3d<f32>;
@group(0) @binding(6) var UH1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var UH2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let k = cut_at(c);
  let u2 = textureLoad(U2, c, 0);
  if (k.x <= 0.0) {
    textureStore(UH1, c, vec4<f32>(0.0));
    textureStore(UH2, c, vec4<f32>(0.0, 0.0, 0.0, u2.w));
    return;
  }
  let u1 = textureLoad(U1, c, 0);
  var a = vec4<f32>(u1.x, u1.yzw + textureLoad(DM, c, 0).xyz);
  var s = u2.z;
  let h = cut_hood(c);
  if (h.small && (h.any || any(h.fallback != vec3<i32>(0)))) {
    // The neighborhood: the cell and its neighbors away from the wall along the axes it faces
    // (or the one face neighbor holding the most gas, cut_hood).
    let n0 = textureLoad(CN, c, 0).x;
    var vol = k.x / n0;
    var sa = a * (k.x / n0);
    var ss = s * (k.x / n0);
    var take = h.take;
    var away = h.away;
    if (!h.any) {
      take = abs(h.fallback);
      away = select(vec3<i32>(1), h.fallback, h.fallback != vec3<i32>(0));
    }
    for (var iz = 0; iz <= take.z; iz++) {
      for (var iy = 0; iy <= take.y; iy++) {
        for (var ix = 0; ix <= take.x; ix++) {
          let o = vec3<i32>(ix, iy, iz) * away;
          if (all(o == vec3<i32>(0))) { continue; }
          let q = c + o;
          let pq = cut_at(q).x;
          if (pq <= 0.0) { continue; }
          let qq = cut_cell(q);
          let nq = textureLoad(CN, qq, 0).x;
          let uq = textureLoad(U1, qq, 0);
          let wq = pq / nq;
          vol += wq;
          sa += wq * vec4<f32>(uq.x, uq.yzw + textureLoad(DM, qq, 0).xyz);
          ss += wq * textureLoad(U2, qq, 0).z;
        }
      }
    }
    a = sa / vol;
    s = ss / vol;
  }
  textureStore(UH1, c, a);
  textureStore(UH2, c, vec4<f32>(s, 0.0, 0.0, u2.w));
}
