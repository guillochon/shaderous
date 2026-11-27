// Applies the editing brush on the uniform grid: reads the current state, writes the other
// parity (every cell, so the two can be swapped).
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "edit/brush_core.wgsl"
#include <refdev>

@group(0) @binding(2) var N1: texture_3d<f32>;
@group(0) @binding(3) var N2: texture_3d<f32>;
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  g_dx = P.dx;
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let x = (vec3<f32>(gid) + 0.5) / vec3<f32>(P.dims.xyz);
  comp_use(comp_tex(c));
  let u1 = textureLoad(N1, c, 0);
  let u2 = textureLoad(N2, c, 0);
  if (REF_DEV) {
    // A gas stored as its deviation from a reference (hydro/ref_on.wgsl): the brush works on the
    // whole state, and a cell it leaves alone keeps its deviation as it was, bit for bit.
    let r = ref_at(c);
    let a1 = vec4<f32>(u1.x + r.x, u1.yzw);
    let a2 = vec4<f32>(u2.x + r.z, u2.y + r.z, u2.zw);
    let o = brush_apply(a1, a2, x);
    if (all(o.u1 == a1) && all(o.u2 == a2)) {
      textureStore(O1, c, u1);
      textureStore(O2, c, u2);
    } else {
      textureStore(O1, c, vec4<f32>(o.u1.x - r.x, o.u1.yzw));
      textureStore(O2, c, vec4<f32>(o.u2.x - r.z, o.u2.y - r.z, o.u2.zw));
    }
    return;
  }
  let out = brush_apply(u1, u2, x);
  textureStore(O1, c, out.u1);
  textureStore(O2, c, out.u2);
}
