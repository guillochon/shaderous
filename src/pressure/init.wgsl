// The pressure a run starts from, before its first solve: that of the state, p(rho). The
// predictor's lagged gradient and the solve's warm start then hold a fluid at rest in a field
// from the first step, where a zero pressure would let it fall for one.
#include "hydro/common.wgsl"
#include <eos>

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var PRS: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  textureStore(PRS, c, vec4<f32>(eos_pressure(max(textureLoad(U1, c, 0).x, P.rho_floor), 0.0), 0.0, 0.0, 0.0));
}
