// First stage of the timestep reduction for MHD on a uniform grid: hydro/rates.wgsl with the
// fastest magnetosonic speed, sqrt(c_s^2 + B^2 / rho), in place of the sound speed.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> RATES: array<f32>;   // (max rate, max |g|) per block

#include "mhd/common.wgsl"

var<workgroup> wg_max: array<atomic<u32>, 8>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let blk = 2u * ((lid.x >> 2u) + 2u * (lid.y >> 2u));
  if (all(gid < P.dims.xyz)) {
    g_dx = P.dx;
    let c = vec3<i32>(gid);
    comp_use(comp_tex(c));
    let w = prim_from_cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
    let cf = fast_max(w.w1.x, w.w2.y, bcell(c));
    let rate = abs(w.w1.y) + abs(w.w1.z) + abs(w.w1.w) + P.geom.y * cf;
    atomicMax(&wg_max[blk], bitcast<u32>(rate));
  }
  workgroupBarrier();
  let nout = (P.dims.xyz + 3u) / 4u;
  let o = gid / 4u;
  if (all((lid & vec3<u32>(3u)) == vec3<u32>(0u)) && all(o < nout)) {
    let base = ((o.z * nout.y + o.y) * nout.x + o.x) * 2u;
    RATES[base] = bitcast<f32>(atomicLoad(&wg_max[blk]));
    RATES[base + 1u] = 0.0;
  }
}
