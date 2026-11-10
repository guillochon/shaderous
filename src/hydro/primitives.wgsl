// Conserved -> primitive on a uniform grid, and the first stage of the timestep reduction.
//   W1 = (rho, vx, vy, vz)    W2 = (p, c_s, scalar, e_int)
//
// The step size needs the largest signal rate and acceleration on the grid. Their first
// reduction stage (blocks of 4^3 cells) is folded in here instead of re-reading the
// primitives in a pass of its own: a workgroup spans 2 x 2 x 1 such blocks, its invocations
// fold their values into shared atomics, and one invocation per block writes the record the
// later stages of gpu/reduce.wgsl expect. Both values are non-negative, so comparing their
// bit patterns as integers orders them as floats.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var W1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var W2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var G: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> RATES: array<f32>;   // (max rate, max |g|) per block

#include "gravity/gfield.wgsl"

var<workgroup> wg_max: array<atomic<u32>, 8>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let blk = 2u * ((lid.x >> 2u) + 2u * (lid.y >> 2u));
  if (all(gid < P.dims.xyz)) {
    g_dx = P.dx;
    let c = vec3<i32>(gid);
    let w = prim_from_cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
    textureStore(W1, c, w.w1);
    textureStore(W2, c, w.w2);
    // Unsplit scheme: the stable step is set by the sum of the per-axis signal speeds.
    let rate = abs(w.w1.y) + abs(w.w1.z) + abs(w.w1.w) + P.geom.y * w.w2.y;
    atomicMax(&wg_max[blk], bitcast<u32>(rate));
    atomicMax(&wg_max[blk + 1u], bitcast<u32>(length(g_load(c))));
  }
  workgroupBarrier();
  let nout = (P.dims.xyz + 3u) / 4u;
  let o = gid / 4u;
  if (all((lid & vec3<u32>(3u)) == vec3<u32>(0u)) && all(o < nout)) {
    let base = ((o.z * nout.y + o.y) * nout.x + o.x) * 2u;
    RATES[base] = bitcast<f32>(atomicLoad(&wg_max[blk]));
    RATES[base + 1u] = bitcast<f32>(atomicLoad(&wg_max[blk + 1u]));
  }
}
