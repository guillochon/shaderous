// Hancock predictor pass on a uniform grid.
//   PA = (rho, vx, vy, vz) at the half step
//   PB = (p, scalar at the half step, shock flag, first-order flag)
// Doing this once per cell, instead of seven times inside the update's gather, is the main
// reason the update pass is cheap.
#include "hydro/common.wgsl"
#include <limiter>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;
@group(0) @binding(3) var W2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;
@group(0) @binding(5) var PA: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var PB: texture_storage_3d<rgba32float, write>;

#include "hydro/recon.wgsl"
#include "hydro/predictor.wgsl"
#include "gravity/gfield.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  g_dx = P.dx;
  let c = vec3<i32>(gid);
  let pr = predictor(
    load_q(c),
    load_q(c + vec3<i32>(-1, 0, 0)), load_q(c + vec3<i32>(1, 0, 0)),
    load_q(c + vec3<i32>(0, -1, 0)), load_q(c + vec3<i32>(0, 1, 0)),
    load_q(c + vec3<i32>(0, 0, -1)), load_q(c + vec3<i32>(0, 0, 1)),
    g_load(c), vec3<f32>(1.0),
  );
  textureStore(PA, c, pr.a);
  textureStore(PB, c, pr.b);
}
