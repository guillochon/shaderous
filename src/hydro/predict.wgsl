// Hancock predictor pass on a uniform grid.
//   PA = (rho, vx, vy, vz) at the half step
//   PB = (p, scalar at the half step, shock flag, first-order flag)
// Doing this once per cell, instead of seven times inside the update's gather, is the main
// reason the update pass is cheap.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;
@group(0) @binding(5) var PA: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var PB: texture_storage_3d<rgba32float, write>;

#include "hydro/region.wgsl"
#include "hydro/recon.wgsl"
#include "hydro/predictor.wgsl"
#include "gravity/gfield.wgsl"

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rc = region_cell(gid);
  if (!rc.mine) { return; }
  g_dx = P.dx;
  let c = rc.c;
  g_inner = stencil_inside(c, 1);
  let q = load_q(c);
  // In 2D the layer is its own neighbour along z.
  var zm = q;
  var zp = q;
  if (P.dims.z != 1u) {
    zm = load_q(c + vec3<i32>(0, 0, -1));
    zp = load_q(c + vec3<i32>(0, 0, 1));
  }
  let pr = predictor(
    q,
    load_q(c + vec3<i32>(-1, 0, 0)), load_q(c + vec3<i32>(1, 0, 0)),
    load_q(c + vec3<i32>(0, -1, 0)), load_q(c + vec3<i32>(0, 1, 0)),
    zm, zp,
    g_load(c), vec3<f32>(1.0),
  );
  textureStore(PA, c, pr.a);
  textureStore(PB, c, pr.b);
}
