// Species on a uniform grid, first pass: every cell's mass fractions half a step ahead,
// X - dt/2 v . grad X with the cell's velocity and the limited slopes (hydro/species_slopes.wgsl),
// as hydro/predictor.wgsl advances the passive scalar. hydro/species.wgsl builds face values
// from these, so species are reconstructed as the passive scalar is, cross terms included.
#include "hydro/species_common.wgsl"

@group(0) @binding(6) var SH: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  g_inner = stencil_inside(c, 1);
  let x = load_x(c);
  var xh = x;
  if (!held(c)) {
    let u = textureLoad(U1, c, 0);
    let v = u.yzw / max(u.x, P.rho_floor);
    let h = 0.5 * clock.dt / P.dx;
    let flat = P.dims.z == 1u;
    for (var a = 0; a < 3; a++) {
      if (a == 2 && flat) { break; }
      let e = unit(a);
      let s = slopes(load_x(c - e), x, load_x(c + e));
      for (var g = 0u; g < NG; g++) { xh[g] -= h * v[a] * s[g]; }
    }
  }
  for (var g = 0u; g < NG; g++) { textureStore(SH, group_at(c, g), xh[g]); }
}
