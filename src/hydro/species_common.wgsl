// Shared by the species passes on a uniform grid (hydro/species_predict.wgsl and
// hydro/species.wgsl): the species at the start of the step and their slopes.
//
// Groups of four are stacked along z: group g of cell c is the texel c + (0, 0, g nz).
#include "hydro/common.wgsl"
#include <species_n>
#include "hydro/species_slopes.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;   // state at the start of the step
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(5) var S: texture_3d<f32>;    // rho X at the start of the step

fn group_at(c: vec3<i32>, g: u32) -> vec3<i32> {
  return c + vec3<i32>(0, 0, i32(g * P.dims.z));
}

fn held(c: vec3<i32>) -> bool {
  return has_solids() && abs(textureLoad(U2, c, 0).w) > 0.5;
}

// Mass fractions of a cell, or of the cell a ghost maps to.
fn load_x(c: vec3<i32>) -> array<vec4<f32>, NG> {
  let m = bc_map(c).c;
  var u: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { u[g] = textureLoad(S, group_at(m, g), 0); }
  return fractions(u);
}
