// Species on a uniform grid, second pass: move every species' mass rho X_i with the mass
// fluxes the update just recorded (hydro/species_on.wgsl), for all NG groups of four at once.
//
// Each face carries F X_i, with F its mass flux and X_i the upwind cell's mass fraction half
// a step ahead (hydro/species_predict.wgsl) plus half its limited slope toward the face: the
// MUSCL-Hancock reconstruction the passive scalar gets. Each species is limited on its own and
// the slopes then made to sum to zero (hydro/species_slopes.wgsl), so face values sum to one and sum_i rho X_i moves
// exactly as rho does: mass fractions stay in [0, 1] and keep summing to one, and a species
// that is the same everywhere stays so. What rounding leaves is removed by scaling the new
// values to the new density. Solid and fixed cells keep theirs. Where the composition sets the
// equation of state (species/mixture.ts), the cell's moments are written beside them.
#include "hydro/species_common.wgsl"

@group(0) @binding(4) var N1: texture_3d<f32>;   // state after the update
@group(0) @binding(6) var SO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> MF: array<f32>;
@group(0) @binding(8) var SH: texture_3d<f32>;   // X half a step ahead

fn mf_index(c: vec3<i32>, axis: i32) -> u32 {
  let n = vec3<u32>(P.dims.xyz) + 1u;
  let u = vec3<u32>(c);
  return ((u.z * n.y + u.y) * n.x + u.x) * 3u + u32(axis);
}

// Net inflow of every species through the two faces of cell c along one axis, given the
// fractions of the five cells around it at the start of the step, k = -2 .. 2 (index k + 2).
fn axis_inflow(c: vec3<i32>, axis: i32, cells: array<array<vec4<f32>, NG>, 5>, acc: ptr<function, array<vec4<f32>, NG>>) {
  let e = unit(axis);
  for (var side = 0; side < 2; side++) {
    // Face between cells k = side - 1 and k = side (index side + 1 and side + 2).
    let F = MF[mf_index(c + side * e, axis)];
    let up = select(side + 2, side + 1, F >= 0.0);      // index of the upwind cell
    let toward = select(-0.5, 0.5, F >= 0.0);           // half a slope toward its upper or lower face
    let s = slopes(cells[up - 1], cells[up], cells[up + 1]);
    let m = bc_map(c + (up - 2) * e).c;
    var xf: array<vec4<f32>, NG>;
    for (var g = 0u; g < NG; g++) { xf[g] = textureLoad(SH, group_at(m, g), 0) + toward * s[g]; }
    xf = face_fractions(xf);
    let w = select(-F, F, side == 0);  // inflow through the lower face, outflow through the upper
    for (var g = 0u; g < NG; g++) { (*acc)[g] += w * xf[g]; }
  }
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  if (held(c)) {
    var u: array<vec4<f32>, NG>;
    for (var g = 0u; g < NG; g++) {
      u[g] = textureLoad(S, group_at(c, g), 0);
      textureStore(SO, group_at(c, g), u[g]);
    }
    if (COMP) { comp_store(c, comp_moments(u)); }
    return;
  }
  // Every cell of the stencil, two each way along the axes in use.
  g_inner = stencil_inside(c, 2);
  var acc: array<vec4<f32>, NG>;
  let flat = P.dims.z == 1u;
  for (var a = 0; a < 3; a++) {
    if (a == 2 && flat) { break; }
    let e = unit(a);
    var cells: array<array<vec4<f32>, NG>, 5>;
    for (var k = 0; k < 5; k++) { cells[k] = load_x(c + (k - 2) * e); }
    axis_inflow(c, a, cells, &acc);
  }
  let dtdx = clock.dt / P.dx;
  var out: array<vec4<f32>, NG>;
  var sum = 0.0;
  for (var g = 0u; g < NG; g++) {
    out[g] = max(textureLoad(S, group_at(c, g), 0) + dtdx * acc[g], vec4<f32>(0.0));
    sum += dot(out[g], vec4<f32>(1.0));
  }
  let scale = textureLoad(N1, c, 0).x / max(sum, 1e-30);
  for (var g = 0u; g < NG; g++) { textureStore(SO, group_at(c, g), out[g] * scale); }
  if (COMP) { comp_store(c, comp_moments(out)); }
}
