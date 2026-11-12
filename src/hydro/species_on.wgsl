// With species (species/species.ts): the update records the mass flux through every face for
// the species pass (hydro/species.wgsl). A cell writes its three lower faces, and its upper
// face along an axis where nothing else would: at the upper edge of the grid, and next to a
// solid or fixed cell (which return before computing fluxes).
//
// MF holds one value per face: face (axis, c) is the lower face of cell c along that axis,
// for c up to dims inclusive, at mf_index(c, axis).
const SPECIES = true;
@group(0) @binding(13) var<storage, read_write> MF: array<f32>;

fn mf_index(c: vec3<i32>, axis: i32) -> u32 {
  let n = vec3<u32>(P.dims.xyz) + 1u;
  let u = vec3<u32>(c);
  return ((u.z * n.y + u.y) * n.x + u.x) * 3u + u32(axis);
}

fn held_cell(c: vec3<i32>) -> bool {
  if (!has_solids() || any(c >= vec3<i32>(P.dims.xyz))) { return false; }
  return abs(textureLoad(U2, c, 0).w) > 0.5;
}

fn store_mass_flux(c: vec3<i32>, fx: Faces, fy: Faces, fz: Faces, flat: bool) {
  let n = vec3<i32>(P.dims.xyz);
  let e = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
  let lo = vec3<f32>(fx.lo.m, fy.lo.m, fz.lo.m);
  let hi = vec3<f32>(fx.hi.m, fy.hi.m, fz.hi.m);
  for (var a = 0; a < 3; a++) {
    if (a == 2 && flat) { break; }
    MF[mf_index(c, a)] = lo[a];
    let up = c + vec3<i32>(e[a]);
    if (up[a] >= n[a] || held_cell(up)) { MF[mf_index(up, a)] = hi[a]; }
  }
}
