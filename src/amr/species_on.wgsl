// With species (species/species.ts): the update records, for every leaf cell, the mass flux
// it computed itself through its three lower faces, and through its upper faces at the edge of
// the block and next to a solid or fixed cell (which return before computing fluxes). These
// are the fluxes before any matching at refinement boundaries: amr/species.wgsl matches the
// species fluxes there itself, from the fine blocks' records. Requires amr/grid.wgsl, U2 and
// hydro/flux.wgsl (Faces).
const SPECIES = true;
@group(0) @binding(12) var<storage, read_write> MF: array<f32>;

#include "amr/species_index.wgsl"

fn store_mass_flux(slot: i32, l: vec3<i32>, c: vec3<i32>, fx: Faces, fy: Faces, fz: Faces) {
  let lo = vec3<f32>(fx.lo.m, fy.lo.m, fz.lo.m);
  let hi = vec3<f32>(fx.hi.m, fy.hi.m, fz.hi.m);
  let nax = select(3, 2, is2d());
  for (var a = 0; a < nax; a++) {
    MF[mf_index(slot, l, a)] = lo[a];
    var e = vec3<i32>(0);
    e[a] = 1;
    let up = l + e;
    let held = up[a] < BS && has_solids() && abs(textureLoad(U2, c + e, 0).w) > 0.5;
    if (up[a] == BS || held) { MF[mf_index(slot, up, a)] = hi[a]; }
  }
}
