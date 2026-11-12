// Species on re-meshing (after amr/regrid.wgsl has filled the new state): every leaf of the new
// tree from the old one. An unchanged block is copied; a merged block takes the average of its
// old children's rho X, as its density is their average; a new child takes its old parent's
// fractions times its own new density, which keeps them summing to one and within bounds. The
// composition moments, where they vary (species/mixture.ts), follow.
#include "amr/species_common.wgsl"

@group(0) @binding(2) var S: texture_3d<f32>;     // rho X on the old mesh
@group(0) @binding(4) var N1: texture_3d<f32>;    // the new state (density)
@group(0) @binding(6) var SO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> RG: array<i32>;

const RG_STRIDE = 12u;

fn load_s(c: vec3<i32>) -> array<vec4<f32>, NG> {
  var u: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { u[g] = textureLoad(S, sp(c, g), 0); }
  return u;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let l = inv.l;
  let r = u32(inv.slot) * RG_STRIDE;
  let kind = RG[r];
  let src = RG[r + 1u];
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  let c = atl(blk_origin(inv.slot), l);
  var u: array<vec4<f32>, NG>;
  if (kind == 2) {
    let oct = RG[r + 2u];
    let lc = (l >> vec3<u32>(1u)) + (BS / 2) * vec3<i32>(oct & 1, (oct >> 1u) & 1, (oct >> 2u) & 1);
    let x = fractions(load_s(atl(blk_origin(src), lc)));
    let rho = textureLoad(N1, c, 0).x;
    for (var g = 0u; g < NG; g++) { u[g] = rho * x[g]; }
  } else if (kind == 3) {
    let ch = RG[r + 4u + u32(h.x | (h.y << 1u) | (h.z << 2u))];
    let o = blk_origin(ch);
    let f = 2 * (l - (BS / 2) * h);
    let nz = select(2, 1, is2d());
    for (var k = 0; k < nz; k++) {
      for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
          let s = load_s(atl(o, f + vec3<i32>(i, j, k)));
          for (var g = 0u; g < NG; g++) { u[g] += s[g]; }
        }
      }
    }
    let w = select(0.125, 0.25, is2d());
    for (var g = 0u; g < NG; g++) { u[g] *= w; }
  } else {
    u = load_s(atl(blk_origin(src), l));
  }
  for (var g = 0u; g < NG; g++) { textureStore(SO, sp(c, g), u[g]); }
  if (COMP) { comp_store(c, comp_moments(u)); }
}
