// After re-meshing with a magnetic field: the total energy of the cells of new and merged blocks
// rebuilt from their kinetic and internal energy and the new field's B^2 / 2 (the field's energy
// is not conserved by prolongation or restriction, the gas's is), and the state and faces moved
// back to the parity the run was in. Copied blocks keep their state bit for bit.
// One invocation per texel [0, BS] of each new leaf: its faces, and its cell where there is one.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;   // the re-meshed state (amr/regrid.wgsl's output)
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var BF: texture_3d<f32>;   // the re-meshed faces (mhd/amr_regrid.wgsl's)
@group(0) @binding(4) var BC: texture_3d<f32>;
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> RG: array<i32>;
@group(0) @binding(8) var OB: texture_storage_3d<rgba32float, write>;
@group(0) @binding(10) var OC: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let row = flat_row(wid);
  let w = BS + 1;
  let n = select(w * w * w, w * w, is2d());
  let i = i32(wid.x * FLAT_WG + lix);
  if (row >= LIST[0] || i >= n) { return; }
  let slot = i32(LIST[1u + row]);
  var l = vec3<i32>(i % w, (i / w) % w, i / (w * w));
  if (is2d()) { l.z = 0; }
  let o = blk_origin(slot);
  let x = atl(o, l);
  let lo = textureLoad(BF, x, 0);
  textureStore(OB, x, lo);
  textureStore(OC, x, textureLoad(BC, x, 0));
  let top = vec3<i32>(BS, BS, select(BS, 1, is2d()));
  if (any(l >= top)) { return; }
  var u1 = textureLoad(U1, x, 0);
  var u2 = textureLoad(U2, x, 0);
  let kind = RG[u32(slot) * 12u];
  if (kind == 2 || kind == 3) {
    var B = lo.xyz;
    for (var a = 0; a < 3; a++) {
      if (a == 2 && is2d()) { break; }
      var e = vec3<i32>(0);
      e[a] = 1;
      B[a] = 0.5 * (lo[a] + textureLoad(BF, atl(o, l + e), 0)[a]);
    }
    let rho = max(u1.x, P.rho_floor);
    u2.x = 0.5 * dot(u1.yzw, u1.yzw) / rho + u2.y + 0.5 * dot(B, B);
  }
  textureStore(O1, x, u1);
  textureStore(O2, x, u2);
}
