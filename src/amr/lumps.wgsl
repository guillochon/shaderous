// The gas as lumps on a block grid (gravity/lumps.wgsl, for gravity/preview.ts): one lump per
// leaf block, its mass at its center of mass. One invocation per block of the leaf list.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"

struct Lump {
  pos: vec4<f32>,   // center of mass, relative to the box center; mass in w
  size: vec4<f32>,  // x: side of the block
}

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<Lump>;
// The leaf list (amr/list.wgsl): the count, then the slots.
@group(0) @binding(31) var<storage, read> LIST: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = id.x;
  if (b >= LIST[0]) { return; }
  let slot = i32(LIST[1u + b]);
  let o = blk_origin(slot);
  let dx = blk_dx(slot);
  let solids = has_solids();
  let nz = select(BS, 1, is2d());
  var m = 0.0;
  var mx = vec3<f32>(0.0);
  for (var z = 0; z < nz; z++) {
    for (var y = 0; y < BS; y++) {
      for (var x = 0; x < BS; x++) {
        let l = vec3<i32>(x, y, z);
        let c = atl(o, l);
        var rho = textureLoad(U1, c, 0).x;
        if (solids && abs(textureLoad(U2, c, 0).w) > 0.5) { rho = 0.0; }
        m += rho;
        mx += rho * blk_pos(slot, l);
      }
    }
  }
  var pos = blk_pos(slot, vec3<i32>(BS / 2)) - 0.5 * dx;
  if (m > 0.0) { pos = mx / m; }
  if (is2d()) { pos.z = 0.0; }
  let dv = select(dx * dx * dx, dx * dx, is2d());
  OUT[b] = Lump(vec4<f32>(pos, m * dv), vec4<f32>(dx * f32(BS), 0.0, 0.0, 0.0));
}
