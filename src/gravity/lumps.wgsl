// The gas as lumps, for the preview of a point mass's orbit (gravity/preview.ts): the box cut
// into L.k regions along each axis, each summed to its mass at its center of mass. One
// invocation per lump. Solid and fixed cells are not gas (gravity/bh_moment.wgsl).
#include "hydro/common.wgsl"

struct LumpGrid { k: vec4<u32> }
// Layout read by gravity/lumps.ts.
struct Lump {
  pos: vec4<f32>,   // center of mass, relative to the box center; mass in w
  size: vec4<f32>,  // x: side of the region summed
}

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var<uniform> L: LumpGrid;
@group(0) @binding(4) var<storage, read_write> OUT: array<Lump>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = L.k.xyz;
  if (any(id >= k)) { return; }
  let lo = vec3<i32>(id * P.dims.xyz / k);
  let hi = vec3<i32>((id + 1u) * P.dims.xyz / k);
  let solids = has_solids();
  var m = 0.0;
  var mx = vec3<f32>(0.0);
  for (var z = lo.z; z < hi.z; z++) {
    for (var y = lo.y; y < hi.y; y++) {
      for (var x = lo.x; x < hi.x; x++) {
        let c = vec3<i32>(x, y, z);
        var rho = textureLoad(U1, c, 0).x;
        if (solids && abs(textureLoad(U2, c, 0).w) > 0.5) { rho = 0.0; }
        m += rho;
        mx += rho * ((vec3<f32>(c) + 0.5) * P.dx + P.org.xyz);
      }
    }
  }
  var pos = (vec3<f32>(lo + hi) * 0.5) * P.dx + P.org.xyz;
  if (m > 0.0) { pos = mx / m; }
  // A 2D run lies in the plane z = 0, where the masses are.
  if (P.dims.z == 1u) { pos.z = 0.0; }
  OUT[id.x + k.x * (id.y + k.y * id.z)] = Lump(vec4<f32>(pos, m * P.geom.x), vec4<f32>(P.dx * f32(P.dims.x) / f32(k.x), 0.0, 0.0, 0.0));
}
