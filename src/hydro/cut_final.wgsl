// Partly filled cells, last of three passes (hydro/cut_common.wgsl): each gas cell's state, the
// mean of the neighborhood means it is in (its own, and the small cells' about it that hold it),
// back into the state the correction wrote. The liquid's energies follow its density. Solid and
// held cells keep theirs.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/cut_flags.wgsl"
#include "hydro/cut_common.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(4) var CN: texture_3d<f32>;
@group(0) @binding(6) var UH1: texture_3d<f32>;
@group(0) @binding(7) var UH2: texture_3d<f32>;
@group(0) @binding(8) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var O2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  if (cut_at(c).x <= 0.0) { return; }
  // A step of nothing (the run holding at a stop): nothing to redistribute. Averaging again at
  // every such step smoothed the flow along the wall each time a run stopped (a runner's output
  // every 0.1 changed a rotor's torque by a factor of 2.5).
  if (!(clock.dt > 0.0)) { return; }
  let h2 = textureLoad(UH2, c, 0);
  if (abs(h2.w) > 0.5) { return; }
  var a = textureLoad(UH1, c, 0);
  var s = h2.x;
  let zr = select(1, 0, P.dims.z == 1u);
  for (var dz = -zr; dz <= zr; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let o = vec3<i32>(dx, dy, dz);
        if (all(o == vec3<i32>(0))) { continue; }
        let q = c - o;
        if (cut_in_hood(q, o)) {
          let qq = cut_cell(q);
          a += textureLoad(UH1, qq, 0);
          s += textureLoad(UH2, qq, 0).x;
        }
      }
    }
  }
  let n = textureLoad(CN, c, 0).x;
  a /= n;
  s /= n;
  let rho = max(a.x, P.rho_floor);
  let eint = eos_eint_from_p(rho, eos_pressure(rho, 0.0));
  textureStore(O1, c, vec4<f32>(rho, a.yzw));
  textureStore(O2, c, vec4<f32>(0.5 * dot(a.yzw, a.yzw) / rho + eint, eint, s, 0.0));
}
