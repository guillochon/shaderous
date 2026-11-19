// The radiation energy that has left the box, summed step by step (radiation/passes.ts): after
// the sweeps of a step, the luminosity through the open faces (as radiation/diagnostics.wgsl
// defines it: the outward F . n dA of the cells on them) times the step, added to ESC with
// Kahan's compensation. A light curve read from ESC between two readouts misses nothing that
// happened between them.
//
// One workgroup: each invocation sums every 256th cell of the faces, and the first adds them up.
#include "hydro/common.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var RR: texture_3d<f32>;
// [energy that has left, its compensation, the luminosity of this step, unused]
@group(0) @binding(4) var<storage, read_write> ESC: array<f32, 4>;

var<workgroup> part: array<f32, 256>;

fn open_face(k: u32) -> bool {
  return k == 1u || k == 3u;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lix: u32) {
  let n = vec3<i32>(P.dims.xyz);
  let flat = n.z == 1;
  var s = 0.0;
  for (var a = 0; a < 3; a++) {
    if (a == 2 && flat) { break; }
    let t1 = (a + 1) % 3;
    let t2 = (a + 2) % 3;
    let count = n[t1] * n[t2];
    for (var side = 0; side < 2; side++) {
      let kind = select(P.bc_lo[a], P.bc_hi[a], side == 1);
      if (!open_face(kind)) { continue; }
      for (var i = i32(lix); i < count; i += 256) {
        var c = vec3<i32>(0);
        c[a] = select(0, n[a] - 1, side == 1);
        c[t1] = i % n[t1];
        c[t2] = i / n[t1];
        if (textureLoad(U2, c, 0).w > 0.5) { continue; }
        let F = textureLoad(RR, c, 0).yzw[a];
        s += max(select(-F, F, side == 1), 0.0);
      }
    }
  }
  part[lix] = s;
  workgroupBarrier();
  if (lix != 0u) { return; }
  var sum = 0.0;
  for (var i = 0u; i < 256u; i++) { sum += part[i]; }
  let lum = sum * select(P.dx * P.dx, P.dx, flat);
  let y = lum * clock.dt - ESC[1];
  let t = ESC[0] + y;
  ESC[1] = (t - ESC[0]) - y;
  ESC[0] = t;
  ESC[2] = lum;
  // The time this total is of: the end of the step (the clock banks it at the next). A readout
  // made while the run goes on reads the clock and this record some steps apart.
  ESC[3] = clock.t + clock.dt;
}
