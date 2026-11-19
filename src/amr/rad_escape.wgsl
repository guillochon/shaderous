// The radiation energy that has left the box, summed step by step, on the block grid:
// radiation/escape.wgsl's, over the leaves' cells on the open faces of the domain, of whatever
// level. One workgroup: each invocation takes every 256th leaf of the list.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var RR: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> ESC: array<f32, 4>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lix: u32) {
  let flat = is2d();
  let depth = select(BS, 1, flat);
  var s = 0.0;
  for (var b = lix; b < LIST[0]; b += 256u) {
    let slot = i32(LIST[1u + b]);
    let o = blk_origin(slot);
    let dx = blk_dx(slot);
    let area = select(dx * dx, dx, flat);
    for (var a = 0; a < 3; a++) {
      if (a == 2 && flat) { break; }
      let t1 = (a + 1) % 3;
      let t2 = (a + 2) % 3;
      for (var side = 0; side < 2; side++) {
        // An open face of the domain: outflow (-2) or the diode (-4).
        let nb = bt(slot, 6u + u32(2 * a + side));
        if (nb != -2 && nb != -4) { continue; }
        var sum = 0.0;
        for (var j = 0; j < BS; j++) {
          for (var i = 0; i < BS; i++) {
            var l = vec3<i32>(0);
            l[a] = select(0, BS - 1, side == 1);
            l[t1] = i;
            l[t2] = j;
            if (l.z >= depth) { continue; }
            let c = atl(o, l);
            if (textureLoad(U2, c, 0).w > 0.5) { continue; }
            let F = textureLoad(RR, c, 0).yzw[a];
            sum += max(select(-F, F, side == 1), 0.0);
          }
        }
        s += sum * area;
      }
    }
  }
  part[lix] = s;
  workgroupBarrier();
  if (lix != 0u) { return; }
  var lum = 0.0;
  for (var i = 0u; i < 256u; i++) { lum += part[i]; }
  let y = lum * clock.dt - ESC[1];
  let t = ESC[0] + y;
  ESC[1] = (t - ESC[0]) - y;
  ESC[0] = t;
  ESC[2] = lum;
  // The time this total is of: the end of the step (the clock banks it at the next). A readout
  // made while the run goes on reads the clock and this record some steps apart.
  ESC[3] = clock.t + clock.dt;
}
