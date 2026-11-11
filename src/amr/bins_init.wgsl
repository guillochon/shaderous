// First guess at every leaf block's bin, at the start of a cycle: the largest power of two of
// the clock's dt that the block's own signal rate and acceleration allow. The rates are the
// records the assembly pass wrote, three by three by five workgroups per block along the list.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read> RATES: array<f32>;
@group(0) @binding(3) var<storage, read_write> KOUT: array<u32>;

override KMAX: u32 = 3u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= LIST[0]) { return; }
  let slot = i32(LIST[1u + i]);
  var k = KMAX;
  if (blk_leaf(slot) && clock.dt > 0.0) {
    let a = P.amr.x;
    let nx = 3u * a * a * P.amr.w;   // 3 per slot
    let nz = select(5u, 1u, is2d());
    var rate = 0.0;
    var g = 0.0;
    for (var z = 0u; z < nz; z++) {
      for (var y = 0u; y < 3u; y++) {
        for (var x = 3u * i; x < 3u * i + 3u; x++) {
          let base = ((z * 3u + y) * nx + x) * 2u;
          rate = max(rate, RATES[base]);
          g = max(g, RATES[base + 1u]);
        }
      }
    }
    // The block's own limit, as hydro/clock.wgsl forms the global one.
    var dtb = P.cfl / max(rate, 1.0e-30);
    if (g > 0.0) { dtb = min(dtb, P.cfl * sqrt(1.0 / g)); }
    k = u32(clamp(floor(log2(dtb / clock.dt)), 0.0, f32(KMAX)));
  }
  KOUT[1u + u32(slot)] = k;
}
