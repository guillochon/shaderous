// The watch for burning (burn/watch.ts): a run that carries only he4, c12 and o16 does not burn,
// but has this pass where the network's would be, on the uniform grid and on the block atlas
// alike. Pointwise: each cell is copied to the other parity as it is (with its species), and,
// where it is hot and dense enough to burn and not flagged as a shock (as the network's pass),
// the energy the network's reactions would release in it over the step, as a share of its
// internal energy, goes into the largest such share since the CPU last read it (WMAX).
//
// The rate is net_watch_rate (the <burn_watch> slot, burn/network.ts netWatchSlot): the eight
// reactions a cell of these three species runs, each rate the term the network takes as an
// input. Float64 mirror: netEnergyRate() in src/burn/network.ts.
#include "hydro/common.wgsl"
#include <eos>
#include <species_n>
#include <burn_watch>
#include "hydro/burn_screen.wgsl"
#include <burn_grid>

struct NetWatch {
  a: vec4<f32>,   // g/cm^3, kelvin, seconds per code unit; code specific energy per erg/g
  b: vec4<f32>,   // coolest T (K), least density (g/cm^3) that would burn; hottest T given (K)
  c: vec4<u32>,   // texels along z from one group of four species to the next, burn in shocks (1)
}

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var PB: texture_3d<f32>;   // predictor's (p, scalar, shock flag, first order)
@group(0) @binding(5) var S: texture_3d<f32>;    // rho X: he4, c12, o16 in the first group
@group(0) @binding(6) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var SO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var<uniform> NW: NetWatch;
// The largest share since the CPU last read it, as the bits of a float (no share is below 0).
@group(0) @binding(12) var<storage, read_write> WMAX: array<atomic<u32>, 1>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let at = burn_texel(gid);
  if (!at.ok) { return; }
  let c = at.c;
  let stride = i32(NW.c.x);
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  var x: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { x[g] = textureLoad(S, c + species_group(i32(g), stride), 0); }
  textureStore(O1, c, u1);
  textureStore(O2, c, u2);
  for (var g = 0u; g < NG; g++) { textureStore(SO, c + species_group(i32(g), stride), x[g]); }

  let rho = u1.x;
  let e = u2.y;
  let rho_cgs = rho * NW.a.x;
  let shocked = NW.c.y == 0u && textureLoad(PB, c, 0).z > 0.5;
  let held = abs(u2.w) > 0.5;
  let x3 = max(x[0].xyz, vec3<f32>(0.0));
  let sum = x3.x + x3.y + x3.z;
  if (!(rho > 0.0) || !(e > 0.0) || !(sum > 0.0) || rho_cgs < NW.b.y || shocked || held || !EOS_HAS_ENERGY) { return; }
  let X = x3 / sum;
  if (COMP) {
    var m: array<vec4<f32>, NG>;
    m[0] = vec4<f32>(X, 0.0);
    comp_use(comp_moments(m));
  }
  let T = eos_temperature(rho, e) * NW.a.y;
  if (!(T >= NW.b.x)) { return; }
  // erg/g over the step, then a share of the cell's internal energy (code units).
  let de = net_watch_rate(rho_cgs, min(T, NW.b.z), X.x, X.y, X.z) * clock.dt * NW.a.z;
  let share = rho * de * NW.a.w / e;
  if (share > 0.0 && share < 3e38) { atomicMax(&WMAX[0], bitcast<u32>(share)); }
}
