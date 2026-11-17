// Nuclear burning by the trained network of the 13-isotope alpha chain (burn/network.ts): a
// pass of its own after the species pass, on the uniform grid and on the block atlas alike,
// in place of hydro/burn.wgsl. Pointwise: each cell is read, burned if it is hot and dense
// enough, and written to the other parity with its species (and, where the species set the
// equation of state, their moments), so a cell the pass does not burn is copied. On a uniform
// grid that is every texel; on the block atlas the interior cells of the leaves, the only cells
// read after it, as after the hydro update (<burn_grid>, burn/network.ts).
//
// The network works in cgs, kelvin and seconds; the cell's state is converted on the way in
// and the energy released on the way out (NB.a). A hydro step longer than NET_MAX_DT is split
// into equal network steps, the temperature taken again from the equation of state after
// each, with the composition as it then is. Cells the shock detector flags do not burn unless
// the problem allows it (as hydro/burn.wgsl).
//
// Burning limiter (Kushnir et al. 2013): energy injected faster than a sound crossing of the
// cell, t_s = dx / c_s, is numerically unstable on a grid much coarser than the burning, and
// lights detonations too early. With NB.d.x = f > 0, each network step of length h is scaled
// down where it would add more than f e h / t_s to the internal energy density e, all of its
// changes alike (as all rates scaled by one factor, to first order): mass stays conserved and
// no species goes below zero. It fades as the grid is refined. The cell size comes from the
// <burn_grid> slot: the grid's, or on the block atlas the cell's block's.
//
// HO gets every texel's heating rate over the step, the energy released per unit volume and
// time (0 where nothing burned), for the view's nuclear heating and the diagnostics' total.
// Float64 mirror of one network step: netStep() in src/burn/network.ts.
#include "hydro/common.wgsl"
#include <eos>
#include <species_n>
#include "hydro/burn_net_core.wgsl"
#include <burn_grid>

struct NetBurn {
  a: vec4<f32>,   // g/cm^3, kelvin, seconds per code unit; code specific energy per erg/g
  b: vec4<f32>,   // coolest T (K), least density (g/cm^3) that burn; longest network step (s); hottest T given (K)
  c: vec4<u32>,   // texels along z from one group of four species to the next, burn in shocks (1), most network steps per hydro step
  d: vec4<f32>,   // burning limiter f (0: none)
}

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var PB: texture_3d<f32>;   // predictor's (p, scalar, shock flag, first order)
@group(0) @binding(5) var S: texture_3d<f32>;    // rho X
@group(0) @binding(6) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var SO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var<uniform> NB: NetBurn;
// store_heat(c, rate): to HO (binding 11), or nowhere where the device has no storage texture to spare.
#include <burn_heat>

// rho X in groups of four, from rho and mass fractions.
fn grouped(rho: f32, X: array<f32, 13>) -> array<vec4<f32>, NG> {
  var u: array<vec4<f32>, NG>;
  for (var i = 0u; i < 13u; i++) { u[i / 4u][i % 4u] = rho * X[i]; }
  return u;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let at = burn_texel(gid);
  if (!at.ok) { return; }
  let c = at.c;
  let stride = i32(NB.c.x);
  var u1 = textureLoad(U1, c, 0);
  var u2 = textureLoad(U2, c, 0);
  var x: array<vec4<f32>, NG>;
  var sum = 0.0;
  for (var g = 0u; g < NG; g++) {
    x[g] = textureLoad(S, c + species_group(i32(g), stride), 0);
    sum += dot(max(x[g], vec4<f32>(0.0)), vec4<f32>(1.0));
  }
  let rho = u1.x;
  let shocked = NB.c.y == 0u && textureLoad(PB, c, 0).z > 0.5;
  // Solid and fixed cells keep what they hold.
  let held = abs(u2.w) > 0.5;
  let rho_cgs = rho * NB.a.x;
  var heat = 0.0;
  if (rho > 0.0 && sum > 0.0 && rho_cgs >= NB.b.y && !shocked && !held && EOS_HAS_ENERGY) {
    var X: array<f32, 13>;
    for (var i = 0u; i < 13u; i++) { X[i] = max(x[i / 4u][i % 4u], 0.0) / sum; }
    let e0 = u2.y;
    var e = e0;
    var dX: array<f32, 13>;
    let dt_s = clock.dt * NB.a.z;
    let steps = clamp(u32(ceil(dt_s / NB.b.z)), 1u, NB.c.z);
    let h = dt_s / f32(steps);
    let log_rho = log(rho_cgs) / LN10;
    let log_h = log(h) / LN10;
    var burned = false;
    for (var k = 0u; k < steps; k++) {
      var Xc: array<f32, 13>;
      for (var i = 0u; i < 13u; i++) { Xc[i] = max(X[i] + dX[i], 0.0); }
      if (COMP) { comp_use(comp_moments(grouped(1.0, Xc))); }
      let T = eos_temperature(rho, e) * NB.a.y;
      if (!(T >= NB.b.x)) { break; }
      var D = net_step(log_rho, log(min(T, NB.b.w)) / LN10, log_h, Xc);
      var de = 0.0;
      for (var i = 0u; i < 13u; i++) { de += NET_Q[i] * D[i]; }
      // 1e18 erg/g per unit of NET_Q.
      var dei = rho * de * 1e18 * NB.a.w;
      if (!(abs(dei) < 3e38)) { break; }
      if (NB.d.x > 0.0) {
        // At most f e h / t_s, h and t_s in code time.
        let cs = eos_sound_speed(rho, eos_pressure(rho, e));
        let cap = NB.d.x * e * (clock.dt / f32(steps)) * cs / burn_cell_dx(c);
        if (abs(dei) > cap && cap >= 0.0) {
          let s = cap / abs(dei);
          for (var i = 0u; i < 13u; i++) { D[i] *= s; }
          dei *= s;
        }
      }
      for (var i = 0u; i < 13u; i++) { dX[i] += D[i]; }
      e += dei;
      burned = true;
    }
    if (burned) {
      for (var i = 0u; i < 13u; i++) {
        let g = i / 4u;
        let j = i % 4u;
        x[g][j] = max(x[g][j] + rho * dX[i], 0.0);
      }
      if (clock.dt > 0.0) { heat = (e - e0) / clock.dt; }
      u2 = vec4<f32>(u2.x + (e - e0), e, u2.zw);
    }
  }
  store_heat(c, heat);
  textureStore(O1, c, u1);
  textureStore(O2, c, u2);
  for (var g = 0u; g < NG; g++) { textureStore(SO, c + species_group(i32(g), stride), x[g]); }
  if (COMP) { comp_store(c, comp_moments(x)); }
}
