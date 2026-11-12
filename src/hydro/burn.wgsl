// Nuclear burning, fuel to ash (burn/burn.ts): a pass of its own after the species pass, on
// the uniform grid and on the block atlas alike. Pointwise: every texel of the state is read,
// burned if it holds fuel, and written to the other parity (with the species), so a texel
// the pass leaves alone is copied. On the atlas that covers ghosts and coarse blocks too,
// which the next assembly overwrites.
//
// One reaction, single-step Arrhenius: the fuel fraction f falls as df/dt = -K f exp(-Ea / T),
// releasing q per unit mass burned into the internal energy at fixed density. T is the
// equation of state's (p / rho for the ideal gas in code units). The step is split into
// substeps, none releasing more than a fifth of the cell's internal energy, each solved
// exactly at its starting temperature (f e^(-k h): it never overshoots, however stiff). The
// last takes what is left of the step, so the burn always covers it; the timestep is never
// shortened for burning.
//
// A shock captured on the grid is a few cells of numerical smearing, through which the gas
// passes through states no real gas visits: burning there sets the front's speed by the
// scheme, not the reaction (Colella, Majda & Roytburd 1986). With B.a.w zero, cells the
// predictor flags as shocks (hydro/predictor.wgsl) do not burn, as FLASH's default.
// Float64 mirror: burnCell() in src/burn/burn.ts.
#include "hydro/common.wgsl"
#include <eos>
#include <species_n>

struct Burn {
  a: vec4<f32>,   // q, Ea, K, burn in shocks (1) or not (0)
  b: vec4<u32>,   // fuel species, ash species, texels along z from one group of four to the next
}

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var PB: texture_3d<f32>;   // predictor's (p, scalar, shock flag, first order)
@group(0) @binding(5) var S: texture_3d<f32>;    // rho X
@group(0) @binding(6) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var SO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var<uniform> B: Burn;

const BURN_SUBSTEPS = 8;

// 1 - e^(-x), accurate for small x as well.
fn one_minus_exp(x: f32) -> f32 {
  if (x < 1e-3) { return x * (1.0 - 0.5 * x * (1.0 - x / 3.0)); }
  return 1.0 - exp(-x);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(U1);
  if (any(gid >= dims)) { return; }
  let c = vec3<i32>(gid);
  let stride = i32(B.b.z);
  var u1 = textureLoad(U1, c, 0);
  var u2 = textureLoad(U2, c, 0);
  var x: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { x[g] = textureLoad(S, c + vec3<i32>(0, 0, i32(g) * stride), 0); }

  let fi = B.b.x;
  let ai = B.b.y;
  let rho = u1.x;
  let fuel = x[fi / 4u][fi % 4u];     // rho f
  let shocked = B.a.w < 0.5 && textureLoad(PB, c, 0).z > 0.5;
  // Solid and fixed cells keep what they hold.
  let held = abs(u2.w) > 0.5;
  if (rho > 0.0 && fuel > 0.0 && !shocked && !held && EOS_HAS_ENERGY) {
    let q = B.a.x;
    let ea = B.a.y;
    let K = B.a.z;
    var f = min(fuel / rho, 1.0);
    let f0 = f;
    var e = u2.y;
    var left = clock.dt;
    for (var i = 0; i < BURN_SUBSTEPS; i++) {
      if (left <= 0.0 || f <= 0.0) { break; }
      let T = eos_temperature(rho, e);
      if (!(T > 0.0)) { break; }
      let k = K * exp(-ea / T);
      if (k <= 0.0) { break; }
      // Fraction burned over the rest of the step at this temperature; if it would release
      // more than a fifth of the energy, a substep that releases just that.
      var h = left;
      let dmax = 0.2 * e / (rho * q);
      if (i < BURN_SUBSTEPS - 1 && f * one_minus_exp(k * left) > dmax && dmax < f) {
        h = -log(1.0 - dmax / f) / k;
      }
      let df = f * one_minus_exp(k * h);
      f -= df;
      e += rho * q * df;
      left -= h;
    }
    let burned = rho * (f0 - f);
    if (burned > 0.0) {
      x[fi / 4u][fi % 4u] = max(fuel - burned, 0.0);
      x[ai / 4u][ai % 4u] += burned;
      u2 = vec4<f32>(u2.x + (e - u2.y), e, u2.zw);
    }
  }
  textureStore(O1, c, u1);
  textureStore(O2, c, u2);
  for (var g = 0u; g < NG; g++) { textureStore(SO, c + vec3<i32>(0, 0, i32(g) * stride), x[g]); }
}
