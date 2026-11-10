// Hancock predictor: advances each cell's primitives half a step using its limited slopes,
// and records the two per-cell decisions the update needs.
//   PA = (rho, vx, vy, vz) at the half step
//   PB = (p, scalar at the half step, shock flag, first-order flag)
// Doing this once per cell, instead of seven times inside the update's gather, is the main
// reason the update pass is cheap.
#include "hydro/common.wgsl"
#include <limiter>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;
@group(0) @binding(3) var W2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;
@group(0) @binding(5) var PA: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var PB: texture_storage_3d<rgba32float, write>;

#include "hydro/recon.wgsl"
#include "gravity/gfield.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= vec3<u32>(P.n))) { return; }
  let c = vec3<i32>(gid);
  let q = load_q(c);
  let xm = load_q(c + vec3<i32>(-1, 0, 0));
  let xp = load_q(c + vec3<i32>(1, 0, 0));
  let ym = load_q(c + vec3<i32>(0, -1, 0));
  let yp = load_q(c + vec3<i32>(0, 1, 0));
  let zm = load_q(c + vec3<i32>(0, 0, -1));
  let zp = load_q(c + vec3<i32>(0, 0, 1));
  let sx = slope(xm, q, xp);
  let sy = slope(ym, q, yp);
  let sz = slope(zm, q, zp);

  // Shock detector: converging flow with a large relative pressure jump.
  let conv = (xp.a.y - xm.a.y) + (yp.a.z - ym.a.z) + (zp.a.w - zm.a.w);
  let jump = max(
    abs(xp.b.x - xm.b.x) / min(xp.b.x, xm.b.x),
    max(abs(yp.b.x - ym.b.x) / min(yp.b.x, ym.b.x), abs(zp.b.x - zm.b.x) / min(zp.b.x, zm.b.x)),
  );
  let shock = select(0.0, 1.0, conv < 0.0 && jump > P.shock_thresh);

  // Predictor in primitive form, plus half a gravity kick.
  let h = 0.5 * clock.dt / P.dx;
  let rho = q.a.x;
  let v = q.a.yzw;
  let cs = q.b.z;
  let dv = sx.a.y + sy.a.z + sz.a.w;
  let da = v.x * sx.a + v.y * sy.a + v.z * sz.a;
  let db = v.x * sx.b + v.y * sy.b + v.z * sz.b;
  let gradp = vec3<f32>(sx.b.x, sy.b.x, sz.b.x);

  var a = vec4<f32>(
    rho - h * (da.x + rho * dv),
    v - h * (da.yzw + gradp / rho) + 0.5 * clock.dt * g_load(c),
  );
  var b = vec2<f32>(q.b.x - h * (db.x + rho * cs * cs * dv), q.b.y - h * db.y);

  // Positivity: if any face state would be non-positive, this cell is first order this step
  // (unpredicted state, and the update zeroes its slopes).
  let rmin = a.x - 0.5 * max(abs(sx.a.x), max(abs(sy.a.x), abs(sz.a.x)));
  let pmin = b.x - 0.5 * max(abs(sx.b.x), max(abs(sy.b.x), abs(sz.b.x)));
  var first_order = 0.0;
  if (!(rmin > 0.0 && pmin > 0.0)) {
    a = q.a;
    b = q.b.xy;
    first_order = 1.0;
  }
  textureStore(PA, c, a);
  textureStore(PB, c, vec4<f32>(b, shock, first_order));
}
