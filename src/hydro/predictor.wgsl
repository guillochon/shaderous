// Hancock predictor for one cell: advances its primitives half a step using its limited
// slopes, and records the two per-cell decisions the update needs. Returns
//   a = (rho, vx, vy, vz) at the half step
//   b = (p, scalar at the half step, shock flag, first-order flag)
// `mask` multiplies the slopes per axis; it is 1 except where a neighbour along that axis is
// unavailable (ghost cells at the edge of a block next to a coarser one).
// Requires P, clock, hydro/q.wgsl, and g_dx set.
fn predictor(q: Q, xm: Q, xp: Q, ym: Q, yp: Q, zm: Q, zp: Q, g: vec3<f32>, mask: vec3<f32>) -> Q {
  var sx = slope(xm, q, xp);
  var sy = slope(ym, q, yp);
  var sz = slope(zm, q, zp);
  sx = Q(sx.a * mask.x, sx.b * mask.x);
  sy = Q(sy.a * mask.y, sy.b * mask.y);
  sz = Q(sz.a * mask.z, sz.b * mask.z);

  // Shock detector: converging flow with a large relative pressure jump.
  let conv = (xp.a.y - xm.a.y) + (yp.a.z - ym.a.z) + (zp.a.w - zm.a.w);
  let jump = max(
    abs(xp.b.x - xm.b.x) / min(xp.b.x, xm.b.x),
    max(abs(yp.b.x - ym.b.x) / min(yp.b.x, ym.b.x), abs(zp.b.x - zm.b.x) / min(zp.b.x, zm.b.x)),
  );
  let shock = select(0.0, 1.0, conv < 0.0 && jump > P.shock_thresh);

  // Predictor in primitive form, plus half a gravity kick.
  let h = 0.5 * clock.dt / g_dx;
  let rho = q.a.x;
  let v = q.a.yzw;
  let cs = q.b.z;
  let dv = sx.a.y + sy.a.z + sz.a.w;
  let da = v.x * sx.a + v.y * sy.a + v.z * sz.a;
  let db = v.x * sx.b + v.y * sy.b + v.z * sz.b;
  let gradp = vec3<f32>(sx.b.x, sy.b.x, sz.b.x);

  var a = vec4<f32>(
    rho - h * (da.x + rho * dv),
    v - h * (da.yzw + gradp / rho) + 0.5 * clock.dt * g,
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
  return Q(a, vec4<f32>(b, shock, first_order));
}
