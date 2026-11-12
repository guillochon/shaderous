// Hancock predictor for one cell: advances its primitives half a step using its limited
// slopes, and records the two per-cell decisions the update needs. Returns
//   a = (rho, vx, vy, vz) at the half step
//   b = (p, scalar at the half step, shock flag, first-order flag)
// `mask` multiplies the slopes per axis; it is 1 except where a neighbor along that axis is
// unavailable (ghost cells at the edge of a block next to a coarser one).
// `lo` and `hi` are the potential differences from the lower neighbor to the cell and from the
// cell to the upper neighbor, for the well-balanced pressure slopes (hydro/q.wgsl): zero
// when that is off.
// Requires P, hydro/q.wgsl, and g_dx and g_dt set.

// The half-step state `pr` of cell q moved toward first order by weight w: at w = 1 the cell as
// it is with no slopes, as the positivity check below makes it.
fn toward_first_order(q: Q, pr: Q, w: f32) -> Q {
  return Q(mix(pr.a, q.a, w), vec4<f32>(mix(pr.b.xy, q.b.xy, w), pr.b.z, max(pr.b.w, w)));
}

fn predictor(q: Q, xm0: Q, xp0: Q, ym0: Q, yp0: Q, zm0: Q, zp0: Q, g: vec3<f32>, mask: vec3<f32>, lo: PhiSteps, hi: PhiSteps) -> Q {
  var dl = lo.seen;
  var dr = hi.seen;
  var xm = xm0;
  var xp = xp0;
  var ym = ym0;
  var yp = yp0;
  var zm = zm0;
  var zp = zp0;
  if (has_solids()) {
    // A solid neighbor is a wall: the cell sees its own image there.
    // (The potential is mirrored with it: no difference across the wall.)
    if (q_solid(xm)) { xm = q_mirror(q, 0); dl.x = 0.0; }
    if (q_solid(xp)) { xp = q_mirror(q, 0); dr.x = 0.0; }
    if (q_solid(ym)) { ym = q_mirror(q, 1); dl.y = 0.0; }
    if (q_solid(yp)) { yp = q_mirror(q, 1); dr.y = 0.0; }
    if (q_solid(zm)) { zm = q_mirror(q, 2); dl.z = 0.0; }
    if (q_solid(zp)) { zp = q_mirror(q, 2); dr.z = 0.0; }
  }
  // Face-to-face difference across the cell along each axis (the mean of its two edge
  // increments), which drives the half-step evolution; and the larger of the two, for the
  // positivity check below.
  let ux = edge_up(xm, q, xp, dl.x, dr.x, hi.face.x);
  let uy = edge_up(ym, q, yp, dl.y, dr.y, hi.face.y);
  let uz = edge_up(zm, q, zp, dl.z, dr.z, hi.face.z);
  var sx = ux;
  var sy = uy;
  var sz = uz;
  var big = vec2<f32>(max(abs(ux.a.x), max(abs(uy.a.x), abs(uz.a.x))), max(abs(ux.b.x), max(abs(uy.b.x), abs(uz.b.x))));
  if (!LIMITER_SYMMETRIC) {
    let lx = edge_dn(xm, q, xp, dl.x, dr.x, lo.face.x);
    let ly = edge_dn(ym, q, yp, dl.y, dr.y, lo.face.y);
    let lz = edge_dn(zm, q, zp, dl.z, dr.z, lo.face.z);
    sx = Q(0.5 * (ux.a + lx.a), 0.5 * (ux.b + lx.b));
    sy = Q(0.5 * (uy.a + ly.a), 0.5 * (uy.b + ly.b));
    sz = Q(0.5 * (uz.a + lz.a), 0.5 * (uz.b + lz.b));
    big = max(big, vec2<f32>(max(abs(lx.a.x), max(abs(ly.a.x), abs(lz.a.x))), max(abs(lx.b.x), max(abs(ly.b.x), abs(lz.b.x)))));
  }
  sx = Q(sx.a * mask.x, sx.b * mask.x);
  sy = Q(sy.a * mask.y, sy.b * mask.y);
  sz = Q(sz.a * mask.z, sz.b * mask.z);

  // Shock detector: converging flow with a large relative pressure jump.
  let conv = (xp.a.y - xm.a.y) + (yp.a.z - ym.a.z) + (zp.a.w - zm.a.w);
  // Measured against hydrostatic balance (hydro/q.wgsl): a steep atmosphere at rest is not a
  // shock, however much its pressure changes across two cells. (hs is zero without gravity.)
  let hs = 0.5 * vec3<f32>(
    (xm.a.x + q.a.x) * dl.x + (q.a.x + xp.a.x) * dr.x,
    (ym.a.x + q.a.x) * dl.y + (q.a.x + yp.a.x) * dr.y,
    (zm.a.x + q.a.x) * dl.z + (q.a.x + zp.a.x) * dr.z,
  );
  let jump = max(
    abs(xp.b.x - xm.b.x + hs.x) / min(xp.b.x, xm.b.x),
    max(abs(yp.b.x - ym.b.x + hs.y) / min(yp.b.x, ym.b.x), abs(zp.b.x - zm.b.x + hs.z) / min(zp.b.x, zm.b.x)),
  );
  let shock = select(0.0, 1.0, conv < 0.0 && jump > P.shock_thresh);

  // Predictor in primitive form, plus half a gravity kick.
  let h = 0.5 * g_dt / g_dx;
  let rho = q.a.x;
  let v = q.a.yzw;
  let cs = q.b.z;
  let dv = sx.a.y + sy.a.z + sz.a.w;
  let da = v.x * sx.a + v.y * sy.a + v.z * sz.a;
  let db = v.x * sx.b + v.y * sy.b + v.z * sz.b;
  let gradp = vec3<f32>(sx.b.x, sy.b.x, sz.b.x);

  var a = vec4<f32>(
    rho - h * (da.x + rho * dv),
    v - h * (da.yzw + gradp / rho) + 0.5 * g_dt * g,
  );
  var b = vec2<f32>(q.b.x - h * (db.x + rho * cs * cs * dv), q.b.y - h * db.y);

  // Positivity: if any face state would be non-positive, this cell is first order this step
  // (unpredicted state, and the update zeroes its slopes).
  let rmin = a.x - 0.5 * big.x;
  let pmin = b.x - 0.5 * big.y;
  var first_order = 0.0;
  if (!(rmin > 0.0 && pmin > 0.0)) {
    a = q.a;
    b = q.b.xy;
    first_order = 1.0;
  }
  return Q(a, vec4<f32>(b, shock, first_order));
}
