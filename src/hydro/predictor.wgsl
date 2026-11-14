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
  var m = array<Q, 3>(xm0, ym0, zm0);
  var p = array<Q, 3>(xp0, yp0, zp0);
  // Per axis, in one loop: compilers inline every call, so the edge differences (two limited
  // slopes each) written out for each axis were compiled three times over; this takes 5-15%
  // off the predict passes' compile time, for no change in step time. (The neighbors' loads
  // are left written out where they are made: in a loop they are issued one after another,
  // and passes bound by memory traffic were 3-10% slower.)
  // Face-to-face difference across the cell along each axis (the mean of its two edge
  // increments), which drives the half-step evolution; and the larger of the two, for the
  // positivity check below.
  var s: array<Q, 3>;
  var big = vec2<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    if (has_solids()) {
      // A solid neighbor is a wall: the cell sees its own image there.
      // (The potential is mirrored with it: no difference across the wall.)
      if (q_solid(m[a])) { m[a] = q_mirror(q, a); dl[a] = 0.0; }
      if (q_solid(p[a])) { p[a] = q_mirror(q, a); dr[a] = 0.0; }
    }
    let u = edge_up(m[a], q, p[a], dl[a], dr[a], hi.face[a]);
    var sa = u;
    big = max(big, vec2<f32>(abs(u.a.x), abs(u.b.x)));
    if (!LIMITER_SYMMETRIC) {
      let d = edge_dn(m[a], q, p[a], dl[a], dr[a], lo.face[a]);
      sa = Q(0.5 * (u.a + d.a), 0.5 * (u.b + d.b));
      big = max(big, vec2<f32>(abs(d.a.x), abs(d.b.x)));
    }
    s[a] = Q(sa.a * mask[a], sa.b * mask[a]);
  }
  let xm = m[0];
  let xp = p[0];
  let ym = m[1];
  let yp = p[1];
  let zm = m[2];
  let zp = p[2];
  let sx = s[0];
  let sy = s[1];
  let sz = s[2];

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
