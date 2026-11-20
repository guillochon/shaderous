// Forces on bodies (plans/wind-tunnel.md 4.2), uniform grid: the momentum each gas cell's wall
// faces hand to the solid cells beyond them this step, as the update forms it. The flux through a
// wall face is the momentum the gas loses there, so the sum over a body's faces is the force on
// it, the very momentum the gas lost: a control volume around the body balances it to round-off
// (test force-balance).
//
// WP: the pressure's part (the flux's momentum less viscosity's), times the face's area, summed
// over the cell's wall faces, and the mark of the body they belong to (1 + its index; the
// larger where a cell touches two). WV: viscosity's part. With implicit pressure the update's
// wall flux is advection's (nothing: a wall's mirror image carries no mass) and viscosity's, and
// the pressure's part is the correction's (pressure/pwalls_on.wgsl), so the update writes WV
// alone, with the whole flux in it. Every cell is written each step (zero away from bodies).
// Requires P, U2, the Flux and Faces of hydro/flux.wgsl, PSPLIT and VISCOSITY.
@group(0) @binding(56) var WP: texture_storage_3d<rgba32float, write>;
@group(0) @binding(57) var WV: texture_storage_3d<rgba32float, write>;

// The mark of the cell across face (a, up) of cell c if it is solid, else 0. A face of the box is
// no body's, unless it is periodic (the body is across it).
fn wall_mark(c: vec3<i32>, a: i32, up: bool) -> f32 {
  var q = c;
  q[a] += select(-1, 1, up);
  let n = vec3<i32>(P.dims.xyz);
  if (q[a] < 0 || q[a] >= n[a]) {
    if (select(P.bc_lo[a], P.bc_hi[a], up) != BC_PERIODIC) { return 0.0; }
    q[a] = (q[a] + n[a]) % n[a];
  }
  let m = textureLoad(U2, q, 0).w;
  return select(0.0, m, m > 0.5);
}

struct WallSum { p: vec3<f32>, v: vec3<f32>, body: f32 }

// One face's flux, if a body is across it: + for an upper face (the gas loses f.mom there), -
// for a lower one.
fn wall_add(s: WallSum, f: Flux, m: f32, sgn: f32) -> WallSum {
  if (m < 0.5) { return s; }
  return WallSum(s.p + sgn * (f.mom - f.d.xyz), s.v + sgn * f.d.xyz, max(s.body, m));
}

fn store_wall_forces(c: vec3<i32>, fx: Faces, fy: Faces, fz: Faces, flat: bool) {
  var s = WallSum(vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
  if (has_solids()) {
    s = wall_add(s, fx.hi, wall_mark(c, 0, true), 1.0);
    s = wall_add(s, fx.lo, wall_mark(c, 0, false), -1.0);
    s = wall_add(s, fy.hi, wall_mark(c, 1, true), 1.0);
    s = wall_add(s, fy.lo, wall_mark(c, 1, false), -1.0);
    if (!flat) {
      s = wall_add(s, fz.hi, wall_mark(c, 2, true), 1.0);
      s = wall_add(s, fz.lo, wall_mark(c, 2, false), -1.0);
    }
  }
  // A face's area: the cell's volume over its side (per unit length along z in 2D).
  let area = P.geom.x / P.dx;
  if (PSPLIT) {
    textureStore(WV, c, vec4<f32>((s.p + s.v) * area, s.body));
    return;
  }
  textureStore(WP, c, vec4<f32>(s.p * area, s.body));
  if (VISCOSITY) { textureStore(WV, c, vec4<f32>(s.v * area, s.body)); }
}

// A solid or held cell: no force of its own.
fn store_wall_none(c: vec3<i32>) {
  if (!PSPLIT) { textureStore(WP, c, vec4<f32>(0.0)); }
  if (VISCOSITY || PSPLIT) { textureStore(WV, c, vec4<f32>(0.0)); }
}
