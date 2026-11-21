// Forces on bodies with implicit pressure (hydro/walls_on.wgsl): the pressure's part, which the
// correction applies, summed over each gas cell's faces on a body as the correction forms them
// (the face pressure the step's momentum change is from), times the face's area, with the body's
// mark. Every cell is written (zero away from bodies). A gas's pressure is taken less the solve's
// reference pressure (pressure/source_gas.wgsl): a constant pressure has no net force on a closed
// body, and the sum keeps its digits for the part that does. Requires P and U2S.
@group(0) @binding(56) var WP: texture_storage_3d<rgba32float, write>;
#include "hydro/cut_flags.wgsl"

var<private> g_pw: vec4<f32>;

// Face (a, up) of cell c, a wall at pressure p: counted if a body's cell is across it.
fn pwall_face(c: vec3<i32>, a: i32, up: bool, p: f32) {
  if (!has_solids()) { return; }
  var q = c;
  q[a] += select(-1, 1, up);
  let n = vec3<i32>(P.dims.xyz);
  if (q[a] < 0 || q[a] >= n[a]) {
    if (select(P.bc_lo[a], P.bc_hi[a], up) != BC_PERIODIC) { return; }
    q[a] = (q[a] + n[a]) % n[a];
  }
  let m = textureLoad(U2S, q, 0).w;
  if (m < 0.5) { return; }
  g_pw[a] += select(-1.0, 1.0, up) * p * P.geom.x / P.dx;
  g_pw.w = max(g_pw.w, m);
}

// Partly filled cells (hydro/cut.ts): the wall's share w of the cell's boundary across axis a
// (its vector by closure, out of the gas), at the cell's pressure p, on the body of the cell's mark.
fn pwall_cut(c: vec3<i32>, a: i32, w: f32, p: f32) {
  if (w == 0.0) { return; }
  g_pw[a] += w * p * P.geom.x / P.dx;
  g_pw.w = max(g_pw.w, cut_wall(c).y);
}

fn pwall_store(c: vec3<i32>) {
  textureStore(WP, c, g_pw);
}
