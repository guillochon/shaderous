// Reduction source for the point masses: gas mass, each mass's total pull on the gas, and what
// each sink takes this step (gravity/bh_cell.wgsl).
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> BH: Masses;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var<storage, read> clock: Clock;

#include "gravity/bh_cell.wgsl"

// A mass's pull on a cell's gas, per unit mass. With implicit pressure (P.ohm.y) the gas is not
// kicked by the pull at its cell's center but by the mean of its two faces' kicks along each
// axis (pressure/kick.wgsl), and this is that, face for face, from the advected state the
// correction kicked: the masses are then given back exactly what the gas was given
// (gravity/bh_kick.wgsl), and the momentum of the two together is conserved to round-off.
fn bh_pull(x: vec3<f32>, b: Mass) -> vec3<f32> {
  let at = bh_accel(x, b);
  if (!(P.ohm.y > 0.0)) { return at; }
  let c = g_rc.xyz;
  let n = vec3<i32>(P.dims.xyz);
  var g = vec3<f32>(0.0);
  let axes = select(3, 2, P.dims.z == 1u);
  for (var a = 0; a < axes; a++) {
    for (var u = 0; u < 2; u++) {
      let up = u == 1;
      // (pressure/faces.wgsl pface: gas across the face, or not.)
      var q = c;
      q[a] += select(-1, 1, up);
      var gas = true;
      if (q[a] < 0 || q[a] >= n[a]) {
        if (select(P.bc_lo[a], P.bc_hi[a], up) == BC_PERIODIC) { q[a] = select(n[a] - 1, 0, up); } else { gas = false; }
      }
      if (gas && abs(textureLoad(U2, q, 0).w) > 0.5) { gas = false; }
      var k = at[a];
      if (gas) {
        let lo = select(q, c, up);
        let hi = select(c, q, up);
        if (hi[a] < lo[a]) {
          k = 0.5 * (bh_accel(cell_pos(lo), b)[a] + bh_accel(cell_pos(hi), b)[a]);
        } else {
          var d = vec3<f32>(0.0);
          d[a] = P.dx;
          k = -bh_dphi(b, cell_pos(lo), d, 0.5 * clock.dt) / P.dx;
        }
      }
      g[a] += 0.5 * k;
    }
  }
  return g;
}

fn red_cell(c: vec3<i32>) -> Red {
  g_rc = vec4<i32>(c, 0);
  var u1 = textureLoad(U1, c, 0);
  // Solid and fixed cells are not gas the masses can pull on or take.
  if (abs(textureLoad(U2, c, 0).w) > 0.5) { u1 = vec4<f32>(0.0); }
  // (An axisymmetric run's cell is a ring of volume 2 pi R dx^2.)
  let vol = select(P.geom.x, 6.2831853 * (f32(c.x) + 0.5) * P.dx * P.dx * P.dx, cylindrical());
  return bh_cell(u1, vol, cell_pos(c), clock.dt);
}
