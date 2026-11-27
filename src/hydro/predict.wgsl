// Hancock predictor pass on a uniform grid.
//   PA = (rho, vx, vy, vz) at the half step
//   PB = (p, scalar at the half step, shock flag, first-order flag)
// Doing this once per cell, instead of seven times inside the update's gather, is the main
// reason the update pass is cheap.
#include "hydro/common.wgsl"
#include "gravity/ext_bind.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;
@group(0) @binding(5) var PA: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var PB: texture_storage_3d<rgba32float, write>;

#include "hydro/region.wgsl"
#include "hydro/recon.wgsl"
#include "hydro/predictor.wgsl"
#include <psplit_grad>
#include "gravity/gfield.wgsl"

// n, as a loop bound the shader compiler cannot see through (P.dims.x is never 0), so that it
// keeps the loop a loop (loop_count in amr/grid.wgsl).
fn pred_count(n: i32) -> i32 {
  return select(n, 0, P.dims.x == 0u);
}

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rc = region_cell(gid);
  if (!rc.mine) { return; }
  set_proper_dx(P.dx, clock.ah);
  if (cosmo_on()) { g_a3 = clock.a * clock.a * clock.a; }
  g_dt = clock.dt;
  // A shearing box: the offset of its x faces, for the ghosts beyond them.
  if (shear_on()) { g_shear = clock.shear / P.dx; }
  let c = rc.c;
  // An axisymmetric run: the cell's radius, for the predictor's geometric terms.
  if (cylindrical()) { g_cyl_r = f32(c.x) + 0.5; }
  // A spherical run: the cell's lengths and its geometric terms.
  if (spherical()) {
    let sg = sph_cell(c.x, c.y);
    g_hr = vec3<f32>(sg.h, 1.0);
    g_sph_geo = vec2<f32>(sg.ir, sg.cot);
  }
  g_inner = stencil_inside(c, 1);
  let q = load_q(c);
  if (PSPLIT) { g_pgrad = pp_gradient(c); }
  // In 2D the layer is its own neighbor along z.
  var zm = q;
  var zp = q;
  var xm: Q;
  var xp: Q;
  var ym: Q;
  var yp: Q;
  let flat = P.dims.z == 1u;
  if (INTERIOR) {
    if (!flat) {
      zm = load_q(c + vec3<i32>(0, 0, -1));
      zp = load_q(c + vec3<i32>(0, 0, 1));
    }
    xm = load_q(c + vec3<i32>(-1, 0, 0));
    xp = load_q(c + vec3<i32>(1, 0, 0));
    ym = load_q(c + vec3<i32>(0, -1, 0));
    yp = load_q(c + vec3<i32>(0, 1, 0));
  } else {
    // The shell, a thin rim of the grid: its loads, each through the boundary mapping and the
    // equation of state, from one call site in a kept loop (with a costly equation of state the
    // six inlined took twice the interior's compile time). In the same order as the interior's,
    // so that the last composition set (comp_use) is the same.
    for (var k = select(0, 2, flat); k < pred_count(6); k++) {
      var o = vec3<i32>(0);
      o[(k / 2 + 2) % 3] = select(-1, 1, (k & 1) == 1);
      let n = load_q(c + o);
      if (k == 0) { zm = n; } else if (k == 1) { zp = n; } else if (k == 2) { xm = n; }
      else if (k == 3) { xp = n; } else if (k == 4) { ym = n; } else { yp = n; }
    }
  }
  // A gas stored as its deviation from a reference (hydro/ref_on.wgsl): the cell's reference,
  // and its density's difference across the cell along each axis.
  if (REF_DEV) {
    let r = ref_at(c);
    g_qs = r.xy;
    for (var a = 0; a < 3; a++) {
      var e = vec3<i32>(0);
      e[a] = 1;
      let rm = ref_at(c - e);
      let rp = ref_at(c + e);
      g_rds[a] = ref_slope(rm, r, rp);
      g_rlo_rho[a] = rm.x;
      g_rlo_p[a] = rm.y;
      g_rhi_rho[a] = rp.x;
      g_rhi_p[a] = rp.y;
    }
  }
  var pr = predictor(
    q,
    xm, xp,
    ym, yp,
    zm, zp,
    g_load(c), vec3<f32>(1.0), dphi_cell(c, -1), dphi_cell(c, 0),
  );
  if (COMP) {
    // Toward first order where two gases with different gammas meet (comp_jump).
    let k = load_comp(c);
    var w = max(max(comp_jump(k, load_comp(c + vec3<i32>(-1, 0, 0))), comp_jump(k, load_comp(c + vec3<i32>(1, 0, 0)))),
                max(comp_jump(k, load_comp(c + vec3<i32>(0, -1, 0))), comp_jump(k, load_comp(c + vec3<i32>(0, 1, 0)))));
    if (P.dims.z != 1u) {
      w = max(w, max(comp_jump(k, load_comp(c + vec3<i32>(0, 0, -1))), comp_jump(k, load_comp(c + vec3<i32>(0, 0, 1)))));
    }
    if (w > 0.0) { pr = toward_first_order(q, pr, w); }
  }
  textureStore(PA, c, pr.a);
  textureStore(PB, c, pr.b);
}
