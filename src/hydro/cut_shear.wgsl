// Partly filled cells, after the pressure's correction (hydro/cut.ts), first of three passes:
// each gas cell's count of neighborhoods (hydro/cut_common.wgsl), and in a cell the wall cuts,
// the wall's shear on the gas: from the velocity at a point CUT_PROBE cells out from the wall
// along its normal (trilinear among the cells with gas about it, those under a quarter full left
// out: their state is yet to be redistributed), over that distance, as the
// prototype's (reference/cutcell.ts). Its change of the momentum density goes to DM, which the
// next pass adds, and the force on the body (the change's opposite, times the gas's volume, over
// dt) to WV, which this pass writes for every cell. Bodies are at rest.
#include "hydro/common.wgsl"
#include "hydro/cut_flags.wgsl"
#include "hydro/cut_common.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(4) var CN: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var DM: texture_storage_3d<rgba32float, write>;
@group(0) @binding(57) var WV: texture_storage_3d<rgba32float, write>;

fn velocity_at(c: vec3<i32>) -> vec3<f32> {
  let u = textureLoad(U1, cut_cell(c), 0);
  return u.yzw / max(u.x, P.rho_floor);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let k = cut_at(c);
  // (A step of nothing, holding at a stop, keeps the last step's force: pressure/pwalls_on.wgsl.)
  let held = !(clock.dt > 0.0);
  if (k.x <= 0.0) {
    textureStore(CN, c, vec4<f32>(1.0, 0.0, 0.0, 0.0));
    textureStore(DM, c, vec4<f32>(0.0));
    if (!held) { textureStore(WV, c, vec4<f32>(0.0)); }
    return;
  }
  textureStore(CN, c, vec4<f32>(cut_count(c), 0.0, 0.0, 0.0));
  let axes = cut_axes();
  let w = cut_wall_vector(c, axes);
  let wl = length(w);
  let nu = P.diff.x;
  let dt = clock.dt;
  if (wl == 0.0 || nu <= 0.0) {
    textureStore(DM, c, vec4<f32>(0.0));
    if (!held) { textureStore(WV, c, vec4<f32>(0.0)); }
    return;
  }
  // The wall's nearest point and the probe, in cells (the cell's center at c + 1/2).
  let n = w / wl;
  let d = cut_wall(c).x / P.dx;
  let xp = vec3<f32>(c) + 0.5 + (d - CUT_PROBE) * n;
  let f = xp - 0.5;
  let i0 = vec3<i32>(floor(f));
  let t = f - floor(f);
  var vs = vec3<f32>(0.0);
  var ws = 0.0;
  let zr = select(1, 0, axes == 2);
  for (var cz = 0; cz <= zr; cz++) {
    for (var cy = 0; cy <= 1; cy++) {
      for (var cx = 0; cx <= 1; cx++) {
        let q = i0 + vec3<i32>(cx, cy, cz);
        var wt = select(1.0 - t.x, t.x, cx == 1) * select(1.0 - t.y, t.y, cy == 1);
        if (axes == 3) { wt *= select(1.0 - t.z, t.z, cz == 1); }
        let phq = cut_at(q).x;
        if (phq <= 0.0 || wt <= 0.0) { continue; }
        // (A cell under a quarter full holds a state redistribution has yet to average: not taken.)
        if (phq < 0.25) { continue; }
        vs += wt * velocity_at(q);
        ws += wt;
      }
    }
  }
  let vp = select(vec3<f32>(0.0), vs / ws, ws > 0.0);
  let rho = textureLoad(U1, c, 0).x;
  // The momentum density's rate of change, nu rho |W| (v_wall - v_probe) / (h phi dx), and over
  // the step (none at a stop, dt = 0, where the force is still the rate's: it is read there).
  let rate = -nu * rho * wl * vp / (CUT_PROBE * P.dx * k.x * P.dx);
  textureStore(DM, c, vec4<f32>(select(vec3<f32>(0.0), dt * rate, dt > 0.0), 0.0));
  if (!held) { textureStore(WV, c, vec4<f32>(-rate * k.x * P.geom.x, cut_wall(c).y)); }
}
