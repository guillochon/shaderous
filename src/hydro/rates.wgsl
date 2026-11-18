// First stage of the timestep reduction on a uniform grid.
//
// The step size needs the largest signal rate and acceleration on the grid. This pass reads
// the conserved state once and writes nothing but one record per block of 4^3 cells: a
// workgroup spans 2 x 2 x 1 such blocks, its invocations fold their values into shared
// atomics, and one invocation per block writes the record the later stages of
// gpu/reduce.wgsl expect. Both values are non-negative, so comparing their bit patterns as
// integers orders them as floats.
#include "hydro/common.wgsl"
#include "gravity/ext_bind.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"
#include "hydro/diff_rate.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(5) var G: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> RATES: array<f32>;   // (max rate, max |g|) per block

#include "gravity/gfield.wgsl"
// Implicit pressure: the last solve's pressure, whose gradient limits the step as gravity's
// acceleration does (below). Off, pp_gradient is zero.
#include <psplit_grad>

var<workgroup> wg_max: array<atomic<u32>, 8>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let blk = 2u * ((lid.x >> 2u) + 2u * (lid.y >> 2u));
  if (all(gid < P.dims.xyz)) {
    g_dx = P.dx;
    let c = vec3<i32>(gid);
    let u2 = textureLoad(U2, c, 0);
    comp_use(comp_tex(c));
    let w = prim_from_cons(textureLoad(U1, c, 0), u2);
    // Unsplit scheme: the stable step is set by the sum of the per-axis signal speeds.
    // (With implicit pressure the sound speed only sets a ceiling: P.ohm.y is one over the
    // acoustic steps a step may span, 0 for the explicit scheme. src/sim.ts packParams.)
    let snd = select(1.0, P.ohm.y, P.ohm.y > 0.0);
    var rate = abs(w.w1.y) + abs(w.w1.z) + abs(w.w1.w) + P.geom.y * w.w2.y * snd + diff_speed(w.w1.x, w.w2.w);
    // A spherical run: per axis over its own cell length (none along the azimuth).
    var acc = length(g_load(c));
    // With implicit pressure the sound speed does not limit the step, and gas at rest with a
    // pressure jump (a shock tube's start) would have no limit at all: the pressure's own
    // acceleration, less the field's that balances it in a column at rest, is one.
    acc = max(acc, length(pp_gradient(c) / (P.dx * w.w1.x) - g_load(c) * select(0.0, 1.0, P.ohm.y > 0.0)) * select(0.0, 1.0, P.ohm.y > 0.0));
    if (cylindrical()) {
      let t = cyl_dt_terms(w.w1.yzw, w.w2.y, f32(c.x) + 0.5, g_dx);
      rate += t.x;
      acc += t.y;
    }
    if (spherical()) {
      let sg = sph_cell(c.x, c.y);
      let h = sph_rate_h(sg);
      rate = (abs(w.w1.y) + w.w2.y) * h.x + (abs(w.w1.z) + w.w2.y) * h.y;
      acc += sph_accel(sg, w.w1.yzw, g_dx);
    }
    // Solid cells hold no gas to keep stable.
    if (u2.w <= 0.5) {
      atomicMax(&wg_max[blk], bitcast<u32>(rate));
      atomicMax(&wg_max[blk + 1u], bitcast<u32>(acc));
    }
  }
  workgroupBarrier();
  let nout = (P.dims.xyz + 3u) / 4u;
  let o = gid / 4u;
  if (all((lid & vec3<u32>(3u)) == vec3<u32>(0u)) && all(o < nout)) {
    let base = ((o.z * nout.y + o.y) * nout.x + o.x) * 2u;
    RATES[base] = bitcast<f32>(atomicLoad(&wg_max[blk]));
    RATES[base + 1u] = bitcast<f32>(atomicLoad(&wg_max[blk + 1u]));
  }
}
