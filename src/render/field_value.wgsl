// The quantities the view can show, in the order of FIELDS in src/problems/types.ts. Shared by
// the display passes of the uniform grid (render/field.wgsl) and the block grid
// (amr/field.wgsl), which supply the derivatives the last few need from neighboring cells.
// Requires P, g_dx, the equation of state, the <cooling> slot and, for range_end(), a storage
// buffer RANGE: array<atomic<u32>, 132>: min, max, smallest positive, unused, then a histogram
// of |value| in HIST_BINS bins of log10 |value| (render/renderer.ts reads it).
#include <cooling>

// Gradients at a cell: of each velocity component and of the density (zero along an axis the
// grid does not have).
struct Grad {
  dvx: vec3<f32>,
  dvy: vec3<f32>,
  dvz: vec3<f32>,
  drho: vec3<f32>,
}

const F_VORTICITY = 11u;
const F_DIVERGENCE = 12u;
const F_DENSITY_GRADIENT = 13u;
const F_LEVEL = 14u;

fn field_needs_grad(f: u32) -> bool {
  return f == F_VORTICITY || f == F_DIVERGENCE || f == F_DENSITY_GRADIENT;
}

// The value of field f at a cell with conserved state u1, u2, gradients g and refinement level.
fn field_value(f: u32, u1: vec4<f32>, u2: vec4<f32>, g: Grad, level: f32) -> f32 {
  let rho = max(u1.x, P.rho_floor);
  let v = u1.yzw / rho;
  let eint = u2.y;
  switch (f) {
    case 1u { return eos_temperature(rho, eint); }
    case 2u { return u2.z / rho; }
    case 3u { return eos_pressure(rho, eint); }
    case 4u { return length(v); }
    case 5u { return v.x; }
    case 6u { return v.y; }
    case 7u { return v.z; }
    case 8u {
      let p = max(eos_pressure(rho, eint), P.p_floor);
      return length(v) / max(eos_sound_speed(rho, p), 1.0e-30);
    }
    // The entropy function p / rho^gamma: constant along a flow line away from shocks.
    case 9u { return eos_pressure(rho, eint) / pow(rho, P.eos.x); }
    // Heating minus cooling, per unit volume (the rate of change of rho e).
    case 10u {
      if (!COOLING || !EOS_HAS_ENERGY) { return 0.0; }
      return cool_rate(rho, eos_temperature(rho, eint));
    }
    // In 2D the one component of the vorticity, with its sign; in 3D its magnitude.
    case 11u {
      let w = vec3<f32>(g.dvz.y - g.dvy.z, g.dvx.z - g.dvz.x, g.dvy.x - g.dvx.y);
      return select(length(w), w.z, P.dims.z == 1u);
    }
    case 12u { return g.dvx.x + g.dvy.y + g.dvz.z; }
    case 13u { return length(g.drho) / rho; }
    case 14u { return level; }
    default { return rho; }
  }
}

// Running range of what the pass draws, for the automatic color range: a minimum, a maximum
// and the smallest positive value (for a log scale), as order-preserving integers (range_key),
// and a histogram of |value|, so that a few extreme cells (inside a sink, at a corner) need not
// set the range. Reduced in each workgroup, then once per workgroup into RANGE.
const HIST_BINS = 128u;
const HIST_LOG_LO = -20.0;          // log10 |value| at the lower edge of the first bin
const HIST_PER_DECADE = 3.2;        // bins per decade: 128 bins span 1e-20 to 1e20
var<workgroup> w_lo: atomic<u32>;
var<workgroup> w_hi: atomic<u32>;
var<workgroup> w_pos: atomic<u32>;
var<workgroup> w_hist: array<atomic<u32>, 128>;

// A float's bits, rearranged so that integer order is float order (negative values too).
fn range_key(x: f32) -> u32 {
  let b = bitcast<u32>(x);
  return select(b ^ 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn range_begin(lix: u32) {
  if (lix < HIST_BINS) { atomicStore(&w_hist[lix], 0u); }
  if (lix == 0u) {
    atomicStore(&w_lo, 0xffffffffu);
    atomicStore(&w_hi, 0u);
    atomicStore(&w_pos, 0xffffffffu);
  }
  workgroupBarrier();
}

fn range_add(x: f32) {
  if (x != x || abs(x) > 3.0e38) { return; }
  let k = range_key(x);
  atomicMin(&w_lo, k);
  atomicMax(&w_hi, k);
  if (x > 0.0) { atomicMin(&w_pos, k); }
  let a = abs(x);
  var bin = 0u;
  if (a > 0.0) { bin = u32(clamp(floor((log(a) * 0.4342944819 - HIST_LOG_LO) * HIST_PER_DECADE), 0.0, f32(HIST_BINS - 1u))); }
  atomicAdd(&w_hist[bin], 1u);
}

fn range_end(lix: u32) {
  workgroupBarrier();
  if (lix < HIST_BINS) {
    let n = atomicLoad(&w_hist[lix]);
    if (n > 0u) { atomicAdd(&RANGE[4u + lix], n); }
  }
  if (lix == 0u) {
    let lo = atomicLoad(&w_lo);
    if (lo != 0xffffffffu) {
      atomicMin(&RANGE[0], lo);
      atomicMax(&RANGE[1], atomicLoad(&w_hi));
      atomicMin(&RANGE[2], atomicLoad(&w_pos));
    }
  }
}
