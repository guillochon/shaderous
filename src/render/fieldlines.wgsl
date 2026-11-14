// Magnetic field lines (render/renderer.ts, FieldLines): seeds that move with the gas, and the
// lines through them traced along the field. Positions are fractions of the box, [0, 1]^3.
//
// The field and the velocity come from the field-line volume (render/bsample.wgsl on the uniform
// grid, amr/bsample.wgsl on blocks), filtered trilinearly. In ideal MHD field lines move with
// the gas, so a seed carried by the flow stays on its line and the picture changes smoothly
// from frame to frame. A seed is placed again where it leaves the box, or where the field
// around it has faded.
//
// In the plane (a 2D run, or the slice view of a 3D one) the lines follow the field's
// components in the plane, and seeds stay in it.
//
// Entry points, dispatched in this order each frame: advect (one invocation per seed), stamp
// (one), trace (one per seed).

struct Lines {
  dims: vec4<f32>,    // xyz: voxels of the volume; w: the box's side in code units
  plane: vec4<f32>,   // x: 1 to trace in a plane; y: its normal axis; z: its position; w: seeds
  trace: vec4<f32>,   // x: points per line; y: step, a fraction of the box; z: frame number; w: 1 to place every seed again
  per: vec4<f32>,     // xyz: 1 along periodic axes
}
@group(0) @binding(0) var<uniform> F: Lines;
@group(0) @binding(1) var BV: texture_3d<f32>;
@group(0) @binding(2) var VV: texture_3d<f32>;
@group(0) @binding(3) var SM: sampler;
@group(0) @binding(4) var<storage, read_write> SEEDS: array<vec4<f32>>;
// Per line, F.trace.x points: position in box units ([-0.5, 0.5]^3) and |B|; w < 0 past its ends.
@group(0) @binding(5) var<storage, read_write> PTS: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> CLOCK: array<f32>;   // dt, t, Kahan term, steps
// The time the seeds were last moved to, and 1 once they have been placed.
@group(0) @binding(7) var<storage, read_write> KEEP: array<f32>;
// Per frame: the largest |B| in the volume (from the sampling pass), then the least and greatest
// |B| on the lines drawn, as float bits.
@group(0) @binding(8) var<storage, read_write> FR: array<atomic<u32>, 4>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(st: ptr<function, u32>) -> f32 {
  *st = pcg(*st);
  return f32(*st >> 8u) / 16777216.0;
}

fn in_plane() -> bool {
  return F.plane.x != 0.0;
}
fn axis() -> i32 {
  return i32(F.plane.y);
}

// The field at p, with its component along the plane's normal dropped when tracing in a plane.
fn field(p: vec3<f32>) -> vec3<f32> {
  var B = textureSampleLevel(BV, SM, p, 0.0).xyz;
  if (in_plane()) { B[axis()] = 0.0; }
  return B;
}

fn bmax() -> f32 {
  return bitcast<f32>(atomicLoad(&FR[0]));
}

fn inside(p: vec3<f32>) -> bool {
  return all(p >= vec3<f32>(0.0)) && all(p <= vec3<f32>(1.0));
}

// A new seed: of 24 random points (in the plane, when tracing in one), one drawn with
// probability in proportion to |B| in the plane, |B|^2 (the magnetic energy) in 3D.
fn place(st: ptr<function, u32>) -> vec3<f32> {
  var best = vec3<f32>(rnd(st), rnd(st), rnd(st));
  var key = -1.0e30;
  for (var k = 0; k < 24; k++) {
    var q = vec3<f32>(rnd(st), rnd(st), rnd(st));
    if (in_plane()) { q[axis()] = F.plane.z; }
    let b = length(field(q));
    let w = select(b * b, b, in_plane());
    if (w <= 0.0) { continue; }
    // Weighted reservoir sampling: the largest log(u) / w.
    let kk = log(max(rnd(st), 1.0e-30)) / w;
    if (kk > key) {
      key = kk;
      best = q;
    }
  }
  if (in_plane()) { best[axis()] = F.plane.z; }
  return best;
}

@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u32(F.plane.w)) { return; }
  var st = pcg(i * 9781u + u32(F.trace.z) * 6271u + 17u);
  var s = SEEDS[i];
  let dt = CLOCK[1] - KEEP[0];
  var fresh = F.trace.w != 0.0 || KEEP[1] == 0.0 || s.w == 0.0 || dt < 0.0;
  if (!fresh) {
    var v = textureSampleLevel(VV, SM, s.xyz, 0.0).xyz;
    if (in_plane()) { v[axis()] = 0.0; }
    var q = s.xyz + v * dt / F.dims.w;
    for (var a = 0; a < 3; a++) {
      if (F.per[a] != 0.0) { q[a] = fract(q[a]); }
    }
    if (in_plane()) { q[axis()] = F.plane.z; }
    // Gone, or where the field has faded: placed again.
    fresh = !inside(q) || length(field(q)) < 0.02 * bmax();
    s = vec4<f32>(q, 1.0);
  }
  if (fresh) { s = vec4<f32>(place(&st), 1.0); }
  SEEDS[i] = s;
}

@compute @workgroup_size(1)
fn stamp() {
  KEEP[0] = CLOCK[1];
  KEEP[1] = 1.0;
}

// One second-order (midpoint) step of length h along the field's direction (or against it,
// h < 0), or the point itself where the field vanishes.
fn step_along(p: vec3<f32>, h: f32) -> vec4<f32> {
  let b1 = field(p);
  let l1 = length(b1);
  if (l1 <= 0.0) { return vec4<f32>(p, 0.0); }
  let m = p + 0.5 * h * b1 / l1;
  let b2 = field(m);
  let l2 = length(b2);
  if (l2 <= 0.0) { return vec4<f32>(p, 0.0); }
  return vec4<f32>(p + h * b2 / l2, 1.0);
}

fn put(at: u32, p: vec3<f32>) {
  let b = length(textureSampleLevel(BV, SM, p, 0.0).xyz);
  var q = p - 0.5;
  // A 2D run is drawn in its plane at z = 0.
  if (F.dims.z == 1.0) { q.z = 0.0; }
  PTS[at] = vec4<f32>(q, b);
  atomicMin(&FR[1], bitcast<u32>(b));
  atomicMax(&FR[2], bitcast<u32>(b));
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u32(F.plane.w)) { return; }
  let n = u32(F.trace.x);
  let half = n / 2u;
  let base = i * n;
  let h = F.trace.y;
  let p0 = SEEDS[i].xyz;
  let gap = vec4<f32>(0.0, 0.0, 0.0, -1.0);
  let faint = 0.01 * bmax();
  put(base + half, p0);
  // Forward, then backward unless the line closed on itself.
  var p = p0;
  var closed = false;
  var k = 1u;
  for (; half + k < n; k++) {
    let q = step_along(p, h);
    if (q.w == 0.0 || !inside(q.xyz) || length(field(q.xyz)) < faint) { break; }
    p = q.xyz;
    put(base + half + k, p);
    if (k > 8u && distance(p, p0) < h) {
      closed = true;
      k++;
      break;
    }
  }
  for (; half + k < n; k++) { PTS[base + half + k] = gap; }
  p = p0;
  k = 1u;
  if (!closed) {
    for (; k <= half; k++) {
      let q = step_along(p, -h);
      if (q.w == 0.0 || !inside(q.xyz) || length(field(q.xyz)) < faint) { break; }
      p = q.xyz;
      put(base + half - k, p);
    }
  }
  for (; k <= half; k++) { PTS[base + half - k] = gap; }
}
