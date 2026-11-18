// What measures the magnetorotational instability of a disk in the x-y plane around the first
// point mass (problems/mri_disk.ts): the directions of the cylinder about the mass, its softened
// Keplerian angular velocity, and the quality factors of Hawley, Guan & Krolik (2011),
//   Q = 2 pi |B_i| / (sqrt(rho) Omega dx),
// the cells across the fastest-growing MRI wavelength of the field's vertical (i = z) and
// azimuthal (i = phi) parts. The MRI grows as it should where Q_z is above some 8 and Q_phi
// above some 20. Where the field dominates the gas (beta below 1, a magnetized corona) the MRI
// does not grow and Q means nothing: it is 0 there. Requires P, gravity/bh.wgsl and a binding
// BH: Masses.
#include "gravity/rot.wgsl"
struct Cyl {
  er: vec3<f32>,     // unit vector away from the mass's axis
  ephi: vec3<f32>,   // unit vector along the orbit
  R: f32,            // distance from the axis
  omega: f32,        // softened Keplerian angular velocity there
}

// The cylinder about the first mass at x (relative to the box center). In a shearing box
// (P.shear) the local frame instead: x radial, y along the orbit, its angular velocity Omega.
fn disk_cyl(x: vec3<f32>) -> Cyl {
  if (P.shear.w != 0.0) { return Cyl(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), 1.0, P.shear.x); }
  let m = BH.list[0];
  let d = x - m.pos.xyz;
  let R = max(length(d.xy), 1.0e-6);
  let eps = m.vel.w;
  let s = R * R + eps * eps;
  return Cyl(vec3<f32>(d.x / R, d.y / R, 0.0), vec3<f32>(-d.y / R, d.x / R, 0.0), R, sqrt(m.pos.w / (s * sqrt(s))));
}

// (Q_z, Q_phi) of a cell of size dx at x with field B, density rho and pressure p; zero without
// a mass, and where beta = 2p / B^2 is below 1.
fn mri_quality(B: vec3<f32>, rho: f32, p: f32, x: vec3<f32>, dx: f32) -> vec2<f32> {
  if ((mass_count() == 0u && P.shear.w == 0.0) || 2.0 * p < dot(B, B)) { return vec2<f32>(0.0); }
  let c = disk_cyl(x);
  let s = 6.28318531 / (sqrt(max(rho, P.rho_floor)) * c.omega * dx);
  return vec2<f32>(abs(B.z), abs(dot(B, c.ephi))) * s;
}

// The disk's span: from the first mass's sink to the damping zone where a preset has one (the
// gas there is held, not free to turn turbulent), else to the box's edge.
fn disk_span() -> vec2<f32> {
  let lo = max(BH.list[0].acc.w, 1.0e-3);
  var hi = 0.5 * P.box;
  if (P.diff.w > 0.0 && P.diff.z > 0.0) { hi = min(hi, P.diff.z); }
  return vec2<f32>(lo, hi);
}

// Annuli the Reynolds stress is measured in, spaced evenly in log R across the span: its
// velocities are measured from each annulus's mean flow, as global disk simulations do
// (dv_phi = v_phi - <v_phi>), so that the disk's mean rotation, and radial drifts, do not count.
// The orbital speed is taken less the softened Keplerian one first, so that its fall across an
// annulus does not count either.
const DISK_BINS = 8;
// Channels: Maxwell stress, pressure, rho Q_z, rho Q_phi, rho (where beta >= 1), then four per
// annulus: rho v_R w, rho v_R, rho w and rho, w = v_phi - R Omega_K.
const DISK_CHANNELS = 5 + 4 * DISK_BINS;

// The disk channels of the magnetic diagnostics (mhd/mhd.ts MAG_OPS, 4 onward), for a cell of
// size dx and volume dv at x, all volume-summed over the disk's body: within two scale heights
// H = c_s / Omega_K of its midplane, as global MRI studies average (the corona above holds gas
// falling in, which spins up past Keplerian, and the field that rises out of the disk).
// Velocities are in the mass's rest frame.
fn disk_channels(B: vec3<f32>, rho: f32, mom: vec3<f32>, p: f32, x: vec3<f32>, dx: f32, dv: f32) -> array<f32, DISK_CHANNELS> {
  var r: array<f32, DISK_CHANNELS>;
  for (var i = 0; i < DISK_CHANNELS; i++) { r[i] = 0.0; }
  // A shearing box: every cell (within two scale heights of the midplane if stratified), its
  // velocity measured from the background flow v_y = -q Omega x, in one annulus.
  if (P.shear.w != 0.0) {
    if (P.shear.z != 0.0 && abs(x.z) > 2.0 * eos_sound_speed(max(rho, P.rho_floor), p) / P.shear.x) { return r; }
    r[0] = -B.x * B.y * dv;
    r[1] = p * dv;
    if (2.0 * p >= dot(B, B)) {
      let q = mri_quality(B, rho, p, x, dx);
      r[2] = rho * q.x * dv;
      r[3] = rho * q.y * dv;
      r[4] = rho * dv;
    }
    let v = mom / max(rho, P.rho_floor);
    let w = v.y + P.shear.y * P.shear.x * x.x;
    r[5] = rho * v.x * w * dv;
    r[6] = rho * v.x * dv;
    r[7] = rho * w * dv;
    r[8] = rho * dv;
    return r;
  }
  if (mass_count() == 0u) { return r; }
  let c = disk_cyl(x);
  let span = disk_span();
  if (c.R <= span.x || c.R >= span.y) { return r; }
  let h = eos_sound_speed(max(rho, P.rho_floor), p) / c.omega;
  if (abs(x.z - BH.list[0].pos.z) > 2.0 * h) { return r; }
  r[0] = -dot(B, c.er) * dot(B, c.ephi) * dv;
  r[1] = p * dv;
  if (2.0 * p >= dot(B, B)) {
    let q = mri_quality(B, rho, p, x, dx);
    r[2] = rho * q.x * dv;
    r[3] = rho * q.y * dv;
    r[4] = rho * dv;
  }
  var v = mom / max(rho, P.rho_floor) - BH.list[0].vel.xyz;
  // In a rotating frame (gravity/rot.wgsl), the velocity about the mass as an inertial
  // observer sees it.
  if (rot_on()) { v += cross(P.rot.xyz, x - BH.list[0].pos.xyz); }
  let vr = dot(v, c.er);
  let vphi = dot(v, c.ephi) - c.R * c.omega;
  let b = min(i32(f32(DISK_BINS) * log(c.R / span.x) / log(span.y / span.x)), DISK_BINS - 1);
  let o = 5 + 4 * b;
  r[o] = rho * vr * vphi * dv;
  r[o + 1] = rho * vr * dv;
  r[o + 2] = rho * vphi * dv;
  r[o + 3] = rho * dv;
  return r;
}
