// Point-mass orbits, second half: hand each mass the gas its sink took, then, from the reduced
// pull on the gas at the new positions, set the accelerations and apply the closing half kick.
//
// A mass (G M) pulls on the gas with total force F = sum of m a over the cells, so the gas pulls
// back on it with -F: an acceleration of -G F / (G M). The masses also pull on one another,
// each pair softened by the larger of its two lengths, so the pair's forces are equal and
// opposite.
//
// The frame is inertial unless P.bh.y is set. At 1 it accelerates with the gas, at abar, the
// mass-weighted mean of the masses' pull on it. In that frame the masses exert no net force on
// the gas, so its center of mass stays where it started, and every mass accelerates by -abar
// more. For one mass that is -(1 + m / M) abar.
//
// At 2 it accelerates with the point masses' center of mass instead, at the mean pull of the
// gas on them (their pulls on one another cancel), so that center stays put and the gas
// accelerates by -abar. The box, its boundaries and any damping zone then go where the masses
// go, as an outer disk held by a damping zone would: held fixed to the box instead, the disk
// takes up none of the masses' recoil, and pulls them on toward whichever side they are nearer.
// The momentum the sinks take in, and rounding in the masses' pulls on one another, would
// still move the center slowly: its velocity is set back to zero after every kick.
//
// Nothing here assumes the gas is a point, so an orbit responds to the tides it raises. Once
// gas leaves the box, "the gas" means what is still inside it.
//
// A mass may be held on its orbit (flag 4, PointMass.onOrbit): the gas then neither pulls on it
// nor changes its mass or momentum through its sink, so it follows the orbit the other point
// masses give it, as a binary of fixed orbit does in most studies of circumbinary disks. Its
// sink still takes gas, and what it took is still counted.
//
// (Two alternatives to the accelerating frame were tried and dropped. A restoring spring on
// the center of mass visibly shoves the remaining gas each time debris exits. Following a
// density-squared-weighted centroid instead keeps the densest gas centered, but lets the true
// center of mass wander by several percent of the stellar radius during a disruption.)
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"
#include "gravity/rot.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read> RED: array<f32>;
@group(0) @binding(3) var<storage, read_write> BH: Masses;

// Mirror symmetry (Params.sym): the reduction summed one 2^k-th of the configuration, and the
// masses lie on the planes. The whole's totals are the grid's unfolded: a scalar times 2^k; a
// vector's component along a mirrored axis cancels against its images and the rest doubles per
// mirror; an angular momentum's (a pseudovector's) component survives only where every mirror
// is normal to it.
fn k_mirrors() -> f32 {
  return f32((P.sym.x & 1u) + (P.sym.y & 1u) + (P.sym.z & 1u));
}
fn unfold_s(s: f32) -> f32 {
  return s * exp2(k_mirrors());
}
fn unfold_v(v: vec3<f32>) -> vec3<f32> {
  var f = vec3<f32>(1.0);
  for (var a = 0; a < 3; a++) {
    if (P.sym[a] != 1u) { continue; }
    var g = vec3<f32>(2.0);
    g[a] = 0.0;
    f *= g;
  }
  return v * f;
}
fn unfold_l(l: vec3<f32>) -> vec3<f32> {
  var f = vec3<f32>(1.0);
  for (var a = 0; a < 3; a++) {
    if (P.sym[a] != 1u) { continue; }
    var g = vec3<f32>(0.0);
    g[a] = 2.0;
    f *= g;
  }
  return l * f;
}
fn red_s(i: u32) -> f32 {
  if (all(P.sym.xyz == vec3<u32>(0u))) { return RED[i]; }
  return unfold_s(RED[i]);
}
fn red_v(i: u32) -> vec3<f32> {
  let v = vec3<f32>(RED[i], RED[i + 1u], RED[i + 2u]);
  if (all(P.sym.xyz == vec3<u32>(0u))) { return v; }
  return unfold_v(v);
}
fn red_l(i: u32) -> vec3<f32> {
  let v = vec3<f32>(RED[i], RED[i + 1u], RED[i + 2u]);
  if (all(P.sym.xyz == vec3<u32>(0u))) { return v; }
  return unfold_l(v);
}

// Cleared for a pipeline that only evaluates the accelerations (a fresh start, or a potential
// solved again in mid-run): no step has been taken, so nothing is accreted and nothing kicked.
override LIVE: bool = true;

@compute @workgroup_size(1)
fn main() {
  let n = mass_count();
  // Accretion: the sink's gas arrives with its momentum. What it brought (mass, momentum, and
  // angular momentum about the mass) is kept for the diagnostics. A step's share is usually
  // far below float32 resolution of the totals, so each is summed with Kahan compensation
  // (as the clock is, hydro/clock.wgsl): added plainly, a binary's G M never grows at all.
  if (LIVE && P.G > 0.0) {
    for (var i = 0u; i < n; i++) {
      let o = 1u + 10u * i;
      let dm = red_s(o + 3u);
      if (dm > 0.0) {
        let b = BH.list[i];
        let dp = red_v(o + 4u);
        let dl = red_l(o + 7u);
        let mass = (b.pos.w - b.cmass.x) / P.G;
        // (M v + dp) / (M + dm), as a change to v so that it too can be compensated
        let dv = (dp - dm * b.vel.xyz) / (mass + dm);
        var gm = kahan1(b.pos.w, b.cmass.x, P.G * dm);
        let acc = kahan1(b.sink.y, b.cmass.y, dm);
        var v = kahan3(b.vel.xyz, b.cvel.xyz, dv);
        // A massless particle keeps count of what it takes, and is otherwise unchanged by it.
        // So does a mass held on its orbit: growing would change the orbit.
        if (mass_massless(b) || mass_on_orbit(b)) { gm = vec2<f32>(b.pos.w, b.cmass.x); }
        // Nor does it move a massless particle, a fixed mass, or one held on its orbit.
        if (mass_massless(b) || mass_fixed(b) || mass_on_orbit(b)) { v = Kahan3(b.vel.xyz, b.cvel.xyz); }
        let p = kahan3(b.accp.xyz, b.cp.xyz, dp);
        let l = kahan3(b.accl.xyz, b.cl.xyz, dl);
        BH.list[i].pos = vec4<f32>(b.pos.xyz, gm.x);
        BH.list[i].vel = vec4<f32>(v.s, b.vel.w);
        BH.list[i].sink = vec4<f32>(b.sink.x, acc.x, b.sink.zw);
        BH.list[i].accp = vec4<f32>(p.s, 0.0);
        BH.list[i].accl = vec4<f32>(l.s, 0.0);
        BH.list[i].cmass = vec4<f32>(gm.y, acc.y, 0.0, 0.0);
        BH.list[i].cvel = vec4<f32>(v.c, 0.0);
        BH.list[i].cp = vec4<f32>(p.c, 0.0);
        BH.list[i].cl = vec4<f32>(l.c, 0.0);
      }
    }
  }

  let m = red_s(0u);
  var abar = vec3<f32>(0.0);
  if (P.bh.y == 1.0 && m > 0.0) {
    for (var i = 0u; i < n; i++) {
      let o = 1u + 10u * i;
      abar += red_v(o) / m;
    }
  } else if (P.bh.y == 2.0) {
    var gm = 0.0;
    for (var i = 0u; i < n; i++) {
      let o = 1u + 10u * i;
      // The gas does not move a mass held on its orbit, nor the center of mass through it.
      if (!mass_on_orbit(BH.list[i])) { abar -= P.G * red_v(o); }
      gm += BH.list[i].pos.w;
    }
    abar /= max(gm, 1e-30);
  }
  BH.frame = vec4<f32>(abar, 0.0);
  for (var i = 0u; i < n; i++) {
    let b = BH.list[i];
    let o = 1u + 10u * i;
    var acc = -abar - P.G * red_v(o) / b.pos.w;
    // Held on its orbit: the frame's acceleration and the other masses' pull, not the gas's.
    if (mass_on_orbit(b)) { acc = -abar; }
    // A rotating frame's centrifugal force.
    if (P.rot.w != 0.0) { acc += rot_accel(b.pos.xyz); }
    for (var j = 0u; j < n; j++) {
      if (j == i) { continue; }
      let other = BH.list[j];
      let d = b.pos.xyz - other.pos.xyz;
      let eps = max(b.vel.w, other.vel.w);
      let r2 = dot(d, d) + eps * eps;
      acc -= other.pos.w * d / (r2 * sqrt(r2));
    }
    BH.list[i].acc = vec4<f32>(acc, b.acc.w);
    if (LIVE && !mass_fixed(b)) {
      // The second half of the step's Coriolis turn (gravity/bh_drift.wgsl has the first), then
      // the closing half kick.
      var v = BH.list[i].vel.xyz;
      if (rot_on()) { v = rot_turn(v, 0.5 * clock.dt); }
      BH.list[i].vel = vec4<f32>(v + 0.5 * clock.dt * acc, b.vel.w);
    }
  }
  if (LIVE && P.bh.y == 2.0) {
    var p = vec3<f32>(0.0);
    var gm = 0.0;
    for (var i = 0u; i < n; i++) {
      p += BH.list[i].pos.w * BH.list[i].vel.xyz;
      gm += BH.list[i].pos.w;
    }
    let vc = p / max(gm, 1e-30);
    for (var i = 0u; i < n; i++) {
      if (mass_fixed(BH.list[i])) { continue; }
      BH.list[i].vel = vec4<f32>(BH.list[i].vel.xyz - vc, BH.list[i].vel.w);
    }
  }
}
