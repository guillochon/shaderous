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
// (Two alternatives to the accelerating frame were tried and dropped. A restoring spring on
// the center of mass visibly shoves the remaining gas each time debris exits. Following a
// density-squared-weighted centroid instead keeps the densest gas centered, but lets the true
// center of mass wander by several percent of the stellar radius during a disruption.)
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read> RED: array<f32>;
@group(0) @binding(3) var<storage, read_write> BH: Masses;

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
      let dm = RED[o + 3u];
      if (dm > 0.0) {
        let b = BH.list[i];
        let dp = vec3<f32>(RED[o + 4u], RED[o + 5u], RED[o + 6u]);
        let dl = vec3<f32>(RED[o + 7u], RED[o + 8u], RED[o + 9u]);
        let mass = (b.pos.w - b.cmass.x) / P.G;
        // (M v + dp) / (M + dm), as a change to v so that it too can be compensated
        let dv = (dp - dm * b.vel.xyz) / (mass + dm);
        let gm = kahan1(b.pos.w, b.cmass.x, P.G * dm);
        let acc = kahan1(b.sink.y, b.cmass.y, dm);
        let v = kahan3(b.vel.xyz, b.cvel.xyz, dv);
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

  let m = RED[0];
  var abar = vec3<f32>(0.0);
  if (P.bh.y == 1.0 && m > 0.0) {
    for (var i = 0u; i < n; i++) {
      let o = 1u + 10u * i;
      abar += vec3<f32>(RED[o], RED[o + 1u], RED[o + 2u]) / m;
    }
  } else if (P.bh.y == 2.0) {
    var gm = 0.0;
    for (var i = 0u; i < n; i++) {
      let o = 1u + 10u * i;
      abar -= P.G * vec3<f32>(RED[o], RED[o + 1u], RED[o + 2u]);
      gm += BH.list[i].pos.w;
    }
    abar /= max(gm, 1e-30);
  }
  BH.frame = vec4<f32>(abar, 0.0);
  for (var i = 0u; i < n; i++) {
    let b = BH.list[i];
    let o = 1u + 10u * i;
    var acc = -abar - P.G * vec3<f32>(RED[o], RED[o + 1u], RED[o + 2u]) / b.pos.w;
    for (var j = 0u; j < n; j++) {
      if (j == i) { continue; }
      let other = BH.list[j];
      let d = b.pos.xyz - other.pos.xyz;
      let eps = max(b.vel.w, other.vel.w);
      let r2 = dot(d, d) + eps * eps;
      acc -= other.pos.w * d / (r2 * sqrt(r2));
    }
    BH.list[i].acc = vec4<f32>(acc, b.acc.w);
    if (LIVE) { BH.list[i].vel = vec4<f32>(BH.list[i].vel.xyz + 0.5 * clock.dt * acc, b.vel.w); }
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
      BH.list[i].vel = vec4<f32>(BH.list[i].vel.xyz - vc, BH.list[i].vel.w);
    }
  }
}
