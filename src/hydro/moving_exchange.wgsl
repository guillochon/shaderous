// Moving bodies (plans/wind-tunnel.md 6.3): the cells that change kind, conservatively, as one
// gather (each cell works out what it gives and takes, with the same arithmetic its neighbors
// use for the same exchange).
//   covered   (gas, now in a body): its mass, momentum, energies and scalar go to its face
//             neighbors that stay gas, shared by their densities;
//   uncovered (in a body, now gas): it takes from each neighbor that stays gas a share,
//             1 / (n + 1) of the neighbor's state for n such neighbors (less, where the neighbor
//             gives to several: at most half of what it holds goes), and moves at the body's
//             velocity there (the body's push: momentum is not conserved, mass is);
//   a moving body's cell: its density kept, the body's velocity in its momentum slot (what the
//             walls' fluxes and viscosity read), its mark.
// Mass, energy and scalar are conserved, unless a covered cell has no neighbor that stays gas.
// Neighbors beyond the box take no part.
#include "hydro/common.wgsl"
#include "hydro/moving_flags.wgsl"

@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var MK: texture_3d<f32>;
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(1) var<storage, read> clock: Clock;
// The force the exchange puts on the body (hydro/forces.ts WX): where an uncovered cell takes
// the body's velocity, the gas gains m v_s less what its donors gave; the body loses as much.
// Over the step's dt, with the body's mark. (A covered cell's momentum stays in the gas.)
@group(0) @binding(7) var WX: texture_storage_3d<rgba32float, write>;

fn n_dirs() -> i32 {
  return select(6, 4, P.dims.z == 1u);
}
// Face neighbor k: +x, -x, +y, -y, +z, -z.
fn dir(k: i32) -> vec3<i32> {
  var e = vec3<i32>(0);
  e[k / 2] = select(1, -1, (k & 1) == 1);
  return e;
}
fn in_box(c: vec3<i32>) -> bool {
  return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(P.dims.xyz));
}
fn gas(m: f32) -> bool {
  return abs(m) < 0.5;
}
fn old_mark(c: vec3<i32>) -> f32 {
  return textureLoad(U2, c, 0).w;
}
fn new_mark(c: vec3<i32>) -> f32 {
  return textureLoad(MK, c, 0).x;
}
fn stays(c: vec3<i32>) -> bool {
  return in_box(c) && gas(old_mark(c)) && gas(new_mark(c));
}
fn covered(c: vec3<i32>) -> bool {
  return in_box(c) && gas(old_mark(c)) && new_mark(c) > 0.5;
}
fn uncovered(c: vec3<i32>) -> bool {
  return in_box(c) && old_mark(c) > 0.5 && gas(new_mark(c));
}
// A covered cell's receivers' total density.
fn receivers(q: vec3<i32>) -> f32 {
  var s = 0.0;
  for (var k = 0; k < n_dirs(); k++) {
    let r = q + dir(k);
    if (stays(r)) { s += textureLoad(U1, r, 0).x; }
  }
  return s;
}
// An uncovered cell's donors: how many of its neighbors stay gas.
fn donors(u: vec3<i32>) -> f32 {
  var n = 0.0;
  for (var k = 0; k < n_dirs(); k++) {
    if (stays(u + dir(k))) { n += 1.0; }
  }
  return n;
}
// What a cell that stays gas gives its uncovered neighbors, before the limit, as a fraction.
fn gifts(r: vec3<i32>) -> f32 {
  var g = 0.0;
  for (var k = 0; k < n_dirs(); k++) {
    let u = r + dir(k);
    if (uncovered(u)) { g += 1.0 / (donors(u) + 1.0); }
  }
  return g;
}
// The limit: at most half of what it holds.
fn gift_scale(r: vec3<i32>) -> f32 {
  let g = gifts(r);
  return select(1.0, min(1.0, 0.5 / g), g > 0.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  let om = u2.w;
  let mk = textureLoad(MK, c, 0);
  let nm = mk.x;
  textureStore(WX, c, vec4<f32>(0.0));
  // Held cells and static bodies' are as they were.
  if (nm < -0.5 || (nm > 0.5 && !moving_mark(nm))) {
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    return;
  }
  // In a moving body.
  if (nm > 0.5) {
    let rho = max(u1.x, P.rho_floor);
    textureStore(O1, c, vec4<f32>(rho, rho * mk.yzw));
    textureStore(O2, c, vec4<f32>(u2.xyz, nm));
    return;
  }
  if (gas(om)) {
    // Stays gas: keeps what it does not give, and takes its share of covered neighbors.
    let keep = 1.0 - gift_scale(c) * gifts(c);
    var a1 = u1 * keep;
    var a2 = u2.xyz * keep;
    for (var k = 0; k < n_dirs(); k++) {
      let q = c + dir(k);
      if (covered(q)) {
        let s = receivers(q);
        if (s > 0.0) {
          let f = u1.x / s;
          a1 += f * textureLoad(U1, q, 0);
          a2 += f * textureLoad(U2, q, 0).xyz;
        }
      }
    }
    textureStore(O1, c, a1);
    textureStore(O2, c, vec4<f32>(a2, 0.0));
    return;
  }
  // Uncovered: what its donors give, moving with the body (its velocity as the cell held it).
  let n = donors(c);
  var a1 = vec4<f32>(0.0);
  var a2 = vec3<f32>(0.0);
  for (var k = 0; k < n_dirs(); k++) {
    let r = c + dir(k);
    if (stays(r)) {
      let f = gift_scale(r) / (n + 1.0);
      a1 += f * textureLoad(U1, r, 0);
      a2 += f * textureLoad(U2, r, 0).xyz;
    }
  }
  let vs = u1.yzw / max(u1.x, P.rho_floor);
  let m = max(a1.x, P.rho_floor);
  let eint = a2.y;
  textureStore(O1, c, vec4<f32>(m, m * vs));
  textureStore(O2, c, vec4<f32>(0.5 * m * dot(vs, vs) + eint, eint, a2.z, 0.0));
  // (Per unit volume as the state is: times the cell's volume. None at a reset, dt = 0.)
  if (clock.dt > 0.0) { textureStore(WX, c, vec4<f32>(-(m * vs - a1.yzw) * P.geom.x / clock.dt, om)); }
}
