// Ideal MHD on the uniform grid: face-centered magnetic fields, their boundary mapping, and the
// primitive state of a cell with its cell-centered field (docs/numerics.md section 21).
//
// Units: B is in units where the magnetic pressure is B^2 / 2 (Heaviside-Lorentz with
// mu_0 = 1, as Athena uses): B_gauss = sqrt(4 pi) B. Plasma beta is 2 p / B^2.
//
// The face texture BF has one more texel than the grid along each axis (one along z in 2D):
// texel t holds (b_x on the lower x face of cell t, b_y on its lower y face, b_z on its lower z
// face, 0). The upper face of the grid along an axis is texel n of that axis; along a periodic
// axis it is the lower face of cell 0 instead, and texel n is never read. The cell-centered
// field is the mean of a cell's two faces along each axis.
//
// Requires P, the equation of state, hydro/pfloor.wgsl, hydro/prim.wgsl, g_dx set, and
// bindings U1, U2 (conserved state) and BF (face field) of the state being read.

// Cells (or faces) per side along each axis.
fn mhd_dims() -> vec3<i32> {
  return vec3<i32>(P.dims.xyz);
}

// Periodic for the face field: the upper face is the lower face of cell 0 and texel n is never
// read. A shearing box's x axis is not, for its faces: its two x planes are sheared images of
// each other, so each keeps its own b_x, which the averaged edge fields (mhd/shear.wgsl) keep in
// step.
fn periodic_axis(a: i32) -> bool {
  return P.bc_lo[a] == 0u && !(a == 0 && shear_on());
}

// Component `a` of the face field on the lower face along `a` of cell t. The face index along
// `a` must be in [0, n]; the indices along the other two axes are cell indices, and may lie up
// to two cells beyond the grid: they are mapped as a cell's would be. A face normal to `a` is
// tangential to the boundaries along the other axes, where its field is even (mirrored as it
// is at a reflecting wall, copied at an outflow).
fn face_texel(a: i32, t: vec3<i32>) -> vec3<i32> {
  let n = mhd_dims();
  var s = t;
  for (var b = 0; b < 3; b++) {
    if (b == a) {
      if (s[a] >= n[a] && periodic_axis(a)) { s[a] -= n[a]; }
      s[a] = clamp(s[a], 0, n[a]);
    } else {
      s[b] = bc_axis(s[b], n[b], P.bc_lo[b], P.bc_hi[b]);
    }
  }
  return s;
}
// Beyond a plane of mirror symmetry a tangential face's field changes sign (B is a pseudovector).
fn bface(a: i32, t: vec3<i32>) -> f32 {
  var s = 1.0;
  for (var b = 0; b < 3; b++) {
    if (b != a && t[b] < 0 && sym_lo(b)) { s = -s; }
  }
  // A shearing box: a face of a ghost beyond an x face lies between two sheared faces along y
  // (as hydro/common.wgsl's bc_map and shear_frac place the ghost's cell), linearly.
  if (a != 0 && shear_ghost(t)) {
    let ny = i32(P.dims.y);
    let J = f32(t.y) + select(-g_shear, g_shear, t.x >= i32(P.dims.x));
    let j0 = floor(J);
    var t0 = t;
    t0.y = ((i32(j0) % ny) + ny) % ny;
    var t1 = t0;
    t1.y = (t0.y + 1) % ny;
    return s * mix(textureLoad(BF, face_texel(a, t0), 0)[a], textureLoad(BF, face_texel(a, t1), 0)[a], J - j0);
  }
  return s * textureLoad(BF, face_texel(a, t), 0)[a];
}

#include "mhd/bcell.wgsl"

#include "mhd/state.wgsl"

// The state of a cell, or of the cell a ghost maps to (velocity and field mirrored at a
// reflecting face: the normal components change sign).
fn mq_at(m: Mapped, c: vec3<i32>) -> MQ {
  let w = prim_from_cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
  return MQ(vec4<f32>(w.w1.x, bc_velocity(w.w1.yzw, m)), vec4<f32>(w.w2.x, w.w2.z, w.w2.y, w.w2.w), bcell(c) * m.bflip);
}
// A shearing box's ghost beyond an x face: between the two sheared cells, linearly.
fn mq_mix(a: MQ, b: MQ, f: f32) -> MQ {
  return MQ(mix(a.a, b.a, f), mix(a.b, b.b, f), mix(a.B, b.B, f));
}
fn mq_load(c: vec3<i32>) -> MQ {
  let m = bc_map(c);
  comp_use(comp_tex(m.c));
  let q = mq_at(m, m.c);
  if (shear_ghost(c)) { return mq_mix(q, mq_at(m, shear_next(m.c)), shear_frac(c)); }
  return q;
}

