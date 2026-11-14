// Ideal MHD on the uniform grid: face-centered magnetic fields, their boundary mapping, and the
// primitive state of a cell with its cell-centered field (docs/mhd-plan.md).
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

fn periodic_axis(a: i32) -> bool {
  return P.bc_lo[a] == 0u;
}

// Component `a` of the face field on the lower face along `a` of cell t. The face index along
// `a` must be in [0, n]; the indices along the other two axes are cell indices, and may lie up
// to two cells beyond the grid: they are mapped as a cell's would be. A face normal to `a` is
// tangential to the boundaries along the other axes, where its field is even (mirrored as it
// is at a reflecting wall, copied at an outflow).
fn bface(a: i32, t: vec3<i32>) -> f32 {
  let n = mhd_dims();
  var s = t;
  for (var b = 0; b < 3; b++) {
    if (b == a) {
      if (s[a] >= n[a] && periodic_axis(a)) { s[a] -= n[a]; }
      s[a] = clamp(s[a], 0, n[a]);
    } else {
      s[b] = bc_axis(s[b], n[b], P.bc_lo[b], P.bc_hi[b]).x;
    }
  }
  return textureLoad(BF, s, 0)[a];
}

// Cell-centered field of an in-range cell.
fn bcell(c: vec3<i32>) -> vec3<f32> {
  return 0.5 * vec3<f32>(
    bface(0, c) + bface(0, c + vec3<i32>(1, 0, 0)),
    bface(1, c) + bface(1, c + vec3<i32>(0, 1, 0)),
    bface(2, c) + bface(2, c + vec3<i32>(0, 0, 1)),
  );
}

// Primitive state of a cell for MHD:
//   a = (rho, vx, vy, vz)    b = (p, scalar, sound speed, e_int per volume)    B = cell-centered field
struct MQ { a: vec4<f32>, b: vec4<f32>, B: vec3<f32> }

// The state of a cell, or of the cell a ghost maps to (velocity and field mirrored at a
// reflecting face: the normal components change sign).
fn mq_load(c: vec3<i32>) -> MQ {
  let m = bc_map(c);
  comp_use(comp_tex(m.c));
  let w = prim_from_cons(textureLoad(U1, m.c, 0), textureLoad(U2, m.c, 0));
  return MQ(vec4<f32>(w.w1.x, bc_velocity(w.w1.yzw, m)), vec4<f32>(w.w2.x, w.w2.z, w.w2.y, w.w2.w), bcell(m.c) * m.flip);
}

// Fastest magnetosonic speed of a state, over all directions: sqrt(c_s^2 + B^2 / rho).
fn fast_max(rho: f32, cs: f32, B: vec3<f32>) -> f32 {
  return sqrt(cs * cs + dot(B, B) / rho);
}

// Electric field -v x B at a cell center.
fn efield(v: vec3<f32>, B: vec3<f32>) -> vec3<f32> {
  return cross(B, v);
}
