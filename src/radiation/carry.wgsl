// The radiation carried through a face with the gas (radiation/advect.wgsl on the uniform grid,
// amr/rad_advect.wgsl on blocks). Requires radiation/common.wgsl.

// What the advection needs of a cell: (E, F) per unit mass, the density, the velocity and the
// row `a` of the pressure tensor, with a ghost's flux and velocity reversed across a wall.
struct ACell { q: vec4<f32>, rho: f32, v: vec3<f32>, prow: vec3<f32> }

// The inflow of (E, F) through the face between q1 and q2 (q0, q1 | q2, q3 along the axis),
// toward +a: MF q_face + (v . P)_a for E.
fn face_carry(a: i32, mf: f32, dtdx: f32, q0: ACell, q1: ACell, q2: ACell, q3: ACell) -> vec4<f32> {
  var qf: vec4<f32>;
  if (mf >= 0.0) {
    let nu = min(mf * dtdx / q1.rho, 1.0);
    qf = q1.q + 0.5 * (1.0 - nu) * rad_mc(q1.q - q0.q, q2.q - q1.q);
  } else {
    let nu = min(-mf * dtdx / q2.rho, 1.0);
    qf = q2.q - 0.5 * (1.0 - nu) * rad_mc(q2.q - q1.q, q3.q - q2.q);
  }
  let work = dot(0.5 * (q1.v + q2.v), 0.5 * (q1.prow + q2.prow));
  return mf * qf + vec4<f32>(work, 0.0, 0.0, 0.0);
}
