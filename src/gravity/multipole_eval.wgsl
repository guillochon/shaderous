// Boundary potential for isolated boundaries from the monopole + quadrupole of the mass
// distribution (the dipole vanishes about the center of mass).
// MP = (M, M x, M y, M z, M xx, M yy, M zz, M xy, M xz, M yz), positions relative to the box center.
@group(0) @binding(10) var<storage, read> MP: array<f32>;

fn phi_boundary(x: vec3<f32>) -> f32 {
  let M = MP[0];
  if (!(M > 0.0)) { return 0.0; }
  let com = vec3<f32>(MP[1], MP[2], MP[3]) / M;
  let d = x - com;
  let r2 = dot(d, d);
  let r = sqrt(r2);
  // Second moments about the center of mass.
  let sxx = MP[4] - M * com.x * com.x;
  let syy = MP[5] - M * com.y * com.y;
  let szz = MP[6] - M * com.z * com.z;
  let sxy = MP[7] - M * com.x * com.y;
  let sxz = MP[8] - M * com.x * com.z;
  let syz = MP[9] - M * com.y * com.z;
  let dSd = sxx * d.x * d.x + syy * d.y * d.y + szz * d.z * d.z
          + 2.0 * (sxy * d.x * d.y + sxz * d.x * d.z + syz * d.y * d.z);
  let tr = sxx + syy + szz;
  // phi = -G [ M / r + Q_ij d_i d_j / (2 r^5) ],  Q_ij = 3 S_ij - tr(S) delta_ij
  return -P.G * (M / r + 0.5 * (3.0 * dSd - tr * r2) / (r2 * r2 * r));
}
