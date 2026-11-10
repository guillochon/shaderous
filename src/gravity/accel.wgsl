// Gravitational acceleration g = -grad(phi) at an in-range cell, central differences.
// Periodic potentials wrap; isolated ones use a one-sided difference in the boundary cells.
// Requires P (hydro/common.wgsl). Only gravity/field.wgsl evaluates this; every other pass
// reads the result from the field texture (gravity/gfield.wgsl).

fn grav_axis(phi: texture_3d<f32>, c: vec3<i32>, e: vec3<i32>, i: i32) -> f32 {
  let n = i32(P.n);
  var lo = c - e;
  var hi = c + e;
  var w = 0.5;
  if (P.grav_bc == 0u) {
    lo = (lo + n) % n;
    hi = hi % n;
  } else {
    if (i == 0) { lo = c; w = 1.0; }
    if (i == n - 1) { hi = c; w = 1.0; }
  }
  return -(textureLoad(phi, hi, 0).x - textureLoad(phi, lo, 0).x) * w / P.dx;
}

fn gravity(phi: texture_3d<f32>, c: vec3<i32>) -> vec3<f32> {
  return vec3<f32>(
    grav_axis(phi, c, vec3<i32>(1, 0, 0), c.x),
    grav_axis(phi, c, vec3<i32>(0, 1, 0), c.y),
    grav_axis(phi, c, vec3<i32>(0, 0, 1), c.z),
  );
}
