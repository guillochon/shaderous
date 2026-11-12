// What one cell contributes to the point masses' reduction. Requires gravity/bh.wgsl, a binding
// BH: Masses, and Red (gpu/reduce.wgsl) with 41 channels:
//   0              gas mass
//   1 + 10 i ...   for mass i: its pull on the gas (m a, 3), then what its sink takes from the
//                  cell this step: mass (1), momentum (3), and angular momentum about the mass
//                  in its rest frame (3)
// `vol` is the cell's volume, `x` its center, u1 its (rho, momentum), `dt` the step.
fn bh_cell(u1: vec4<f32>, vol: f32, x: vec3<f32>, dt: f32) -> Red {
  var r: Red;
  for (var k = 0u; k < RED_K; k++) { r[k] = 0.0; }
  let m = u1.x * vol;
  r[0] = m;
  // The sinks take their share one after another, as gravity/sink.wgsl removes it.
  var rho = u1.x;
  var mom = u1.yzw;
  for (var i = 0u; i < mass_count(); i++) {
    let b = BH.list[i];
    let a = bh_accel(x, b);
    let o = 1u + 10u * i;
    r[o] = m * a.x;
    r[o + 1u] = m * a.y;
    r[o + 2u] = m * a.z;
    let f = sink_fraction(b, x, rho, dt);
    if (f > 0.0) {
      r[o + 3u] = f * rho * vol;
      r[o + 4u] = f * mom.x * vol;
      r[o + 5u] = f * mom.y * vol;
      r[o + 6u] = f * mom.z * vol;
      let l = cross(x - b.pos.xyz, mom - rho * b.vel.xyz) * (f * vol);
      r[o + 7u] = l.x;
      r[o + 8u] = l.y;
      r[o + 9u] = l.z;
      rho *= 1.0 - f;
      mom *= 1.0 - f;
    }
  }
  return r;
}
