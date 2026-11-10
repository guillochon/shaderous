// Removal of the gas the point masses' sinks take, from one cell. The gas that goes takes its
// momentum, energy and scalar with it; gravity/bh_cell.wgsl reduces the same amounts for the
// masses. Requires gravity/bh.wgsl and a binding BH: Masses.
struct Sunk { u1: vec4<f32>, u2: vec4<f32> }

fn apply_sinks(u1: vec4<f32>, u2: vec4<f32>, x: vec3<f32>, dt: f32) -> Sunk {
  var a = u1;
  var b = u2.xyz;
  for (var i = 0u; i < mass_count(); i++) {
    let f = sink_fraction(BH.list[i], x, a.x, dt);
    if (f > 0.0) {
      a *= 1.0 - f;
      b *= 1.0 - f;
    }
  }
  return Sunk(a, vec4<f32>(b, u2.w));
}
