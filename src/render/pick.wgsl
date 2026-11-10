// Depth pick for the volume view: marches one ray like the ray-marcher and reports how far
// along it the gas that the viewer sees lies (the mean distance, weighted by each sample's
// contribution to the pixel). The editing tools use it to place a brush on what was clicked.
#include "render/camera.wgsl"

struct PickIn { uv: vec2<f32> }   // [-1, 1], y up
@group(0) @binding(4) var<uniform> I: PickIn;
@group(0) @binding(5) var<storage, read_write> O: array<f32, 4>;   // distance (-1: nothing seen), opacity

@compute @workgroup_size(1)
fn main() {
  let ro = C.eye.xyz;
  let rd = normalize(C.fwd.xyz + I.uv.x * C.p.y * C.p.x * C.right.xyz + I.uv.y * C.p.x * C.up.xyz);
  let inv = 1.0 / rd;
  let ta = (vec3<f32>(-0.5) - ro) * inv;
  let tb = (vec3<f32>(0.5) - ro) * inv;
  let tlo = min(ta, tb);
  let thi = max(ta, tb);
  let t0 = max(max(tlo.x, max(tlo.y, tlo.z)), 0.0);
  let t1 = min(thi.x, min(thi.y, thi.z));
  var seen = 0.0;
  var sum = 0.0;
  if (t1 > t0) {
    let ds = 1.0 / C.p.w;
    let count = min(i32(ceil((t1 - t0) / ds)), 1024);
    var alpha = 0.0;
    for (var i = 0; i < count; i++) {
      let t = t0 + (f32(i) + 0.5) * ds;
      let v = textureSampleLevel(VOL, S, ro + rd * t + 0.5, 0.0).r;
      let tf = textureSampleLevel(TF, S, vec2<f32>(v, 0.5), 0.0);
      // An obstacle (-1) stops the ray.
      let a = (1.0 - alpha) * select(1.0 - exp(-tf.a * C.p.z * ds), 1.0, v < -0.5);
      seen += a;
      sum += a * t;
      alpha += a;
      if (alpha > 0.995) { break; }
    }
  }
  O[0] = select(-1.0, sum / max(seen, 1.0e-6), seen > 0.02);
  O[1] = seen;
}
