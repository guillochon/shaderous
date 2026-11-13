// Volume lookup for the ray-marcher and the depth pick (the <volume> slot): the uniform display
// volume VOL, which spans the box. Requires render/camera.wgsl.
struct Look {
  s: vec4<f32>,   // the display volume's texel, filtered (render/field.wgsl, amr/field.wgsl)
  cell: f32,      // size of a cell there, in units of the box side
}

// At p, in the box [-0.5, 0.5]^3.
fn vol_look(p: vec3<f32>) -> Look {
  return Look(textureSampleLevel(VOL, S, p + 0.5, 0.0), 1.0 / C.q.z);
}

// How far the ray moves past a sample: C.p.w samples across the box.
fn vol_step(l: Look) -> f32 {
  return 1.0 / C.p.w;
}

// Most samples along a ray: enough to cross the box diagonally at the step above.
const VOL_MAX_SAMPLES = 1024;
