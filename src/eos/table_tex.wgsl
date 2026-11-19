// The texture every tabulated equation of state reads (EOS_TAB, binding 22: whichever the
// run's equation of state uses, eos/table.ts bindTable) and the bicubic Hermite basis on it:
// per node (f, f_x, f_y, f_xy), x along the texture's rows and y along its columns. A free
// energy's lookup and solve (eos/table.wgsl) are built on this, and so are the ANEOS materials'
// pressure and energy (eos/aneos.wgsl), which are not a free energy's derivatives.
// Float64 mirror: hermite, hermite1 in src/eos/table.ts.

@group(0) @binding(22) var EOS_TAB: texture_2d<f32>;

const TAB_LN10: f32 = 2.30258509;

fn tab_hb(t: f32) -> vec4<f32> { return vec4<f32>(2.0*t*t*t - 3.0*t*t + 1.0, t*t*t - 2.0*t*t + t, -2.0*t*t*t + 3.0*t*t, t*t*t - t*t); }
fn tab_hb1(t: f32) -> vec4<f32> { return vec4<f32>(6.0*t*t - 6.0*t, 3.0*t*t - 4.0*t + 1.0, -6.0*t*t + 6.0*t, 3.0*t*t - 2.0*t); }
fn tab_hb2(t: f32) -> vec4<f32> { return vec4<f32>(12.0*t - 6.0, 6.0*t - 4.0, -12.0*t + 6.0, 6.0*t - 2.0); }
// sum over the four nodes of A_a(u) B_b(v) (f, f_x, f_y, f_xy): basis entries 0 and 2 weigh the
// value at nodes 0 and 1, entries 1 and 3 the slope there. Written out node by node, with no
// array: arrays in a function live in each thread's local memory, and on Firefox (Direct3D,
// NVIDIA) the adaptive mesh's update ran out of it (SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE) and lost the device.
fn tab_node(q: vec4<f32>, a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
  return a0 * b0 * q.x + a1 * b0 * q.y + a0 * b1 * q.z + a1 * b1 * q.w;
}
// The nodes (u, v) = (0, 0), (0, 1), (1, 0), (1, 1).
struct TabCell { c00: vec4<f32>, c01: vec4<f32>, c10: vec4<f32>, c11: vec4<f32> }
fn tab_comb(c: TabCell, A: vec4<f32>, B: vec4<f32>) -> f32 {
  return tab_node(c.c00, A.x, A.y, B.x, B.y) + tab_node(c.c01, A.x, A.y, B.z, B.w)
       + tab_node(c.c10, A.z, A.w, B.x, B.y) + tab_node(c.c11, A.z, A.w, B.z, B.w);
}
