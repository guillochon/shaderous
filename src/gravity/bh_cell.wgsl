// What one cell contributes to the point masses' reduction. Requires gravity/bh.wgsl, a binding
// BH: Masses, and Red (gpu/reduce.wgsl) with 41 channels. A reduction sums a group of four
// masses, those from BHG.base.x on (gravity/mass_sums.ts runs one per group in use):
//   0              gas mass
//   1 + 10 i ...   for the group's mass i: its pull on the gas (m a, 3), then what its sink
//                  takes from the cell this step: mass (1), momentum (3), and angular momentum
//                  about the mass in its rest frame (3)
// `vol` is the cell's volume, `x` its center, u1 its (rho, momentum), `dt` the step.
// Mass i's share of a cell: its pull on the gas (m a), then what its sink takes: mass and
// momentum, and angular momentum about the mass in its rest frame. The sink leaves the rest
// of rho and mom for the next mass.
struct BhTake { pull: vec3<f32>, taken: vec4<f32>, l: vec3<f32> }
// The cell being reduced, for a source whose pull is not the cell center's (bh_pull): a
// uniform grid's (c, 0), a block grid's (l, slot). red_cell() sets it.
var<private> g_rc: vec4<i32>;
fn bh_take(i: u32, x: vec3<f32>, m: f32, vol: f32, dt: f32, rho: ptr<function, f32>, mom: ptr<function, vec3<f32>>) -> BhTake {
  let b = BH.list[i];
  var t: BhTake;
  t.pull = m * bh_pull(x, b);
  t.taken = vec4<f32>(0.0);
  t.l = vec3<f32>(0.0);
  let f = sink_fraction(b, x, *rho, dt);
  if (f > 0.0) {
    t.taken = vec4<f32>(f * *rho * vol, f * (*mom).x * vol, f * (*mom).y * vol, f * (*mom).z * vol);
    t.l = cross(x - b.pos.xyz, *mom - *rho * b.vel.xyz) * (f * vol);
    *rho *= 1.0 - f;
    *mom *= 1.0 - f;
  }
  return t;
}

// Written out mass by mass, for the four masses of a group (MASS_GROUP, gravity/mass_sums.ts):
// with every index into the record a constant it stays in registers, where a loop kept it in
// each thread's local memory (gpu/reduce.ts says why that matters).
struct MassGroup { base: vec4<u32> }   // x: the group's first mass
@group(0) @binding(6) var<uniform> BHG: MassGroup;

fn bh_cell(u1: vec4<f32>, vol: f32, x: vec3<f32>, dt: f32) -> Red {
  var r = red_identity(); // all sums: zeros
  let m = u1.x * vol;
  r[0] = m;
  // The sinks take their share one after another, as gravity/sink.wgsl removes it: first those
  // of the masses before this group, whose takes other reductions sum.
  var rho = u1.x;
  var mom = u1.yzw;
  let n = mass_count();
  let b = BHG.base.x;
  for (var j = 0u; j < min(b, n); j++) {
    let f = sink_fraction(BH.list[j], x, rho, dt);
    if (f > 0.0) {
      rho *= 1.0 - f;
      mom *= 1.0 - f;
    }
  }
  if (b + 0u < n) {
    let t = bh_take(b + 0u, x, m, vol, dt, &rho, &mom);
    r[1] = t.pull.x;
    r[2] = t.pull.y;
    r[3] = t.pull.z;
    r[4] = t.taken.x;
    r[5] = t.taken.y;
    r[6] = t.taken.z;
    r[7] = t.taken.w;
    r[8] = t.l.x;
    r[9] = t.l.y;
    r[10] = t.l.z;
  }
  if (b + 1u < n) {
    let t = bh_take(b + 1u, x, m, vol, dt, &rho, &mom);
    r[11] = t.pull.x;
    r[12] = t.pull.y;
    r[13] = t.pull.z;
    r[14] = t.taken.x;
    r[15] = t.taken.y;
    r[16] = t.taken.z;
    r[17] = t.taken.w;
    r[18] = t.l.x;
    r[19] = t.l.y;
    r[20] = t.l.z;
  }
  if (b + 2u < n) {
    let t = bh_take(b + 2u, x, m, vol, dt, &rho, &mom);
    r[21] = t.pull.x;
    r[22] = t.pull.y;
    r[23] = t.pull.z;
    r[24] = t.taken.x;
    r[25] = t.taken.y;
    r[26] = t.taken.z;
    r[27] = t.taken.w;
    r[28] = t.l.x;
    r[29] = t.l.y;
    r[30] = t.l.z;
  }
  if (b + 3u < n) {
    let t = bh_take(b + 3u, x, m, vol, dt, &rho, &mom);
    r[31] = t.pull.x;
    r[32] = t.pull.y;
    r[33] = t.pull.z;
    r[34] = t.taken.x;
    r[35] = t.taken.y;
    r[36] = t.taken.z;
    r[37] = t.taken.w;
    r[38] = t.l.x;
    r[39] = t.l.y;
    r[40] = t.l.z;
  }
  return r;
}
