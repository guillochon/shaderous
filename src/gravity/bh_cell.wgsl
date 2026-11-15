// What one cell contributes to the point masses' reduction. Requires gravity/bh.wgsl, a binding
// BH: Masses, and Red (gpu/reduce.wgsl) with 41 channels:
//   0              gas mass
//   1 + 10 i ...   for mass i: its pull on the gas (m a, 3), then what its sink takes from the
//                  cell this step: mass (1), momentum (3), and angular momentum about the mass
//                  in its rest frame (3)
// `vol` is the cell's volume, `x` its center, u1 its (rho, momentum), `dt` the step.
// Mass i's share of a cell: its pull on the gas (m a), then what its sink takes: mass and
// momentum, and angular momentum about the mass in its rest frame. The sink leaves the rest
// of rho and mom for the next mass.
struct BhTake { pull: vec3<f32>, taken: vec4<f32>, l: vec3<f32> }
fn bh_take(i: u32, x: vec3<f32>, m: f32, vol: f32, dt: f32, rho: ptr<function, f32>, mom: ptr<function, vec3<f32>>) -> BhTake {
  let b = BH.list[i];
  var t: BhTake;
  t.pull = m * bh_accel(x, b);
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

// Written out mass by mass, for the four masses there can be (MAX_MASSES, src/sim.ts): with
// every index a constant the record stays in registers, where a loop kept it in each thread's
// local memory (gpu/reduce.ts says why that matters).
fn bh_cell(u1: vec4<f32>, vol: f32, x: vec3<f32>, dt: f32) -> Red {
  var r = red_identity(); // all sums: zeros
  let m = u1.x * vol;
  r[0] = m;
  // The sinks take their share one after another, as gravity/sink.wgsl removes it.
  var rho = u1.x;
  var mom = u1.yzw;
  let n = mass_count();
  if (0u < n) {
    let t = bh_take(0u, x, m, vol, dt, &rho, &mom);
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
  if (1u < n) {
    let t = bh_take(1u, x, m, vol, dt, &rho, &mom);
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
  if (2u < n) {
    let t = bh_take(2u, x, m, vol, dt, &rho, &mom);
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
  if (3u < n) {
    let t = bh_take(3u, x, m, vol, dt, &rho, &mom);
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
