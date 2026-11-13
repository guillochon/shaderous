// Debris energies (debris/debris.ts): gas binned by its specific orbital energy about one point
// mass. The update passes add what leaves the box through an open face (the ledger); a census
// pass adds what is still inside, apart from a surviving self-bound core.
//
// The totals are 64-bit fixed-point sums, kept as (low, high) pairs of u32 atomics with the
// carry propagated by hand. Integer addition is exact and order-free, so the totals do not drift
// however many small amounts go into them, and the same run gives the same bits every time.
// Requires P and gravity/bh.wgsl.
@group(0) @binding(20) var<storage, read> BH: Masses;

struct DebrisHead {
  bins: vec4<f32>,  // (energy at the lower edge of the first bin, bins per unit energy,
                    //  fixed-point units per unit mass, per unit momentum)
  core: vec4<f32>,  // census: velocity of the surviving core (xyz); w is 1 to set apart the gas
                    //  bound to it, 2 to set apart the gas denser than a threshold (the seed
                    //  of the iteration that finds the core), 0 for no core
  sel: vec4<u32>,   // (index of the point mass, number of bins, 1 to bin only the gas the
                    //  passive scalar marks, the seed's density threshold as f32 bits)
}
// Slots of two words each: 0 is below the first bin, 1 to nb the bins, nb + 1 above the last,
// nb + 2 the gas left out of the bins (ambient gas, where the scalar marks the debris); then
// (census only) the core's mass and its momentum along x, y and z.
struct Debris { head: DebrisHead, acc: array<atomic<u32>> }
@group(0) @binding(21) var<storage, read_write> DEB: Debris;

// Add v fixed-point units (any sign, rounded to the nearest) to a slot.
fn deb_add(slot: u32, v: f32) {
  let r = round(v);
  if (r == 0.0) { return; }
  let hi = floor(r / 4294967296.0);
  let lo = u32(min(r - hi * 4294967296.0, 4294967295.0));
  var h = bitcast<u32>(i32(hi));
  let old = atomicAdd(&DEB.acc[2u * slot], lo);
  if (old > 0xffffffffu - lo) { h += 1u; }
  if (h != 0u) { atomicAdd(&DEB.acc[2u * slot + 1u], h); }
}

// The slot of a specific energy.
fn deb_slot(e: f32) -> u32 {
  let nb = DEB.head.sel.y;
  if (e != e) { return nb + 1u; }
  let k = floor((e - DEB.head.bins.x) * DEB.head.bins.y);
  return u32(clamp(k + 1.0, 0.0, f32(nb + 1u)));
}

// Specific orbital energy about the chosen mass of gas moving at v where the true potential
// (self-gravity and the point masses) is phi. Velocities are relative to the frame, so their
// difference is the inertial one.
fn deb_energy(v: vec3<f32>, phi: f32) -> f32 {
  let m = BH.list[DEB.head.sel.x];
  let dv = v - m.vel.xyz;
  return 0.5 * dot(dv, dv) + phi;
}

// Of mass dm of gas at x moving at v, the part to bin: all of it, or where the scalar marks
// the debris (a star's gas, say, apart from the ambient medium), the scalar's mass ds.
fn deb_weight(dm: f32, ds: f32) -> f32 {
  return select(dm, ds, DEB.head.sel.z != 0u);
}

// Mass dm of gas (ds of it marked by the scalar) moving at v in the true potential phi: into
// its energy bin, and what is not binned into the slot for the rest.
fn deb_bin(v: vec3<f32>, phi: f32, dm: f32, ds: f32) {
  let w = deb_weight(dm, ds);
  if (w != dm) { deb_add(DEB.head.sel.y + 2u, (dm - w) * DEB.head.bins.z); }
  if (w == 0.0) { return; }
  deb_add(deb_slot(deb_energy(v, phi)), w * DEB.head.bins.z);
}

// The update's tally: as deb_bin, with `phi` the potential the hydro passes use (the gravity
// field's w, computed for this step: self-gravity, the point masses half a step ahead, and the
// frame's fictitious term, which is taken out here).
fn deb_tally(x: vec3<f32>, v: vec3<f32>, phi: f32, dm: f32, ds: f32) {
  deb_bin(v, phi - dot(BH.frame.xyz, x), dm, ds);
}

// Census of one cell (conserved state u1, u2, the gas's own potential phi_self, volume dv):
// into the core if it is bound to it, otherwise into its energy bin. The census runs between
// steps, after the masses have moved on from where the gravity field last saw them, so the
// potential is put together here: the gas's own plus the masses' where they are now. Bound to the core means
// 1/2 |v - v_core|^2 + phi_self < 0, phi_self being the gas's own potential. In the seed pass,
// the core is the gas denser than the threshold instead.
fn deb_census(x: vec3<f32>, u1: vec4<f32>, u2: vec4<f32>, phi_self: f32, dv: f32) {
  if (abs(u2.w) > 0.5) { return; }
  let rho = max(u1.x, P.rho_floor);
  let v = u1.yzw / rho;
  let dm = u1.x * dv;
  let ds = clamp(u2.z, 0.0, u1.x) * dv;
  let mode = DEB.head.core.w;
  if (mode > 0.0) {
    var in_core = false;
    if (mode > 1.5) {
      in_core = u1.x > bitcast<f32>(DEB.head.sel.w);
    } else {
      let dc = v - DEB.head.core.xyz;
      in_core = 0.5 * dot(dc, dc) + phi_self < 0.0;
    }
    if (in_core) {
      let w = deb_weight(dm, ds);
      let s = DEB.head.sel.y + 3u;
      if (w != dm) { deb_add(s - 1u, (dm - w) * DEB.head.bins.z); }
      deb_add(s, w * DEB.head.bins.z);
      let p = w * v * DEB.head.bins.w;
      deb_add(s + 1u, p.x);
      deb_add(s + 2u, p.y);
      deb_add(s + 3u, p.z);
      return;
    }
  }
  var phi = phi_self;
  for (var i = 0u; i < mass_count(); i++) { phi += bh_phi(x, BH.list[i], 0.0); }
  deb_bin(v, phi, dm, ds);
}
