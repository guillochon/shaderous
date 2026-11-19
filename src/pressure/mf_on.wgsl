// Implicit pressure with species (hydro/species.wgsl): the correction leaves in MF, the update's
// face mass fluxes, the mass each face carried over the whole step, advection and correction
// together. The species, moved after the correction with these fluxes, then move exactly as the
// density did, and each species' mass is conserved. (Moved with the advective fluxes alone and
// scaled to the corrected density, a liquid's species drifted by the order of the correction:
// 2e-3 of their mass at Mach 0.1.) Each cell writes the faces below it along each axis, and the
// last cell along an axis the face above it too.
@group(0) @binding(13) var<storage, read_write> MF: array<f32>;

fn mf_at(c: vec3<i32>, axis: i32) -> u32 {
  let n = vec3<u32>(P.dims.xyz) + 1u;
  let u = vec3<u32>(c);
  return ((u.z * n.y + u.y) * n.x + u.x) * 3u + u32(axis);
}

// The face below cell c along `axis` (up false) or above it (up true), if this cell writes it.
fn mf_mine(c: vec3<i32>, axis: i32, up: bool) -> bool {
  return !up || c[axis] == i32(P.dims[axis]) - 1;
}

// Set the face's mass flux to F.
fn mf_set(c: vec3<i32>, axis: i32, up: bool, F: f32) {
  if (!mf_mine(c, axis, up)) { return; }
  var q = c;
  if (up) { q[axis] += 1; }
  MF[mf_at(q, axis)] = F;
}

// Add dF to the face's mass flux.
fn mf_add(c: vec3<i32>, axis: i32, up: bool, dF: f32) {
  if (!mf_mine(c, axis, up)) { return; }
  var q = c;
  if (up) { q[axis] += 1; }
  let i = mf_at(q, axis);
  MF[i] = MF[i] + dF;
}
