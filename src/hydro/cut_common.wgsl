// State redistribution for partly filled cells (hydro/cut.ts; Berger & Giuliani 2021, as
// src/reference/cutcell.ts redistribute): a cell holding under CUT_SMALL of its volume in gas
// is merged with its neighbors away from the wall: along each axis the wall faces at least half
// as squarely as the most (one neighbor; two axes, the three of their square; three, the seven
// of their cube). Every gas cell's state becomes the mean of the merged neighborhoods' means it
// is in, each neighborhood's taken by volume with each of its cells counted over the number of
// neighborhoods it is in: conservative. Written as gathers: a cell finds the small cells whose
// neighborhoods hold it among its 26 (8 in 2D) neighbors.
// Requires P and hydro/cut_flags.wgsl.

fn cut_axes() -> i32 {
  return select(3, 2, P.dims.z == 1u);
}

// A small cell's neighbors away from the wall: along each axis the wall faces at least half as
// squarely as the most, the offsets' signs (`away`) and which axes (`take`), and whether any of
// them holds gas. Where none does (a sliver between a thin body's two walls), the cell merges
// with the face neighbor holding the most gas instead (`fallback`; 0 where none has any).
struct CutHood { small: bool, away: vec3<i32>, take: vec3<i32>, any: bool, fallback: vec3<i32> }

fn cut_hood(q: vec3<i32>) -> CutHood {
  var h = CutHood(false, vec3<i32>(1), vec3<i32>(0), false, vec3<i32>(0));
  let kq = cut_at(q);
  if (!(kq.x > 0.0 && kq.x < CUT_SMALL)) { return h; }
  let axes = cut_axes();
  let w = cut_wall_vector(q, axes);
  let m = max(max(abs(w.x), abs(w.y)), abs(w.z));
  if (m == 0.0) { return h; }
  h.small = true;
  h.away = vec3<i32>(select(vec3<i32>(1), vec3<i32>(-1), w > vec3<f32>(0.0)));
  h.take = vec3<i32>(select(vec3<i32>(0), vec3<i32>(1), abs(w) >= vec3<f32>(0.5 * m)));
  if (axes == 2) { h.take.z = 0; }
  for (var iz = 0; iz <= h.take.z; iz++) {
    for (var iy = 0; iy <= h.take.y; iy++) {
      for (var ix = 0; ix <= h.take.x; ix++) {
        let o = vec3<i32>(ix, iy, iz) * h.away;
        if (all(o == vec3<i32>(0))) { continue; }
        if (cut_at(q + o).x > 0.0) { h.any = true; }
      }
    }
  }
  if (!h.any) {
    var best = 0.0;
    for (var a = 0; a < axes; a++) {
      for (var sgn = -1; sgn <= 1; sgn += 2) {
        var o = vec3<i32>(0);
        o[a] = sgn;
        let p = cut_at(q + o).x;
        if (p > best) {
          best = p;
          h.fallback = o;
        }
      }
    }
  }
  return h;
}

// Whether cell q's neighborhood holds the cell at offset o from it (o not 0).
fn cut_in_hood(q: vec3<i32>, o: vec3<i32>) -> bool {
  if (cut_at(q + o).x <= 0.0) { return false; }
  let h = cut_hood(q);
  if (!h.small) { return false; }
  if (!h.any) { return all(o == h.fallback) && any(h.fallback != vec3<i32>(0)); }
  for (var a = 0; a < 3; a++) {
    if (o[a] == 0) { continue; }
    if (h.take[a] == 0 || o[a] != h.away[a]) { return false; }
  }
  return true;
}

// The number of neighborhoods cell c is in: its own, and the small cells' about it that hold it.
fn cut_count(c: vec3<i32>) -> f32 {
  var n = 1.0;
  let zr = select(1, 0, P.dims.z == 1u);
  for (var dz = -zr; dz <= zr; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let o = vec3<i32>(dx, dy, dz);
        if (all(o == vec3<i32>(0))) { continue; }
        if (cut_in_hood(c - o, o)) { n += 1.0; }
      }
    }
  }
  return n;
}
