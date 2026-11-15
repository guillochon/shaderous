// Per-block timesteps with a magnetic field (plans/mhd-plan.md, section 5b): blocks that share
// only an edge (3D) or a vertex (2D) are neighbors too, since constrained transport defines each
// edge's field from every block around it. These are found through the leaf index IDX. Off
// (EDGES false) for hydro, which needs face neighbors alone. Requires P, amr/grid.wgsl and
// bindings IDX (texture_3d<u32>) and, for diag_min, the bins read (`k_of`).
override EDGES: bool = false;

// The leaves sharing an edge or vertex of leaf `slot` (not a face), passed to `visit` as the
// slots found: two points along each edge in 3D (finer leaves split it in two), one per vertex
// in 2D. Writes them to `out` (at most 24) and returns how many.
fn diag_leaves(slot: i32, out: ptr<function, array<i32, 24>>) -> i32 {
  var count = 0;
  let level = u32(bt(slot, 0u));
  let s = P.amr.z - level;
  let span = 1 << s;
  let m = i32(P.amr.y) << P.amr.z;   // finest-level blocks per side
  let c = blk_coord(slot);
  let d = select(3, 2, is2d());
  for (var a = 0; a < d; a++) {
    for (var b = a + 1; b < d; b++) {
      for (var sa = -1; sa <= 1; sa += 2) {
        for (var sb = -1; sb <= 1; sb += 2) {
          var p = c * span;
          var ok = true;
          for (var k = 0; k < 2; k++) {
            let ax = select(b, a, k == 0);
            let sg = select(sb, sa, k == 0);
            var v = select(c[ax] * span - 1, (c[ax] + 1) * span, sg > 0);
            if (v < 0 || v >= m) {
              let bc = select(P.bc_lo[ax], P.bc_hi[ax], v >= m);
              if (bc != 0u) { ok = false; }
              v = ((v % m) + m) % m;
            }
            p[ax] = v;
          }
          if (!ok) { continue; }
          // Along the edge (3D): its first and last finest block.
          let e = 3 - a - b;
          let n_along = select(2, 1, is2d() || span == 1);
          for (var q = 0; q < n_along; q++) {
            var pp = p;
            if (!is2d()) { pp[e] = c[e] * span + select(0, span - 1, q == 1); }
            if (is2d()) { pp.z = 0; }
            (*out)[count] = i32(textureLoad(IDX, pp, 0).x);
            count++;
          }
        }
      }
    }
  }
  return count;
}
