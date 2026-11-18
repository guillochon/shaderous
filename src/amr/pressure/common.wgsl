// Implicit pressure on the block grid (plans/incompressible.md, stage 3): shared definitions.
//
// The equation is the uniform grid's (pressure/source.wgsl), alpha p - lap p = b with
// alpha = 1 / (c^2 dt^2), on the leaves. The pressure lives on every block, parents included,
// in the hydro atlas's layout (its ghost texels are not used: a neighbor is read where it is).
//
// Across a face of a cell:
//   - a cell of the same level, in this block or the neighboring one (which may be a parent,
//     holding the restriction of its children);
//   - where the region beyond is one level coarser, the coarse cell there. The fine cell's
//     ghost is on the line between the two centers, p_g = p_i + (2/3) (p_c - p_i), so the
//     flux through the fine face is (2/3) (p_c - p_i) / h;
//   - a wall of the box (nothing crosses), or an open face (p = 0 at the face).
// A leaf cell whose neighbor block is refined takes, through that face, the mean of the fluxes
// of the fine faces that tile it: (4/3) (mean of the fine cells along the face - p) / h in its
// own cell size. That is the same number the fine cells form, so the operator is conservative
// across levels: what plans/incompressible.md 5.2 asks, and what the correction pass
// (amr/pressure/correct.wgsl) then moves as mass.
//
// The solve is multigrid over the tree's levels in full-approximation form: every level
// relaxes the plain operator (the parent's restricted value across a refined face), with the
// difference between that and the composite operator, frozen when the finer level was last
// restricted, added to its right-hand side (TAU). At convergence the leaves satisfy the
// composite equation. The root level, a uniform grid, is solved by the uniform grid's
// multigrid (pressure/multigrid.ts).
#include "hydro/common.wgsl"
#include <eos>
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;

const AP_TWO_THIRDS = 0.6666667;

// 1 / (c^2 dt^2): a liquid's sound speed is one number.
fn ap_alpha() -> f32 {
  let cs = eos_sound_speed(P.eos.x, 0.0);
  return 1.0 / (cs * cs * clock.dt * clock.dt);
}
fn ap_axes() -> i32 {
  return select(3, 2, is2d());
}

// What lies across face (axis a, upper or lower) of cell l of block `slot`.
//   kind 0: cell `l` of block `slot`, at the same level; `fine` if that block is refined (and is
//           not this one)
//   kind 1: cell `l` of the coarser leaf `slot`
//   kind 2: a wall      kind 3: an open face (`only`: it lets nothing in)
struct ANb { kind: i32, slot: i32, l: vec3<i32>, fine: bool, only: bool }

fn ap_nb(slot: i32, l: vec3<i32>, a: i32, up: bool) -> ANb {
  var q = l;
  q[a] += select(-1, 1, up);
  if (q[a] >= 0 && q[a] < BS) { return ANb(0, slot, q, false, false); }
  let f = u32(2 * a) + u32(up);
  let nb = bt(slot, 6u + f);
  if (nb >= 0) {
    q[a] += select(BS, -BS, up);
    return ANb(0, nb, q, !blk_leaf(nb), false);
  }
  if (nb == -1) {
    let cs = bt(slot, 20u + f);
    var lc = ((blk_coord(slot) * BS + q) >> vec3<u32>(1u)) - blk_coord(cs) * BS;
    lc[a] = select(BS - 1, 0, up);
    if (is2d()) { lc.z = 0; }
    return ANb(1, cs, lc, false, false);
  }
  let k = u32(-1 - nb);
  if (bc_wall(k)) { return ANb(2, slot, l, false, false); }
  return ANb(3, slot, l, false, k == BC_OUTFLOW_ONLY);
}

// Fine faces that tile a coarse one: 4, or 2 in 2D.
fn ap_nfine() -> i32 {
  return select(4, 2, is2d());
}
// The t-th fine cell just across face (a, up) of a coarse cell, where `n` (ap_nb, kind 0, fine)
// is the parent's cell there: its block and local index. It sits on its block's edge, and its
// ghost toward the coarse cell is one step back along a.
fn ap_fine(n: ANb, a: i32, up: bool, t: i32) -> ANb {
  var g = 2 * (blk_coord(n.slot) * BS + n.l);
  g[a] += select(1, 0, up);
  let b = (a + 1) % 3;
  let c = (a + 2) % 3;
  g[b] += t & 1;
  if (!is2d()) { g[c] += (t >> 1) & 1; } else if (b == 2) { g[c] += t & 1; g[b] -= t & 1; }
  let fb = g >> vec3<u32>(4u);
  let oct = (fb.x & 1) | ((fb.y & 1) << 1u) | ((fb.z & 1) << 2u);
  return ANb(0, bt(n.slot, 12u + u32(oct)), g - fb * BS, false, false);
}
