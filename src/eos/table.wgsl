// What the tabulated equations of state share (eos/helmholtz.wgsl, eos/hhe.wgsl): a free
// energy h(x, y) on a grid, x the log10 of a density along the texture's rows and y = log10 T
// along its columns, with (h, h_x, h_y, h_xy) per node and bicubic Hermite polynomials between
// them; the lookup of its derivatives, and the solve for the temperature by Newton's method in
// ln T. Pressure and energy are first derivatives of h, so they are consistent. The texture
// and the Hermite basis are eos/table_tex.wgsl, which eos/aneos.wgsl reads too.
// Float64 mirror: src/eos/table.ts.
//
// The module that includes this defines:
//   TAB_X0, TAB_DX, TAB_NX   the grid along x: first node, spacing (dex), nodes
//   TAB_Y0, TAB_DY, TAB_NY   the same along y
//   TAB_DIFF                 false: a node's first channel is h. True: it is h's difference to
//                            the next node along y, and rows TAB_NX .. 2 TAB_NX - 1 hold the
//                            difference to the next node along x. Only h's derivatives are
//                            ever needed, and where h is large (hundreds, against steps of
//                            0.01 between nodes) single precision rounds it too coarsely.
//   TAB_LNT_MIN, TAB_LNT_MAX the table's ends in ln T
//   struct TabCtx            what its own terms need at a state
//   fn tab_fd(at: TabAt, lnT: f32, byP: bool, ctx: TabCtx) -> vec2<f32>
//                            the energy per gram (byP false) or p / rho (true) at ln T, and its
//                            derivative in ln T
//   fn tab_above(fd: vec2<f32>, goal: f32, byP: bool, ctx: TabCtx) -> f32
//                            how far past TAB_LNT_MAX, in ln T, the goal lies, from tab_fd there

#include "eos/table_tex.wgsl"

// The four nodes of the table's cell (i, j): i along x, j along y. With differences
// (TAB_DIFF), h is counted from the cell's first node.
fn tab_cell(i: i32, j: i32) -> TabCell {
  var c = TabCell(
    textureLoad(EOS_TAB, vec2<i32>(j, i), 0),
    textureLoad(EOS_TAB, vec2<i32>(j + 1, i), 0),
    textureLoad(EOS_TAB, vec2<i32>(j, i + 1), 0),
    textureLoad(EOS_TAB, vec2<i32>(j + 1, i + 1), 0),
  );
  if (TAB_DIFF) {
    let dx = textureLoad(EOS_TAB, vec2<i32>(j, TAB_NX + i), 0).x;
    c.c11.x = dx + c.c10.x;
    c.c01.x = c.c00.x;
    c.c10.x = dx;
    c.c00.x = 0.0;
  }
  return c;
}
// Where x lies on the grid, in nodes.
fn tab_x(lx: f32) -> f32 { return clamp((lx - TAB_X0) / TAB_DX, 0.0, f32(TAB_NX) - 1.0001); }
fn tab_y(ly: f32) -> f32 { return clamp((ly - TAB_Y0) / TAB_DY, 0.0, f32(TAB_NY) - 1.0001); }
// The table's cell at (x, y), and where in it the point lies.
struct TabAt { c: TabCell, u: f32, v: f32 }
fn tab_at(lx: f32, ly: f32) -> TabAt {
  let x = tab_x(lx);
  let y = tab_y(ly);
  let i = i32(floor(x));
  let j = i32(floor(y));
  return TabAt(tab_cell(i, j), x - f32(i), y - f32(j));
}
// h's first and second derivatives per ln of the density (n) and ln T there (h itself is never
// needed).
struct TabH { hn: f32, ht: f32, hnn: f32, hnt: f32, htt: f32 }
fn tab_lookup(a: TabAt) -> TabH {
  let sx = 1.0 / (TAB_DX * TAB_LN10);
  let sy = 1.0 / (TAB_DY * TAB_LN10);
  let A0 = tab_hb(a.u);
  let A1 = tab_hb1(a.u);
  let B0 = tab_hb(a.v);
  let B1 = tab_hb1(a.v);
  var r: TabH;
  r.hn = tab_comb(a.c, A1, B0) * sx;
  r.ht = tab_comb(a.c, A0, B1) * sy;
  r.hnn = tab_comb(a.c, tab_hb2(a.u), B0) * sx * sx;
  r.hnt = tab_comb(a.c, A1, B1) * sx * sy;
  r.htt = tab_comb(a.c, A0, tab_hb2(a.v)) * sy * sy;
  return r;
}
// The two of them Newton's method needs: (h_T, h_TT) to solve by the energy, (h_n, h_nT) by
// the pressure. A third of the work of the whole set.
fn tab_lookup2(a: TabAt, byP: bool) -> vec2<f32> {
  let sx = 1.0 / (TAB_DX * TAB_LN10);
  let sy = 1.0 / (TAB_DY * TAB_LN10);
  let B1 = tab_hb1(a.v);
  if (byP) {
    let A1 = tab_hb1(a.u);
    return vec2<f32>(tab_comb(a.c, A1, tab_hb(a.v)) * sx, tab_comb(a.c, A1, B1) * sx * sy);
  }
  let A0 = tab_hb(a.u);
  return vec2<f32>(tab_comb(a.c, A0, B1) * sy, tab_comb(a.c, A0, tab_hb2(a.v)) * sy * sy);
}

// A table of Newton's starting points, by bilinear interpolation: the texel at (gx, gy) nodes
// along the rows from `row` and along the columns.
fn tab_bilinear(row: i32, gx: f32, gy: f32) -> vec4<f32> {
  let i = i32(floor(gx));
  let j = i32(floor(gy));
  let u = gx - f32(i);
  let v = gy - f32(j);
  return mix(mix(textureLoad(EOS_TAB, vec2<i32>(j, row + i), 0), textureLoad(EOS_TAB, vec2<i32>(j + 1, row + i), 0), v),
             mix(textureLoad(EOS_TAB, vec2<i32>(j, row + i + 1), 0), textureLoad(EOS_TAB, vec2<i32>(j + 1, row + i + 1), 0), v), u);
}

// Newton's method converges quadratically, so a step this small in ln T leaves an error of its
// square times half the curvature (at most a few): under the 1.5e-6 single precision holds
// ln T to. The step is taken and not checked by another evaluation; what the callers return is
// then corrected from the evaluation they make anyway (tab_fix).
const TAB_LAST_STEP: f32 = 3e-4;

// ln T at which the module's energy per gram (byP false) or p / rho (byP true) is `goal`, from
// the starting point lnT0, at x = lx; `hi` is the most ln T may be on the way (TAB_LNT_MAX, or
// less where the module knows a bound). Past the table's ends the module's own extrapolation:
// linear in T below, tab_above above.
fn tab_solve(lnT0: f32, lx: f32, goal: f32, byP: bool, hi: f32, ctx: TabCtx) -> f32 {
  var lnT = lnT0;
  // The density's place in the table does not change, and after the first step or two neither
  // does the temperature's cell: its nodes are fetched only when it does.
  let x = tab_x(lx);
  let i = i32(floor(x));
  var j = -1;
  var cell: TabCell;
  for (var it = 0; it < 10; it++) {
    let y = tab_y(lnT / TAB_LN10);
    let jn = i32(floor(y));
    if (jn != j) {
      j = jn;
      cell = tab_cell(i, j);
    }
    let fd = tab_fd(TabAt(cell, x - f32(i), y - f32(j)), lnT, byP, ctx);
    if (lnT <= TAB_LNT_MIN && fd.x > goal) { return TAB_LNT_MIN + log(goal / fd.x); }
    if (lnT >= TAB_LNT_MAX && fd.x < goal) { return TAB_LNT_MAX + tab_above(fd, goal, byP, ctx); }
    let step = clamp(-(fd.x - goal) / fd.y, -1.0, 1.0);
    lnT = clamp(lnT + step, TAB_LNT_MIN, hi);
    // At an end of the table the next pass decides whether the answer lies beyond it.
    if (abs(step) < TAB_LAST_STEP && lnT > TAB_LNT_MIN && lnT < TAB_LNT_MAX) { break; }
  }
  return lnT;
}

// The evaluation at the temperature found gives what is returned, and with it one more
// Newton step for nothing: the step in ln T that would take the solved quantity, f with slope
// df there, onto `goal`. Callers move what they return by its own slope times this, so the pair
// (energy, pressure) they hand back is consistent to second order in what the iteration left,
// with no evaluation the iteration did not already owe. Inside the table only: beyond it the
// temperature is an extrapolation.
fn tab_fix(f: f32, df: f32, goal: f32, lnT: f32) -> f32 {
  if (lnT <= TAB_LNT_MIN || lnT >= TAB_LNT_MAX) { return 0.0; }
  return clamp((goal - f) / df, -1e-3, 1e-3);
}
