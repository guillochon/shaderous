// The tables of the hydrogen and helium equation of state (src/eos/hhe.ts, eos/hhe.wgsl): the
// free energy of src/eos/hhe_model.ts on a grid in log10 rho and log10 T, for bicubic Hermite
// interpolation, with the starting points of the solve for the temperature.
//
//   node tools/eos/make_hhe_table.mjs            write the tables, their layout, and report accuracy
//   node tools/eos/make_hhe_table.mjs --check    report only (also: --dt 0.01, --dr 0.1 to try a grid)
//   node tools/eos/make_hhe_table.mjs --ortho-para equilibrium
//
// One rgba32float texture per composition, rows of NT texels, x along log10 rho (rows) and y
// along log10 T (columns):
//   rows 0 .. NR-1     per node (h(i, j+1) - h(i, j), h_x, h_y, h_xy): h = F m_u / (k T) per
//                      atomic mass unit, its exact first derivatives and its cross derivative
//                      (central differences of the exact h_x), all per grid step
//   rows NR .. 2NR-1   (h(i+1, j) - h(i, j), log10 T at which the energy is 10^(E0 + k DE) erg/g,
//                      log10 T at which p / rho is 10^(P0 + k DP) erg/g, unused), k the column
// h itself runs from -100 to 100 over the table and is never needed, only its derivatives; in
// single precision its rounding would be 1e-4 of the pressure and 1e-2 of the sound speed. So
// the table holds h's differences between neighbouring nodes, exact to 1e-7 of themselves.
//
// Writes src/eos/hhe_<name>.bin, src/eos/hhe_meta.ts and src/eos/hhe_consts.wgsl.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { hheSecond, hheState } from '../../src/eos/hhe_model.ts';

const LN10 = Math.LN10;
/** The compositions that ship: the code's solar gas with its metals counted as helium, and hydrogen. */
export const TABLES = [
  { name: 'solar', X: 0.7381 },
  { name: 'hydrogen', X: 1 },
];
export const GRID = { r0: -20, r1: 3, dr: 0.1, t0: 1, t1: 8, dt: 0.01, e0: 8.6, e1: 16.5, p0: 8.4, p1: 16.4 };

const hb = (t) => [2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2];
const hb1 = (t) => [6 * t ** 2 - 6 * t, 3 * t ** 2 - 4 * t + 1, -6 * t ** 2 + 6 * t, 3 * t ** 2 - 2 * t];
const hb2 = (t) => [12 * t - 6, 6 * t - 4, -12 * t + 6, 6 * t - 2];

/** The texture of one composition on grid g. */
export function build(o, g = GRID) {
  const nr = Math.round((g.r1 - g.r0) / g.dr) + 1;
  const nt = Math.round((g.t1 - g.t0) / g.dt) + 1;
  const sx = g.dr * LN10;
  const sy = g.dt * LN10;
  const tex = new Float32Array(2 * nr * nt * 4);
  const H = new Float64Array(nr * nt);
  const E = new Float64Array(nr * nt);
  const P = new Float64Array(nr * nt);
  const step = 1e-4;
  for (let i = 0; i < nr; i++) {
    const rho = 10 ** (g.r0 + i * g.dr);
    for (let j = 0; j < nt; j++) {
      const T = 10 ** (g.t0 + j * g.dt);
      const s = hheState(rho, T, o);
      const hxy = (hheState(rho, T * Math.exp(step), o).hn - hheState(rho, T * Math.exp(-step), o).hn) / (2 * step);
      const k = i * nt + j;
      H[k] = s.h;
      E[k] = Math.log10(s.e);
      P[k] = Math.log10(s.pr);
      tex.set([0, s.hn * sx, s.ht * sy, hxy * sx * sy], k * 4);
    }
  }
  const ne = (x0, x1) => (x1 - x0) / (nt - 1);
  const de = ne(g.e0, g.e1);
  const dp = ne(g.p0, g.p1);
  // log10 T at which the row's column `a` (rising with T, or made so) reaches x.
  const invert = (a, i, x) => {
    const at = (j) => a[i * nt + j];
    if (x <= at(0)) return g.t0;
    let top = at(0);
    for (let j = 1; j < nt; j++) {
      const v = at(j);
      if (v >= x) return g.t0 + (j - 1 + (x - top) / (v - top)) * g.dt;
      // Not rising (the pressure, where the stand-in for pressure ionization gives way): the
      // first temperature at which it is reached.
      top = Math.max(top, v);
    }
    return g.t1;
  };
  for (let i = 0; i < nr; i++)
    for (let j = 0; j < nt; j++) {
      const k = i * nt + j;
      if (j + 1 < nt) tex[k * 4] = H[k + 1] - H[k];
      const m = ((nr + i) * nt + j) * 4;
      if (i + 1 < nr) tex[m] = H[k + nt] - H[k];
      tex[m + 1] = invert(E, i, g.e0 + j * de);
      tex[m + 2] = invert(P, i, g.p0 + j * dp);
    }
  return { tex, nr, nt, layout: { r0: g.r0, dr: g.dr, nr, t0: g.t0, dt: g.dt, nt, e0: g.e0, de, p0: g.p0, dp } };
}

/** h's derivatives from the texture, as the shader and the float64 mirror have them. */
export function lookup(t, lr, lt) {
  const { tex, nr, nt, layout: L } = t;
  const x = Math.min(Math.max((lr - L.r0) / L.dr, 0), nr - 1.0001);
  const y = Math.min(Math.max((lt - L.t0) / L.dt, 0), nt - 1.0001);
  const i = Math.floor(x);
  const j = Math.floor(y);
  const u = x - i;
  const v = y - j;
  const node = (a, b) => Array.from(tex.subarray(((i + a) * nt + j + b) * 4, ((i + a) * nt + j + b) * 4 + 4));
  const c = [node(0, 0), node(0, 1), node(1, 0), node(1, 1)];
  const dx = tex[((nr + i) * nt + j) * 4];
  c[3][0] = dx + c[2][0];
  c[1][0] = c[0][0];
  c[2][0] = dx;
  c[0][0] = 0;
  const comb = (A, B) => {
    let s = 0;
    for (let a = 0; a < 2; a++)
      for (let b = 0; b < 2; b++) {
        const q = c[a * 2 + b];
        s += A[2 * a] * B[2 * b] * q[0] + A[2 * a + 1] * B[2 * b] * q[1] + A[2 * a] * B[2 * b + 1] * q[2] + A[2 * a + 1] * B[2 * b + 1] * q[3];
      }
    return s;
  };
  const sx = 1 / (L.dr * LN10);
  const sy = 1 / (L.dt * LN10);
  return { hn: comb(hb1(u), hb(v)) * sx, ht: comb(hb(u), hb1(v)) * sy, hnn: comb(hb2(u), hb(v)) * sx * sx, hnt: comb(hb1(u), hb1(v)) * sx * sy, htt: comb(hb(u), hb2(v)) * sy * sy };
}

/** How far the table is from the model at n states between the nodes: [p99, worst] of e, p and c_s^2. */
export function accuracy(t, o, n = 4000, rhoMax = -1) {
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  const err = { e: [], p: [], cs2: [] };
  const L = t.layout;
  let worst = null;
  for (let k = 0; k < n; k++) {
    const lr = L.r0 + (rhoMax - L.r0) * rnd();
    const lt = L.t0 + (L.t0 + (L.nt - 1) * L.dt - L.t0) * rnd();
    const x = hheSecond(10 ** lr, 10 ** lt, o);
    const q = lookup(t, lr, lt);
    const c = (q.hn + q.hnn + (q.hn + q.hnt) ** 2 / -(q.ht + q.htt)) / (x.cs2 / (x.pr / x.hn));
    err.e.push(Math.abs(q.ht / x.ht - 1));
    err.p.push(Math.abs(q.hn / x.hn - 1));
    err.cs2.push(Math.abs(c - 1));
    if (!worst || err.cs2[k] > worst[0]) worst = [err.cs2[k], lr, lt];
  }
  const out = {};
  for (const [k, a] of Object.entries(err)) {
    a.sort((p, q) => p - q);
    out[k] = [a[Math.floor(0.99 * (n - 1))], a[n - 1]];
  }
  out.worstAt = worst;
  return out;
}

function main() {
  const arg = (k, d) => {
    const i = process.argv.indexOf(k);
    return i < 0 ? d : process.argv[i + 1];
  };
  const check = process.argv.includes('--check');
  const g = { ...GRID, dt: Number(arg('--dt', GRID.dt)), dr: Number(arg('--dr', GRID.dr)) };
  const orthoParaEquilibrium = arg('--ortho-para', 'fixed') === 'equilibrium';
  const src = fileURLToPath(new URL('../../src/eos/', import.meta.url));
  let layout;
  for (const c of TABLES) {
    const o = { X: c.X, orthoParaEquilibrium };
    const t0 = Date.now();
    const t = build(o, g);
    layout = t.layout;
    const a = accuracy(t, o);
    const f = (v) => v.map((x) => x.toExponential(1)).join(' / ');
    console.log(`${c.name}: ${t.nr} x ${t.nt} nodes, ${(t.tex.byteLength / 1e6).toFixed(2)} MB, ${((Date.now() - t0) / 1e3).toFixed(0)} s`);
    console.log(`  against the model below 0.1 g/cm^3 (p99 / worst): e ${f(a.e)}, p ${f(a.p)}, c_s^2 ${f(a.cs2)} (worst at log rho ${a.worstAt[1].toFixed(2)}, log T ${a.worstAt[2].toFixed(2)})`);
    if (!check) writeFileSync(`${src}hhe_${c.name}.bin`, new Uint8Array(t.tex.buffer));
  }
  if (check) return;
  const meta = { width: layout.nt, rows: 2 * layout.nr, table: layout, orthoPara: orthoParaEquilibrium ? 'equilibrium' : '3:1', tables: TABLES };
  writeFileSync(`${src}hhe_meta.ts`,
    '// Generated by tools/eos/make_hhe_table.mjs: do not edit. The layout of the hydrogen and\n'
    + '// helium tables hhe_<name>.bin (src/eos/hhe.ts), and the compositions they are for.\n'
    + `export const HHE_META = ${JSON.stringify(meta)} as const;\n`);
  const f = (x) => {
    const s = String(Math.fround(x));
    return /[.e]/.test(s) ? s : `${s}.0`;
  };
  const L = layout;
  writeFileSync(`${src}hhe_consts.wgsl`, [
    '// Generated by tools/eos/make_hhe_table.mjs: do not edit. The layout of the table bound as',
    '// EOS_TAB for eos/hhe.wgsl (eos/table.wgsl, hhe_<name>.bin).',
    `const TAB_X0: f32 = ${f(L.r0)};`, `const TAB_DX: f32 = ${f(L.dr)};`, `const TAB_NX: i32 = ${L.nr};`,
    `const TAB_Y0: f32 = ${f(L.t0)};`, `const TAB_DY: f32 = ${f(L.dt)};`, `const TAB_NY: i32 = ${L.nt};`,
    `const HH_E0: f32 = ${f(L.e0)};`, `const HH_DE: f32 = ${f(L.de)};`,
    `const HH_P0: f32 = ${f(L.p0)};`, `const HH_DP: f32 = ${f(L.dp)};`, ''].join('\n'));
  console.log('wrote src/eos/hhe_*.bin, hhe_meta.ts, hhe_consts.wgsl');
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
