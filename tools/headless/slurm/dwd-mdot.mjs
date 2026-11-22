// Mass transfer in the double white dwarf fill sweep (plans/dwd-second-run.md section 1), from the
// runs' .athdf files. For each file: the donor-dyed mass on the accretor's side of the L1 plane,
// and the density at L1 over the donor's central density. For each fill: Mdot, the slope of that
// mass over time after the first quarter orbit. Across the fills: a fit of Mdot = A (f - f0)^3
// above the floor.
//
//   node tools/headless/slurm/dwd-mdot.mjs SWEEP_DIR            every f<fill>/ folder in it
//   node tools/headless/slurm/dwd-mdot.mjs SWEEP_DIR/f0.950     one fill
//   node tools/headless/slurm/dwd-mdot.mjs FILE.athdf ...       files alone (no fits)
// Options: --skip T   leave out outputs before T (default 5, a quarter orbit)
//          --json     print the results as JSON
// Each file's measurement is kept beside it as FILE.mdot.json and reused (delete it to measure
// again). On Unity, on a CPU node, in the container:
//   srun -p umd-cscdr-cpu -A pi_rfisher1_umassd_edu -c 2 --mem=16G -t 2:00:00 \
//     apptainer exec -B /work/pi_rfisher1_umassd_edu ~/containers/playwright-v1.65.0-noble.sif \
//     node tools/headless/slurm/dwd-mdot.mjs /work/pi_rfisher1_umassd_edu/guilloch_access-ci_org/dwd-sweep
//
// The stars carry dye: the donor (the setup's first star) 1, the accretor 0.5, the background 0.
// Gas is the donor's where the dye is above 0.75 and the accretor's between 0.25 and 0.75; their
// centers of mass give the axis, which stays on x since the frame follows the pair. L1 is the
// Roche one of two points of those masses at those centers. Donor gas past the plane through L1
// across x is measured two ways: m75, the mass of gas with dye above 0.75; and mdye, the mass
// times the donor's share if the gas is a mix of the two stars' (2 s - 1, from 0 to 1). The
// density at L1 is read along the x axis (the cells holding y = z = 0), at the Roche L1. Only
// leaf blocks are counted.
import fs from 'node:fs';
import path from 'node:path';
import h5wasm from 'h5wasm/node';

const DONOR = 0.75, ACCRETOR = 0.25; // dye thresholds
const BINS = 8192; // bins along x, for the mass past the plane

function measure(file) {
  const f = new h5wasm.File(file, 'r');
  const a = (k) => f.attrs[k]?.value;
  const t = Number(a('Time'));
  const names = [].concat(a('VariableNames')).map(String);
  const iRho = names.indexOf('rho') >= 0 ? names.indexOf('rho') : 0;
  const iDye = names.indexOf('r0');
  if (iDye < 0) throw new Error(`${file}: no dye (r0) among ${names.join(', ')}`);
  const lo = Array.from(a('RootGridX1')), y0 = Array.from(a('RootGridX2')), z0 = Array.from(a('RootGridX3'));
  const prim = f.get('prim');
  const [, nb, nz, ny, nx] = prim.shape;
  const levels = Array.from(f.get('Levels').value);
  const locs = Array.from(f.get('LogicalLocations').value, Number);
  const x1f = f.get('x1f').value, x2f = f.get('x2f').value, x3f = f.get('x3f').value;
  const x1v = f.get('x1v').value, x2v = f.get('x2v').value, x3v = f.get('x3v').value;

  // A block is a leaf unless a block one level finer lies inside it.
  const key = (l, i, j, k) => `${l}:${i}:${j}:${k}`;
  const parents = new Set();
  for (let b = 0; b < nb; b++) if (levels[b] > 0) parents.add(key(levels[b] - 1, locs[3 * b] >> 1, locs[3 * b + 1] >> 1, locs[3 * b + 2] >> 1));
  const leaf = levels.map((l, b) => !parents.has(key(l, locs[3 * b], locs[3 * b + 1], locs[3 * b + 2])));

  const xmin = lo[0], xmax = lo[1], bw = (xmax - xmin) / BINS;
  const m75 = new Float64Array(BINS), mdye = new Float64Array(BINS), mall = new Float64Array(BINS);
  // Density along the x axis: the finest cell holding (x, 0, 0), per bin.
  const lineRho = new Float64Array(BINS).fill(NaN), lineLev = new Int32Array(BINS).fill(-1);
  const S = { d: { m: 0, x: 0, y: 0 }, a: { m: 0, x: 0, y: 0 } };
  let rhoc = 0, rhocAt = null, mass = 0;
  const SLAB = 512;
  for (let b0 = 0; b0 < nb; b0 += SLAB) {
    const b1 = Math.min(nb, b0 + SLAB);
    const rho = prim.slice([[iRho, iRho + 1], [b0, b1], [], [], []]);
    const dye = prim.slice([[iDye, iDye + 1], [b0, b1], [], [], []]);
    for (let b = b0; b < b1; b++) {
      if (!leaf[b]) continue;
      const dxb = x1f[b * (nx + 1) + 1] - x1f[b * (nx + 1)];
      const dv = dxb * (x2f[b * (ny + 1) + 1] - x2f[b * (ny + 1)]) * (x3f[b * (nz + 1) + 1] - x3f[b * (nz + 1)]);
      // the row of this block on the x axis, if it holds y = z = 0
      const yl = x2f[b * (ny + 1)], yh = x2f[b * (ny + 1) + ny], zl = x3f[b * (nz + 1)], zh = x3f[b * (nz + 1) + nz];
      const onAxis = yl <= 0 && 0 < yh && zl <= 0 && 0 < zh;
      const jAxis = onAxis ? Math.min(ny - 1, Math.floor((0 - yl) / (yh - yl) * ny)) : -1;
      const kAxis = onAxis ? Math.min(nz - 1, Math.floor((0 - zl) / (zh - zl) * nz)) : -1;
      const base = (b - b0) * nz * ny * nx;
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
        const y = x2v[b * ny + j];
        const row = base + (k * ny + j) * nx;
        const axisRow = k === kAxis && j === jAxis;
        for (let i = 0; i < nx; i++) {
          const r = rho[row + i], s = dye[row + i];
          const x = x1v[b * nx + i];
          const m = r * dv;
          mass += m;
          const bin = Math.min(BINS - 1, Math.max(0, Math.floor((x - xmin) / bw)));
          mall[bin] += m;
          const w = Math.min(1, Math.max(0, 2 * s - 1));
          if (w > 0) mdye[bin] += m * w;
          if (s > DONOR) {
            m75[bin] += m;
            const c = S.d; c.m += m; c.x += m * x; c.y += m * y;
            if (r > rhoc) { rhoc = r; rhocAt = [x, y, x3v[b * nz + k]]; }
          } else if (s > ACCRETOR) {
            const c = S.a; c.m += m; c.x += m * x; c.y += m * y;
          }
          if (axisRow) {
            // every bin this cell covers along x
            const c0 = Math.max(0, Math.floor((x - dxb / 2 - xmin) / bw)), c1 = Math.min(BINS - 1, Math.floor((x + dxb / 2 - xmin) / bw - 1e-9));
            for (let c = c0; c <= c1; c++) if (levels[b] > lineLev[c]) { lineLev[c] = levels[b]; lineRho[c] = r; }
          }
        }
      }
    }
  }
  f.close();

  const D = { m: S.d.m, x: S.d.x / S.d.m, y: S.d.y / S.d.m }, A = { m: S.a.m, x: S.a.x / S.a.m, y: S.a.y / S.a.m };
  const sep = Math.hypot(A.x - D.x, A.y - D.y);
  const tilt = (Math.atan2(A.y - D.y, A.x - D.x) * 180) / Math.PI; // 0 or 180 when on x
  // Roche L1 of two points, xi from the donor toward the accretor (G cancels).
  const xcm = (A.m * sep) / (D.m + A.m), w2 = (D.m + A.m) / sep ** 3;
  const g = (xi) => -D.m / xi ** 2 + A.m / (sep - xi) ** 2 + w2 * (xi - xcm);
  let l = 1e-3 * sep, h = (1 - 1e-3) * sep;
  for (let it = 0; it < 100; it++) { const mid = (l + h) / 2; if (g(mid) < 0) l = mid; else h = mid; }
  const dir = Math.sign(A.x - D.x);
  const xL1 = D.x + dir * ((l + h) / 2);
  const binL1 = Math.floor((xL1 - xmin) / bw);
  // past the plane, toward the accretor (the bin holding L1 split by where L1 falls in it)
  const frac = (xL1 - xmin) / bw - binL1;
  const past = (arr) => {
    let s = 0;
    if (dir > 0) { for (let c = binL1 + 1; c < BINS; c++) s += arr[c]; s += arr[binL1] * (1 - frac); }
    else { for (let c = 0; c < binL1; c++) s += arr[c]; s += arr[binL1] * frac; }
    return s;
  };
  // density along the axis, at L1
  const at = (x) => lineRho[Math.min(BINS - 1, Math.max(0, Math.floor((x - xmin) / bw)))];
  return {
    file: path.basename(file), t, blocks: nb, leaves: leaf.filter(Boolean).length, mass,
    donor: { m: D.m, x: D.x, y: D.y, rhoc, rhocAt }, accretor: { m: A.m, x: A.x, y: A.y },
    sep, tilt, xL1,
    m75: past(m75), mdye: past(mdye), mPast: past(mall),
    rhoL1: at(xL1), rhoL1Rel: at(xL1) / rhoc,
  };
}

function cached(file) {
  const c = `${file}.mdot.json`;
  if (fs.existsSync(c) && fs.statSync(c).mtimeMs >= fs.statSync(file).mtimeMs) return JSON.parse(fs.readFileSync(c, 'utf8'));
  const t0 = Date.now();
  const r = measure(file);
  r.seconds = (Date.now() - t0) / 1000;
  fs.writeFileSync(c, JSON.stringify(r, null, 1));
  return r;
}

// .athdf files under a folder (the runner writes them at its top level; dwd.sbatch-style runs in seg-*/)
function athdfs(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory() && e.name.startsWith('seg-')) out.push(...athdfs(p));
    else if (e.isFile() && e.name.endsWith('.athdf')) out.push(p);
  }
  return out;
}

// least squares y = a + b x, with b's standard error
function line(xs, ys) {
  const n = xs.length, mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
  const b = sxy / sxx, a = my - b * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) rss += (ys[i] - a - b * xs[i]) ** 2;
  return { a, b, se: n > 2 ? Math.sqrt(rss / (n - 2) / sxx) : NaN };
}

function rate(rows, skip, k) {
  const use = rows.filter((r) => r.t >= skip - 1e-6);
  if (use.length < 2) return { mdot: NaN, se: NaN, n: use.length };
  const L = line(use.map((r) => r.t), use.map((r) => r[k]));
  return { mdot: L.b, se: L.se, n: use.length };
}

// Mdot = A (f - f0)^3 above the floor: the floor is the least Mdot; fit Mdot^(1/3) on the fills
// whose Mdot is above 3 times it, plainly and with the floor taken off.
function cubeFit(fills, k) {
  const pts = fills.map((F) => ({ f: F.fill, m: F[k].mdot })).filter((p) => Number.isFinite(p.m));
  if (pts.length < 2) return null;
  const floor = Math.max(0, Math.min(...pts.map((p) => p.m)));
  const above = pts.filter((p) => p.m > 3 * floor && p.m > 0);
  const fit = (sub) => {
    const q = above.filter((p) => p.m - sub > 0);
    if (q.length < 2) return null;
    const L = line(q.map((p) => p.f), q.map((p) => Math.cbrt(p.m - sub)));
    return { A: L.b ** 3, f0: -L.a / L.b, used: q.map((p) => p.f) };
  };
  return { floor, plain: fit(0), lessFloor: fit(floor) };
}

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); if (i < 0) return d; const v = args[i + 1]; args.splice(i, 2); return v; };
const skip = Number(opt('--skip', 5));
const asJson = args.includes('--json') ? (args.splice(args.indexOf('--json'), 1), true) : false;
if (!args.length) { console.error('usage: dwd-mdot.mjs SWEEP_DIR | FILL_DIR | FILE.athdf ... [--skip T] [--json]'); process.exit(1); }
await h5wasm.ready;

const g3 = (v) => (Number.isFinite(v) ? v.toPrecision(3) : String(v));
const row = (r) => `t=${r.t.toFixed(2)} blocks=${r.leaves}/${r.blocks} M=${r.mass.toFixed(4)} Md=${r.donor.m.toFixed(4)} Ma=${r.accretor.m.toFixed(4)} sep=${r.sep.toFixed(4)} tilt=${r.tilt.toFixed(1)} xL1=${r.xL1.toFixed(4)} m75=${g3(r.m75)} mdye=${g3(r.mdye)} mPast=${g3(r.mPast)} rhoc=${g3(r.donor.rhoc)} rhoL1/rhoc=${g3(r.rhoL1Rel)}${r.seconds ? ` (${r.seconds.toFixed(0)} s)` : ''}`;

const files = args.filter((a) => a.endsWith('.athdf'));
if (files.length) {
  const rows = files.map(cached);
  if (asJson) console.log(JSON.stringify(rows, null, 1)); else rows.forEach((r) => console.log(row(r)));
  process.exit(0);
}

const fillDirs = [];
for (const d of args) {
  const base = path.basename(path.resolve(d));
  if (/^f\d/.test(base)) fillDirs.push(d);
  else for (const e of fs.readdirSync(d).sort()) if (/^f\d/.test(e) && fs.statSync(path.join(d, e)).isDirectory()) fillDirs.push(path.join(d, e));
}
const fills = [];
for (const d of fillDirs) {
  const fill = Number(path.basename(path.resolve(d)).slice(1));
  const rows = athdfs(d).map((p) => { const r = cached(p); if (!asJson) console.log(`f=${fill} ${row(r)}`); return r; }).sort((x, y) => x.t - y.t);
  const late = rows.filter((r) => r.t >= skip - 1e-6);
  const mean = (k) => late.reduce((s, r) => s + r[k], 0) / late.length;
  fills.push({
    fill, outputs: rows.length, tLast: rows.at(-1)?.t,
    m75: rate(rows, skip, 'm75'), mdye: rate(rows, skip, 'mdye'),
    rhoL1Rel: late.length ? mean('rhoL1Rel') : NaN, rhoL1RelLast: rows.at(-1)?.rhoL1Rel,
    rows,
  });
}
const fit75 = cubeFit(fills, 'm75'), fitDye = cubeFit(fills, 'mdye');
if (asJson) {
  console.log(JSON.stringify({ skip, fills, fit: { m75: fit75, mdye: fitDye } }, null, 1));
} else {
  console.log(`\nMdot in Msun per code time unit, from t >= ${skip}; rhoL1/rhoc the mean over those outputs and the last`);
  console.log('fill   outputs t_last  Mdot(m75)               Mdot(mdye)              rhoL1/rhoc mean, last');
  for (const F of fills) {
    const pm = (q) => `${g3(q.mdot)} ± ${g3(q.se)}`.padEnd(24);
    console.log(`${F.fill.toFixed(3)}  ${String(F.outputs).padEnd(7)} ${String(F.tLast?.toFixed(1)).padEnd(7)} ${pm(F.m75)}${pm(F.mdye)}${g3(F.rhoL1Rel)}, ${g3(F.rhoL1RelLast)}`);
  }
  const say = (name, fit) => {
    if (!fit) return console.log(`${name}: too few fills to fit`);
    const one = (k, x) => (x ? `A=${g3(x.A)} f0=${x.f0.toFixed(4)} (fills ${x.used.join(', ')})` : 'too few fills above the floor');
    console.log(`${name}: floor ${g3(fit.floor)}; Mdot = A (f - f0)^3: ${one('plain', fit.plain)}; less the floor: ${one('less', fit.lessFloor)}`);
  };
  console.log('');
  say('m75', fit75);
  say('mdye', fitDye);
}
