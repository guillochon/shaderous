// GPU benchmark of two ways to get (T, p_th, c_s^2) from what the hydro holds, (rho, e_th, Abar):
//   table   bicubic Hermite table of h(log rho Ye, log T) (the electron-positron free energy, as
//           Helmholtz tabulates it), pairs by a Chebyshev series, ions and radiation exact, and
//           Newton's method in ln T on the energy; then p and c_s from the converged lookup
//   network the inverse network (fit_inverse.py): one forward pass, no iteration
// and a copy kernel for the cost of reading and writing the states alone. Accuracy against the
// exact gas (gen_inverse.py) on held-out states; timing on N states (the held-out ones tiled).
// Assets from prep.py. Run by run.mjs.
const N = 1 << 20;
const REPS = 20;

const f32 = async (name) => new Float32Array(await (await fetch(`assets/${name}`)).arrayBuffer());
const fmt = (x) => {
  const s = Math.fround(x).toPrecision(9);
  return /[.e]/.test(s) ? s : `${s}.0`;
};

async function main() {
  const meta = await (await fetch('assets/meta.json')).json();
  const [table, states0, truth] = await Promise.all([f32('table.f32'), f32('states.f32'), f32('truth.f32')]);
  const n0 = states0.length / 4;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const ts = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredFeatures: ts ? ['timestamp-query'] : [] });
  const info = adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}` : '';

  const states = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) states.set(states0.subarray((i % n0) * 4, (i % n0) * 4 + 4), i * 4);
  const buf = (data, usage) => {
    const b = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const sIn = buf(states, GPUBufferUsage.STORAGE);
  const out = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: n0 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const w = buf(new Float32Array(meta.net.weights), GPUBufferUsage.STORAGE);
  const tex = device.createTexture({ size: [meta.table.nt, meta.table.nr], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, table, { bytesPerRow: meta.table.nt * 16 }, [meta.table.nt, meta.table.nr]);

  const C = meta.consts;
  const common = `
const N_STATES: u32 = ${N}u;
const KB: f32 = ${fmt(C.k)};
const MU: f32 = ${fmt(C.mu)};
const MEC2: f32 = ${fmt(C.mec2)};
const ARAD: f32 = ${fmt(C.arad)};
const YE: f32 = 0.5;
const PREF3: f32 = ${fmt(3 / C.pref)};
const ACOLD: f32 = ${fmt((C.pref * C.mec2) / 24)};
const LN10: f32 = 2.30258509;
@group(0) @binding(0) var<storage, read> S: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> O: array<vec4<f32>>;
// The zero-temperature electrons' share of c_s^2, dP_cold/drho (cgs).
fn c0sq(rho: f32) -> f32 {
  let n = rho * YE / MU;
  let x = pow(PREF3 * n, 1.0 / 3.0);
  return ACOLD * 8.0 * x * x * x * x * x / (3.0 * rho * sqrt(1.0 + x * x));
}
`;
  const copy = `${common}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= N_STATES) { return; }
  let s = S[g.x];
  O[g.x] = vec4<f32>(s.x, s.y, c0sq(s.x), 0.0);
}`;

  const T = meta.table;
  const P = meta.pair;
  const series = (name, c) => `const ${name} = array<f32, ${c.length}>(${c.map(fmt).join(', ')});`;
  const tableK = `${common}
@group(0) @binding(2) var TB: texture_2d<f32>;
${series('Q0', P.q)}
${series('Q1', P.dq)}
${series('Q2', P.d2q)}
const NQ: i32 = ${P.q.length};
fn cheb(c: ptr<function, array<f32, ${P.q.length}>>, x: f32) -> f32 {
  var b1 = 0.0;
  var b2 = 0.0;
  for (var k = NQ - 1; k >= 1; k--) {
    let b0 = 2.0 * x * b1 - b2 + (*c)[k];
    b2 = b1;
    b1 = b0;
  }
  return x * b1 - b2 + (*c)[0];
}
struct H { h: f32, hn: f32, ht: f32, hnn: f32, hnt: f32, htt: f32 }
fn hb(t: f32) -> vec4<f32> { return vec4<f32>(2.0*t*t*t - 3.0*t*t + 1.0, t*t*t - 2.0*t*t + t, -2.0*t*t*t + 3.0*t*t, t*t*t - t*t); }
fn hb1(t: f32) -> vec4<f32> { return vec4<f32>(6.0*t*t - 6.0*t, 3.0*t*t - 4.0*t + 1.0, -6.0*t*t + 6.0*t, 3.0*t*t - 2.0*t); }
fn hb2(t: f32) -> vec4<f32> { return vec4<f32>(12.0*t - 6.0, 6.0*t - 4.0, -12.0*t + 6.0, 6.0*t - 2.0); }
// f(u, v) = sum over corners a, b of A_a(u) B_b(v) of (h, h_x, h_y, h_xy): basis entries 0/2 are the
// value at corner 0/1, 1/3 the slope there.
fn comb(c: array<vec4<f32>, 4>, A: vec4<f32>, B: vec4<f32>) -> f32 {
  var s = 0.0;
  for (var a = 0; a < 2; a++) {
    for (var b = 0; b < 2; b++) {
      let q = c[a * 2 + b];
      s += A[2*a] * B[2*b] * q.x + A[2*a+1] * B[2*b] * q.y + A[2*a] * B[2*b+1] * q.z + A[2*a+1] * B[2*b+1] * q.w;
    }
  }
  return s;
}
fn lookup(lrye: f32, lt: f32) -> H {
  let x = clamp((lrye - ${fmt(T.r0)}) / ${fmt(T.d)}, 0.0, ${fmt(T.nr - 1.0001)});
  let y = clamp((lt - ${fmt(T.t0)}) / ${fmt(T.d)}, 0.0, ${fmt(T.nt - 1.0001)});
  let i = i32(floor(x));
  let j = i32(floor(y));
  let u = x - f32(i);
  let v = y - f32(j);
  var c: array<vec4<f32>, 4>;
  c[0] = textureLoad(TB, vec2<i32>(j, i), 0);
  c[1] = textureLoad(TB, vec2<i32>(j + 1, i), 0);
  c[2] = textureLoad(TB, vec2<i32>(j, i + 1), 0);
  c[3] = textureLoad(TB, vec2<i32>(j + 1, i + 1), 0);
  let s = 1.0 / (${fmt(T.d)} * LN10);
  var r: H;
  r.h = comb(c, hb(u), hb(v));
  r.hn = comb(c, hb1(u), hb(v)) * s;
  r.ht = comb(c, hb(u), hb1(v)) * s;
  r.hnn = comb(c, hb2(u), hb(v)) * s * s;
  r.hnt = comb(c, hb1(u), hb1(v)) * s * s;
  r.htt = comb(c, hb(u), hb2(v)) * s * s;
  return r;
}
// The pairs at zero net charge: (P_p, dln P_p/dln T, d^2 ln P_p/dln T^2).
fn pairs(lt: f32) -> vec3<f32> {
  let T = exp(LN10 * lt);
  let beta = KB * T / MEC2;
  let x = clamp((lt - ${fmt(P.mid)}) / ${fmt(P.half)}, -1.0, 1.0);
  var c0 = Q0;
  var c1 = Q1;
  var c2 = Q2;
  let s = 1.0 / (${fmt(P.half)} * LN10);
  let q = cheb(&c0, x);
  let lnP = log(4.0 * KB * T) + 1.5 * log(${fmt(2 * Math.PI * C.me * C.k / C.h ** 2)} * T) - 1.0 / beta + q;
  return vec3<f32>(exp(lnP), 2.5 + 1.0 / beta + cheb(&c1, x) * s, -1.0 / beta + cheb(&c2, x) * s * s);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= N_STATES) { return; }
  let st = S[g.x];
  let rho = st.x;
  let et = st.y;
  let n = rho * YE / MU;
  let ni = rho / (st.z * MU);
  let lrye = log(rho * YE) / LN10;
  // Start from everything classical: 1.5 (n + n_i) k T + a T^4 = rho e (the smaller of the two
  // limits), and step in ln T.
  // In logs: rho e / a passes single precision's 3e38 in hot, thin gas.
  let lre = log(rho) + log(et);
  var lnT = clamp(min(lre - log(1.5 * KB * (n + ni)), 0.25 * (lre - log(ARAD))), 9.21, 25.33);
  var it = 0;
  var hh: H;
  var pp: vec3<f32>;
  for (; it < 40; it++) {
    let lt = lnT / LN10;
    hh = lookup(lrye, lt);
    pp = pairs(lt);
    let T = exp(lnT);
    let s = n * KB * T;
    let Ei = 1.5 * ni * KB * T;
    let Er = ARAD * T * T * T * T;
    let Ee = pp.x * (pp.y - 1.0) - s * hh.ht;
    let dE = pp.x * (pp.y * pp.y - pp.y + pp.z) - s * (hh.ht + hh.htt) + Ei + 4.0 * Er;
    let step = clamp(-(Ee + Ei + Er - rho * et) / dE, -2.0, 2.0);
    lnT = clamp(lnT + step, 9.21, 25.33);   // within the table: 1e4 to 1e11 K
    if (abs(step) < 1e-6) { break; }
  }
  let lt = lnT / LN10;
  hh = lookup(lrye, lt);
  pp = pairs(lt);
  let T = exp(lnT);
  let s = n * KB * T;
  let Pi = ni * KB * T;
  let Pr = ARAD * T * T * T * T / 3.0;
  let p = pp.x + s * hh.hn + Pi + Pr;
  let dPdrho = s * (hh.hn + hh.hnn) / rho + Pi / rho;
  let dPdT = (pp.x * pp.y + s * (hh.hn + hh.hnt) + Pi + 4.0 * Pr) / T;
  let cv = (pp.x * (pp.y * pp.y - pp.y + pp.z) - s * (hh.ht + hh.htt) + 1.5 * Pi + 12.0 * Pr) / (rho * T);
  let dPdTm = dPdT / rho;                  // per unit mass: (dP/dT)^2 alone passes 3e38
  let cs2 = c0sq(rho) + dPdrho + T * dPdTm * (dPdTm / cv);
  O[g.x] = vec4<f32>(T, p, cs2, f32(it));
}`;

  const nt = meta.net;
  const layers = nt.layers;
  const offs = [];
  let o = 0;
  for (const [a, b] of layers) {
    offs.push(o);
    o += a * b + b;
  }
  const H = layers[0][1];
  const netK = `${common}
@group(0) @binding(3) var<storage, read> W: array<f32>;
const HID: u32 = ${H}u;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= N_STATES) { return; }
  let st = S[g.x];
  let rho = st.x;
  var x = array<f32, 3>(log(rho) / LN10, log(st.y) / LN10, log(st.z) / LN10);
  let lo = array<f32, 3>(${nt.lo.map(fmt).join(', ')});
  let hi = array<f32, 3>(${nt.hi.map(fmt).join(', ')});
  for (var i = 0; i < 3; i++) { x[i] = 2.0 * (x[i] - lo[i]) / (hi[i] - lo[i]) - 1.0; }
  var a: array<f32, ${H}>;
  var b: array<f32, ${H}>;
  for (var j = 0u; j < HID; j++) {
    var s = W[${offs[0] + 3 * H}u + j];
    for (var i = 0u; i < 3u; i++) { s += W[${offs[0]}u + j * 3u + i] * x[i]; }
    a[j] = tanh(s);
  }
${layers.slice(1, -1).map((_, k) => `  for (var j = 0u; j < HID; j++) {
    var s = W[${offs[k + 1] + H * H}u + j];
    for (var i = 0u; i < HID; i++) { s += W[${offs[k + 1]}u + j * HID + i] * ${k % 2 ? 'b' : 'a'}[i]; }
    ${k % 2 ? 'a' : 'b'}[j] = tanh(s);
  }`).join('\n')}
  var y: array<f32, 3>;
  let ylo = array<f32, 3>(${nt.ylo.map(fmt).join(', ')});
  let yhi = array<f32, 3>(${nt.yhi.map(fmt).join(', ')});
  for (var j = 0u; j < 3u; j++) {
    var s = W[${offs[layers.length - 1] + 3 * H}u + j];
    for (var i = 0u; i < HID; i++) { s += W[${offs[layers.length - 1]}u + j * HID + i] * ${(layers.length - 2) % 2 ? 'b' : 'a'}[i]; }
    y[j] = ylo[j] + s * (yhi[j] - ylo[j]);
  }
  let T = exp(LN10 * y[0]);
  let p = y[1] * rho * st.y;
  O[g.x] = vec4<f32>(T, p, c0sq(rho) + y[2] * p / rho, 0.0);
}`;

  const results = {};
  for (const [name, code, extra] of [
    ['copy', copy, {}],
    ['table', tableK, { 2: tex.createView() }],
    ['network', netK, { 3: { buffer: w } }],
  ]) {
    const mod = device.createShaderModule({ code });
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error(`${name}: ${errs.map((m) => `${m.lineNum}: ${m.message}`).join('\n')}`);
    const pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    const entries = [{ binding: 0, resource: { buffer: sIn } }, { binding: 1, resource: { buffer: out } }];
    for (const [k, v] of Object.entries(extra)) entries.push({ binding: Number(k), resource: v });
    const group = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const qs = ts ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    const qb = ts ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
    const qr = ts ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
    const times = [];
    for (let r = 0; r < REPS + 3; r++) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(ts ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
      pass.setPipeline(pipe);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(N / 64);
      pass.end();
      if (ts) {
        enc.resolveQuerySet(qs, 0, 2, qb, 0);
        enc.copyBufferToBuffer(qb, 0, qr, 0, 16);
      }
      const t0 = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      let ms = performance.now() - t0;
      if (ts) {
        await qr.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(qr.getMappedRange().slice(0));
        qr.unmap();
        ms = Number(t[1] - t[0]) / 1e6;
      }
      if (r >= 3) times.push(ms);
    }
    times.sort((a, b) => a - b);
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(out, 0, read, 0, n0 * 16);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const o4 = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    const r = { ms_median: times[times.length >> 1], ms_min: times[0], ns_per_state: (times[times.length >> 1] * 1e6) / N };
    if (name !== 'copy') {
      const rel = [[], [], []];
      let iters = 0;
      for (let i = 0; i < n0; i++) {
        for (let k = 0; k < 3; k++) rel[k].push(Math.abs(o4[i * 4 + k] / truth[i * 4 + k] - 1));
        iters += o4[i * 4 + 3];
      }
      const q = (a, f) => a.slice().sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(f * a.length))];
      ['T', 'p_th', 'cs2'].forEach((nm, k) => {
        const finite = rel[k].filter(Number.isFinite);
        r[nm] = { median: q(finite, 0.5), p99: q(finite, 0.99), max: q(finite, 1), nonfinite: rel[k].length - finite.length };
      });
      if (name === 'table') r.newton_iterations_mean = iters / n0;
      // The worst states for T: (log rho, log e_th, Abar, true log T, got log T, iterations).
      const idx = [...rel[0].keys()].sort((a, b) => (rel[0][b] || 0) - (rel[0][a] || 0)).slice(0, 8);
      r.worst_T = idx.map((i) => [Math.log10(states0[i*4]), Math.log10(states0[i*4+1]), states0[i*4+2], Math.log10(truth[i*4]), Math.log10(o4[i*4]), o4[i*4+3]].map((x) => +x.toFixed(3)));
      const bad = rel[0].filter((x) => x > 1e-3).length;
      r.T_bad_fraction = bad / n0;
    }
    results[name] = r;
  }
  window.benchResult = { gpu: info, timestamps: ts, states_timed: N, states_checked: n0, results };
}

main().catch((e) => {
  window.benchResult = { error: String(e && e.stack ? e.stack : e) };
});
