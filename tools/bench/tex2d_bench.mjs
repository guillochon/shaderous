#!/usr/bin/env node
// Is a 2D grid faster in texture_2d than in a texture_3d one cell deep (how uniform 2D grids are
// stored now)? A stand-in for hydro/update.wgsl's memory traffic, not the real update shader:
// per cell, the 13-cell stencil (the cell and two neighbors each way along x and y, plus the z
// neighbors, which in 2D are the cell itself) of two rgba32float state textures, the half-step
// pair at the cell and its face neighbors, and two rgba32float writes. Same 16x16x1 workgroups
// as the uniform 2D update. Alternates the two layouts over several rounds of 400 passes and
// reports the median time per pass. A benchmark: run it with the GPU otherwise idle, at normal
// priority.
//
//   node tools/bench/tex2d_bench.mjs [N=2048] [ROUNDS=7]
//
// Run from anywhere in the repository after npm install; it starts Vite and headless Chrome
// itself (tools/headless/session.mjs).
import { openSession } from '../headless/session.mjs';

const N = Number(process.argv[2] ?? 2048);
const ROUNDS = Number(process.argv[3] ?? 7);
const s = await openSession({ setup: 'problem=sedov&dim=2&n=64', log: () => {} });
try {
  const r = await s.page.evaluate(async ([N, ROUNDS]) => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const device = await adapter.requestDevice();
    const errs = [];
    device.addEventListener('uncapturederror', (e) => errs.push(e.error.message));

    const kernel = (two) => {
      const T = two ? 'texture_2d<f32>' : 'texture_3d<f32>';
      const S = two ? 'texture_storage_2d<rgba32float, write>' : 'texture_storage_3d<rgba32float, write>';
      const at = two ? 'c.xy' : 'c';
      const p = two ? 'vec2<i32>' : 'vec3<i32>';
      return /* wgsl */ `
@group(0) @binding(0) var U1: ${T};
@group(0) @binding(1) var U2: ${T};
@group(0) @binding(2) var PA: ${T};
@group(0) @binding(3) var PB: ${T};
@group(0) @binding(4) var O1: ${S};
@group(0) @binding(5) var O2: ${S};
const n = ${N};
fn wrap(c: vec3<i32>) -> vec3<i32> { return vec3<i32>((c.x + n) % n, (c.y + n) % n, 0); }
fn ld1(c0: vec3<i32>) -> vec4<f32> { let c = wrap(c0); return textureLoad(U1, ${at}, 0); }
fn ld2(c0: vec3<i32>) -> vec4<f32> { let c = wrap(c0); return textureLoad(U2, ${at}, 0); }
fn lda(c0: vec3<i32>) -> vec4<f32> { let c = wrap(c0); return textureLoad(PA, ${at}, 0); }
fn ldb(c0: vec3<i32>) -> vec4<f32> { let c = wrap(c0); return textureLoad(PB, ${at}, 0); }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (c.x >= n || c.y >= n) { return; }
  var a = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  // 13-cell stencil of the state: the cell, two each way along x and y, and the z pair
  // (the cell itself in 2D, as the uniform update reads it).
  let offs = array<vec3<i32>, 13>(
    vec3<i32>(0, 0, 0), vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(-2, 0, 0), vec3<i32>(2, 0, 0),
    vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, -2, 0), vec3<i32>(0, 2, 0),
    vec3<i32>(0, 0, 0), vec3<i32>(0, 0, 0), vec3<i32>(0, 0, 0), vec3<i32>(0, 0, 0));
  for (var k = 0; k < 13; k++) {
    let u = ld1(c + offs[k]);
    let v = ld2(c + offs[k]);
    a += u * (1.0 + 0.01 * f32(k)) - 0.5 * v.yzwx;
    b += v * u.x;
  }
  // Half-step states at the cell and its six face neighbors.
  for (var k = 0; k < 7; k++) {
    let q = lda(c + offs[min(k, 8)]);
    let r = ldb(c + offs[min(k, 8)]);
    a += 0.1 * q * r.x;
    b -= 0.1 * r;
  }
  textureStore(O1, ${two ? 'c.xy' : 'c'}, a);
  textureStore(O2, ${two ? 'c.xy' : 'c'}, b);
}`;
    };

    const make = (two) => {
      const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST;
      const tex = (label) => device.createTexture({ label, size: [N, N, 1], dimension: two ? '2d' : '3d', format: 'rgba32float', usage });
      const ins = [tex('U1'), tex('U2'), tex('PA'), tex('PB')];
      const outs = [tex('O1'), tex('O2')];
      // Fill the inputs with something nonzero.
      const row = new Float32Array(N * 4);
      for (let i = 0; i < row.length; i++) row[i] = 1 + (i % 7) * 0.1;
      for (const t of ins) for (let y = 0; y < N; y++) device.queue.writeTexture({ texture: t, origin: [0, y, 0] }, row, { bytesPerRow: N * 16 }, [N, 1, 1]);
      const module = device.createShaderModule({ code: kernel(two) });
      const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [...ins, ...outs].map((t, i) => ({ binding: i, resource: t.createView() })),
      });
      return { pipeline, bind, textures: [...ins, ...outs] };
    };

    const ts = adapter.features.has('timestamp-query');
    const time = async (k, passes) => {
      const enc = device.createCommandEncoder();
      for (let i = 0; i < passes; i++) {
        const pass = enc.beginComputePass();
        pass.setPipeline(k.pipeline);
        pass.setBindGroup(0, k.bind);
        pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(N / 16), 1);
        pass.end();
      }
      await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      return (performance.now() - t0) / passes;
    };

    const k3 = make(false);
    const k2 = make(true);
    await device.queue.onSubmittedWorkDone();
    const PASSES = 400;
    await time(k3, 50);
    await time(k2, 50);
    const t3 = [];
    const t2 = [];
    for (let r = 0; r < ROUNDS; r++) {
      t3.push(await time(k3, PASSES));
      t2.push(await time(k2, PASSES));
    }
    const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
    for (const t of [...k3.textures, ...k2.textures]) t.destroy();
    return { N, ROUNDS, PASSES, ms3d: t3, ms2d: t2, median3d: med(t3), median2d: med(t2), speedup: med(t3) / med(t2), errs, adapter: `${adapter.info.vendor} ${adapter.info.architecture}`, timestampQuery: ts };
  }, [N, ROUNDS]);
  console.log(JSON.stringify(r, null, 1));
} finally {
  await s.close();
}
