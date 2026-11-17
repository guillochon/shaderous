// Device setup, limit checks, and small texture / pipeline helpers.
import { usedBindings } from './reach';
import { countCompile } from './progress';

export interface Gpu {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: { vendor: string; architecture: string; description: string; fallback: boolean };
  limits: GPUSupportedLimits;
  maxGrid: number; // largest cubic grid this device can hold
}

// What the simulator needs, against the WebGPU defaults (plan section 4.2).
const NEEDED: Partial<Record<keyof GPUSupportedLimits, number>> = {
  maxStorageTexturesPerShaderStage: 4,
  maxSampledTexturesPerShaderStage: 16,
  maxStorageBuffersPerShaderStage: 8,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 8,
  maxComputeWorkgroupSizeY: 8,
  maxComputeWorkgroupSizeZ: 4,
  maxTextureDimension3D: 256,
};

// Shown when there is no WebGPU at all, or no GPU behind it: what to try, most likely first.
const NO_API_HELP = `WebGPU is not available in this browser.

This browser is too old or does not support WebGPU. Try a recent Chrome or Edge.
Recent Firefox and Safari support WebGPU on some systems but not all.`;

const NO_ADAPTER_HELP = `No WebGPU adapter found.

The browser supports WebGPU but could not get a GPU to run it on. Things to try:

1. Check what the browser reports: open chrome://gpu (edge://gpu in Edge) and look
   at the "WebGPU" line. If it says disabled or software only, the page says why.
2. Turn on hardware acceleration: Settings > System > "Use graphics acceleration
   when available", then restart the browser.
3. Update the graphics driver from NVIDIA, AMD or Intel directly (not Windows Update).
   Browsers block WebGPU on old or known-bad drivers.
4. Try another browser. Chrome or Edge is the safest choice. On Linux, Chrome may
   need chrome://flags/#enable-unsafe-webgpu (and #enable-vulkan) turned on.
5. Remote Desktop, virtual machines, some managed work browsers and very old
   integrated GPUs often have no WebGPU adapter. Run it locally on a real GPU.
6. On a laptop with two GPUs, put the browser on the dedicated one (Windows:
   Settings > System > Display > Graphics > the browser > High performance).`;

const DEVICE_LOST_HELP = `The GPU device was lost.

The browser stopped this page's use of the GPU: a GPU reset (a pass that ran too long, or
a driver fault), the GPU process crashing, or the driver being updated. Reload the page to
start again. Firefox turns WebGPU off after such a reset until it is restarted: if the
reloaded page finds no WebGPU adapter, quit Firefox and open it again.`;

export async function initGpu(): Promise<Gpu> {
  if (!('gpu' in navigator)) throw new Error(NO_API_HELP);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error(NO_ADAPTER_HELP);

  const problems: string[] = [];
  for (const [k, v] of Object.entries(NEEDED)) {
    const have = adapter.limits[k as keyof GPUSupportedLimits] as number;
    if (have < v!) problems.push(`${k}: need ${v}, adapter has ${have}`);
  }
  if (!navigator.gpu.wgslLanguageFeatures?.has('readonly_and_readwrite_storage_textures')) {
    problems.push('WGSL feature readonly_and_readwrite_storage_textures (read-write r32float) is missing');
  }
  if (problems.length) throw new Error('WebGPU adapter is below the minimum configuration:\n' + problems.join('\n'));

  // Ask for the adapter's full buffer size so large readbacks (validation) work, and for
  // timestamps where there are any (the optional per-pass timing, src/gpu/timing.ts).
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      // For the flux update's stencil tile (hydro/tile.wgsl).
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      // Up to 8 (most adapters): the network's burn pass writes a fifth, its heating rate (burn/network.ts).
      maxStorageTexturesPerShaderStage: Math.min(8, adapter.limits.maxStorageTexturesPerShaderStage),
    },
  });
  device.addEventListener('uncapturederror', (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    console.error('WebGPU error:', message);
    // The page shows these in its error panel (src/ui/errors.ts).
    globalThis.dispatchEvent?.(new CustomEvent('shaderous-gpu-error', { detail: 'WebGPU error: ' + message }));
  });
  // A lost device fails every readback after it ("Context lost"): say once what happened instead.
  device.lost.then((info) => {
    if (info.reason === 'destroyed') return; // the page let it go itself
    const message = `${DEVICE_LOST_HELP}\n\nThe browser says: ${info.message || '(nothing)'}`;
    console.error(message);
    globalThis.dispatchEvent?.(new CustomEvent('shaderous-gpu-error', { detail: message }));
  });
  await probeRenderable3d(device);

  const ai = adapter.info;
  const maxGrid = Math.min(adapter.limits.maxTextureDimension3D, 512);
  return {
    adapter,
    device,
    info: {
      vendor: ai?.vendor ?? '',
      architecture: ai?.architecture ?? '',
      description: ai?.description ?? '',
      fallback: Boolean((ai as { isFallbackAdapter?: boolean } | undefined)?.isFallbackAdapter),
    },
    limits: device.limits,
    maxGrid,
  };
}

// Evaluated on use, so the module can be imported where WebGPU does not exist (unit tests).
const stateUsage = () =>
  GPUTextureUsage.STORAGE_BINDING |
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST;

/**
 * Devices whose 3D storage textures may also be render attachments. WebGPU zero-fills a texture
 * on first use; Dawn does it with a render pass when the texture may be rendered to, and
 * otherwise by copying from a staging buffer of the whole texture's size, which may be no
 * larger than a buffer may be (maxBufferSize, 4 GiB on NVIDIA). A large mesh's atlas textures
 * pass that (5.5 GB at 35^3 blocks lost the device), so a texture past it is made renderable
 * where it can be; smaller ones are not, since steps ran some 2.5% slower with renderable atlas
 * textures (MRI disk, 256^3 with 2 levels, RTX 3080). Where it cannot be, the atlas keeps each
 * texture within maxBufferSize (amr/amr_sim.ts).
 */
const renderable3d = new WeakMap<GPUDevice, boolean>();
export const rendersTo3d = (device: GPUDevice) => renderable3d.get(device) ?? false;

/** Whether this device takes a 3D rgba32float storage texture that is also a render attachment. */
export async function probeRenderable3d(device: GPUDevice): Promise<boolean> {
  device.pushErrorScope('validation');
  const t = device.createTexture({ size: [4, 4, 4], dimension: '3d', format: 'rgba32float', usage: stateUsage() | GPUTextureUsage.RENDER_ATTACHMENT });
  const error = await device.popErrorScope();
  t.destroy();
  renderable3d.set(device, !error);
  return !error;
}

/** Bytes per texel of the formats the atlases use (16 for any other, to be safe). */
const texelBytes = (format: GPUTextureFormat) => (format.startsWith('r32') ? 4 : format.startsWith('rg32') ? 8 : 16);

/** A 3D state texture n x ny x nz (ny = n unless given: the species atlas stacks its groups along y too). */
export function tex3d(device: GPUDevice, n: number, format: GPUTextureFormat, label: string, nz = n, ny = n): GPUTexture {
  // Past one buffer's size, cleared by a render pass instead of a staging buffer (see above).
  const tooBig = n * ny * nz * texelBytes(format) > device.limits.maxBufferSize;
  return device.createTexture({
    label,
    size: [n, ny, nz],
    dimension: '3d',
    format,
    usage: stateUsage() | (tooBig && rendersTo3d(device) ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
  });
}

// Compute pipelines by device and by what they compile: the same shader asked for twice (by
// two passes, or by the next run of the same setup) is compiled once. Pipelines with layout
// 'auto' are immutable and hold no resources, so passes can share them. The oldest go first
// past a few hundred, so a long session of different setups does not keep them all.
const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline>>>();
const PIPELINE_CACHE_MAX = 400;

/**
 * A compute pipeline, compiled in the background. Ask for every pipeline a setup needs before
 * awaiting any of them: the browser compiles them in parallel, so a build costs about its
 * slowest compile rather than the sum of them all (measured on a cold cache: 3.4 s for the
 * 2D adaptive mesh when each was awaited in turn, dominated by the update pass's 1.7 s).
 */
export function computePipeline(
  device: GPUDevice,
  code: string,
  label: string,
  constants?: Record<string, number>,
  entryPoint = 'main',
): Promise<GPUComputePipeline> {
  let cache = pipelineCache.get(device);
  if (!cache) pipelineCache.set(device, (cache = new Map()));
  const key = `${entryPoint}\n${constants ? JSON.stringify(Object.entries(constants).sort()) : ''}\n${code}`;
  let p = cache.get(key);
  if (p) {
    // Most recently used last, so the oldest is the first one dropped.
    cache.delete(key);
    cache.set(key, p);
    countCompile(p); // part of the load's total all the same (gpu/progress.ts)
    return p;
  }
  p = compilePipeline(device, code, label, constants, entryPoint);
  countCompile(p);
  cache.set(key, p);
  p.catch(() => cache.delete(key));
  if (cache.size > PIPELINE_CACHE_MAX) cache.delete(cache.keys().next().value!);
  return p;
}

async function compilePipeline(
  device: GPUDevice,
  code: string,
  label: string,
  constants: Record<string, number> | undefined,
  entryPoint: string,
): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label, code });
  try {
    const pipe = await device.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint, constants } });
    registerPipelineCode(pipe, code, [entryPoint]);
    return pipe;
  } catch (e) {
    // The compiler's messages, with the offending lines, say more than the pipeline's error.
    // Asked for only on failure: waiting for them first holds every compile up by a round trip.
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (!errors.length) throw e;
    const lines = code.split('\n');
    const msg = errors.map((m) => `${label}:${m.lineNum}:${m.linePos} ${m.message}\n    ${lines[m.lineNum - 1] ?? ''}`).join('\n');
    throw new Error('WGSL compile error\n' + msg);
  }
}

/** A resource to bind: a whole buffer, a window of one ({ buffer, offset, size }), a view or a sampler. */
export type Bindable = GPUBuffer | GPUBufferBinding | GPUTextureView | GPUSampler;

/** Bind group from a sparse {binding: resource} map against a pipeline's auto layout. */
/**
 * Resources every bind group of a device gets where its pipeline uses their binding (the
 * equation of state's table, eos/helmholtz.ts): set once, bound wherever a shader reaches them,
 * with no call site to change. Which bindings a pipeline uses comes from its WGSL (gpu/reach.ts),
 * recorded as computePipeline builds it, or by registerPipelineCode for pipelines made elsewhere.
 */
const sharedBindings = new WeakMap<GPUDevice, Map<number, Bindable>>();
const pipelineCode = new WeakMap<GPUComputePipeline | GPURenderPipeline, { code: string; entryPoints: string[] }>();

export function setSharedBinding(device: GPUDevice, binding: number, resource: Bindable) {
  let m = sharedBindings.get(device);
  if (!m) sharedBindings.set(device, (m = new Map()));
  m.set(binding, resource);
}

export function registerPipelineCode(pipeline: GPUComputePipeline | GPURenderPipeline, code: string, entryPoints: string[]) {
  pipelineCode.set(pipeline, { code, entryPoints });
}

export function bindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline | GPURenderPipeline,
  resources: Record<number, Bindable>,
  label = '',
): GPUBindGroup {
  const shared = sharedBindings.get(device);
  const known = pipelineCode.get(pipeline);
  if (shared && known) {
    const used = usedBindings(known.code, known.entryPoints);
    for (const [b, r] of shared) if (used.has(b) && !(b in resources)) resources = { ...resources, [b]: r };
  }
  const entries: GPUBindGroupEntry[] = Object.entries(resources).map(([b, r]) => ({
    binding: Number(b),
    resource: r instanceof GPUBuffer ? { buffer: r } : 'buffer' in r ? r : (r as GPUTextureView | GPUSampler),
  }));
  return device.createBindGroup({ label, layout: pipeline.getBindGroupLayout(0), entries });
}

/** Read a whole 3D texture back as floats (validation only; this is slow). */
export async function readTexture(
  device: GPUDevice,
  tex: GPUTexture,
  n: number,
  channels: 1 | 4,
  nz = n,
): Promise<Float32Array> {
  const bytesPerRow = n * 4 * channels;
  const padded = Math.ceil(bytesPerRow / 256) * 256;
  const size = padded * n * nz;
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: padded, rowsPerImage: n }, [n, n, nz]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Float32Array(buf.getMappedRange());
  const out = new Float32Array(n * n * nz * channels);
  const rowF = n * channels;
  const padF = padded / 4;
  for (let r = 0; r < n * nz; r++) out.set(src.subarray(r * padF, r * padF + rowF), r * rowF);
  buf.unmap();
  buf.destroy();
  return out;
}

export function writeTexture(device: GPUDevice, tex: GPUTexture, n: number, channels: 1 | 4, data: Float32Array, nz = n) {
  device.queue.writeTexture({ texture: tex }, data as Float32Array<ArrayBuffer>, { bytesPerRow: n * 4 * channels, rowsPerImage: n }, [n, n, nz]);
}
