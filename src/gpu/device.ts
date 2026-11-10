// Device setup, limit checks, and small texture / pipeline helpers.

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

export async function initGpu(): Promise<Gpu> {
  if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');

  const problems: string[] = [];
  for (const [k, v] of Object.entries(NEEDED)) {
    const have = adapter.limits[k as keyof GPUSupportedLimits] as number;
    if (have < v!) problems.push(`${k}: need ${v}, adapter has ${have}`);
  }
  if (!navigator.gpu.wgslLanguageFeatures?.has('readonly_and_readwrite_storage_textures')) {
    problems.push('WGSL feature readonly_and_readwrite_storage_textures (read-write r32float) is missing');
  }
  if (problems.length) throw new Error('WebGPU adapter is below the minimum configuration:\n' + problems.join('\n'));

  // Ask for the adapter's full buffer size so large readbacks (validation) work.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  device.addEventListener('uncapturederror', (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    console.error('WebGPU error:', message);
    // The page shows these in its error panel (src/ui/errors.ts).
    globalThis.dispatchEvent?.(new CustomEvent('shaderous-gpu-error', { detail: 'WebGPU error: ' + message }));
  });

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

const STATE_USAGE =
  GPUTextureUsage.STORAGE_BINDING |
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST;

export function tex3d(device: GPUDevice, n: number, format: GPUTextureFormat, label: string, nz = n): GPUTexture {
  return device.createTexture({
    label,
    size: [n, n, nz],
    dimension: '3d',
    format,
    usage: STATE_USAGE,
  });
}

export async function computePipeline(
  device: GPUDevice,
  code: string,
  label: string,
  constants?: Record<string, number>,
): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    const lines = code.split('\n');
    const msg = errors
      .map((m) => `${label}:${m.lineNum}:${m.linePos} ${m.message}\n    ${lines[m.lineNum - 1] ?? ''}`)
      .join('\n');
    throw new Error('WGSL compile error\n' + msg);
  }
  return device.createComputePipelineAsync({
    label,
    layout: 'auto',
    compute: { module, entryPoint: 'main', constants },
  });
}

export type Bindable = GPUBuffer | GPUTextureView | GPUSampler;

/** Bind group from a sparse {binding: resource} map against a pipeline's auto layout. */
export function bindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline | GPURenderPipeline,
  resources: Record<number, Bindable>,
  label = '',
): GPUBindGroup {
  const entries: GPUBindGroupEntry[] = Object.entries(resources).map(([b, r]) => ({
    binding: Number(b),
    resource: r instanceof GPUBuffer ? { buffer: r } : (r as GPUTextureView | GPUSampler),
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
