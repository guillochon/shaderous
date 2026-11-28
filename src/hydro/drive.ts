// Turbulence driving: a random acceleration field, stirring the gas at large scales. The idea
// follows the WebGPU reduced-MHD page of Mallet & Abraham, built on their code taranis
// (https://alfredmallet.github.io/taranis/webgpu/rmhd2d.html); the method is Eswaran & Pope's
// (1988) Ornstein-Uhlenbeck forcing in a band of wavenumbers, with the solenoidal / compressive
// projection of Federrath et al. (2010). Written from those papers, not from the page's code.
//
// The modes are every wavevector k = 2 pi n / L with integer n and kMin <= |n| <= kMax (one of
// each pair +-n; in the plane in 2D), up to DRIVE_MODES of them, shortest first. Their complex
// amplitudes follow an Ornstein-Uhlenbeck process of correlation time tau, advanced on the GPU
// by the clock pass (hydro/drive_ou.wgsl), and each cell sums them at its own position (no FFT,
// so any grid works: hydro/drive.wgsl). The noise is scaled so that the acceleration's mean
// square over the box is accel^2 in the stationary state.
//
// The buffer is one per device, bound by every pass that declares it (gpu/device.ts
// setSharedBinding): a simulation writes its modes when it is built or starts again. The
// acceleration has zero mean, but its force, rho a, does not where the density varies: the box
// may pick up a net momentum, which the diagnostics report.
import { setSharedBinding } from '../gpu/device';

/** Binding of the driving buffer in every pass that reads it (hydro/drive.wgsl, drive_ou.wgsl). */
export const DRIVE_BINDING = 33;
/** Most modes the shaders hold (hydro/drive_common.wgsl DRIVE_MODES). */
export const DRIVE_MODES = 64;
/** Size of the buffer: two vec4 of header, three vec4 per mode. */
export const DRIVE_BYTES = 32 + DRIVE_MODES * 48;

export interface DriveParams {
  /** root-mean-square acceleration over the box, stationary; 0 is off */
  accel: number;
  /** correlation time of the amplitudes */
  tau: number;
  /** the band, in wavelengths per box side */
  kMin: number;
  kMax: number;
  /** solenoidal weight: 1 solenoidal (divergence-free), 0 compressive, 0.5 the natural mix */
  zeta: number;
  /** which random stream */
  seed: number;
}

export const defaultDrive = (): DriveParams => ({ accel: 0, tau: 0.5, kMin: 1, kMax: 2, zeta: 1, seed: 1 });

/** The integer wavevectors of the band (one of each pair), shortest first, at most DRIVE_MODES. */
export function driveModes(ndim: 1 | 2 | 3, kMin: number, kMax: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  const m = Math.floor(kMax);
  // (A 1D run's modes lie along x alone, a 2D run's in the plane.)
  const mj = ndim === 1 ? 0 : m;
  const mk = ndim === 3 ? m : 0;
  for (let i = -m; i <= m; i++)
    for (let j = -mj; j <= mj; j++)
      for (let k = -mk; k <= mk; k++) {
        const r = Math.hypot(i, j, k);
        if (r < kMin || r > kMax || r === 0) continue;
        // One of each pair: the first nonzero component positive.
        const first = i !== 0 ? i : j !== 0 ? j : k;
        if (first < 0) continue;
        out.push([i, j + 0, k + 0]); // (+ 0: no -0 from a bound of 0)
      }
  out.sort((a, b) => Math.hypot(...a) - Math.hypot(...b) || a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  return out.slice(0, DRIVE_MODES);
}

/**
 * The noise amplitude sigma for an rms acceleration `accel` from `count` modes: each mode's two
 * real vectors have stationary variance sigma^2 per component before the projection, which
 * keeps zeta^2 (d - 1) + (1 - zeta)^2 of it, and the box mean of |a|^2 is half the sum over
 * modes of |ar|^2 + |ai|^2.
 */
export function driveSigma(p: DriveParams, ndim: 1 | 2 | 3, count: number): number {
  const kept = p.zeta * p.zeta * (ndim - 1) + (1 - p.zeta) * (1 - p.zeta);
  return count > 0 && kept > 0 ? p.accel / Math.sqrt(count * kept) : 0;
}

/** A small seeded generator (mulberry32) and unit normal deviates from it. */
function normals(seed: number): () => number {
  let s = (seed * 2654435761) >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 1) / 4294967297;
  };
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

/**
 * The buffer's contents for a run: the modes' wavevectors and amplitudes drawn from the
 * stationary distribution (so that the driving is at full strength from the start), or a
 * header of no modes when the driving is off.
 */
export function packDrive(p: DriveParams, ndim: 1 | 2 | 3, box: number): ArrayBuffer {
  const buf = new ArrayBuffer(DRIVE_BYTES);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  const on = p.accel > 0 && p.tau > 0 && p.kMax >= p.kMin && p.kMax >= 1;
  const modes = on ? driveModes(ndim, p.kMin, p.kMax) : [];
  const sigma = driveSigma(p, ndim, modes.length);
  const seed = Math.max(0, Math.round(p.seed)) >>> 0;
  u.set([modes.length, 0, seed, ndim], 0);
  f.set([p.tau, sigma, p.zeta, 0], 4);
  const xi = normals(seed + 1);
  modes.forEach((n, m) => {
    const k = n.map((c) => (2 * Math.PI * c) / box) as [number, number, number];
    const kk = Math.hypot(...k);
    const amp = () => {
      const g = [xi(), xi(), ndim === 2 ? 0 : xi()];
      if (ndim === 2) xi();
      // (A 1D run is driven along x alone: compressively, the only way there is.)
      if (ndim === 1) g[1] = g[2] = 0;
      const along = (g[0] * k[0] + g[1] * k[1] + g[2] * k[2]) / (kk * kk);
      return g.map((c, i) => sigma * (p.zeta * (c - along * k[i]) + (1 - p.zeta) * along * k[i]));
    };
    f.set([...k, 0], 8 + 12 * m);
    f.set([...amp(), 0], 8 + 12 * m + 4);
    f.set([...amp(), 0], 8 + 12 * m + 8);
  });
  return buf;
}

const buffers = new WeakMap<GPUDevice, GPUBuffer>();

/**
 * Write a run's driving into the device's buffer (made and shared on first use). Every
 * simulation calls this when it is built and when it starts again, driving or not, so that the
 * passes that read the buffer find it bound, and a run without driving finds no modes.
 */
export function writeDrive(device: GPUDevice, p: DriveParams, ndim: 1 | 2 | 3, box: number): void {
  let b = buffers.get(device);
  if (!b) {
    b = device.createBuffer({ label: 'turbulence driving', size: DRIVE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    buffers.set(device, b);
    setSharedBinding(device, DRIVE_BINDING, b);
  }
  device.queue.writeBuffer(b, 0, packDrive(p, ndim, box));
}
