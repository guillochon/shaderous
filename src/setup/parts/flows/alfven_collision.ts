// Counter-propagating Alfven waves along a guide field: the interaction behind the turbulent
// cascade. The setups follow the taranis WebGPU reduced-MHD page of Mallet & Abraham
// (https://alfredmallet.github.io/taranis/webgpu/rmhd2d.html), written from the papers below,
// not from its code.
//
// Units: magnetic pressure B^2 / 2, density 1, so the Alfven speed is the guide field B0, and
// b = B / sqrt(rho) is B. The Elsasser fields are z+- = u +- b: a wave with z- = 0 (u = b)
// travels against the guide field, one with z+ = 0 (u = -b) along it. Coordinates are relative
// to the box center; the guide field is along z.
//
// The z+ wave is polarized along y and varies across the field along x, the z- wave polarized
// along x and varying along y, so that each one's velocity carries it across the other's
// gradient: (z-+ . grad) z+- is not zero. Each face component depends only on the other two
// coordinates (b_y on x and z, b_x on y and z), so the face field is divergence-free exactly.
import type { FlowContext, FlowDef } from '../flow';

interface Waves {
  b0: number;
  amp: number;
  beta: number;
  kperp: number;
  kpar: number;
}

/** The two waves' field components and the parallel envelope each is multiplied by. */
function waves(ctx: FlowContext, q: Waves, env: (z: number, sign: number) => number) {
  const L = ctx.box;
  const kp = (2 * Math.PI * q.kperp) / L;
  const kz = (2 * Math.PI * q.kpar) / L;
  const a = q.amp * q.b0;
  return {
    // z+ (u = b, travelling toward -z): b_y(x, z).
    by: (x: number, z: number) => a * env(z, 1) * Math.cos(kp * x - kz * z),
    // z- (u = -b, travelling toward +z): b_x(y, z).
    bx: (y: number, z: number) => a * env(z, -1) * Math.cos(kp * y + kz * z),
  };
}

const COMMON = {
  b0: { label: 'Guide field', default: 1, min: 0.01, max: 100, factor: 1.25, tip: 'B0 along z: the Alfven speed, with density 1 (magnetic pressure is B² / 2)' },
  amp: { label: 'Wave amplitude', default: 0.25, min: 0, max: 2, step: 0.05, tip: 'Amplitude of each wave’s field (and velocity) over the guide field' },
  beta: { label: 'Plasma beta', default: 10, min: 0.01, max: 1000, factor: 1.5, tip: 'Gas pressure over magnetic pressure of the guide field, 2p / B0²: the higher it is, the closer the flow is to incompressible' },
  kperp: { label: 'Waves across the field', default: 1, min: 1, max: 16, step: 1, tip: 'Wavelengths of each wave across the field (along x for one, y for the other) in a box side' },
} as const;

const gas = (q: Waves, w: ReturnType<typeof waves>) => (x: number, y: number, z: number) => {
  const by = w.by(x, z);
  const bx = w.bx(y, z);
  // u = b for z+, u = -b for z- (density 1).
  return { rho: 1, vx: -bx, vy: by, vz: 0, p: 0.5 * q.beta * q.b0 * q.b0, s: z > 0 ? 1 : 0 };
};

const faces = (q: Waves, w: ReturnType<typeof waves>) => (axis: number, x: number, y: number, z: number) =>
  axis === 0 ? w.bx(y, z) : axis === 1 ? w.by(x, z) : q.b0;

/**
 * Two plane Alfven waves counter-propagating along the guide field, filling the periodic box
 * (Howes & Nielson 2013): their collision builds a secondary field fluctuation with no
 * parallel variation, then tertiary waves at higher perpendicular wavenumber, growing with time.
 */
const collision: FlowDef<Waves & Record<string, number>> = {
  name: 'alfven-collision',
  label: 'Alfven-wave collision',
  tip: 'Two plane Alfven waves travelling in opposite directions along a guide field B0 (along z), in uniform gas of density 1: one varies along x and is polarized along y, the other varies along y and is polarized along x (Howes & Nielson 2013). Needs a 3D periodic box.',
  wholeBox: true,
  fields: {
    ...COMMON,
    kpar: { label: 'Waves along the field', default: 1, min: 1, max: 16, step: 1, tip: 'Wavelengths of each wave along the guide field in a box side' },
  },
  face: (ctx, q) => faces(q, waves(ctx, q, () => 1)),
  init: (ctx, q) => gas(q, waves(ctx, q, () => 1)),
};

/**
 * Two Alfven-wave packets, localized along the guide field, heading for each other (Verniero
 * et al. 2018): they interact only while they overlap, then separate, each carrying the
 * smaller-scale structure the collision put on it.
 */
const packets: FlowDef<Waves & { width: number } & Record<string, number>> = {
  name: 'alfven-packets',
  label: 'Elsasser packets',
  tip: 'Two Alfven-wave packets on a guide field B0 (along z), each with a Gaussian profile along the field: a z+ packet above the center heading down and a z- packet below it heading up, in uniform gas of density 1 (Verniero et al. 2018). They meet at the center at t = box / (4 B0). Needs a 3D periodic box.',
  wholeBox: true,
  fields: {
    ...COMMON,
    kpar: { label: 'Wavelengths in a packet', default: 0, min: 0, max: 8, step: 1, tip: 'Wavelengths along the field under each packet’s profile, over a box side: 0 is a packet of one sign' },
    width: { label: 'Packet width', default: 0.08, min: 0.02, max: 0.25, step: 0.01, tip: 'Width (standard deviation) of each packet along the field, in box sides' },
  },
  face: (ctx, q) => faces(q, waves(ctx, q, envelope(ctx, q.width))),
  init: (ctx, q) => gas(q, waves(ctx, q, envelope(ctx, q.width))),
};

/** The packets' profile along z: the z+ packet (sign 1) centered a quarter box above the center, the z- one a quarter below; periodic. */
function envelope(ctx: FlowContext, width: number): (z: number, sign: number) => number {
  const L = ctx.box;
  const s = width * L;
  return (z, sign) => {
    let d = z - sign * 0.25 * L;
    d -= L * Math.round(d / L);
    return Math.exp(-(d * d) / (2 * s * s));
  };
}

export const flows = [collision, packets];
