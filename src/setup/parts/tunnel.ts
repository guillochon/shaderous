// A wind tunnel (plans/wind-tunnel.md 4.1): one part that writes what a person would otherwise
// assemble by hand. The stream flows along `axis` at `speed` from an inlet (cells held at the
// stream's state at the upstream face) to an outlet (an open face, with a sponge: a damping
// zone over the last `sponge` of the box that pulls the gas toward the stream's state, so that
// a wake leaves without reflecting). Sides and roof are slip walls, open or periodic; the floor
// (the lower face along the box's up axis: z in 3D, y in 2D) may be the sides' kind, a slip
// wall, a no-slip wall or a wall that moves with the stream (a car on a road). Below Mach 0.3
// the gas is the liquid with implicit pressure; above, an ideal gas (gamma 1.4) with the
// explicit scheme. The whole box starts at the stream's speed.
import type { ForceRef } from '../../problems/types';
import type { Prim } from '../../problems/types';
import { AXES, type SetupSpec, type TunnelPart, type Vec3 } from '../spec';
export { AXES, TUNNEL_FLOORS, TUNNEL_SIDES, type TunnelPart } from '../spec';

export const TUNNEL_DEFAULTS = { speed: 1, axis: 'x', floor: 'none', sides: 'slip', sponge: 0.1, mach: 0.1, rho: 1, inlet: 2 } as const;

/** The setup's tunnel, if it has one. */
export function tunnelOf(spec: SetupSpec): TunnelPart | undefined {
  return (spec.parts ?? []).find((p) => p.kind === 'tunnel') as TunnelPart | undefined;
}

export interface Tunnel {
  part: TunnelPart;
  speed: number;
  a: 0 | 1 | 2;
  /** the box's up axis: z in 3D, y in 2D */
  up: 0 | 1 | 2;
  rho: number;
  /** sound speed, and the pressure of the stream (0 for the liquid) */
  cs: number;
  p: number;
  liquid: boolean;
  sponge: number;
  inlet: number;
}

export function tunnelParams(t: TunnelPart, ndim: 2 | 3): Tunnel {
  const speed = t.speed ?? TUNNEL_DEFAULTS.speed;
  const a = AXES.indexOf(t.axis ?? TUNNEL_DEFAULTS.axis) as 0 | 1 | 2;
  const up = (ndim === 2 ? 1 : 2) as 0 | 1 | 2;
  const mach = t.mach ?? TUNNEL_DEFAULTS.mach;
  const rho = t.rho ?? TUNNEL_DEFAULTS.rho;
  const cs = Math.abs(speed) / Math.max(mach, 1e-6);
  const liquid = mach < 0.3;
  const gamma = 1.4;
  return { part: t, speed, a, up, rho, cs, p: liquid ? 0 : (rho * cs * cs) / gamma, liquid, sponge: t.sponge ?? TUNNEL_DEFAULTS.sponge, inlet: t.inlet ?? TUNNEL_DEFAULTS.inlet };
}

/**
 * The setup as the tunnel makes it: the box's faces, the gas, how the pressure acts, and the
 * background at the stream's state. What the document says of these is replaced; the rest
 * (viscosity, the grid, the parts) is its own.
 */
export function expandTunnel(spec: SetupSpec): SetupSpec {
  const t = tunnelOf(spec);
  if (!t) return spec;
  const ndim = spec.domain?.ndim === 3 ? 3 : 2;
  const k = tunnelParams(t, ndim);
  const side = { slip: 2, open: 1, periodic: 0 }[t.sides ?? TUNNEL_DEFAULTS.sides];
  const lo: [number, number, number] = [side, side, side];
  const hi: [number, number, number] = [side, side, side];
  lo[k.a] = 1;
  hi[k.a] = 1;
  if (ndim === 2) lo[2] = hi[2] = 0;
  const floor = t.floor ?? TUNNEL_DEFAULTS.floor;
  const wallVelocity: Record<string, Vec3> = {};
  if (k.up !== k.a && floor !== 'none') {
    lo[k.up] = { slip: 2, fixed: 4, moving: 5 }[floor];
    if (floor === 'moving') {
      const v: Vec3 = [0, 0, 0];
      v[k.a] = k.speed;
      wallVelocity[`${'xyz'[k.up]}lo`] = v;
    }
  }
  const v: Vec3 = [0, 0, 0];
  v[k.a] = k.speed;
  const ph = { ...(spec.physics ?? {}) };
  if (k.liquid) {
    ph.eos = 'liquid';
    ph.eosParams = { ...(ph.eosParams ?? {}), liqRho: k.rho, liqC: k.cs };
    ph.pressure = 'implicit';
  } else {
    ph.eos = 'ideal';
    ph.eosParams = { ...(ph.eosParams ?? {}), gamma: 1.4 };
    ph.pressure = 'explicit';
  }
  ph.gravity = ph.gravity ?? 'off';
  return {
    ...spec,
    domain: { ...(spec.domain ?? {}), bc: { lo, hi }, ...(Object.keys(wallVelocity).length ? { wallVelocity: { ...(spec.domain?.wallVelocity ?? {}), ...wallVelocity } } : {}) },
    physics: ph,
    background: { ...(spec.background ?? {}), rho: k.rho, p: k.p, v },
  };
}

/** The inlet: cells within `inlet` cells of the upstream face, held at the stream's state. */
export function tunnelPainter(t: TunnelPart, ctx: { box: number; n: number; ndim: 2 | 3 }): (w: Prim, x: number, y: number, z: number) => Prim {
  const k = tunnelParams(t, ctx.ndim);
  const edge = -ctx.box / 2 + (k.inlet * ctx.box) / ctx.n;
  const v = [0, 0, 0];
  v[k.a] = k.speed;
  return (w, x, y, z) => {
    if ([x, y, z][k.a] >= edge || w.solid) return w;
    return { ...w, rho: k.rho, vx: v[0], vy: v[1], vz: v[2], p: k.p, fixed: true };
  };
}

const f32 = (x: number) => {
  const s = String(Math.fround(x));
  return /[.eE]/.test(s) ? s : `${s}.0`;
};

/**
 * The sponge: a damping zone (hydro/damping.wgsl) over the last `sponge` of the box's length
 * along the stream, its rate rising as the square of the depth into it to `4 U / (sponge L)` at
 * the outlet, pulling the gas to the stream's state.
 */
export function tunnelDamping(t: TunnelPart, box: number, ndim: 2 | 3): { wgsl: string; radius: number; rate: number } | undefined {
  const k = tunnelParams(t, ndim);
  if (!(k.sponge > 0)) return undefined;
  const len = k.sponge * box;
  const start = box / 2 - len;
  const rate = (4 * Math.abs(k.speed)) / len;
  const v = [0, 0, 0];
  v[k.a] = k.speed;
  const wgsl = `const DAMPING = true;
fn damping_target(x: vec3<f32>) -> DampTarget {
  let s = (x[${k.a}] - ${f32(start)}) / ${f32(len)};
  if (s <= 0.0) { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }
  let ramp = min(s, 1.0);
  return DampTarget(${f32(rate)} * ramp * ramp, ${f32(k.rho)}, vec3<f32>(${v.map(f32).join(', ')}), ${f32(k.p)});
}
`;
  return { wgsl, radius: 0, rate: 0 };
}

/**
 * The coefficients' reference: the stream's density and speed, an area (the tunnel's `area`, or
 * the shadow of the first obstacle along the stream: `shadow`), and a length (`length`, or the
 * square root of the area; in 2D the area is a length itself). Drag along the stream, lift
 * along the box's up axis (or, for a stream along it, the next axis).
 */
export function tunnelForceRef(t: TunnelPart, ndim: 2 | 3, shadow: number | undefined): ForceRef | undefined {
  const k = tunnelParams(t, ndim);
  const area = t.area ?? shadow;
  if (!(area !== undefined && area > 0)) return undefined;
  const lift = (k.up !== k.a ? k.up : ((k.a + 1) % 3)) as 0 | 1 | 2;
  return { rho: k.rho, speed: Math.abs(k.speed), area, length: t.length ?? (ndim === 2 ? area : Math.sqrt(area)), drag: k.a, lift };
}
