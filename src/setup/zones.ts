// A setup's refinement zones (mesh.zones): the shapes they come to once its bodies are built,
// what is wrong with them, and the set a binary of two stars usually wants. The geometry itself
// (which zone a point is in, a block's tier) is amr/zones.ts.
import type { Zone, ZoneCut } from '../amr/zones';
import { innerLagrange } from '../problems/roche';
import type { Bodies } from './parts/bodies';
import type { SetupSpec, ZoneAnchor, ZoneSpec } from './spec';

const AXES = { x: 0, y: 1, z: 2 } as const;

/** Where an anchor is and its radius, or why it cannot be placed. */
function anchorOf(a: ZoneAnchor | undefined, bodies: Bodies): { c: number[]; r: number } | string {
  if (!a || typeof a.at !== 'string') return 'it says nowhere to be';
  if (!(a.r > 0)) return 'its radius is not above 0';
  if (a.at === 'L1') {
    const bin = bodies.binary;
    if (!bin?.lobe) return 'it is at L1, which needs a star on a circular binary orbit';
    return { c: innerLagrange(bin.lobe.bodies, bin.omega).at.map((v) => v ?? 0), r: a.r };
  }
  const star = bodies.stars.find((s) => s.id === a.at);
  if (!star) return `it is at "${a.at}", which is neither L1 nor a star`;
  return { c: [star.center[0], star.center[1], star.center[2] ?? 0], r: a.radii ? a.r * star.R : a.r };
}

/** A zone's shape in the box, or why it cannot be had. */
export function zoneOf(z: ZoneSpec, bodies: Bodies): Zone | string {
  const cuts: ZoneCut[] = [
    ...Object.entries(z.below ?? {}).filter(([, v]) => typeof v === 'number').map(([k, v]) => ({ axis: AXES[k as keyof typeof AXES], value: v as number })),
    ...Object.entries(z.above ?? {}).filter(([, v]) => typeof v === 'number').map(([k, v]) => ({ axis: AXES[k as keyof typeof AXES], value: v as number, above: true })),
  ];
  const tier = z.tier;
  if (!(typeof tier === 'number' && tier >= 1)) return 'its tier is not 1 or more';
  if (z.shape === 'cone') {
    const a = anchorOf(z.from, bodies);
    if (typeof a === 'string') return `its first sphere: ${a}`;
    const b = anchorOf(z.to, bodies);
    if (typeof b === 'string') return `its second sphere: ${b}`;
    if (!(Math.hypot(...[0, 1, 2].map((k) => b.c[k] - a.c[k])) > Math.abs(b.r - a.r))) return 'one of its spheres lies inside the other: no cone touches both';
    return { shape: 'cone', c1: a.c, r1: a.r, c2: b.c, r2: b.r, tier, cuts };
  }
  const a = anchorOf(z, bodies);
  if (typeof a === 'string') return a;
  const rIn = z.rIn === undefined ? 0 : z.radii ? z.rIn * (a.r / z.r) : z.rIn;
  if (!(rIn < a.r)) return 'its inner radius is not inside its outer one';
  return { shape: 'sphere', c: a.c, r: a.r, rIn, tier, cuts };
}

/** A setup's zones as shapes in the box. Throws at the first that cannot be placed. */
export function zonesFrom(spec: SetupSpec, bodies: Bodies): Zone[] {
  return (spec.mesh?.zones ?? []).map((z, i) => {
    const out = zoneOf(z, bodies);
    if (typeof out === 'string') throw new Error(`refinement zone ${i + 1}: ${out}`);
    return out;
  });
}

/**
 * The zones a binary of two stars on a circular orbit usually wants, seen from above with the
 * donor (the orbit's first body) at -x: L1 kept longest, then the half of a shell from one to
 * two accretor radii below the line of centers (where the stream arrives), the other half, the
 * accretor, the donor; and two cones joining a sphere about L1 to the donor's surface and to the
 * shell's outer one, the first tier where nothing listed before them holds them (the stream's
 * cut off above the plane tangent to the top of the sphere about L1). The sphere about L1 is a
 * quarter of the donor's radius. Null if the setup has no such binary.
 */
export function binaryZones(spec: SetupSpec, bodies: Bodies): ZoneSpec[] | null {
  const orbit = (spec.parts ?? []).find((p) => p.kind === 'orbit' && p.type === 'binary' && !p.ecc);
  if (!orbit || orbit.kind !== 'orbit' || !bodies.binary?.lobe) return null;
  const [donorId, accId] = orbit.of ?? [];
  const donor = bodies.stars.find((s) => s.id === donorId);
  const acc = bodies.stars.find((s) => s.id === accId);
  if (!donor || !acc) return null;
  const r1 = +(0.26 * donor.R).toPrecision(2);
  return [
    { name: 'L1', shape: 'sphere', at: 'L1', r: r1, tier: 1 },
    { name: 'shell, stream side', shape: 'sphere', at: acc.id, rIn: 1, r: 2, radii: true, below: { y: 0 }, tier: 2 },
    { name: 'shell, far side', shape: 'sphere', at: acc.id, rIn: 1, r: 2, radii: true, above: { y: 0 }, tier: 3 },
    { name: 'accretor', shape: 'sphere', at: acc.id, r: 1, radii: true, tier: 4 },
    { name: 'donor', shape: 'sphere', at: donor.id, r: 1, radii: true, tier: 5 },
    { name: 'stream', shape: 'cone', from: { at: 'L1', r: r1 }, to: { at: acc.id, r: 2, radii: true }, below: { y: r1 }, tier: 1 },
    { name: 'neck', shape: 'cone', from: { at: 'L1', r: r1 }, to: { at: donor.id, r: 1, radii: true }, tier: 1 },
  ];
}
