import { all, OUTFLOW_ONLY, type Problem } from './types';
import { COMPOSITIONS } from '../units/composition';
import { codeG } from '../units/units';
import { dwarfsInit, dwarfsSettling, scalesOf, whiteDwarf, WD_AMBIENT, WD_UNITS, type PlacedDwarf } from './white_dwarf';
import type { Eos } from '../eos/eos';

// Stellar collisions: two stars on a parabolic orbit about their common center of mass, set
// up a little way apart and left to fall together under their own gravity (isolated
// self-gravity, no point masses).
//
// White dwarfs. Each star is a cold carbon-oxygen white dwarf, in hydrostatic equilibrium
// under the pressure of its degenerate electrons (problems/white_dwarf.ts), and the gas obeys
// the matching equation of state (degenerate electrons plus ideal gas and radiation), so the
// heat of the collision shows up as the thermal part. No nuclear burning yet.
//
// Units: 10^4 km, one solar mass, and the time that makes G = 1 (2.75 s). A 1 M_sun white
// dwarf is then 0.54 across in radius, with a central density of 17 (3.4e7 g/cm^3); the speed
// unit is 3640 km/s.
//
// The orbit lies in the x-y plane with pericenter along +x. The impact parameter b is the
// pericenter distance in units of the sum of the two radii: 0 is head-on, 1 a graze. The stars
// start STARTS times that sum apart (less if they would not fit in the box), moving as the
// parabolic orbit has them there. Both are then settled into the grid's own hydrostatic
// balance, each in its own potential, keeping their velocities (problems/settle.ts). Mapped
// as they are, they would ring: a 1 M_sun dwarf's core is mildly relativistic, with Gamma_1 not
// far above 4/3, so the small mismatch of the mapping meets little restoring force, and its
// central density swung by 20% in its fundamental mode (period about 6 s).
export const WD_COLLISION = {
  box: 8,
  /** starting separation, in sums of the radii */
  starts: 2,
};

/** The two dwarfs as the parameters place them. */
function placed(p: Record<string, number>, eos: Eos): PlacedDwarf[] {
  const G = codeG(WD_UNITS);
  const s = scalesOf(eos);
  const wds = [whiteDwarf(p.m1, s, G), whiteDwarf(p.m2, s, G)];
  const o = collisionOrbit(p.m1, p.m2, wds[0].R, wds[1].R, p.b, G, WD_COLLISION.box);
  // The passive scalar tells the two stars apart.
  return wds.map((wd, i) => ({ wd, center: o.pos[i], vel: o.vel[i], scalar: i === 0 ? 1 : 0.5 }));
}

/** Positions and velocities of the two stars (relative to the center of mass) at the start. */
export function collisionOrbit(m1: number, m2: number, R1: number, R2: number, b: number, G: number, box: number) {
  const M = m1 + m2;
  const mu = G * M;
  const rp = b * (R1 + R2);
  // STARTS sums of the radii apart, or closer if that would put either star's far side more
  // than 45% of the box's side from the center; never closer than just touching, or than pericenter.
  const half = 0.45 * box;
  const fit = Math.min(((half - R1) * M) / m2, ((half - R2) * M) / m1);
  const d = Math.max(Math.min(WD_COLLISION.starts * (R1 + R2), fit), Math.max(1.05 * (R1 + R2), 1.05 * rp));
  // Parabola r = 2 rp / (1 + cos f); on the way in, f < 0. Speed sqrt(2 mu / r), of which
  // h / r is across the line of centers, h = sqrt(2 mu rp).
  const f = -Math.acos(Math.min(1, Math.max(-1, (2 * rp) / d - 1)));
  const v = Math.sqrt((2 * mu) / d);
  const vt = Math.sqrt(2 * mu * rp) / d;
  const vr = -Math.sqrt(Math.max(v * v - vt * vt, 0));
  const [c, s] = [Math.cos(f), Math.sin(f)];
  // Star 2 relative to star 1.
  const rel = [d * c, d * s, 0];
  const vrel = [vr * c - vt * s, vr * s + vt * c, 0];
  return {
    d,
    rp,
    pos: [rel.map((x) => (-m2 / M) * x), rel.map((x) => (m1 / M) * x)],
    vel: [vrel.map((x) => (-m2 / M) * x), vrel.map((x) => (m1 / M) * x)],
  };
}

export const wdCollision: Problem = {
  id: 'wd-collision',
  label: 'White dwarf collision',
  blurb: 'Two cold carbon-oxygen white dwarfs on a parabolic orbit, falling together. Masses in solar masses; the impact parameter is the pericenter distance over the sum of the radii (0 is head-on).',
  eos: 'degenerate',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW_ONLY),
  bcHi: all(OUTFLOW_ONLY),
  gravity: 'isolated',
  G: codeG(WD_UNITS),
  units: WD_UNITS,
  composition: COMPOSITIONS.carbonOxygen,
  box: WD_COLLISION.box,
  tEnd: Infinity,
  rhoFloor: 1e-10,
  pFloor: 1e-14,
  // Ambient gas and the thinnest debris: no hotter than the stars' outer layers get when
  // shocked, no faster than a few times the speed at contact.
  fluff: { rho: 10 * WD_AMBIENT.rhoAmb, u: 3, v: 10 },
  view: { field: 0, log: true, mode: 'volume', pitch: 1.15 },
  params: [
    { id: 'm1', label: 'Mass 1 (M☉)', title: 'Mass of the first white dwarf, in solar masses (carbon-oxygen white dwarfs are 0.5 to 1.3). Changing it starts the run again', value: 1.0, min: 0.5, max: 1.3, step: 0.05 },
    { id: 'm2', label: 'Mass 2 (M☉)', title: 'Mass of the second white dwarf, in solar masses. Changing it starts the run again', value: 0.8, min: 0.5, max: 1.3, step: 0.05 },
    { id: 'b', label: 'Impact parameter', title: 'Pericenter distance of the parabolic orbit over the sum of the two radii: 0 is head-on, 1 just grazes. Changing it starts the run again', value: 0, min: 0, max: 1.5, step: 0.1 },
  ],
  settle: (p, eos) => dwarfsSettling(scalesOf(eos), placed(p, eos), codeG(WD_UNITS)),
  init: (_n, eos, _ndim, p) => dwarfsInit(scalesOf(eos), placed(p, eos)),
};
