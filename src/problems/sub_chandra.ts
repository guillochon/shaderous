import { all, OUTFLOW_ONLY, type Problem } from './types';
import { COMPOSITIONS } from '../units/composition';
import { codeG } from '../units/units';
import { dwarfsInit, dwarfsSettling, scalesOf, whiteDwarf, WD_AMBIENT, WD_UNITS, type PlacedDwarf, type WhiteDwarf } from './white_dwarf';
import { makeEos, type Eos } from '../eos/eos';
import { NET_SPECIES } from '../burn/network';
import { hydrogenParticles, mixMoments, problemCoefficients } from '../species/mixture';

// A sub-Chandrasekhar Type Ia supernova by double detonation (Livne 1990; Fink et al. 2007,
// 2010; Shen et al. 2018): a 1 M_sun white dwarf, a carbon-oxygen core under a thin shell of
// helium, set alight at the base of the shell at its north pole. The helium detonation runs
// around the star and its shock goes into the core, where it converges on the far side, off
// center, and can ignite a carbon detonation that burns the whole core: mostly Ni56 for a
// 1 M_sun dwarf (about 0.5-0.6 M_sun in Shen et al. 2018), intermediate-mass elements outside.
//
// The star is the cold white dwarf of problems/white_dwarf.ts (helium and carbon-oxygen both
// have two nucleons per electron, so one structure serves both), settled into the mesh's
// hydrostatic balance; the outermost mass, down to the shell's mass, is helium. Settling
// rewrites the star as cold gas, so the hot spot goes in afterward (Problem.heat). Burning is
// the trained aprox13 network (burn/network.ts); the equation of state follows each cell's
// composition (eos/helmholtz.wgsl). Units as for the white dwarf collision: 10^4 km, one
// solar mass, G = 1 (2.75 s).
//
// The box is a little larger than the star, so the ejecta leave it within a few seconds;
// nearly everything has burned by then. On the default 98 km cells (2026-11-16): the helium
// detonation passes the equator at t = 0.4 (1.1 s) and meets itself at the south pole near
// t = 0.55, where the carbon lights; by t = 0.8 (2.2 s) the core has burned (0.003 M_sun of
// carbon left), with 0.51 M_sun of Ni56 in the box and 0.11 M_sun of the outer layers gone.
// The settled star is puffier than the analytic one on cells this size, so the cells marked
// helium hold more than the shell's mass asked for: 0.079 M_sun for 0.05.
export const SUB_CHANDRA = { box: 1.25 };

const COMP = COMPOSITIONS.carbonOxygen;
const CO_X = NET_SPECIES.map((s) => (s === 'c12' || s === 'o16' ? 0.5 : 0));
const HE_X = NET_SPECIES.map((s) => (s === 'he4' ? 1 : 0));

/** Radius (code units) inside which a white dwarf holds mass m. */
function radiusOfMass(wd: WhiteDwarf, m: number): number {
  const steps = 20000;
  const h = wd.R / steps;
  let M = 0;
  let prev = 0;
  for (let i = 1; i <= steps; i++) {
    const r = i * h;
    const f = 4 * Math.PI * r * r * wd.rho(r);
    const dM = 0.5 * (prev + f) * h;
    if (M + dM >= m) return r - h + (h * (m - M)) / dM;
    M += dM;
    prev = f;
  }
  return wd.R;
}

/** The star, the base of its shell, and the hot spot. */
function layout(p: Record<string, number>, eos: Eos) {
  const wd = whiteDwarf(p.m, scalesOf(eos), codeG(WD_UNITS));
  const rBase = radiusOfMass(wd, wd.M - p.mhe);
  const rSpot = (p.spot * 1e3) / WD_UNITS.length; // km to code units (lengths in m)
  // In the shell on +z, one spot radius clear of the core: a spot touching the core lights its
  // carbon directly (an edge-lit detonation) on cells of tens of km.
  const center = [0, 0, rBase + 2 * rSpot];
  return { wd, rBase, rSpot, center };
}

const placed = (p: Record<string, number>, eos: Eos): PlacedDwarf[] => [{ wd: layout(p, eos).wd, center: [0, 0, 0], vel: [0, 0, 0], scalar: 1 }];

/** The helium's equation of state, for the hot spot's temperature. */
function heliumEos(eos: Eos): Eos {
  const k = problemCoefficients({ species: NET_SPECIES, composition: COMP }, eos.params.gamma)!;
  return makeEos('helmholtz', eos.params, mixMoments(k, HE_X), hydrogenParticles(COMP));
}

/** The hot spot's internal energy density at a point, or undefined outside it (or outside the star's helium). */
function spot(p: Record<string, number>, eos: Eos, ndim: 2 | 3) {
  const { rBase, rSpot, center } = layout(p, eos);
  const he = heliumEos(eos);
  const T = (p.tign * 1e9) / eos.params.helmT;
  const c = ndim === 3 ? center : [0, center[2], 0];
  return (x: number, y: number, z: number, rho: number): number | undefined => {
    if (!(rho > WD_AMBIENT.rhoCut) || Math.hypot(x, y, z) < rBase) return undefined;
    if (Math.hypot(x - c[0], y - c[1], z - c[2]) > rSpot) return undefined;
    return he.eintFromT(rho, T);
  };
}

export const subChandra: Problem = {
  id: 'sub-chandra',
  label: 'Sub-Chandrasekhar Type Ia (double detonation)',
  blurb:
    'A 1 M☉ carbon-oxygen white dwarf under a thin helium shell, set alight at the base of the shell at its pole. The helium detonation wraps around the star and drives a shock into the core, which converges off center and can ignite carbon: the double-detonation model of Type Ia supernovae. Burned by the trained aprox13 network.',
  eos: 'helmholtz',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW_ONLY),
  bcHi: all(OUTFLOW_ONLY),
  gravity: 'isolated',
  G: codeG(WD_UNITS),
  units: WD_UNITS,
  composition: COMP,
  species: NET_SPECIES,
  burn: (p) => ({ network: 'aprox13', inShocks: p.shockburn > 0.5, limiter: p.limiter }),
  box: SUB_CHANDRA.box,
  // The core has burned by t = 0.8 (2.2 s); by t = 1 most of the ejecta have left the box.
  tEnd: 1,
  checkpointDt: 0.1,
  tDyn: 0.4, // sqrt(R^3 / G M) of a 1 M_sun dwarf
  supports2d: false,
  rhoFloor: 1e-10,
  pFloor: 1e-14,
  fluff: { rho: 10 * WD_AMBIENT.rhoAmb, u: 3, v: 10 },
  // The whole star at the finest level (2e4 g/cm^3 and up): the shell and its detonation too.
  refineDense: 0.01,
  // 98 km cells: the whole star at 49 km (n 256, amr 2) needs more than the default 2.5 GB of
  // blocks, and the mesh then stays at 98 km.
  grid: { ndim: 3, n: 128, amr: 1 },
  view: { field: 0, log: true, mode: 'slice', axis: 0 },
  params: [
    { id: 'm', label: 'Mass (M☉)', title: 'Mass of the white dwarf, core and shell, in solar masses. Changing it starts the run again', value: 1.0, min: 0.8, max: 1.2, step: 0.05 },
    { id: 'mhe', label: 'Helium shell (M☉)', title: 'Mass of the helium shell, in solar masses: the outermost mass of the star. Thin shells (0.01-0.05) are what make normal Type Ia supernovae; too thin and the helium does not detonate at this resolution. Changing it starts the run again', value: 0.05, min: 0.005, max: 0.15, step: 0.005 },
    { id: 'tign', label: 'Ignition temperature (10⁹ K)', title: 'Temperature of the hot spot that sets the helium alight, in 1e9 K. Changing it starts the run again', value: 3, min: 1, max: 6, step: 0.25 },
    { id: 'spot', label: 'Hot spot radius (km)', title: 'Radius of the hot spot, in km, centered one radius above the base of the shell on the +z axis. Changing it starts the run again', value: 300, min: 50, max: 1000, step: 25 },
    { id: 'shockburn', label: 'Burn inside shocks', title: 'Let cells the shock detector flags burn (1) or not (0). Off, as in most supernova codes: a captured shock is a few cells of numerical smearing. Changing it starts the run again', value: 0, min: 0, max: 1, step: 1 },
    { id: 'limiter', label: 'Burning limiter f', title: 'No cell takes in burning energy faster than f times its internal energy per sound crossing of the cell (Kushnir et al. 2013); 0 switches it off (the default). Changing it starts the run again', value: 0, min: 0, max: 1, step: 0.05 },
  ],
  settle: (p, eos) => dwarfsSettling(scalesOf(eos), placed(p, eos), codeG(WD_UNITS)),
  heat: (p, eos) => spot(p, eos, 3),
  init: (_n, eos, ndim, p) => {
    const at = dwarfsInit(scalesOf(eos), placed(p, eos));
    const { rBase } = layout(p, eos);
    const hot = spot(p, eos, ndim);
    const he = heliumEos(eos);
    return (x, y, z) => {
      const w = at(x, y, z);
      const X = w.rho > WD_AMBIENT.rhoCut && Math.hypot(x, y, z) < rBase ? CO_X : HE_X;
      const e = hot(x, y, z, w.rho);
      return { ...w, X, p: e === undefined ? w.p : he.pressure(w.rho, e) };
    };
  },
};
