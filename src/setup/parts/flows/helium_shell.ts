// A helium shell on a carbon-oxygen white dwarf, set alight by a hot spot at its base: the
// double-detonation model of sub-Chandrasekhar Type Ia supernovae (Livne 1990; Fink et al.
// 2007, 2010; Shen et al. 2018).
//
// The star is a `star` part (a cold white dwarf: helium and carbon-oxygen both have two
// nucleons per electron, so one structure serves both). This flow changes what that part
// paints: the outermost mass, down to the shell's mass, is helium, the rest carbon and oxygen;
// and a hot spot in the shell on the +z axis, one spot radius clear of the core. Settling
// rewrites the star as cold gas, so the hot spot also goes in afterward (Problem.heat).
import { NET_SPECIES } from '../../../burn/network';
import { makeEos, type Eos } from '../../../eos/eos';
import { hydrogenParticles, mixMoments, problemCoefficients } from '../../../species/mixture';
import { COMPOSITIONS } from '../../../units/composition';
import type { BuiltStar } from '../bodies';
import type { FlowDef, FlowInitContext } from '../flow';

const COMP = COMPOSITIONS.carbonOxygen;
const CO_X = NET_SPECIES.map((s) => (s === 'c12' || s === 'o16' ? 0.5 : 0));
const HE_X = NET_SPECIES.map((s) => (s === 'he4' ? 1 : 0));

/** Radius (code units) inside which a star holds mass m. */
function radiusOfMass(wd: BuiltStar, m: number): number {
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

interface Shell {
  mhe: number;
  tign: number;
  spot: number;
}

/** The base of the shell and the hot spot. */
function layout(ctx: FlowInitContext, p: Shell) {
  const wd = ctx.bodies.stars[0];
  if (!wd) throw new Error('the helium shell needs a star');
  if (!ctx.units) throw new Error('the helium shell needs physical units');
  const rBase = radiusOfMass(wd, wd.M - p.mhe);
  const rSpot = (p.spot * 1e3) / ctx.units.length; // km to code units (lengths in m)
  // In the shell on +z, one spot radius clear of the core: a spot touching the core lights its
  // carbon directly (an edge-lit detonation) on cells of tens of km.
  const center = [0, 0, rBase + 2 * rSpot];
  return { rBase, rSpot, center };
}

/** The helium's equation of state, for the hot spot's temperature. */
function heliumEos(eos: Eos): Eos {
  const k = problemCoefficients({ species: NET_SPECIES, composition: COMP }, eos.params.gamma)!;
  return makeEos('helmholtz', eos.params, mixMoments(k, HE_X), hydrogenParticles(COMP));
}

/** The hot spot's internal energy density at a point, or undefined outside it (or outside the star's helium). */
function spot(ctx: FlowInitContext, p: Shell, ndim: 2 | 3) {
  const { rBase, rSpot, center } = layout(ctx, p);
  const he = heliumEos(ctx.eos);
  const T = (p.tign * 1e9) / ctx.eos.params.helmT;
  const cut = ctx.spec.background?.cut ?? 1e-4;
  const c = ndim === 3 ? center : [0, center[2], 0];
  return (x: number, y: number, z: number, rho: number): number | undefined => {
    if (!(rho > cut) || Math.hypot(x, y, z) < rBase) return undefined;
    if (Math.hypot(x - c[0], y - c[1], z - c[2]) > rSpot) return undefined;
    return he.eintFromT(rho, T);
  };
}

const heliumShell: FlowDef<Shell & Record<string, number | boolean>> = {
  name: 'helium-shell-ignition',
  label: 'Helium shell with a hot spot',
  tip: 'Turns the outermost mass of the first star into helium over a carbon-oxygen core and heats a spot at the base of that shell, on the +z axis. Needs the 13 species of the burning network, the tabulated equation of state and physical units.',
  wholeBox: false,
  requires: [{ fact: 'parts.star', reason: 'Needs a star: the shell is laid on it' }, { fact: 'units' }, { fact: 'species.aprox13' }],
  fields: {
    mhe: { label: 'Helium shell (M☉)', default: 0.05, min: 0.005, max: 0.15, step: 0.005, tip: 'Mass of the helium shell: the outermost mass of the star' },
    tign: { label: 'Ignition temperature (10⁹ K)', default: 3, min: 1, max: 6, step: 0.25, tip: 'Temperature of the hot spot that sets the helium alight, in 1e9 K' },
    spot: { label: 'Hot spot radius (km)', default: 300, min: 50, max: 1000, step: 25, tip: 'Radius of the hot spot, in km, centered one radius above the base of the shell on the +z axis' },
  },
  decorate: (ctx, p, at) => {
    const { rBase } = layout(ctx, p);
    const hot = spot(ctx, p, ctx.ndim);
    const he = heliumEos(ctx.eos);
    const cut = ctx.spec.background?.cut ?? 1e-4;
    return (x, y, z) => {
      const w = at(x, y, z);
      const X = w.rho > cut && Math.hypot(x, y, z) < rBase ? CO_X : HE_X;
      const e = hot(x, y, z, w.rho);
      return { ...w, X, p: e === undefined ? w.p : he.pressure(w.rho, e) };
    };
  },
  heat: (ctx, p) => spot(ctx, p, 3),
};

export const flows = [heliumShell];
