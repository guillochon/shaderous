// The Zel'dovich pancake (Zel'dovich 1970; the test of Bryan et al. 1995): one plane wave along
// x in an Einstein-de Sitter universe, on its growing mode, whose Lagrangian displacement
// reaches a caustic (an infinitely dense sheet) at redshift zCaustic. With D = a / a_c, the gas
// from comoving q sits at x = q - D sin(kq) / k, with comoving density 1 / (1 - D cos(kq)) and
// proper peculiar velocity -a H D sin(kq) / k; k is one wavelength across the box. Needs
// physics.cosmology, whose starting redshift sets a.
import type { FlowDef } from '../flow';
import { hubble } from '../../../cosmo';
import { cosmologyOfSetup } from '../../compile';

/** Comoving density and peculiar velocity of the pancake at x, at scale factor a (caustic at aC, box L). */
export function zeldovichState(x: number, a: number, aC: number, L: number, H: number): { rho: number; v: number } {
  const k = (2 * Math.PI) / L;
  const D = a / aC;
  // Solve x = q - D sin(kq) / k for q: monotonic while D < 1.
  let q = x;
  for (let i = 0; i < 60; i++) q -= (q - (D * Math.sin(k * q)) / k - x) / (1 - D * Math.cos(k * q));
  return { rho: 1 / (1 - D * Math.cos(k * q)), v: (-a * H * D * Math.sin(k * q)) / k };
}

const zeldovich: FlowDef<{ zCaustic: number; temperature: number }> = {
  name: 'zeldovich',
  label: 'Zel’dovich pancake',
  tip: 'One plane wave along x on its growing mode in an Einstein-de Sitter universe (physics.cosmology with Omega_m = 1): the gas converges into a sheet, a caustic, at the given redshift, and then shocks. Cold gas: its specific internal energy at the start is given.',
  wholeBox: true,
  fields: {
    zCaustic: { label: 'Caustic redshift', default: 1, min: 0, max: 1e3, factor: 1.5, tip: 'Redshift at which the wave would collapse to a sheet' },
    temperature: { label: 'Specific internal energy', default: 1e-10, min: 0, max: 1, factor: 10, tip: 'Internal energy per unit mass of the gas at the start, in (box length times H0) squared' },
  },
  init: (ctx, q) => {
    const k = cosmologyOfSetup(ctx.spec);
    if (!k) throw new Error('the Zel’dovich pancake needs physics.cosmology');
    const aC = 1 / (1 + q.zCaustic);
    const a = k.aStart;
    if (a >= aC) throw new Error('the pancake has already collapsed at the starting redshift');
    const H = hubble(k, a);
    return (x) => {
      const s = zeldovichState(x, a, aC, ctx.box, H);
      return { rho: s.rho, vx: s.v, vy: 0, vz: 0, p: (2 / 3) * s.rho * q.temperature, s: 0 };
    };
  },
};

export const flows = [zeldovich];
