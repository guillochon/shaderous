// Magnetic reconnection with resistivity (mhd/ohm.wgsl): current sheets that tear into islands,
// and islands that merge through the current sheet their X-point collapses into. The choice of
// setups follows the taranis WebGPU reduced-MHD page of Mallet & Abraham
// (https://alfredmallet.github.io/taranis/webgpu/rmhd2d.html); the equilibria are the standard
// ones of the papers named below, not taken from its code.
//
// Units: magnetic pressure B^2 / 2, density 1. The fields come from a vector potential A_z(x, y)
// (B = (dA/dy, -dA/dx)), plus a uniform guide field B_z on the faces. Coordinates are relative
// to the box center; the box is periodic along x and closed by reflecting walls along y, which
// the field lines do not cross.
import type { FlowDef } from '../flow';

/**
 * A Harris current sheet (Harris 1962) along y = 0, B_x = B0 tanh(y / a), held by the pressure
 * of the gas in it, with a small perturbation of the flux that seeds the tearing mode (Furth,
 * Killeen & Rosenbluth 1963). Unstable to wavelengths longer than 2 pi a: the sheet breaks into
 * a chain of islands, at a rate set by the Lundquist number a v_A / eta.
 */
const harris: FlowDef<{ b0: number; width: number; beta: number; guide: number; psi: number; modes: number }> = {
  name: 'harris-sheet',
  label: 'Harris current sheet',
  tip: 'A current sheet along y = 0: B_x = B0 tanh(y / a), with the gas pressure higher in the sheet so that p + B² / 2 is uniform, density 1, an optional uniform guide field B_z, and a perturbation A_z = psi B0 L cos(k x) cos(pi y / L) that seeds the tearing mode (Harris 1962; Furth, Killeen & Rosenbluth 1963). Periodic along x, walls along y.',
  wholeBox: true,
  fields: {
    b0: { label: 'Field strength', default: 1, min: 0.01, max: 100, factor: 1.25, tip: 'B0, the field either side of the sheet: the Alfven speed there, with density 1' },
    width: { label: 'Sheet half-width', default: 0.05, min: 0.005, max: 0.2, step: 0.005, tip: 'a, the sheet’s half-width, in box sides. Wavelengths longer than 2 pi a tear' },
    beta: { label: 'Plasma beta outside', default: 1, min: 0.01, max: 100, factor: 1.5, tip: 'Gas pressure over magnetic pressure away from the sheet, 2p / B0²' },
    guide: { label: 'Guide field', default: 0, min: 0, max: 100, step: 0.1, tip: 'A uniform B_z along the sheet, over B0: it stiffens the gas against compression' },
    psi: { label: 'Perturbation', default: 1e-4, min: 0, max: 0.1, factor: 2, tip: 'Amplitude of the seeded flux perturbation, over B0 times the box side' },
    modes: { label: 'Wavelengths across', default: 1, min: 1, max: 16, step: 1, tip: 'Wavelengths of the perturbation along the sheet, in a box side' },
  },
  potential: (ctx, q) => {
    const L = ctx.box;
    const a = q.width * L;
    const k = (2 * Math.PI * q.modes) / L;
    return (x, y) => [0, 0, q.b0 * (a * lnCosh(y / a) + q.psi * L * Math.cos(k * x) * Math.cos((Math.PI * y) / L))];
  },
  face: (_ctx, q) => (axis) => (axis === 2 ? q.guide * q.b0 : 0),
  init: (ctx, q) => {
    const a = q.width * ctx.box;
    return (_x, y) => {
      const s = 1 / Math.cosh(y / a);
      return { rho: 1, vx: 0, vy: 0, vz: 0, p: 0.5 * q.b0 * q.b0 * (q.beta + s * s), s: y > 0 ? 1 : 0 };
    };
  },
};

/** ln cosh(u), without overflow for large |u|. */
const lnCosh = (u: number) => {
  const v = Math.abs(u);
  return v + Math.log1p(Math.exp(-2 * v)) - Math.LN2;
};

/**
 * Island coalescence: a chain of magnetic islands in a current sheet, the Fadeev equilibrium
 * A_z = -lambda B0 ln(cosh(y / lambda) + eps cos(x / lambda)) (Fadeev et al. 1965), pushed
 * toward each other. Neighboring islands attract, like parallel currents: they merge through a
 * current sheet between them, at a rate set by the resistivity (Finn & Kaw 1977; Knoll &
 * Chacon 2006).
 */
const islands: FlowDef<{ b0: number; eps: number; islands: number; beta: number; kick: number }> = {
  name: 'island-coalescence',
  label: 'Island coalescence',
  tip: 'A chain of magnetic islands along y = 0, the Fadeev equilibrium A_z = -lambda B0 ln(cosh(y / lambda) + eps cos(x / lambda)), with the gas pressure p = p_out + B0² (1 - eps²) / (2 (cosh(y / lambda) + eps cos(x / lambda))²) that holds it, density 1, and a flow along x that pushes pairs of islands together (Fadeev et al. 1965; Finn & Kaw 1977). Periodic along x, walls along y.',
  wholeBox: true,
  fields: {
    b0: { label: 'Field strength', default: 1, min: 0.01, max: 100, factor: 1.25, tip: 'B0, the field far from the islands: the Alfven speed there, with density 1' },
    eps: { label: 'Island size', default: 0.2, min: 0.01, max: 0.9, step: 0.05, tip: 'eps: 0 is a plain current sheet, larger values make fatter islands' },
    islands: { label: 'Islands across', default: 2, min: 2, max: 16, step: 2, tip: 'Islands along the sheet in a box side (an even number: they merge in pairs)' },
    beta: { label: 'Plasma beta outside', default: 0.4, min: 0.01, max: 100, factor: 1.5, tip: 'Gas pressure over magnetic pressure far from the islands, 2 p_out / B0²' },
    kick: { label: 'Push', default: 0.1, min: 0, max: 1, step: 0.01, tip: 'Speed of the flow that pushes each pair of islands together, over B0' },
  },
  potential: (ctx, q) => {
    const lam = ctx.box / (2 * Math.PI * q.islands);
    return (x, y) => [0, 0, -lam * q.b0 * Math.log(Math.cosh(y / lam) + q.eps * Math.cos(x / lam))];
  },
  init: (ctx, q) => {
    const L = ctx.box;
    const lam = L / (2 * Math.PI * q.islands);
    // Islands (O-points) where cos(x / lambda) = -1; the push moves each pair toward the
    // X-point between them, at the zeros of sin(pi islands x / L) that lie between O-points.
    const k = (Math.PI * q.islands) / L;
    return (x, y) => {
      const u = Math.cosh(Math.min(Math.abs(y) / lam, 80)) + q.eps * Math.cos(x / lam);
      const env = Math.exp(-((y / (2 * lam)) ** 2));
      return {
        rho: 1, vx: -q.kick * q.b0 * Math.sin(k * x) * env, vy: 0, vz: 0,
        p: 0.5 * q.b0 * q.b0 * (q.beta + (1 - q.eps * q.eps) / (u * u)), s: Math.cos(k * x) > 0 ? 1 : 0,
      };
    };
  },
};

export const flows = [harris, islands];
