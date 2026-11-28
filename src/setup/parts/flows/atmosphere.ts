// An isothermal atmosphere at rest in a uniform field, between walls, over many scale heights:
// the column of the hydro-proofs failure maps (column_discrete_balance), built in the grid's own
// hydrostatic balance so that it is an exact solution of the scheme. Its density falls by e every
// scale height; past a total contrast of about 1e-10, rounding of order 1e-7 in its state sends
// sound waves up the column whose speed grows as rho^(-1/2), and it stirs itself up. Stored as its
// deviation from a reference column (docs/numerics.md section 7), it stays at rest however deep.
// "Up" is y in 2D and z in 3D.
import type { FlowDef } from '../flow';

const atmosphere: FlowDef<{ T: number; rho0: number }> = {
  name: 'atmosphere',
  label: 'Isothermal atmosphere',
  tip: 'Gas at one temperature at rest in a unit box, in hydrostatic balance with the uniform downward acceleration g of the setup (Gravity: Uniform acceleration), between walls at the bottom and top: the density falls by e every scale height, T / g. Built in the grid’s own balance, cell by cell, so nothing should move. Up is y in 2D and z in 3D.',
  wholeBox: true,
  fields: {
    T: { label: 'Temperature', default: 0.025, min: 1e-8, max: 1e8, factor: 1.25, tip: 'p / rho, the square of the isothermal sound speed. With the setup’s acceleration g it sets the scale height T / g, and the box holds g / T of them (at the box side 1): the density at the top is e to the minus that of the bottom’s' },
    rho0: { label: 'Density at the bottom', default: 1, min: 1e-20, max: 1e20, factor: 2, tip: 'Density of the lowest cells' },
  },
  init: (ctx, q) => {
    const ndim = ctx.ndim;
    const n = Math.max(1, Math.round(ctx.n));
    const dx = ctx.box / n;
    // The setup's own downward acceleration, which the atmosphere is in balance with.
    const gext = ctx.spec.physics?.gext;
    const g = gext === undefined ? 1 : Array.isArray(gext) ? -gext[ndim - 1] : gext.down;
    // Cell by cell, the trapezoid rule of the balanced reconstruction for an isothermal gas,
    // T (rho[j+1] - rho[j]) = -g dx (rho[j] + rho[j+1]) / 2: rho[j+1] = rho[j] (1 - a) / (1 + a).
    const a = (g * dx) / (2 * q.T);
    const ratio = Math.max(0, (1 - a) / (1 + a));
    const rho = new Float64Array(n);
    for (let j = 0; j < n; j++) rho[j] = q.rho0 * Math.pow(ratio, j);
    return (_x, y, z) => {
      const h = [_x, y, z][ndim - 1] / ctx.box + 0.5; // height, 0 to 1 (up: x in 1D, y in 2D, z in 3D)
      const j = Math.min(n - 1, Math.max(0, Math.floor(h * n)));
      return { rho: rho[j], vx: 0, vy: 0, vz: 0, p: q.T * rho[j], s: h };
    };
  },
  issues: (ctx) => {
    const gext = ctx.spec.physics?.gext;
    const g = gext === undefined ? 1 : Array.isArray(gext) ? -gext[ctx.ndim - 1] : gext.down;
    return g <= 0 ? 'the atmosphere needs a uniform acceleration pointing down' : null;
  },
};

export const flows = [atmosphere];
