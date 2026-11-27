// Convection in a stratified layer: a polytropic atmosphere of ideal gas in a uniform field,
// between walls, steeper than the adiabat by a superadiabatic excess delta, so that it overturns
// in convection cells. "Up" is y in 2D and z in 3D, as in the Rayleigh-Taylor layers.
//
// The layer is built in the grid's own hydrostatic balance, cell by cell, with rho = C p^(1/gamma
// - delta), so that on the grid it is delta off the adiabat to rounding. (Sampled from the smooth
// polytrope instead, the grid's layer differs from it by (dx / H)^2, 1e-4 on 128 cells, which
// would set how unstable it is in place of delta.) The scheme's own error in a layer's buoyancy
// is a floor all the same: on 128 cells a layer stable by 1e-5 convects as a neutral one does,
// and one stable by 1e-3 does not (docs/numerics.md section 7). Below the floor the flow is the
// grid's, not the layer's.
import type { FlowDef } from '../flow';

const convection: FlowDef<{ delta: number; beta: number; seed: number; cells: number }> = {
  name: 'convection',
  label: 'Convecting layer',
  tip: 'A layer of ideal gas in a unit box, stratified as a polytrope steeper than the adiabat, in balance with the uniform downward acceleration of the setup (Gravity: Uniform acceleration), between walls at the bottom and top: it overturns in convection cells. Seeded with cells at a given Mach number. Up is y in 2D and z in 3D. Below an excess of about 1e-3 on 128 cells the scheme’s own error in the layer’s buoyancy drives the flow, not the excess',
  wholeBox: true,
  fields: {
    delta: { label: 'Superadiabatic excess', default: 0.01, min: 1e-6, max: 0.2, factor: 2, tip: 'How much steeper the layer is than the adiabat: dln T / dln p less its adiabatic value (gamma - 1) / gamma. The more, the faster it overturns. Below about 1e-3 on 128 cells (less on finer grids) the scheme’s own error in the buoyancy is larger than the excess, and the flow is the grid’s' },
    beta: { label: 'Temperature drop', default: 0.5, min: 0.01, max: 0.95, step: 0.01, tip: 'The fraction by which the temperature falls from the bottom to the top. The density falls by about its power 1.5, the pressure by its power 2.5' },
    seed: { label: 'Seed Mach number', default: 1e-3, min: 0, max: 0.5, factor: 2, tip: 'The peak speed of the seeded convection cells over the sound speed at the bottom' },
    cells: { label: 'Cells across', default: 2, min: 1, max: 16, step: 1, tip: 'Pairs of seeded convection cells across the box (along x, and y in 3D)' },
  },
  init: (ctx, q) => {
    const ndim = ctx.ndim;
    const gamma = ctx.eos.params.gamma;
    // The setup's own downward acceleration, which the layer is in balance with.
    const gext = ctx.spec.physics?.gext;
    const g = gext === undefined ? 1 : Array.isArray(gext) ? -gext[ndim === 2 ? 1 : 2] : gext.down;
    // The smooth polytrope T = 1 - beta z', rho = T^m, p = p_b T^(m+1), in balance for p_b = g / ((m + 1) beta),
    // with m set by the excess: 1 / (m + 1) = (gamma - 1) / gamma + delta.
    const m = 1 / ((gamma - 1) / gamma + q.delta) - 1;
    const pb = g / ((m + 1) * q.beta);
    const cb = Math.sqrt(gamma * pb);
    const L = ctx.box;
    const k = (2 * Math.PI * Math.round(q.cells)) / L;
    // A stream function psi = A sin(pi z') cos(k x) (mass flux rho v = curl psi in 2D, free of
    // divergence): its peak speed, at the bottom, q.seed * cb.
    const amp = (q.seed * cb) / Math.max(Math.PI / L, k);
    // The layer in the grid's balance: p[j+1] = p[j] - (rho[j] + rho[j+1]) g dx / 2 (the trapezoid
    // rule of the balanced reconstruction), rho = C p^(1/gamma - delta), from the bottom cell up.
    const n = Math.max(1, Math.round(ctx.n));
    const dx = L / n;
    const ex = 1 / gamma - q.delta;
    const lr = new Float64Array(n);
    const lp = new Float64Array(n);
    const T0 = 1 - q.beta * (0.5 / n);
    lr[0] = Math.pow(T0, m);
    lp[0] = pb * Math.pow(T0, m + 1);
    const C = lr[0] / Math.pow(lp[0], ex);
    for (let j = 0; j + 1 < n; j++) {
      // Newton on f(p) = p - p[j] + (rho[j] + C p^ex) g dx / 2, from the smooth profile's ratio.
      let p = lp[j] * Math.pow((1 - q.beta * ((j + 1.5) / n)) / (1 - q.beta * ((j + 0.5) / n)), m + 1);
      for (let it = 0; it < 30; it++) {
        const f = p - lp[j] + 0.5 * (lr[j] + C * Math.pow(p, ex)) * g * dx;
        const df = 1 + 0.5 * C * ex * Math.pow(p, ex - 1) * g * dx;
        const step = f / df;
        p -= step;
        if (Math.abs(step) <= 1e-16 * p) break;
      }
      lp[j + 1] = p;
      lr[j + 1] = C * Math.pow(p, ex);
    }
    return (x, y, z) => {
      const h = (ndim === 2 ? y : z) / L + 0.5; // height, 0 to 1
      const j = Math.min(n - 1, Math.max(0, Math.floor(h * n)));
      const rho = lr[j];
      const p = lp[j];
      const sz = Math.sin(Math.PI * h);
      const cz = Math.cos(Math.PI * h) * (Math.PI / L);
      // rho v = (d psi / dz, -d psi / dx) in the vertical plane through x (and y in 3D).
      const across = ndim === 2 ? Math.cos(k * x) : Math.cos(k * x) * Math.cos(k * y);
      const dxacross = ndim === 2 ? -k * Math.sin(k * x) : -k * Math.sin(k * x) * Math.cos(k * y);
      const vh = (amp * cz * across) / rho;
      const vz = (-amp * sz * dxacross) / rho;
      return ndim === 2
        ? { rho, vx: vh, vy: vz, vz: 0, p, s: h < 0.5 ? 1 : 0 }
        : { rho, vx: vh, vy: 0, vz, p, s: h < 0.5 ? 1 : 0 };
    };
  },
};

export const flows = [convection];
