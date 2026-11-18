// The radiation a run starts with (Prim.erad, Prim.frad): where a problem does not give it, E is
// a T^4 at the gas's temperature and F its diffusion value -c / (3 kappa_R rho) grad E, from
// the grid's own centered differences. A star built in balance with the gas-and-radiation
// equation of state then starts in balance (plan section 7): without F the first step would have
// no radiation force.
import { makeEos, type Eos } from '../eos/eos';
import { opacity } from '../opacity/opacity';
import { PERIODIC } from '../problems/boundaries';
import type { SimConfig } from '../sim';

/** Equations of state with a radiation term of their own, which the run with transport leaves out. */
const WITH_RADIATION = new Set(['gas_radiation', 'degenerate']);

/**
 * A cell's gas internal energy and temperature from a problem's (total) pressure: where the
 * equation of state has its own radiation term, the pressure is split between gas and radiation
 * at the temperature the full equation of state gives; otherwise it is all the gas's.
 * `gas` is the run's equation of state (without radiation), `moments` the cell's composition.
 */
export function splitPressure(c: SimConfig, gas: Eos, rho: number, p: number, moments?: readonly number[], zh1?: number): { eint: number; T: number } {
  if (WITH_RADIATION.has(c.eos) && c.radA > 0) {
    const full = makeEos(c.eos, { ...c.eosParams, arad: c.radA }, moments, zh1);
    const T = full.temperature(rho, full.eintFromP(rho, p));
    return { eint: gas.eintFromT(rho, T), T };
  }
  const eint = gas.eintFromP(rho, p);
  return { eint, T: gas.temperature(rho, eint) };
}

/**
 * Fill F (channels 1 to 3 of `R`, cells in x-fastest order) with the diffusion flux of E
 * (channel 0) where `given` is not set, from centered differences (one-sided at a face that is
 * not periodic), never past c E.
 */
export function diffusionFlux(c: SimConfig, dims: number[], dx: number, R: Float32Array, rho: Float32Array, T: Float32Array, given: Uint8Array) {
  const [nx, ny, nz] = dims;
  const at = (i: number, j: number, k: number) => ((k * ny + j) * nx + i) * 4;
  const n = [nx, ny, nz];
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const o = at(i, j, k);
        const q = o / 4;
        if (given[q]) continue;
        const kr = opacity(c.opacity, c.opacityParams, rho[q], T[q]).rosseland;
        const D = c.radC / (3 * Math.max(kr * rho[q], 1e-30));
        const F = [0, 0, 0];
        for (let a = 0; a < 3; a++) {
          if (n[a] === 1) continue;
          const idx = [i, j, k];
          const nb = (s: number) => {
            const x = [...idx];
            x[a] += s;
            if (x[a] < 0 || x[a] >= n[a]) {
              if (c.bcLo[a] !== PERIODIC) return null;
              x[a] = (x[a] + n[a]) % n[a];
            }
            return R[at(x[0], x[1], x[2])];
          };
          const lo = nb(-1);
          const hi = nb(1);
          const grad = lo !== null && hi !== null ? (hi - lo) / (2 * dx) : hi !== null ? (hi - R[o]) / dx : lo !== null ? (R[o] - lo) / dx : 0;
          F[a] = -D * grad;
        }
        const mag = Math.hypot(...F);
        const s = mag > c.radC * R[o] ? (c.radC * R[o]) / mag : 1;
        R[o + 1] = F[0] * s;
        R[o + 2] = F[1] * s;
        R[o + 3] = F[2] * s;
      }
}
