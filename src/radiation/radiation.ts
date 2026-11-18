// Gray radiation transport with the M1 closure (plans/radiation-m1.md, docs/numerics.md
// section 24): the settings, the constants the shaders take, and float64 versions of the
// closure, the reduced speed of light per cell and the local implicit exchange, which the 1D
// reference (reference/radiation.ts) and the tests share.
//
// Radiation energy E and flux F are evolved in the gas's frame. The hydro step carries them
// with the gas (radiation/advect.wgsl); a sweep, run N_r times per hydro step, transports them
// between cells and exchanges energy and momentum with the gas, implicitly and cell by cell
// (radiation/sweep.wgsl). The gas's side of the exchange collects in a texture of its own and
// is folded into the state once the sweeps are done (radiation/commit.wgsl).
import type { OpacityKind, OpacityParams } from '../opacity/opacity';
import type { EosKind } from '../eos/eos';
import type { SimConfig } from '../sim';

export type RadiationKind = 'off' | 'm1';

export const RADIATION_LABELS: Record<RadiationKind, string> = { off: 'Off', m1: 'M1 (gray)' };

/**
 * The sweep's Courant number: its step is RAD_CFL dx / (d chat). Linear reconstruction with one
 * forward Euler stage is stable to 0.5 (at 0.8 a streaming pulse gained energy sevenfold in the
 * reference, reference/radiation.ts).
 */
export const RAD_CFL = 0.5;

/** The face flux of the sweep: HLL with the M1 wave speeds, or global Lax-Friedrichs (speed c). */
export type RadSolver = 'hll' | 'glf';

/** Most sweeps per hydro step: past this the hydro step is shortened instead. */
export const MAX_RAD_SWEEPS = 16;

/**
 * A cell is opaque enough for the true speed of light at an optical depth tau_0 c / chat:
 * there diffusion, not streaming, limits the sweep's step (section 3.4 of the plan). With the
 * timestep's rates summed over the axes (hydro/rates.wgsl), the streaming rate is d chat and the
 * diffusion rate 2.5 d D / dx with D = r c / (3 kappa rho) (the conduction's, numerics.md
 * section 3): they are equal at tau_c = 3 r c / (2.5 chat), so tau_0 = 2.5 / 3.
 */
export const TAU_0 = 2.5 / 3;

/** The radiation settings of a run (SimConfig). */
export interface RadiationSettings {
  radiation: RadiationKind;
  opacity: OpacityKind;
  opacityParams: OpacityParams;
  /** The reduced speed of light chat, in code units (at most c). */
  radChat: number;
  /** The speed of light, in code units: from the problem's units, else 1 unless given. */
  radC: number;
  /** The radiation constant a, in code units: the equation of state's own, which it then leaves out. */
  radA: number;
  /** The reduced speed of light chosen per cell (true: section 3.4) or the same everywhere. */
  radLocalC: boolean;
  radSolver: RadSolver;
  /** Sweeps per hydro step (1 to MAX_RAD_SWEEPS); the hydro step is at most this many sweeps' worth. */
  radSweeps: number;
}

/** Levermore's Eddington factor chi(f) for the reduced flux f = |F| / (c E) (clamped to [0, 1]). */
export function eddington(f: number): number {
  const g = Math.min(Math.max(f, 0), 1);
  return (3 + 4 * g * g) / (5 + 2 * Math.sqrt(4 - 3 * g * g));
}

/**
 * The Eddington tensor D (3x3, row major) for a flux F and energy E: P = D E, with
 * D = (1 - chi)/2 I + (3 chi - 1)/2 n n, n = F / |F|.
 */
export function eddingtonTensor(E: number, F: readonly number[], c: number): number[] {
  const mag = Math.hypot(F[0], F[1] ?? 0, F[2] ?? 0);
  const f = E > 0 ? mag / (c * E) : 0;
  const chi = eddington(f);
  const iso = (1 - chi) / 2;
  const dir = (3 * chi - 1) / 2;
  const n = mag > 0 ? [F[0] / mag, (F[1] ?? 0) / mag, (F[2] ?? 0) / mag] : [0, 0, 0];
  const D: number[] = [];
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) D.push((a === b ? iso : 0) + dir * n[a] * n[b]);
  return D;
}

/**
 * The reduction factor r of the speed of light in a cell of optical depth tau_c = kappa_R rho dx:
 * r = min(1, (chat / c) max(1, tau_c / tau_0)), or chat / c everywhere without the per-cell rule.
 */
export function reduction(tauCell: number, chat: number, c: number, local: boolean): number {
  const base = Math.min(1, chat / c);
  if (!local) return base;
  return Math.min(1, base * Math.max(1, tauCell / TAU_0));
}

/**
 * The local implicit exchange of a cell over a sweep of length dt (plan section 3.3), after the
 * transport has made E* and F*: the new flux, gas momentum gained, temperature, radiation and gas
 * internal energies. `eintFromT` and `temperature` are the gas's equation of state (without
 * radiation), kP and kR the opacities per unit mass, r the cell's reduction factor.
 */
export interface Exchange {
  F: number[];
  dmom: number[];
  T: number;
  E: number;
  eint: number;
}

export function exchange(
  rho: number,
  eint: number,
  Es: number,
  Fs: readonly number[],
  kP: number,
  kR: number,
  r: number,
  c: number,
  a: number,
  dt: number,
  eintFromT: (rho: number, T: number) => number,
  temperature: (rho: number, eint: number) => number,
): Exchange {
  const thF = r * c * kR * rho * dt;
  const F = Fs.map((x) => x / (1 + thF));
  const dmom = Fs.map((x, i) => (x - F[i]) / (r * c * c));
  const th = r * c * kP * rho * dt;
  const A = th / (r * (1 + th));
  const C = eint + A * Es;
  const T = solveTemperature(rho, A * a, C, eintFromT, temperature);
  const aT4 = a * T ** 4;
  let E = (Es + th * aT4) / (1 + th);
  let e = eintFromT(rho, T);
  // The larger of the two takes what the root's rounding left over (as rad_exchange).
  if (e >= E / r) e = eint + (Es - E) / r;
  else E = Es + r * (eint - e);
  return { F, dmom, T, E, eint: e };
}

/**
 * The root T of eint(rho, T) + K T^4 = C, K >= 0: bracketed below the smaller of the two one-term
 * solutions, then Newton from that upper bound (the left side is increasing and convex in T for
 * the equations of state here, so Newton descends monotonically), as rad_temperature in
 * radiation/exchange.wgsl.
 */
export function solveTemperature(
  rho: number,
  K: number,
  C: number,
  eintFromT: (rho: number, T: number) => number,
  temperature: (rho: number, eint: number) => number,
  iters = 12,
): number {
  if (!(C > 0)) return 0;
  let T = temperature(rho, C);
  if (K > 0) T = Math.min(T, Math.sqrt(Math.sqrt(C / K)));
  const e0 = eintFromT(rho, 0);
  for (let i = 0; i < iters; i++) {
    const e = eintFromT(rho, T);
    // The heat capacity from the change over a small step (exact for a linear e(T)).
    const h = Math.max(T * 1e-3, 1e-30);
    const cv = (eintFromT(rho, T + h) - e) / h;
    const f = e + K * T ** 4 - C;
    const df = cv + 4 * K * T ** 3;
    if (!(df > 0)) break;
    const next = T - f / df;
    T = Math.max(next, 0);
    if (e0 >= C) break;
  }
  return T;
}

/** Radiation transport is on. */
export const radiationOn = (c: Pick<SimConfig, 'radiation'>): boolean => c.radiation === 'm1';

/** Equations of state radiation transport runs with: those with an energy, and no table of their own radiation. */
const RAD_EOS: readonly EosKind[] = ['ideal', 'gas_radiation', 'degenerate', 'ionizing'];

/**
 * Why a run with radiation cannot go as configured, or null. The first version: the uniform grid
 * in Cartesian coordinates, without a magnetic field (plans/radiation-m1.md, section 1).
 */
export function radiationUnsupported(c: SimConfig): string | null {
  if (!radiationOn(c)) return null;
  if (c.amrLevels > 0) return 'radiation transport is not available on the adaptive mesh yet';
  if (c.problem.mhd) return 'radiation transport with a magnetic field is not available yet';
  if (!RAD_EOS.includes(c.eos)) return `radiation transport needs an equation of state with an energy and no radiation table of its own (ideal, gas + radiation, degenerate, ionizing), not ${c.eos}`;
  if (c.problem.geometry === 'cylindrical' || c.problem.geometry === 'spherical') return 'radiation transport is Cartesian only for now';
  if (c.problem.cosmology || c.problem.shear) return 'radiation transport is not available in comoving coordinates or a shearing box';
  if (!(c.radC > 0) || !(c.radChat > 0)) return 'the speed of light and its reduced value must be positive';
  if (!(c.radA > 0)) return 'the radiation constant must be positive (the equation of state’s, or the problem’s units’)';
  return null;
}

/**
 * The equation-of-state constants a run's setting asks for: with radiation transport on, the
 * equation of state leaves its own radiation term out (its arad is 0) and E carries it; radA
 * keeps the constant either way. Off again, the equation of state gets it back.
 */
export function syncRadiationEos<C extends SimConfig>(c: C): C {
  const a = c.eosParams.arad > 0 ? c.eosParams.arad : c.radA;
  if (radiationOn(c)) return { ...c, radA: a, eosParams: { ...c.eosParams, arad: 0 } };
  return { ...c, radA: a, eosParams: { ...c.eosParams, arad: a } };
}

/** P.rad and P.rad2 (hydro/common.wgsl): (c, chat, a, flags), (sweeps, Courant number, tau_0, 0). */
export function packRadiation(c: SimConfig): number[] {
  if (!radiationOn(c)) return [0, 0, 0, 0, 0, 0, 0, 0];
  const flags = 1 + (c.radLocalC ? 2 : 0) + (c.radSolver === 'glf' ? 4 : 0);
  return [c.radC, Math.min(c.radChat, c.radC), c.radA, flags, radSweeps(c), RAD_CFL, TAU_0, 0];
}

/** Sweeps per hydro step: the setting, within 1 to MAX_RAD_SWEEPS. */
export const radSweeps = (c: Pick<SimConfig, 'radSweeps'>): number => Math.min(MAX_RAD_SWEEPS, Math.max(1, Math.round(c.radSweeps || 1)));
