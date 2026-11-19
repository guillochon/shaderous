// Hydrogen and helium in chemical equilibrium among H2, H, H+, He, He+, He++ and electrons:
// the free energy the tabulated equation of state (eos/hhe.ts, eos/hhe.wgsl) is made from, in
// float64. tools/eos/make_hhe_table.mjs puts it on a grid; the tests check the table against
// it. This file imports nothing, so that Node runs it as it stands.
//
// An ideal mixture. Per atomic mass unit of gas and over k T, with y_s the number of species s
// per atomic mass unit and n_s = rho y_s / m_u its number density,
//   h = F m_u / (k T) = sum_s y_s [ ln(n_s / (nQ_s z_s)) - 1 + eps_s / (k T) + b_s phi(rho) ]
// nQ_s = (2 pi m_s k T / h^2)^(3/2), z_s the internal partition function, eps_s the energy of
// the species' ground state, and the last term the stand-in for pressure ionization (below).
// The species numbers are those that minimize h at fixed nuclei, which gives the dissociation
// and Saha equations. Since h is at a minimum, its first derivatives need none of the
// composition's:
//   p / rho = (k T / m_u) dh/dln rho = (k T / m_u) sum_s y_s (1 + b_s phi)
//   e       = -(k T / m_u) dh/dln T  = (k T / m_u) sum_s y_s (3/2 + dln z_s/dln T + eps_s / (k T))
//
// The model is that of Tomida et al. (2013, ApJ 763, 6, appendix) and Vaytet et al. (2013),
// with their constants, and these differences:
//   - Masses are nucleon numbers times m_u (H 1, He 4), the convention of units/composition.ts,
//     and whatever is not hydrogen is counted as helium.
//   - Energies are counted from the ground states of H2 and He, so every one is positive and
//     the energy goes to zero with the temperature: eps = D/2 per H atom, D/2 + chi_H per
//     proton, chi_1 and chi_1 + chi_2 for He+ and He++. The vibration has no zero-point term,
//     and D is the dissociation energy from the ground state.
//   - Statistical weights are the textbook ones: nuclear spin 2 for each proton, electron spin
//     2 for H, 1 for H2's singlet ground state. With ortho and para hydrogen held at 3:1 the
//     rotation is z_rot = 4 z_para^(1/4) z_ortho^(3/4), the two forms' sums over even and odd
//     j (the odd one counted from j = 1), which includes the entropy of mixing them and goes
//     over to the equilibrium sum when hot. Tomida et al. have 2 x 3^(3/4) where this has 1,
//     which weights H2 4.6 times more: their dissociation sits about 6% hotter.
//   - Pressure ionization. Bare Saha equations recombine the gas at stellar-interior densities
//     (and turn it molecular above 0.1 g/cm^3, as Tomida et al. note). Until a calibrated
//     model replaces it (plans/hydrogen-helium-eos.md, step 5), each bound electron costs
//     k T phi, phi = rho / RHO_PI: the excess chemical potential of hard spheres of a Bohr
//     radius at low packing, 8 eta. It is part of h, so pressure and energy stay consistent;
//     it leaves the gas alone below about 0.01 g/cm^3 and ionizes it above about 1.
// Not in it: electron degeneracy (it matters above about 100 g/cm^3 at 1e7 K), Coulomb terms,
// excited states of the atoms, metals, dust.

/** cgs */
export const HHE = {
  k: 1.380649e-16,
  mu: 1.6605390666e-24,
  me: 9.1093837015e-28,
  h: 6.62607015e-27,
  arad: 7.565723e-15,
  /** H2: rotational temperature (the levels are j (j + 1) thetaRot), vibrational temperature, dissociation energy over k */
  thetaRot: 85.32,
  thetaVib: 5984.48,
  tDis: (4.47807 * 1.602176634e-12) / 1.380649e-16,
  /** ionization energies over k: H, He, He+ */
  tH: (13.598434 * 1.602176634e-12) / 1.380649e-16,
  tHe1: (24.587387 * 1.602176634e-12) / 1.380649e-16,
  tHe2: (54.41776 * 1.602176634e-12) / 1.380649e-16,
  /** the density scale of pressure ionization, g/cm^3: m_u / (8 x 4 pi a0^3 / 3) */
  rhoPi: 0.334,
} as const;

export interface HheOptions {
  /** hydrogen's mass fraction; the rest is helium */
  X: number;
  /** ortho and para hydrogen in equilibrium at every temperature, instead of fixed at 3:1 */
  orthoParaEquilibrium?: boolean;
  /** the density scale of pressure ionization (HHE.rhoPi); Infinity for none */
  rhoPi?: number;
}

/** The species, in the order of HheState.y. */
export const HHE_SPECIES = ['H2', 'H', 'H+', 'He', 'He+', 'He++', 'e'] as const;
/** bound electrons of each */
const BOUND = [2, 1, 0, 2, 1, 0, 0];

export interface HheState {
  /** species per atomic mass unit, in the order of HHE_SPECIES */
  y: number[];
  /** h = F m_u / (k T), per atomic mass unit */
  h: number;
  /** dh/dln rho at fixed T: p m_u / (rho k T) */
  hn: number;
  /** dh/dln T at fixed rho: -e m_u / (k T) */
  ht: number;
  /** p / rho and e, erg/g */
  pr: number;
  e: number;
}

const J_MAX = 200;

/** ln of H2's rotational partition function (nuclear spin included) and its slope in ln T. */
export function h2Rotation(T: number, equilibrium = false): [number, number] {
  const a = HHE.thetaRot / T;
  // Sums of (2j + 1) exp(-(j(j+1) - j0(j0+1)) a) over even and odd j, and of the exponents times them.
  const s = [0, 0];
  const d = [0, 0];
  for (let j = 0; j <= J_MAX; j++) {
    const par = j & 1;
    const x = (j * (j + 1) - (equilibrium ? 0 : 2 * par)) * a;
    if (x > 745) break;
    const w = (2 * j + 1) * Math.exp(-x);
    s[par] += w;
    d[par] += w * x;
  }
  if (equilibrium) {
    const z = s[0] + 3 * s[1];
    return [Math.log(z), (d[0] + 3 * d[1]) / z];
  }
  return [Math.log(4) + 0.25 * Math.log(s[0]) + 0.75 * Math.log(s[1]), 0.25 * (d[0] / s[0]) + 0.75 * (d[1] / s[1])];
}

/** ln of H2's vibrational partition function, counted from the ground state, and its slope in ln T. */
export function h2Vibration(T: number): [number, number] {
  const a = HHE.thetaVib / T;
  const q = Math.exp(-a);
  return [-Math.log1p(-q), (a * q) / (1 - q)];
}

const softplus = (x: number) => (x > 36 ? x : Math.log1p(Math.exp(x)));
/** ln(exp(a) + exp(b) + exp(c)) */
const lse3 = (a: number, b: number, c: number) => {
  const m = Math.max(a, b, c);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m) + Math.exp(c - m));
};

/** What depends on the temperature alone: ln nQ z of each species, the energies over k T, and u_s. */
function atT(T: number, o: HheOptions) {
  const lnq = (m: number) => 1.5 * Math.log((2 * Math.PI * m * HHE.k * T) / HHE.h ** 2);
  const [lr, dr] = h2Rotation(T, o.orthoParaEquilibrium);
  const [lv, dv] = h2Vibration(T);
  const { mu, me } = HHE;
  const L2 = Math.LN2;
  // ln(nQ_s z_s)
  const lz = [lnq(2 * mu) + lr + lv, lnq(mu) + 2 * L2, lnq(mu) + L2, lnq(4 * mu), lnq(4 * mu) + L2, lnq(4 * mu), lnq(me) + L2];
  const eps = [0, 0.5 * HHE.tDis, 0.5 * HHE.tDis + HHE.tH, 0, HHE.tHe1, HHE.tHe1 + HHE.tHe2, 0].map((t) => t / T);
  // 3/2 + dln z/dln T + eps / (k T)
  const u = eps.map((x, s) => 1.5 + x + (s === 0 ? dr + dv : 0));
  return { lz, eps, u };
}

/** The equilibrium at density rho (g/cm^3) and temperature T (K). */
export function hheState(rho: number, T: number, o: HheOptions): HheState {
  const { lz, eps, u } = atT(T, o);
  const phi = rho / (o.rhoPi ?? HHE.rhoPi);
  const lnNu = Math.log(rho / HHE.mu);
  const aH = o.X;
  const aHe = (1 - o.X) / 4;
  const lnNH = aH > 0 ? Math.log(aH) + lnNu : -Infinity;
  const lnNHe = aHe > 0 ? Math.log(aHe) + lnNu : -Infinity;
  // ln of each species' n / activity at unit activity: n_s = exp(a_s + mu_s / k T).
  const a = lz.map((l, s) => l - eps[s] - BOUND[s] * phi);
  // n_H^2 / n_H2, n_p n_e / n_H, n_He+ n_e / n_He, n_He++ n_e / n_He+
  const lnKD = 2 * a[1] - a[0];
  const lnKH = a[2] + a[6] - a[1];
  const lnK1 = a[4] + a[6] - a[3];
  const lnK2 = a[5] + a[6] - a[4];
  // Everything in logarithms: at 10 K the electron density is exp(-8000) of the gas's.
  const species = (lne: number): number[] => {
    const ln = new Array<number>(7).fill(-Infinity);
    ln[6] = lne;
    if (aH > 0) {
      const lr = lnKH - lne;
      const lu = -softplus(lr);                       // ln of 1 / (1 + n_p / n_H)
      const lq = Math.log(8) + lnNH + 2 * lu - lnKD;
      const root = lq > 700 ? 0.5 * lq : Math.log1p(Math.sqrt(1 + Math.exp(lq)));
      ln[1] = Math.log(2) + lnNH + lu - root;
      ln[2] = ln[1] + lr;
      ln[0] = 2 * ln[1] - lnKD;
    }
    if (aHe > 0) {
      const l1 = lnK1 - lne;
      const l2 = lnK2 - lne;
      ln[3] = lnNHe - lse3(0, l1, l1 + l2);
      ln[4] = ln[3] + l1;
      ln[5] = ln[4] + l2;
    }
    return ln;
  };
  // Charge neutrality, n_e = n_p + n_He+ + 2 n_He++: the right side falls as n_e rises.
  let lo = -3e4;
  let hi = lnNu + Math.log(aH + 2 * aHe);
  for (let it = 0; it < 90; it++) {
    const mid = 0.5 * (lo + hi);
    const ln = species(mid);
    if (lse3(ln[2], ln[4], ln[5] + Math.LN2) > mid) lo = mid;
    else hi = mid;
  }
  const ln = species(0.5 * (lo + hi));
  const y = ln.map((l) => Math.exp(l - lnNu));
  let h = 0;
  let hn = 0;
  let ht = 0;
  for (let s = 0; s < 7; s++) {
    if (!(y[s] > 0)) continue;
    h += y[s] * (ln[s] - lz[s] - 1 + eps[s] + BOUND[s] * phi);
    hn += y[s] * (1 + BOUND[s] * phi);
    ht -= y[s] * u[s];
  }
  const kt = (HHE.k * T) / HHE.mu;
  return { y, h, hn, ht, pr: kt * hn, e: -kt * ht };
}

/** Particles per atomic mass unit, 1 / mu. */
export const hheParticles = (s: HheState): number => s.y.reduce((t, x) => t + x, 0);

/**
 * Second derivatives of h by central differences of its exact first ones, and with them the
 * heat capacity, the adiabatic index and the sound speed (gas alone, no radiation).
 */
export function hheSecond(rho: number, T: number, o: HheOptions, step = 1e-4) {
  const s = hheState(rho, T, o);
  const tp = hheState(rho, T * Math.exp(step), o);
  const tm = hheState(rho, T * Math.exp(-step), o);
  const rp = hheState(rho * Math.exp(step), T, o);
  const rm = hheState(rho * Math.exp(-step), T, o);
  const hnt = (tp.hn - tm.hn) / (2 * step);
  const htt = (tp.ht - tm.ht) / (2 * step);
  const hnn = (rp.hn - rm.hn) / (2 * step);
  const kt = (HHE.k * T) / HHE.mu;
  const de = -kt * (s.ht + htt);       // de/dln T
  const dpr = kt * (s.hn + hnt);       // d(p / rho)/dln T
  const dprho = kt * (s.hn + hnn);     // dp/drho at fixed T
  const cs2 = dprho + (dpr * dpr) / de;
  return { ...s, hnn, hnt, htt, de, dpr, dprho, cs2, gamma1: cs2 / s.pr };
}
