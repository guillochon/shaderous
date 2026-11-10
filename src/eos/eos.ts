// Float64 mirrors of the WGSL equation-of-state modules, used to build initial conditions,
// by the CPU reference solver, and by the tests.

export type EosKind = 'ideal' | 'barotropic' | 'gas_radiation';

export interface EosParams {
  gamma: number;
  K: number; // barotropic constant (c_s^2 when gamma = 1)
  R: number; // gas constant in code units
  arad: number; // radiation constant in code units
}

export interface Eos {
  kind: EosKind;
  hasEnergy: boolean;
  params: EosParams;
  pressure(rho: number, eint: number): number;
  soundSpeed(rho: number, p: number): number;
  eintFromP(rho: number, p: number): number;
  temperature(rho: number, eint: number): number;
}

export const EOS_MODULES: Record<EosKind, string> = {
  ideal: 'eos/ideal.wgsl',
  barotropic: 'eos/barotropic.wgsl',
  gas_radiation: 'eos/gas_radiation.wgsl',
};

export const EOS_LABELS: Record<EosKind, string> = {
  ideal: 'Ideal gas',
  barotropic: 'Barotropic',
  gas_radiation: 'Gas + radiation',
};

export const defaultEosParams = (): EosParams => ({ gamma: 5 / 3, K: 1, R: 1, arad: 1 });

/** Root of A T^4 + B T = C; same iteration as eos_quartic in gas_radiation.wgsl. */
export function quartic(A: number, B: number, C: number, iters = 8): number {
  let T = C / B;
  if (A > 0) T = Math.min(T, Math.sqrt(Math.sqrt(C / A)));
  for (let i = 0; i < iters; i++) {
    const T3 = T * T * T;
    T = (3 * A * T3 * T + C) / (4 * A * T3 + B);
  }
  return T;
}

export function makeEos(kind: EosKind, params: EosParams): Eos {
  const { gamma: g, K, R, arad: a } = params;
  switch (kind) {
    case 'ideal':
      return {
        kind,
        hasEnergy: true,
        params,
        pressure: (_rho, eint) => (g - 1) * eint,
        soundSpeed: (rho, p) => Math.sqrt((g * p) / rho),
        eintFromP: (_rho, p) => p / (g - 1),
        temperature: (rho, eint) => ((g - 1) * eint) / rho,
      };
    case 'barotropic': {
      const pressure = (rho: number) => (g === 1 ? K * rho : K * Math.pow(rho, g));
      return {
        kind,
        hasEnergy: false,
        params,
        pressure,
        soundSpeed: (rho, p) => Math.sqrt((g * p) / rho),
        eintFromP: (_rho, p) => (g < 1.001 ? 1.5 * p : p / (g - 1)),
        temperature: (rho) => pressure(rho) / rho,
      };
    }
    case 'gas_radiation':
      return {
        kind,
        hasEnergy: true,
        params,
        pressure: (rho, eint) => {
          const T = quartic(a, (R * rho) / (g - 1), eint);
          return R * rho * T + (a * T ** 4) / 3;
        },
        soundSpeed: (rho, p) => {
          const T = quartic(a / 3, R * rho, p);
          const beta = Math.min(1, Math.max(0, (R * rho * T) / p));
          const k = 4 - 3 * beta;
          const gamma1 = beta + (k * k * (g - 1)) / (beta + 12 * (g - 1) * (1 - beta));
          return Math.sqrt((gamma1 * p) / rho);
        },
        eintFromP: (rho, p) => {
          const T = quartic(a / 3, R * rho, p);
          return (R * rho * T) / (g - 1) + a * T ** 4;
        },
        temperature: (rho, eint) => quartic(a, (R * rho) / (g - 1), eint),
      };
  }
}
