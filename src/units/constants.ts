// Physical constants and astronomical scales, in SI. CODATA 2018 and IAU 2015 nominal values.

export const SI = {
  G: 6.6743e-11, // m^3 kg^-1 s^-2
  kB: 1.380649e-23, // J K^-1
  mu: 1.6605390666e-27, // atomic mass unit, kg
  c: 299792458, // m s^-1
  arad: 7.565733250280007e-16, // radiation constant 4 sigma / c, J m^-3 K^-4
} as const;

export const ASTRO = {
  /** IAU nominal G M_sun; M_sun itself is only as good as G. */
  GMsun: 1.3271244e20, // m^3 s^-2
  Msun: 1.3271244e20 / SI.G, // kg
  Rsun: 6.957e8, // m
  au: 1.495978707e11, // m
  pc: 3.0856775814913673e16, // m
  yr: 3.15576e7, // Julian year, s
  km: 1e3,
} as const;
