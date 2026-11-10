// Optically thin cooling and heating of the atomic interstellar medium, the fit of Koyama &
// Inutsuka (2002) with the exponent as corrected by Vazquez-Semadeni et al. (2007):
//   de/dt = heat rho (1 - rho L(T_K))
//   L(T_K) = 1e7 exp(-118400 / (T_K + 1000)) + 0.014 sqrt(T_K) exp(-92 / T_K)
// with the density in hydrogen atoms per cm^3 and T_K the temperature in kelvin. Gas between
// about 300 K and 6000 K in thermal balance is unstable and separates into a cold and a warm
// phase at the same pressure.
// P.cool = (heat, kelvin per code unit of temperature, unused, temperature floor in code units).
// Float64 mirror: src/cooling/cooling.ts.
const COOLING = true;

fn cool_rate(rho: f32, T: f32) -> f32 {
  var lam = 0.0;
  if (T > P.cool.w && T > 0.0) {
    let tk = T * P.cool.y;
    lam = 1.0e7 * exp(-118400.0 / (tk + 1000.0)) + 0.014 * sqrt(tk) * exp(-92.0 / tk);
  }
  return P.cool.x * rho * (1.0 - rho * lam);
}
