// Ideal gas whose hydrogen ionizes in Saha equilibrium; every other species keeps the free
// electrons the composition gives it (src/units/composition.ts). With x the ionized fraction
// of the hydrogen and T the code temperature,
//   p     = rho T N(x),  N(x) = h (1 + x) + o
//   e_int = rho (T N(x) / (gamma - 1) + h x T_i)
//   x^2 / (1 - x) = (rho_s / (h rho)) (T / T_i)^(3/2) exp(-T_i / T)
// h and o are the particles per atomic mass unit from hydrogen (its nuclei) and from the other
// species, both times the composition's own mean molecular weight mu_ref; T_i is 13.6 eV / k
// in code temperature, and rho_s = mu_ref m_u (2 pi m_e k T_i / h^2)^(3/2) in code density.
// Scaling by mu_ref keeps the code temperature what the other modules and the kelvin readout
// take it to be: T_K = mu_ref theta T (units/units.ts), so p = rho T wherever the hydrogen is
// as ionized as the composition says.
//
// The ionization energy is part of e_int, which is what makes the gas soften as it ionizes:
// Gamma_1 dips well below gamma in the ionization zone. T comes from e_int, or p, by Newton's
// method kept inside a bracket (the two values of x bound it), in a fixed number of steps.
//
// P.eos = (gamma, unused, unused, unused), P.eos2 = (h, o, T_i, rho_s).
const EOS_HAS_ENERGY = true;
// Where E - KE cannot be trusted, rebuild E from the advected e_int (hydro/dual_energy.wgsl).
const EOS_KEEP_TOTAL = false;

// (x, dx/dlnS) at (rho, T). Written so neither end loses precision: x = 2 / (1 + sqrt(1 + 4/S)),
// and S dx/dS = x (1 - x) / (2 - x). Beyond |ln S| = 80 the gas is taken as fully neutral or
// fully ionized (x is then within 1e-17 of that), with no dependence on rho or T.
fn ion_x(rho: f32, T: f32) -> vec2<f32> {
  if (!(P.eos2.x > 0.0) || !(T > 0.0)) { return vec2<f32>(0.0); }
  let lnS = log(P.eos2.w / (P.eos2.x * rho)) + 1.5 * log(T / P.eos2.z) - P.eos2.z / T;
  if (lnS < -80.0) { return vec2<f32>(0.0); }
  if (lnS > 80.0) { return vec2<f32>(1.0, 0.0); }
  let x = 2.0 / (1.0 + sqrt(1.0 + 4.0 * exp(-lnS)));
  return vec2<f32>(x, x * (1.0 - x) / (2.0 - x));
}

// dx/dT from dx/dlnS = g: d lnS / dT = (3/2 + T_i / T) / T. Zero wherever x is pinned, which
// also keeps T -> 0 finite.
fn ion_dxdT(g: f32, T: f32) -> f32 {
  if (!(g > 0.0)) { return 0.0; }
  return g * (1.5 + P.eos2.z / T) / T;
}

fn ion_n(x: f32) -> f32 {
  return P.eos2.x * (1.0 + x) + P.eos2.y;
}

// Code temperature at which e_int / rho (if `by_p` is false) or p / rho (if true) is `goal`.
// Both grow with T. Between the fully neutral and fully ionized answers; Newton's step is taken
// when it lands inside the bracket, the midpoint otherwise.
fn ion_solve(rho: f32, goal: f32, by_p: bool) -> f32 {
  let h = P.eos2.x;
  let o = P.eos2.y;
  let Ti = P.eos2.z;
  let gm1 = P.eos.x - 1.0;
  var lo: f32;
  var hi: f32;
  if (by_p) {
    lo = goal / (2.0 * h + o);
    hi = goal / (h + o);
  } else {
    lo = max(goal - h * Ti, 0.0) * gm1 / (2.0 * h + o);
    hi = goal * gm1 / (h + o);
  }
  var T = hi;
  for (var i = 0; i < 24; i++) {
    let xs = ion_x(rho, T);
    let n = ion_n(xs.x);
    let xT = ion_dxdT(xs.y, T);
    var f: f32;
    var df: f32;
    if (by_p) {
      f = T * n - goal;
      df = n + T * h * xT;
    } else {
      f = T * n / gm1 + h * xs.x * Ti - goal;
      df = n / gm1 + h * (T / gm1 + Ti) * xT;
    }
    if (f > 0.0) { hi = T; } else { lo = T; }
    var next = T - f / df;
    if (!(next > lo && next < hi)) { next = 0.5 * (lo + hi); }
    T = next;
  }
  return T;
}

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  let T = ion_solve(rho, eint / rho, false);
  return rho * T * ion_n(ion_x(rho, T).x);
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  // (dp/drho) at constant entropy, from p(rho, T) and u(rho, T) = e_int / rho:
  //   c^2 = p_rho + p_T dT/drho,  dT/drho = (p / rho^2 - u_rho) / u_T.
  let T = ion_solve(rho, p / rho, true);
  let xs = ion_x(rho, T);
  let h = P.eos2.x;
  let Ti = P.eos2.z;
  let gm1 = P.eos.x - 1.0;
  let n = ion_n(xs.x);
  let xT = ion_dxdT(xs.y, T);
  let w = T / gm1 + Ti;                        // what ionizing one more nucleus costs, per h
  let p_rho = T * (n - h * xs.y);
  let p_T = rho * (n + T * h * xT);
  let u_T = n / gm1 + h * w * xT;
  let dT = (T * n + h * xs.y * w) / (rho * u_T);
  return sqrt(max(p_rho + p_T * dT, 0.0));
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  return eos_eint_from_T(rho, ion_solve(rho, p / rho, true));
}
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  return ion_solve(rho, eint / rho, false);
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  let x = ion_x(rho, T).x;
  return rho * (T * ion_n(x) / (P.eos.x - 1.0) + P.eos2.x * x * P.eos2.z);
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl): that of the pressure floor.
fn eos_eint_floor(rho: f32) -> f32 {
  return eos_eint_from_p(rho, P.p_floor);
}
