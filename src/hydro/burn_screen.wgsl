// Screening of the network's reaction rates (pynucastro's screen5: Graboske weak, Alastuey-Itoh
// strong), and logaddexp: shared by the network (hydro/burn_net_core.wgsl) and the watch for
// burning (hydro/burn_watch.wgsl). Needs SCREEN_CO2 and SCREEN_AA, which their generated slots
// define (burn/network.ts).

fn logaddexp(a: f32, b: f32) -> f32 {
  return max(a, b) + log(1.0 + exp(-abs(a - b)));
}

struct ScreenCommon {
  qlam0z: f32,
  taufac: f32,
  gamp: f32,
}

// What every pair's screening shares, from the gas's moles per gram ytot and the mean charge
// and mean squared charge of its ions.
fn screen_moments(T: f32, rho: f32, ytot: f32, zbar: f32, z2bar: f32) -> ScreenCommon {
  let rr = rho * ytot;
  var s: ScreenCommon;
  s.qlam0z = 1.88e8 / T * sqrt(rr / T * (z2bar + zbar));
  s.taufac = SCREEN_CO2 / pow(T, 1.0 / 3.0);
  s.gamp = SCREEN_AA / T * pow(rr * zbar, 1.0 / 3.0);
  return s;
}

// ln of the factor one pair's rate is multiplied by, from the pair's constants.
fn screen5(s: ScreenCommon, bb: f32, zs13: f32, zhat: f32, zhat2: f32, lzav: f32, aznut: f32) -> f32 {
  let fact = 1.25992105;
  var gamp = s.gamp;
  var gamef = fact * bb / zs13 * gamp;
  let tau12 = s.taufac * aznut;
  var alph12 = gamef / tau12;
  if (alph12 > 1.6) {
    alph12 = 1.6;
    gamef = 1.6 * tau12;
    gamp = gamef * zs13 / (fact * bb);
  }
  let h12w = bb * s.qlam0z;
  let cc = 0.896434 * gamp * zhat - 3.4474 * pow(gamp, 0.25) * zhat2 - 0.5551 * (log(gamp) + lzav) - 2.996;
  let a3 = alph12 * alph12 * alph12;
  let ss = tau12 * (5.0 / 32.0 - alph12 * (0.014 + 0.0128 * alph12));
  let vv = gamef * alph12 * (0.0055 + alph12 * (-0.0098 + 0.0048 * alph12));
  let h12 = cc - a3 * (ss + vv) + log(max(1.0 - 0.0562 * a3, 0.77));
  let gx = 0.3;
  let gs = 0.8;
  var h = h12w;
  if (gamef > gx) {
    h = select(h12, (h12w * (gs - gamef) + h12 * (gamef - gx)) / (gs - gx), gamef <= gs);
  }
  return max(h, 0.0);
}

