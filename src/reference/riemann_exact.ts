// Exact solution of the 1D Riemann problem for an ideal gas (Toro, chapter 4).

export interface State1D {
  rho: number;
  u: number;
  p: number;
}

export function exactRiemann(L: State1D, R: State1D, gamma: number): (xi: number) => State1D {
  const g = gamma;
  const cL = Math.sqrt((g * L.p) / L.rho);
  const cR = Math.sqrt((g * R.p) / R.rho);

  // f_K(p) and its derivative
  const fK = (p: number, s: State1D, c: number): [number, number] => {
    if (p > s.p) {
      const A = 2 / ((g + 1) * s.rho);
      const B = ((g - 1) / (g + 1)) * s.p;
      const q = Math.sqrt(A / (p + B));
      return [(p - s.p) * q, q * (1 - (p - s.p) / (2 * (B + p)))];
    }
    const pr = p / s.p;
    return [((2 * c) / (g - 1)) * (Math.pow(pr, (g - 1) / (2 * g)) - 1), (1 / (s.rho * c)) * Math.pow(pr, -(g + 1) / (2 * g))];
  };

  // Newton iteration for p*, started from the two-rarefaction estimate.
  const z = (g - 1) / (2 * g);
  let p = Math.pow((cL + cR - 0.5 * (g - 1) * (R.u - L.u)) / (cL / Math.pow(L.p, z) + cR / Math.pow(R.p, z)), 1 / z);
  p = Math.max(p, 1e-12);
  for (let i = 0; i < 100; i++) {
    const [fl, dl] = fK(p, L, cL);
    const [fr, dr] = fK(p, R, cR);
    const pn = Math.max(p - (fl + fr + R.u - L.u) / (dl + dr), 1e-14);
    const change = (2 * Math.abs(pn - p)) / (pn + p);
    p = pn;
    if (change < 1e-14) break;
  }
  const ps = p;
  const us = 0.5 * (L.u + R.u) + 0.5 * (fK(ps, R, cR)[0] - fK(ps, L, cL)[0]);
  const gm = (g - 1) / (g + 1);

  return (xi: number): State1D => {
    if (xi <= us) {
      // left of the contact
      if (ps > L.p) {
        const sL = L.u - cL * Math.sqrt(((g + 1) / (2 * g)) * (ps / L.p) + (g - 1) / (2 * g));
        if (xi <= sL) return L;
        return { rho: (L.rho * (ps / L.p + gm)) / (gm * (ps / L.p) + 1), u: us, p: ps };
      }
      const head = L.u - cL;
      const cs = cL * Math.pow(ps / L.p, z);
      const tail = us - cs;
      if (xi <= head) return L;
      if (xi >= tail) return { rho: L.rho * Math.pow(ps / L.p, 1 / g), u: us, p: ps };
      const c = (2 / (g + 1)) * (cL + 0.5 * (g - 1) * (L.u - xi));
      return { rho: L.rho * Math.pow(c / cL, 2 / (g - 1)), u: (2 / (g + 1)) * (cL + 0.5 * (g - 1) * L.u + xi), p: L.p * Math.pow(c / cL, (2 * g) / (g - 1)) };
    }
    if (ps > R.p) {
      const sR = R.u + cR * Math.sqrt(((g + 1) / (2 * g)) * (ps / R.p) + (g - 1) / (2 * g));
      if (xi >= sR) return R;
      return { rho: (R.rho * (ps / R.p + gm)) / (gm * (ps / R.p) + 1), u: us, p: ps };
    }
    const head = R.u + cR;
    const cs = cR * Math.pow(ps / R.p, z);
    const tail = us + cs;
    if (xi >= head) return R;
    if (xi <= tail) return { rho: R.rho * Math.pow(ps / R.p, 1 / g), u: us, p: ps };
    const c = (2 / (g + 1)) * (cR - 0.5 * (g - 1) * (R.u - xi));
    return { rho: R.rho * Math.pow(c / cR, 2 / (g - 1)), u: (2 / (g + 1)) * (-cR + 0.5 * (g - 1) * R.u + xi), p: R.p * Math.pow(c / cR, (2 * g) / (g - 1)) };
  };
}
