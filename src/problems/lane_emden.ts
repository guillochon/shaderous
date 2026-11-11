// Lane-Emden polytropes: theta'' + (2 / xi) theta' + theta^n = 0, theta(0) = 1.

export interface Polytrope {
  n: number;
  xi1: number; // first zero of theta
  dtheta1: number; // theta'(xi1), negative
  /** theta at a fraction s = r / R of the radius (0 outside) */
  theta(s: number): number;
}

export function laneEmden(n: number, steps = 4000): Polytrope {
  const f = (xi: number, th: number, dth: number): [number, number] => [dth, -Math.pow(Math.max(th, 0), n) - (xi > 0 ? (2 / xi) * dth : 0)];
  // Start from the series solution to step off the singular point.
  const h = 1e-3;
  let xi = h;
  let th = 1 - (h * h) / 6 + (n * h ** 4) / 120;
  let dth = -h / 3 + (n * h ** 3) / 30;
  const xs = [0, xi];
  const ths = [1, th];
  const dx = 8 / steps;
  let prev = { xi, th, dth };
  while (th > 0 && xi < 20) {
    prev = { xi, th, dth };
    const [k1a, k1b] = f(xi, th, dth);
    const [k2a, k2b] = f(xi + dx / 2, th + (dx / 2) * k1a, dth + (dx / 2) * k1b);
    const [k3a, k3b] = f(xi + dx / 2, th + (dx / 2) * k2a, dth + (dx / 2) * k2b);
    const [k4a, k4b] = f(xi + dx, th + dx * k3a, dth + dx * k3b);
    th += (dx / 6) * (k1a + 2 * k2a + 2 * k3a + k4a);
    dth += (dx / 6) * (k1b + 2 * k2b + 2 * k3b + k4b);
    xi += dx;
    xs.push(xi);
    ths.push(th);
  }
  // Linear interpolation to the zero crossing.
  const w = prev.th / (prev.th - th);
  const xi1 = prev.xi + w * (xi - prev.xi);
  const dtheta1 = prev.dth + w * (dth - prev.dth);
  return {
    n,
    xi1,
    dtheta1,
    theta(s: number): number {
      if (s >= 1) return 0;
      const x = s * xi1;
      let i = x <= h ? 0 : Math.min(xs.length - 2, 1 + Math.floor((x - h) / dx));
      const t = (x - xs[i]) / (xs[i + 1] - xs[i]);
      return Math.max(0, ths[i] + t * (ths[i + 1] - ths[i]));
    },
  };
}

/** Central density and polytropic constant of a polytrope of mass M and radius R (p = K rho^(1 + 1/n)). */
export function polytropeScales(poly: Polytrope, M: number, R: number, G: number) {
  const rhoc = (M * poly.xi1) / (4 * Math.PI * R ** 3 * -poly.dtheta1);
  const K = (4 * Math.PI * G * R * R * Math.pow(rhoc, 1 - 1 / poly.n)) / ((poly.n + 1) * poly.xi1 ** 2);
  return { rhoc, K };
}

/** K in p = K rho^(1 + 1/n) of a polytrope of central density `rhoc` and radius R. */
export function polytropeK(poly: Polytrope, rhoc: number, R: number, G: number): number {
  return (4 * Math.PI * G * R * R * Math.pow(rhoc, 1 - 1 / poly.n)) / ((poly.n + 1) * poly.xi1 ** 2);
}

/** Mass of a polytrope of central density `rhoc` and radius R. */
export function polytropeMass(poly: Polytrope, rhoc: number, R: number): number {
  return (4 * Math.PI * R ** 3 * rhoc * -poly.dtheta1) / poly.xi1;
}

const solved = new Map<number, Polytrope>();
/** laneEmden(n), solved once per index. */
export function laneEmdenCached(n: number): Polytrope {
  let p = solved.get(n);
  if (!p) solved.set(n, (p = laneEmden(n)));
  return p;
}
