// Magnetic fields inside stars (docs/numerics.md section 21, Fields inside stars), as vector potentials for
// MhdSpec.potential: the face field is their discrete curl, with each edge's line integral (the
// block grid's summed over the finest level's segments), so it is divergence-free to rounding
// and coarse faces are the mean of the fine ones.
//
// Both are confined to the star by a taper, (p - p_cut) / (p_c - p_cut) where positive, from the
// star's own pressure profile (the preset's, not the settled one: any A gives a divergence-free
// field, and at the plasma betas these are meant for the field does not change the balance).

/** A star's pressure at radius r from its center (code units), and the cut below which there is no field. */
export interface StarProfile {
  pressure: (r: number) => number;
  /** the pressure at the center */
  pc: number;
  /** below this pressure the field is zero */
  pcut: number;
  /** the star's radius: the random field's largest scale */
  radius: number;
}

type Potential = (x: number, y: number, z: number) => [number, number, number];

/**
 * A dipole along z: A = A0 max(p - p_cut, 0) (-y, x, 0), A_phi = A0 f(r) varpi, whose curl at
 * the center is B_z = 2 A0 (p_c - p_cut). A0 makes the plasma beta there `beta`.
 */
export function dipoleField(star: StarProfile, beta: number): Potential {
  const bc = Math.sqrt((2 * star.pc) / beta);
  const a0 = bc / (2 * (star.pc - star.pcut));
  return (x, y, z) => {
    const f = Math.max(star.pressure(Math.hypot(x, y, z)) - star.pcut, 0);
    return [-a0 * f * y, a0 * f * x, 0];
  };
}

/** A small seeded generator (mulberry32), so a random field is the same on every load. */
function generator(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A random field with a Kolmogorov spectrum: A is a sum of `modes` Fourier modes, wavenumbers
 * spread evenly in log k from 2 pi / R to `kmaxPerR` times that, each with a random direction,
 * phase and polarization (perpendicular to k), amplitude ~ k^(-4/3). (|A_k| ~ k^(-17/6) per
 * unit volume of k-space, with the modes' density in log k, k^3, folded in: B = curl A then has
 * E(k) ~ k^(-5/3).) Tapered to the star, then scaled so that the mean B^2 within a fifth of the
 * radius of the center makes the plasma beta there `beta`.
 */
export function randomField(star: StarProfile, beta: number, seed = 1, modes = 64, kmaxPerR = 8): Potential {
  const rnd = generator(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd());
  const kmin = (2 * Math.PI) / star.radius;
  const ms: { k: number[]; a: number[]; phase: number }[] = [];
  for (let i = 0; i < modes; i++) {
    const kk = kmin * kmaxPerR ** ((i + rnd()) / modes);
    // A direction for k, uniform on the sphere.
    const cz = 2 * rnd() - 1;
    const ph = 2 * Math.PI * rnd();
    const sz = Math.sqrt(1 - cz * cz);
    const n = [sz * Math.cos(ph), sz * Math.sin(ph), cz];
    // A direction perpendicular to it.
    const t = Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    const u = [n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]];
    const ul = Math.hypot(u[0], u[1], u[2]);
    const amp = gauss() * kk ** (-4 / 3);
    ms.push({ k: n.map((c) => c * kk), a: u.map((c) => (amp * c) / ul), phase: 2 * Math.PI * rnd() });
  }
  const raw: Potential = (x, y, z) => {
    const f = Math.max(star.pressure(Math.hypot(x, y, z)) - star.pcut, 0) / (star.pc - star.pcut);
    if (f === 0) return [0, 0, 0];
    let ax = 0;
    let ay = 0;
    let az = 0;
    for (const m of ms) {
      const s = Math.sin(m.k[0] * x + m.k[1] * y + m.k[2] * z + m.phase);
      ax += m.a[0] * s;
      ay += m.a[1] * s;
      az += m.a[2] * s;
    }
    return [f * ax, f * ay, f * az];
  };
  // The mean B^2 near the center, by central differences of A at points within R / 5.
  const h = star.radius * 1e-3;
  const sample = generator(seed + 7919);
  let b2 = 0;
  const count = 200;
  for (let i = 0; i < count; i++) {
    let p: number[];
    do p = [0, 0, 0].map(() => (2 * sample() - 1) * 0.2 * star.radius);
    while (Math.hypot(p[0], p[1], p[2]) > 0.2 * star.radius);
    const d = (a: number, axis: number) => {
      const q1 = [...p];
      const q0 = [...p];
      q1[axis] += h;
      q0[axis] -= h;
      return (raw(q1[0], q1[1], q1[2])[a] - raw(q0[0], q0[1], q0[2])[a]) / (2 * h);
    };
    const bx = d(2, 1) - d(1, 2);
    const by = d(0, 2) - d(2, 0);
    const bz = d(1, 0) - d(0, 1);
    b2 += bx * bx + by * by + bz * bz;
  }
  b2 /= count;
  const scale = b2 > 0 ? Math.sqrt((2 * star.pc) / beta / b2) : 0;
  return (x, y, z) => {
    const a = raw(x, y, z);
    return [scale * a[0], scale * a[1], scale * a[2]];
  };
}

// ---------------------------------------------------------------------------------------------
// The fields of Guillochon & McCourt (2017), as their FLASH setup made them
// (Simulation_initBlock.F90 of MultiTidalPoly, and the notebook that wrote its mags.dat).
//
// Both are A = sqrt(8 pi p Az) times a shape, inside r < R - h and zero outside, with p the
// star's pressure there and h a hundredth of its diameter (their finest cell: 100 across the
// star). Here B^2 / 2 is the magnetic pressure, so sqrt(8 pi p Az) is sqrt(2 p Az). The paper's
// beta_M is 1 / Az: the field follows the square root of the pressure, so its plasma beta is
// about the same throughout the star, not set at the center alone (the dipole's is
// 1 / (4 Az) there).
//   dipole   shape (-y, x, 0): B_z = 2 sqrt(2 p Az) at the center.
//   random   shape the lattice M below, one value per hundredth of the diameter, read at each
//            point of the star. (Their code multiplied it by the cell size as well; the curl then
//            leaves the lattice's per-cell differences, a few hundredths, and a field 800 times
//            weaker than the paper's figure 1 shows. Without it the energy is that figure's.)

/** The cut and the lattice's resolution: a hundredth of the star's diameter. */
const GM17_CELLS = 100;

/** The GM17 dipole along z, of strength beta_M = 1 / Az (see above). */
export function gm17Dipole(star: StarProfile, betaM: number): Potential {
  const az = 1 / betaM;
  const rcut = star.radius * (1 - 2 / GM17_CELLS);
  return (x, y, z) => {
    if (Math.hypot(x, y, z) >= rcut) return [0, 0, 0];
    const f = Math.sqrt(2 * Math.max(star.pressure(Math.hypot(x, y, z)), 0) * az);
    return [-f * y, f * x, 0];
  };
}

/**
 * A complex FFT of length n in place, any n (mixed radix, recursive; n = 100 here). The sign
 * of the exponent and the normalization do not matter: only the magnitude is kept.
 */
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;
  let p = 2;
  while (n % p !== 0) p++;
  const m = n / p;
  const sr: Float64Array[] = [];
  const si: Float64Array[] = [];
  for (let q = 0; q < p; q++) {
    const a = new Float64Array(m);
    const b = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      a[j] = re[j * p + q];
      b[j] = im[j * p + q];
    }
    fft(a, b);
    sr.push(a);
    si.push(b);
  }
  for (let k = 0; k < n; k++) {
    let xr = 0;
    let xi = 0;
    for (let q = 0; q < p; q++) {
      const ang = (-2 * Math.PI * q * k) / n;
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      const ar = sr[q][k % m];
      const ai = si[q][k % m];
      xr += ar * c - ai * s;
      xi += ar * s + ai * c;
    }
    re[k] = xr;
    im[k] = xi;
  }
}

/**
 * GM17's random lattice, as their notebook made mags.dat (mkplot[100, 17/6]): for each of the
 * three components, amplitudes k^(-17/6) times a normal deviate with a uniform random phase on
 * an n^3 lattice of wavenumbers, k = |index - (n - 1) / 2| (never zero), transformed, its
 * magnitude taken, and scaled to a largest value of 1. Every value is positive (a mean near 0.43),
 * the spectrum is Kolmogorov's (A_k ~ k^(-17/6), B_k ~ k^(-11/6) per mode) from the lattice's
 * size down to two of its cells. (Their notebook shifts the lattice half a period along the first
 * axis only before transforming: a sign pattern in space, which the magnitude removes.)
 */
function gm17Lattice(seed: number, n = GM17_CELLS): Float32Array[] {
  const rnd = generator(seed);
  const out: Float32Array[] = [];
  const re = new Float64Array(n * n * n);
  const im = new Float64Array(n * n * n);
  const lr = new Float64Array(n);
  const li = new Float64Array(n);
  const c0 = (n - 1) / 2;
  for (let comp = 0; comp < 3; comp++) {
    for (let a = 0; a < n; a++)
      for (let b = 0; b < n; b++)
        for (let c = 0; c < n; c++) {
          const k = Math.hypot(a - c0, b - c0, c - c0);
          const g = Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-300))) * Math.cos(2 * Math.PI * rnd());
          const ph = 2 * Math.PI * rnd();
          const amp = k ** (-17 / 6) * g;
          const o = (a * n + b) * n + c;
          re[o] = amp * Math.cos(ph);
          im[o] = amp * Math.sin(ph);
        }
    // The 3D transform, one axis at a time.
    for (const [s0, s1, s2] of [[n * n, n, 1], [n, 1, n * n], [1, n * n, n]]) {
      for (let u = 0; u < n; u++)
        for (let v = 0; v < n; v++) {
          for (let w = 0; w < n; w++) {
            lr[w] = re[u * s0 + v * s1 + w * s2];
            li[w] = im[u * s0 + v * s1 + w * s2];
          }
          fft(lr, li);
          for (let w = 0; w < n; w++) {
            re[u * s0 + v * s1 + w * s2] = lr[w];
            im[u * s0 + v * s1 + w * s2] = li[w];
          }
        }
    }
    const mag = new Float32Array(n * n * n);
    let max = 0;
    for (let o = 0; o < mag.length; o++) {
      mag[o] = Math.hypot(re[o], im[o]);
      max = Math.max(max, mag[o]);
    }
    for (let o = 0; o < mag.length; o++) mag[o] /= max;
    out.push(mag);
  }
  return out;
}

const latticeCache = new Map<number, Float32Array[]>();

/**
 * The GM17 random field of strength beta_M = 1 / Az (see above): A = sqrt(2 p Az) M(x), the
 * lattice M spanning the star's diameter, a hundredth of it per lattice cell, read trilinearly
 * (their code took the nearest value, at a resolution where the two are the same). On a grid
 * coarser than the lattice, the edges' line integrals sample it, and the smallest scales are
 * lost (mhd-tidal-disruption measures how much).
 */
export function gm17Random(star: StarProfile, betaM: number, seed = 1): Potential {
  const az = 1 / betaM;
  const n = GM17_CELLS;
  const key = Math.round(seed);
  let lat = latticeCache.get(key);
  if (!lat) {
    lat = gm17Lattice(key, n);
    latticeCache.set(key, lat);
  }
  const L = lat;
  const h = (2 * star.radius) / n;
  const rcut = star.radius - h;
  // Lattice coordinates of a point: cell centers at (i - n/2 + 1/2) h.
  const at = (v: number) => Math.min(Math.max(v / h + n / 2 - 0.5, 0), n - 1);
  return (x, y, z) => {
    const r = Math.hypot(x, y, z);
    if (r >= rcut) return [0, 0, 0];
    const f = Math.sqrt(2 * Math.max(star.pressure(r), 0) * az);
    const u = [at(x), at(y), at(z)];
    const i0 = u.map((v) => Math.min(Math.floor(v), n - 2));
    const t = u.map((v, a) => v - i0[a]);
    const res: [number, number, number] = [0, 0, 0];
    for (let comp = 0; comp < 3; comp++) {
      const M = L[comp];
      let s = 0;
      for (let dz = 0; dz < 2; dz++)
        for (let dy = 0; dy < 2; dy++)
          for (let dxx = 0; dxx < 2; dxx++) {
            const w = (dxx ? t[0] : 1 - t[0]) * (dy ? t[1] : 1 - t[1]) * (dz ? t[2] : 1 - t[2]);
            s += w * M[((i0[0] + dxx) * n + (i0[1] + dy)) * n + (i0[2] + dz)];
          }
      res[comp] = f * s;
    }
    return res;
  };
}
