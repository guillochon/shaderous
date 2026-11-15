// Magnetic fields inside stars (docs/mhd-plan.md, section 8), as vector potentials for
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
