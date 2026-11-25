// The equations the query readout shows for a cell, with the size of each term there.
//
// Each term is the rate it changes the cell's conserved state at (per unit volume per unit
// time), estimated from the cell and its neighbors (Simulation.probe with a stencil): the flux
// divergences, gradients and curls by centered differences of the cell-centered state (one
// sided where a neighbor is missing), the source terms from the cell itself. This is the
// equations' content, not the scheme's: the solver's own fluxes come from a Riemann solver on
// reconstructed faces, and near a shock or at a sharp contact the two can differ a lot. Each
// term's importance is its size against the largest term of its equation; a vector equation's
// terms are compared by their magnitudes.
//
// Magnetic fields are in units where the magnetic pressure is B^2 / 2 (mhd/common.wgsl), and the
// total energy E then includes B^2 / 2. Radiation's coupling to the gas is that of the M1 exchange
// (radiation/exchange.wgsl), whose reduced speed of light cancels from what the gas gains.
import type { CellStencil } from '../sim';
import type { Eos } from '../eos/eos';
import { coolRate, type CoolingKind, type CoolingParams } from '../cooling/cooling';

export interface TermMass {
  /** G M (code units) */
  gm: number;
  /** position, relative to the box center */
  pos: number[];
  /** Plummer softening length */
  soft: number;
}

export interface TermRadiation {
  /** the speed of light and the radiation constant (code units) */
  c: number;
  a: number;
  /** Planck and Rosseland mean opacities per unit mass at (rho, T) */
  opacity: (rho: number, T: number) => { planck: number; rosseland: number };
}

export interface TermInput {
  ndim: 2 | 3;
  dx: number;
  stencil: CellStencil;
  /** cell center, relative to the box center */
  pos: number[];
  eos: Eos;
  /** uniform external acceleration */
  gext: number[];
  /** the point masses that pull on the gas */
  masses: TermMass[];
  cooling: CoolingKind;
  coolParams: CoolingParams;
  /** start-up damping's rate now (0 when it is off or over) */
  calm: number;
  /** MHD: the magnetic diffusivity (0 for ideal MHD); the field itself is the stencil's b */
  resistivity?: number;
  /** radiation transport's constants; the radiation itself is the stencil's rad */
  radiation?: TermRadiation | null;
}

export interface Term {
  id: string;
  /** the term as MathML (the content of an mrow), with its sign */
  math: string;
  /** what it is, in words */
  label: string;
  /** its value: a scalar's rate, or a vector's components */
  value: number[];
  /** its size: |value| */
  size: number;
}

export interface Equation {
  id: 'mass' | 'momentum' | 'energy' | 'induction';
  /** left-hand side as MathML */
  lhs: string;
  terms: Term[];
  /** the sum of the terms (vector for momentum and induction) and its size */
  net: number[];
  netSize: number;
  /** the largest term's size */
  largest: number;
}

const mi = (s: string) => `<mi>${s}</mi>`;
const mo = (s: string) => `<mo>${s}</mo>`;
const mn = (s: string) => `<mn>${s}</mn>`;
const bold = (s: string) => `<mi mathvariant="bold">${s}</mi>`;
const sub = (a: string, b: string) => `<msub>${a}${b}</msub>`;
const sup = (a: string, b: string) => `<msup>${a}${b}</msup>`;
const frac = (a: string, b: string) => `<mfrac>${a}${b}</mfrac>`;
const row = (...a: string[]) => `<mrow>${a.join('')}</mrow>`;
const paren = (...a: string[]) => row(mo('('), ...a, mo(')'));
const brack = (...a: string[]) => row(mo('['), ...a, mo(']'));
const DIV = row(mo('∇'), mo('·'));
const CURL = row(mo('∇'), mo('×'));
const V = bold('v');
const B = bold('B');
const RHO = mi('ρ');
const dt = (q: string) => row(frac(row(mo('∂'), q), row(mo('∂'), mi('t'))));

interface Prim {
  rho: number;
  v: [number, number, number];
  p: number;
  E: number;
  /** cell-centered field (zero without MHD), and whether it is known */
  b: [number, number, number];
  hasB: boolean;
}

function prims(s: CellStencil, eos: Eos): (Prim | null)[] {
  return s.u1.map((u1, i) => {
    const u2 = s.u2[i];
    if (!u1 || !u2 || u2[3] > 0.5 || !(u1[0] > 0)) return null;
    const rho = u1[0];
    const b = s.b?.[i];
    return { rho, v: [u1[1] / rho, u1[2] / rho, u1[3] / rho], p: eos.pressure(rho, u2[1]), E: u2[0], b: b ? [b[0], b[1], b[2]] : [0, 0, 0], hasB: !!b };
  });
}

/**
 * The state is float32: a difference below the round-off of what it subtracts is noise, and
 * counts as zero. Otherwise a term that is zero in exact arithmetic (a uniform flow's flux
 * divergence, radiation in equilibrium with the gas) can be the largest of its equation.
 */
const ROUNDOFF = 1e-5;
export const diff = (a: number, b: number): number => (Math.abs(a - b) <= ROUNDOFF * (Math.abs(a) + Math.abs(b)) ? 0 : a - b);

const slot = (o: number[]) => o[0] + 1 + 3 * (o[1] + 1) + 9 * (o[2] + 1);

/**
 * d f / dx along axis a at the center: centered, one-sided where a neighbor is missing, else 0.
 * Cells for which `usable` is false count as missing.
 */
function deriv(c: (Prim | null)[], f: (q: Prim) => number, a: number, dx: number, usable: (q: Prim) => boolean = () => true): number {
  const lo = [0, 0, 0];
  const hi = [0, 0, 0];
  lo[a] = -1;
  hi[a] = 1;
  const pick = (q: Prim | null) => (q && usable(q) ? q : null);
  const m = pick(c[slot(lo)]);
  const z = c[13]!;
  const p = pick(c[slot(hi)]);
  if (m && p) return diff(f(p), f(m)) / (2 * dx);
  if (p) return diff(f(p), f(z)) / dx;
  if (m) return diff(f(z), f(m)) / dx;
  return 0;
}

/** d^2 f / dx^2 along axis a at the center, where both neighbors are there; else 0. */
function deriv2(c: (Prim | null)[], f: (q: Prim) => number, a: number, dx: number, usable: (q: Prim) => boolean): number {
  const lo = [0, 0, 0];
  const hi = [0, 0, 0];
  lo[a] = -1;
  hi[a] = 1;
  const m = c[slot(lo)];
  const p = c[slot(hi)];
  if (!m || !p || !usable(m) || !usable(p)) return 0;
  const [fp, fz, fm] = [f(p), f(c[13]!), f(m)];
  const s = fp - 2 * fz + fm;
  return Math.abs(s) <= ROUNDOFF * (Math.abs(fp) + 2 * Math.abs(fz) + Math.abs(fm)) ? 0 : s / (dx * dx);
}

const norm = (v: number[]) => Math.hypot(...v);
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: number[], b: number[]): [number, number, number] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** The gravitational acceleration at the cell: self-gravity's (from the field), the point masses', the uniform field. */
export function gravityAt(x: TermInput): [number, number, number] {
  const g: [number, number, number] = [x.stencil.g[0] + x.gext[0], x.stencil.g[1] + x.gext[1], x.stencil.g[2] + x.gext[2]];
  for (const m of x.masses) {
    if (!(m.gm > 0)) continue;
    const d = [0, 1, 2].map((a) => (a < x.ndim ? x.pos[a] - m.pos[a] : 0));
    const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + m.soft * m.soft;
    if (!(r2 > 0)) continue;
    const k = -m.gm / (r2 * Math.sqrt(r2));
    for (let a = 0; a < 3; a++) g[a] += k * d[a];
  }
  if (x.ndim === 2) g[2] = 0;
  return g;
}

function equation(id: Equation['id'], lhs: string, all: (Term | null)[]): Equation {
  const terms = all.filter((t): t is Term => !!t);
  const n = terms[0].value.length;
  const net = Array.from({ length: n }, (_, i) => terms.reduce((s, t) => s + t.value[i], 0));
  return { id, lhs, terms, net, netSize: norm(net), largest: Math.max(0, ...terms.map((t) => t.size)) };
}

const term = (id: string, math: string, label: string, value: number[]): Term => ({ id, math, label, value, size: norm(value) });

/** The cell's equations and their terms; null for a solid cell or one without gas. */
export function cellEquations(x: TermInput): Equation[] | null {
  const c = prims(x.stencil, x.eos);
  const z = c[13];
  if (!z) return null;
  const axes = x.ndim;
  const mhd = !!x.stencil.b && z.hasB;
  const withB = (q: Prim) => q.hasB;
  // The divergence of a flux whose component along axis a is flux(q, a).
  const div = (flux: (q: Prim, a: number) => number, usable?: (q: Prim) => boolean) => {
    let s = 0;
    for (let a = 0; a < axes; a++) s += deriv(c, (q) => flux(q, a), a, x.dx, usable);
    return s;
  };
  // All three components, in 2D too: a 2D run may carry v_z and B_z, though nothing varies along z.
  const vec = (f: (a: number) => number) => [0, 1, 2].map(f);
  const g = gravityAt(x);
  const gOn = norm(g) > 0 || x.masses.some((m) => m.gm > 0);
  const u2 = x.stencil.u2[13]!;
  const out: Equation[] = [];

  // Radiation's coupling at the cell: the force on the gas and the heat it gains.
  let radForce: number[] | null = null;
  let radHeat = 0;
  const rad = x.radiation && x.stencil.rad ? x.stencil.rad : null;
  if (rad && x.radiation) {
    const T = x.eos.temperature(z.rho, u2[1]);
    const k = x.radiation.opacity(z.rho, T);
    radForce = [0, 1, 2].map((a) => (k.rosseland * z.rho * rad[1 + a]) / x.radiation!.c);
    radHeat = x.radiation.c * k.planck * z.rho * diff(rad[0], x.radiation.a * T ** 4);
  }

  out.push(equation('mass', dt(RHO), [term('mass-flux', row(mo('−'), DIV, paren(RHO, V)), 'mass carried in or out', [-div((q, a) => q.rho * q.v[a])])]));

  // The Lorentz force as the divergence of the magnetic stress, B B - (B^2 / 2) I.
  const lorentz = mhd ? [0, 1, 2].map((b) => div((q, a) => q.b[a] * q.b[b] - (a === b ? 0.5 * dot(q.b, q.b) : 0), withB)) : null;
  out.push(
    equation('momentum', dt(row(RHO, V)), [
      term('mom-flux', row(mo('−'), DIV, paren(RHO, V, V)), 'momentum carried in or out', vec((b) => -div((q, a) => q.rho * q.v[a] * q.v[b]))),
      term('pressure', row(mo('−'), mo('∇'), mi('P')), 'gas pressure gradient', vec((a) => (a < axes ? -deriv(c, (q) => q.p, a, x.dx) : 0))),
      lorentz ? term('lorentz', row(mo('+'), paren(CURL, B), mo('×'), B), 'magnetic force (tension and pressure)', lorentz) : null,
      gOn ? term('gravity', row(mo('+'), RHO, bold('g')), 'gravity', vec((a) => z.rho * g[a])) : null,
      radForce ? term('rad-force', row(mo('+'), frac(row(sub(mi('κ'), mi('R')), RHO), mi('c')), bold('F')), 'radiation pressure on the gas', radForce) : null,
      x.calm > 0 ? term('calm', row(mo('−'), sub(mi('k'), mi('d')), RHO, V), 'start-up damping', vec((a) => -x.calm * z.rho * z.v[a])) : null,
    ]),
  );

  if (x.eos.hasEnergy) {
    const heat = x.cooling !== 'off' ? coolRate(x.cooling, x.coolParams, z.rho, x.eos.temperature(z.rho, u2[1])) : 0;
    const v2 = dot(z.v, z.v);
    const ptot = mhd ? row(mi('P'), mo('+'), frac(sup(mi('B'), mn('2')), mn('2'))) : mi('P');
    out.push(
      equation('energy', dt(mi('E')), [
        term(
          'energy-flux',
          row(mo('−'), DIV, brack(paren(mi('E'), mo('+'), ptot), V)),
          mhd ? 'energy carried in or out, and pressure work (gas and magnetic)' : 'energy carried in or out, and pressure work',
          [-div((q, a) => (q.E + q.p + (mhd ? 0.5 * dot(q.b, q.b) : 0)) * q.v[a], mhd ? withB : undefined)],
        ),
        mhd ? term('mag-work', row(mo('+'), DIV, brack(paren(V, mo('·'), B), B)), 'work of magnetic tension', [div((q, a) => dot(q.v, q.b) * q.b[a], withB)]) : null,
        gOn ? term('grav-work', row(mo('+'), RHO, V, mo('·'), bold('g')), 'work done by gravity', [z.rho * dot(z.v, g)]) : null,
        x.cooling !== 'off' ? term('cooling', row(mo('+'), mi('Γ'), mo('−'), mi('Λ')), 'heating less cooling', [heat]) : null,
        x.stencil.heat !== undefined ? term('burn', row(mo('+'), RHO, sub(mi('ε'), mi('nuc'))), 'nuclear burning (last step)', [x.stencil.heat]) : null,
        radForce
          ? term('rad-heat', row(mo('+'), mi('c'), sub(mi('κ'), mi('P')), RHO, paren(sub(mi('E'), mi('r')), mo('−'), mi('a'), sup(mi('T'), mn('4')))), 'radiation absorbed less emitted', [radHeat])
          : null,
        radForce ? term('rad-work', row(mo('+'), V, mo('·'), frac(row(sub(mi('κ'), mi('R')), RHO), mi('c')), bold('F')), 'work of radiation pressure', [dot(z.v, radForce)]) : null,
        x.calm > 0 ? term('calm-e', row(mo('−'), sub(mi('k'), mi('d')), RHO, sup(mi('v'), mn('2'))), 'kinetic energy start-up damping removes', [-x.calm * z.rho * v2]) : null,
      ]),
    );
  }

  if (mhd) {
    // dB/dt = curl(v x B) + eta lap B; curl(v x B) from the cell-centered v x B.
    const e = (q: Prim) => cross(q.v, q.b);
    const d = (b: number, a: number) => (a < axes ? deriv(c, (q) => e(q)[b], a, x.dx, withB) : 0);
    const curl = [d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)];
    const eta = x.resistivity ?? 0;
    const lap = eta > 0 ? [0, 1, 2].map((b) => eta * [0, 1, 2].reduce((s, a) => s + (a < axes ? deriv2(c, (q) => q.b[b], a, x.dx, withB) : 0), 0)) : null;
    out.push(
      equation('induction', dt(B), [
        term('induction', row(mo('+'), CURL, paren(V, mo('×'), B)), 'field carried, stretched and compressed by the flow', curl),
        lap ? term('resistive', row(mo('+'), mi('η'), sup(mo('∇'), mn('2')), B), 'resistive diffusion', lap) : null,
      ]),
    );
  }
  return out;
}

/** Where a term's size sits on the importance ramp: 0 at `decades` below the largest or less, 1 for the largest. */
export function importance(size: number, largest: number, decades = 6): number {
  if (!(largest > 0) || !(size > 0)) return 0;
  return Math.min(1, Math.max(0, 1 + Math.log10(size / largest) / decades));
}

/** The ramp's color for an importance in [0, 1]: faint grey through cyan to warm yellow, for a dark panel. */
export function rampColor(s: number): string {
  const stops: [number, number, number][] = [
    [74, 81, 98],
    [51, 160, 220],
    [120, 220, 200],
    [255, 224, 102],
  ];
  const t = Math.min(1, Math.max(0, s)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const f = t - i;
  const col = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${col[0]}, ${col[1]}, ${col[2]})`;
}
