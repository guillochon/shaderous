// The equations the query readout shows for a cell, with the size of each term there.
//
// Each term is the rate it changes the cell's conserved state at (per unit volume per unit
// time), estimated from the cell and its neighbors (Simulation.probe with a stencil): the flux
// divergences and the pressure gradient by centered differences of the cell-centered state (one
// sided where a neighbor is missing), the source terms from the cell itself. This is the
// equations' content, not the scheme's: the solver's own fluxes come from a Riemann solver on
// reconstructed faces, and near a shock or at a sharp contact the two can differ a lot. Each
// term's importance is its size against the largest term of its equation; a vector equation's
// terms are compared by their magnitudes.
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
  id: 'mass' | 'momentum' | 'energy';
  /** left-hand side as MathML */
  lhs: string;
  terms: Term[];
  /** the sum of the terms (vector for momentum) and its size */
  net: number[];
  netSize: number;
  /** the largest term's size */
  largest: number;
}

const mi = (s: string) => `<mi>${s}</mi>`;
const mo = (s: string) => `<mo>${s}</mo>`;
const bold = (s: string) => `<mi mathvariant="bold">${s}</mi>`;
const sub = (a: string, b: string) => `<msub>${a}${b}</msub>`;
const sup = (a: string, b: string) => `<msup>${a}${b}</msup>`;
const row = (...a: string[]) => `<mrow>${a.join('')}</mrow>`;
const paren = (...a: string[]) => row(mo('('), ...a, mo(')'));
const DIV = row(mo('∇'), mo('·'));
const V = bold('v');
const RHO = mi('ρ');
const dt = (q: string) => row(`<mfrac>${row(mo('∂'), q)}${row(mo('∂'), mi('t'))}</mfrac>`);

interface Prim {
  rho: number;
  v: [number, number, number];
  p: number;
  E: number;
}

function prims(s: CellStencil, eos: Eos): (Prim | null)[] {
  return s.u1.map((u1, i) => {
    const u2 = s.u2[i];
    if (!u1 || !u2 || u2[3] > 0.5 || !(u1[0] > 0)) return null;
    const rho = u1[0];
    return { rho, v: [u1[1] / rho, u1[2] / rho, u1[3] / rho], p: eos.pressure(rho, u2[1]), E: u2[0] };
  });
}

const slot = (o: number[]) => o[0] + 1 + 3 * (o[1] + 1) + 9 * (o[2] + 1);

/** d f / dx along axis a at the center: centered, one-sided where a neighbor is missing, else 0. */
function deriv(c: (Prim | null)[], f: (q: Prim) => number, a: number, dx: number): number {
  const lo = [0, 0, 0];
  const hi = [0, 0, 0];
  lo[a] = -1;
  hi[a] = 1;
  const m = c[slot(lo)];
  const z = c[13]!;
  const p = c[slot(hi)];
  if (m && p) return (f(p) - f(m)) / (2 * dx);
  if (p) return (f(p) - f(z)) / dx;
  if (m) return (f(z) - f(m)) / dx;
  return 0;
}

const norm = (v: number[]) => Math.hypot(...v);

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
  const div = (flux: (q: Prim, a: number) => number) => {
    let s = 0;
    for (let a = 0; a < axes; a++) s += deriv(c, (q) => flux(q, a), a, x.dx);
    return s;
  };
  const vec = (f: (a: number) => number) => [0, 1, 2].map((a) => (a < axes ? f(a) : 0));
  const g = gravityAt(x);
  const gOn = norm(g) > 0 || x.masses.some((m) => m.gm > 0);
  const out: Equation[] = [];

  out.push(equation('mass', dt(RHO), [term('mass-flux', row(mo('−'), DIV, paren(RHO, V)), 'mass carried in or out', [-div((q, a) => q.rho * q.v[a])])]));

  out.push(
    equation('momentum', dt(row(RHO, V)), [
      term('mom-flux', row(mo('−'), DIV, paren(RHO, V, V)), 'momentum carried in or out', vec((b) => -div((q, a) => q.rho * q.v[a] * q.v[b]))),
      term('pressure', row(mo('−'), mo('∇'), mi('P')), 'pressure gradient', vec((a) => -deriv(c, (q) => q.p, a, x.dx))),
      gOn ? term('gravity', row(mo('+'), RHO, bold('g')), 'gravity', vec((a) => z.rho * g[a])) : null,
      x.calm > 0 ? term('calm', row(mo('−'), sub(mi('k'), mi('d')), RHO, V), 'start-up damping', vec((a) => -x.calm * z.rho * z.v[a])) : null,
    ]),
  );

  if (x.eos.hasEnergy) {
    const u2 = x.stencil.u2[13]!;
    const heat = x.cooling !== 'off' ? coolRate(x.cooling, x.coolParams, z.rho, x.eos.temperature(z.rho, u2[1])) : 0;
    const v2 = z.v[0] ** 2 + z.v[1] ** 2 + z.v[2] ** 2;
    out.push(
      equation('energy', dt(mi('E')), [
        term('energy-flux', row(mo('−'), DIV, paren(paren(mi('E'), mo('+'), mi('P')), V)), 'energy carried in or out, and pressure work', [-div((q, a) => (q.E + q.p) * q.v[a])]),
        gOn ? term('grav-work', row(mo('+'), RHO, V, mo('·'), bold('g')), 'work done by gravity', [z.rho * (z.v[0] * g[0] + z.v[1] * g[1] + z.v[2] * g[2])]) : null,
        x.cooling !== 'off' ? term('cooling', row(mo('+'), mi('Γ'), mo('−'), mi('Λ')), 'heating less cooling', [heat]) : null,
        x.calm > 0 ? term('calm-e', row(mo('−'), sub(mi('k'), mi('d')), RHO, sup(mi('v'), '<mn>2</mn>')), 'kinetic energy start-up damping removes', [-x.calm * z.rho * v2]) : null,
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
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
