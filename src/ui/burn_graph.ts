// The burning network as a graph, for the query readout (plans/burn-network-graph.md): the 13
// nuclei of the alpha chain as nodes, and the reactions between them as edges, each labelled with
// its rate in the cell read. The rates are the reference's (aprox13, at the cell's density,
// temperature and composition: burn/network.ts netReactionRates), the ones the trained network
// takes as inputs, not the step the network took. The readout's species equations show that.
//
// The pure part (reactionEdges, cellReactions) has no DOM and is unit-tested; BurnGraphView
// builds its SVG once and changes attributes in place on every read, as the readout's other parts
// do (ui/readout_view.ts), so that a hover survives the readout refreshing.
import { NET_REACTIONS } from '../burn/aprox13_reactions';
import { isNetworkBurn, type AnyBurnSpec } from '../burn/burn';
import { NET_DATA } from '../burn/aprox13_net';
import { NET_SPECIES, NET_T_MAX, netReactionRates, netUnits, watchesOnly, WATCH_SPECIES } from '../burn/network';
import type { Composition } from '../units/composition';
import type { UnitSystem } from '../units/units';
import { fmt3 } from './diag_plot';
import { importance, rampColor } from './terms';

/** A net change smaller than this share of the larger direction: the reaction is in equilibrium. */
export const EQUILIBRIUM = 0.01;
/** Below this mass fraction a species is drawn faint (and its reactions' rates are taken as 0). */
export const ABSENT = 1e-12;

/** One reaction, its forward and backward rates (mol/g/s) netted. */
export interface ReactionEdge {
  pair: number;
  /** the forward reaction's label (C12(a,g)O16), and the backward's */
  label: string;
  revLabel: string;
  fwd: number;
  rev: number;
  /** fwd - rev */
  signed: number;
  /** |fwd - rev| */
  net: number;
  /** the way the net flux goes: 1 forward, -1 backward, 0 neither */
  dir: 1 | -1 | 0;
  /** the net change is under EQUILIBRIUM of the larger direction */
  eq: boolean;
  /** energy released by one mole of the forward reaction, erg/mol */
  Q: number;
  /** energy released by the net reaction, erg/g/s */
  energy: number;
  /** the forward reaction's change in each species' count, by species index */
  nu: Map<number, number>;
}

/** Each species' change in count by a term, as a map. */
const nuOf = (t: number) => new Map(NET_REACTIONS[t].nu.map(([i, n]) => [i, n] as [number, number]));

/**
 * The 32 terms' rates (mol/g/s, NET_DATA.terms' order) paired into the network's 16 reactions,
 * and each species' net dY/dt (mol/g/s) from all of them. With X, a term that needs a species the
 * cell does not hold (X < ABSENT) is taken as 0, rather than the 1e-30-per-power floor of the rates.
 */
export function reactionEdges(rates: readonly number[], X?: readonly number[]): { edges: ReactionEdge[]; dYdt: number[] } {
  const r = rates.map((v, t) => (X && NET_REACTIONS[t].nu.some(([i, n]) => n < 0 && !(X[i] >= ABSENT)) ? 0 : v));
  const dYdt = new Array<number>(NET_SPECIES.length).fill(0);
  r.forEach((v, t) => NET_REACTIONS[t].nu.forEach(([i, n]) => (dYdt[i] += n * v)));
  const edges: ReactionEdge[] = [];
  NET_REACTIONS.forEach((re, t) => {
    if (re.dir !== 'fwd') return;
    const back = NET_REACTIONS.findIndex((o) => o.pair === re.pair && o.dir === 'rev');
    const fwd = r[t];
    const rev = back >= 0 ? r[back] : 0;
    const signed = fwd - rev;
    const net = Math.abs(signed);
    const Q = 1e18 * re.nu.reduce((q, [i, n]) => q + NET_DATA.q[i] * NET_DATA.A[i] * n, 0);
    const big = Math.max(fwd, rev);
    edges.push({
      pair: re.pair, label: re.label, revLabel: back >= 0 ? NET_REACTIONS[back].label : '', fwd, rev, signed, net,
      dir: signed > 0 ? 1 : signed < 0 ? -1 : 0, eq: big > 0 && net < EQUILIBRIUM * big, Q, energy: Q * signed, nu: nuOf(t),
    });
  });
  edges.sort((a, b) => a.pair - b.pair);
  return { edges, dYdt };
}

/** The reactions at a cell, as the graph shows them. */
export interface CellReactions {
  /** density (g/cm^3) and temperature (K) the rates are at */
  rho: number;
  T: number;
  /** the 13 mass fractions (the watched three padded with zeros) */
  X: number[];
  edges: ReactionEdge[];
  dYdt: number[];
  /** the burn pass works on this cell; if not, why not */
  burning: boolean;
  why: string;
  /** a run that carries he4, c12 and o16 only, and watches for burning rather than burning */
  watching: boolean;
}

/**
 * The reactions at a cell of a run that burns with the network, from its density, temperature
 * (code units, as the state holds them) and mass fractions (the run's species). null for a run
 * that does not burn with the network, or does not carry its species.
 */
export function cellReactions(
  spec: AnyBurnSpec | null,
  p: { species?: readonly string[]; units?: UnitSystem; composition?: Composition },
  rho: number,
  T: number,
  Xrun: readonly number[],
): CellReactions | null {
  if (!spec || !isNetworkBurn(spec) || !p.species || !p.units) return null;
  const watching = watchesOnly(p.species);
  const full = p.species.length === NET_SPECIES.length && p.species.every((k, i) => k === NET_SPECIES[i]);
  if (!watching && !full) return null;
  const X = NET_SPECIES.map((_, i) => (i < Xrun.length ? Math.max(Xrun[i] ?? 0, 0) : 0));
  if (watching) X.fill(0, WATCH_SPECIES.length);
  const u = netUnits(p.units, p.composition);
  const rhoCgs = rho * u.rho;
  const tK = Math.min(T * u.T, NET_T_MAX);
  const tMin = spec.tMin ?? 1e8;
  const rhoMin = spec.rhoMin ?? 1e4;
  const ok = rhoCgs > 0 && tK > 0;
  const { edges, dYdt } = ok ? reactionEdges(netReactionRates(Math.log10(rhoCgs), Math.log10(tK), X), X) : reactionEdges(new Array(NET_REACTIONS.length).fill(0));
  const why = watching ? 'watching: only He4, C12 and O16 are carried, nothing burns' : !(tK >= tMin) ? `not burning: T < ${fmt3(tMin)} K` : !(rhoCgs >= rhoMin) ? `not burning: ρ < ${fmt3(rhoMin)} g/cm³` : '';
  return { rho: rhoCgs, T: tK, X, edges, dYdt, burning: !why, why, watching };
}

// ---- The view ----

const SVG = 'http://www.w3.org/2000/svg';
const W = 304;
const H = 176;
const R = 12;
/** the column x of each of the 7 columns */
const col = (c: number) => 20 + c * 44;
const TOP = 76;
const BOTTOM = 142;
const MID = (TOP + BOTTOM) / 2;

type Pt = [number, number];

/** Where each species is drawn: he4 to the left, the chain snaking c12 .. s32 along the top and ar36 .. ni56 back along the bottom. */
const NODE_AT: Pt[] = [
  [col(0), MID],
  ...[1, 2, 3, 4, 5, 6].map((c) => [col(c), TOP] as Pt),
  ...[6, 5, 4, 3, 2, 1].map((c) => [col(c), BOTTOM] as Pt),
];
const NODE_NAME = NET_SPECIES.map((s) => s[0].toUpperCase() + s.slice(1));

/** One stroke of an edge: its path, and the direction an arrow at each end points (into the node there). */
interface Stroke {
  d: string;
  start: Pt;
  startDir: Pt;
  end: Pt;
  endDir: Pt;
}

interface EdgeShape {
  pair: number;
  /** the strokes: for a reaction of two heavy nuclei, each reactant's into the junction, then the junction's to the product */
  strokes: Stroke[];
  /** arrowheads: where they go for a net forward flux, and for a net backward one */
  fwdHeads: { at: Pt; dir: Pt }[];
  revHeads: { at: Pt; dir: Pt }[];
  label: { at: Pt; anchor: 'start' | 'middle' | 'end' };
  junction?: Pt;
}

const unit = (v: Pt): Pt => {
  const l = Math.hypot(v[0], v[1]) || 1;
  return [v[0] / l, v[1] / l];
};
const sub = (a: Pt, b: Pt): Pt => [a[0] - b[0], a[1] - b[1]];
const f1 = (v: number) => +v.toFixed(1);

/** A straight stroke from node a to node b, trimmed to their outlines. */
function line(a: number, b: number): Stroke {
  const [pa, pb] = [NODE_AT[a], NODE_AT[b]];
  const d = unit(sub(pb, pa));
  const s: Pt = [pa[0] + d[0] * (R + 1), pa[1] + d[1] * (R + 1)];
  const e: Pt = [pb[0] - d[0] * (R + 1), pb[1] - d[1] * (R + 1)];
  return { d: `M${f1(s[0])},${f1(s[1])}L${f1(e[0])},${f1(e[1])}`, start: s, startDir: [-d[0], -d[1]], end: e, endDir: d };
}

/** A quadratic stroke from s to e by control c. */
function quad(s: Pt, c: Pt, e: Pt): Stroke {
  return { d: `M${f1(s[0])},${f1(s[1])}Q${f1(c[0])},${f1(c[1])} ${f1(e[0])},${f1(e[1])}`, start: s, startDir: unit(sub(s, c)), end: e, endDir: unit(sub(e, c)) };
}

const topOf = (i: number): Pt => [NODE_AT[i][0], NODE_AT[i][1] - R - 1];

/** An arc over the top row from node a to node b, peaking h above the nodes. */
function arc(pair: number, a: number, b: number, h: number): EdgeShape {
  const [s, e] = [topOf(a), topOf(b)];
  const st = quad(s, [(s[0] + e[0]) / 2, s[1] - 2 * h], e);
  return { pair, strokes: [st], fwdHeads: [{ at: st.end, dir: st.endDir }], revHeads: [{ at: st.start, dir: st.startDir }], label: { at: [(s[0] + e[0]) / 2, s[1] - h - 3], anchor: 'middle' } };
}

/** A reaction of two heavy nuclei a and b to c: both into a junction h above the row at x = jx, and one stroke on to c. */
function hyper(pair: number, a: number, b: number, c: number, jx: number, h: number): EdgeShape {
  const J: Pt = [jx, TOP - R - 1 - h];
  const ins = [a, b].map((i) => quad(topOf(i), [NODE_AT[i][0], J[1]], J));
  const out = quad(J, [NODE_AT[c][0], J[1]], topOf(c));
  return {
    pair, strokes: [...ins, out], junction: J,
    fwdHeads: [{ at: out.end, dir: out.endDir }],
    revHeads: ins.map((s) => ({ at: s.start, dir: s.startDir })),
    label: { at: [(J[0] + NODE_AT[c][0]) / 2, J[1] - 3], anchor: 'middle' },
  };
}

/** A capture along the chain, from node a to node b: labelled below the top row, below the bottom row, or beside the turn. */
function chain(pair: number, a: number, b: number): EdgeShape {
  const st = line(a, b);
  const [pa, pb] = [NODE_AT[a], NODE_AT[b]];
  const label: EdgeShape['label'] =
    pa[0] === pb[0] ? { at: [pa[0] - R + 2, (pa[1] + pb[1]) / 2 + 3], anchor: 'end' }
      : pa[1] === TOP && pb[1] === TOP ? { at: [(pa[0] + pb[0]) / 2, TOP + R + 9], anchor: 'middle' }
        : pa[1] === BOTTOM && pb[1] === BOTTOM ? { at: [(pa[0] + pb[0]) / 2, BOTTOM + R + 11], anchor: 'middle' }
          : { at: [(pa[0] + pb[0]) / 2 - 6, (pa[1] + pb[1]) / 2 - 2], anchor: 'end' };
  return { pair, strokes: [st], fwdHeads: [{ at: st.end, dir: st.endDir }], revHeads: [{ at: st.start, dir: st.startDir }], label };
}

/** Every reaction's shape, by pair (aprox13_reactions.ts: the first eight, then the captures Mg24 .. Fe52). */
function shapes(): EdgeShape[] {
  const out: EdgeShape[] = [
    chain(0, 1, 2), // C12(a,g)O16
    chain(1, 2, 3), // O16(a,g)Ne20
    chain(2, 3, 4), // Ne20(a,g)Mg24
    arc(3, 1, 3, 14), // C12 + C12 -> Ne20 + a
    hyper(4, 1, 2, 4, col(2.4), 28), // C12 + O16 -> Mg24 + a
    arc(5, 2, 5, 42), // O16 + O16 -> Si28 + a
    hyper(6, 1, 3, 5, col(2.6), 56), // C12 + Ne20 -> Si28 + a
    chain(7, 0, 1), // 3a -> C12
  ];
  for (let k = 0; k < 8; k++) out.push(chain(8 + k, 4 + k, 5 + k));
  return out;
}

const fmtRate = (v: number) => (v === 0 ? '0' : v.toExponential(1).replace('e+', 'e').replace(/\.0e/, 'e'));
const ARROW = 6;
const arrowPath = (at: Pt, dir: Pt) => {
  const n: Pt = [-dir[1], dir[0]];
  const b: Pt = [at[0] - dir[0] * ARROW, at[1] - dir[1] * ARROW];
  return `M${f1(at[0])},${f1(at[1])}L${f1(b[0] + n[0] * 3.2)},${f1(b[1] + n[1] * 3.2)}L${f1(b[0] - n[0] * 3.2)},${f1(b[1] - n[1] * 3.2)}Z`;
};

/** What a graph read shows: the reactions, and the network's own change of each species (null where it does not burn). */
export interface GraphState {
  cell: CellReactions;
  /** the network's change of each species over a step (any units; only its sign and size against the others are drawn) */
  network: number[] | null;
}

/** The graph of the network's reactions, built once (on the first update) and changed in place. */
export class BurnGraphView {
  private svg: SVGSVGElement | null = null;
  private edges: { shape: EdgeShape; g: SVGGElement; strokes: SVGPathElement[]; heads: SVGPathElement[]; label: SVGTextElement; junction?: SVGCircleElement }[] = [];
  private nodes: { g: SVGGElement; circle: SVGCircleElement; mark: SVGCircleElement }[] = [];
  private caption!: HTMLElement;
  private detail!: HTMLElement;
  private last: GraphState | null = null;
  /** what the pointer is over: an edge's pair, or a node */
  private hover: { edge: number } | { node: number } | null = null;

  /** `open`: the graph starts expanded (else only its header shows); `toggled` hears each click on the header. */
  constructor(private root: HTMLElement, open = false, private toggled: (open: boolean) => void = () => {}) {
    root.classList.toggle('collapsed', !open);
  }

  /** The graph is expanded (not just its header). */
  get open(): boolean {
    return !this.root.classList.contains('collapsed');
  }

  /** The series a pinned readout records: each reaction's net rate (forward minus backward, mol/g/s), and each species' mass fraction. */
  series(): Map<string, { label: string; raw: number[]; scale: number; unit: string }> {
    const out = new Map<string, { label: string; raw: number[]; scale: number; unit: string }>();
    const cell = this.last?.cell;
    if (!cell) return out;
    for (const e of cell.edges) out.set(`r:${e.pair}`, { label: `r:${e.label}`, raw: [e.signed], scale: 1, unit: 'mol/g/s' });
    cell.X.forEach((x, i) => out.set(`x:${i}`, { label: `X:${NODE_NAME[i]}`, raw: [x], scale: 1, unit: '' }));
    return out;
  }

  /** Show the graph for a read, or hide it (null). */
  update(state: GraphState | null) {
    this.last = state;
    this.root.hidden = !state;
    if (!state) return;
    if (!this.svg) this.build();
    const { cell, network } = state;
    this.svg!.classList.toggle('cold', !cell.burning);
    const largest = Math.max(0, ...cell.edges.map((e) => e.net));
    cell.edges.forEach((e, k) => {
      const v = this.edges[k];
      const imp = importance(e.net, largest);
      const color = rampColor(imp);
      const width = String(f1(0.8 + 2.4 * imp));
      for (const s of v.strokes) {
        s.setAttribute('stroke', color);
        s.setAttribute('stroke-width', width);
        s.classList.toggle('eq', e.eq);
      }
      const heads = e.eq || e.dir === 0 ? [] : e.dir > 0 ? v.shape.fwdHeads : v.shape.revHeads;
      v.heads.forEach((h, i) => {
        const at = heads[i];
        h.setAttribute('visibility', at ? 'visible' : 'hidden');
        if (at) {
          h.setAttribute('d', arrowPath(at.at, at.dir));
          h.setAttribute('fill', color);
        }
      });
      if (v.junction) v.junction.setAttribute('fill', color);
      // A reaction with no flux at all (a species the cell does not hold) goes unlabelled.
      const text = e.eq ? '≈ eq' : e.net > 0 ? fmtRate(e.net) : '';
      if (v.label.textContent !== text) v.label.textContent = text;
      v.label.setAttribute('fill', imp > 0 ? color : 'var(--dim)');
    });
    const maxY = Math.max(0, ...cell.dYdt.map(Math.abs));
    const maxN = network ? Math.max(0, ...network.map(Math.abs)) : 0;
    this.nodes.forEach((n, i) => {
      const x = cell.X[i];
      n.g.classList.toggle('absent', !(x >= ABSENT));
      n.circle.setAttribute('fill-opacity', String(f1(0.08 + 0.72 * Math.min(1, Math.max(0, (Math.log10(Math.max(x, 1e-30)) + 6) / 6)))));
      const y = cell.dYdt[i];
      const imp = importance(Math.abs(y), maxY);
      n.circle.setAttribute('stroke', imp > 0 ? (y > 0 ? 'var(--good)' : 'var(--bad)') : 'var(--line)');
      n.circle.setAttribute('stroke-width', String(f1(0.8 + 2.2 * imp)));
      const m = network?.[i] ?? 0;
      const shown = !!network && importance(Math.abs(m), maxN) > 0;
      n.mark.setAttribute('visibility', shown ? 'visible' : 'hidden');
      if (shown) n.mark.setAttribute('fill', m > 0 ? 'var(--good)' : 'var(--bad)');
    });
    const cap = cell.why || 'aprox13 reference rates here, mol/g/s';
    if (this.caption.textContent !== cap) this.caption.textContent = cap;
    this.showDetail();
  }

  private build() {
    const el = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent: Element): SVGElementTagNameMap[K] => {
      const e = document.createElementNS(SVG, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
      parent.appendChild(e);
      return e;
    };
    this.root.innerHTML = '<div class="head" title="Show or hide the graph of the burning network’s reactions at this cell"><span class="chev">▸</span>Network <span class="caption"></span></div>';
    this.caption = this.root.querySelector('.caption')!;
    this.root.querySelector('.head')!.addEventListener('click', () => {
      this.root.classList.toggle('collapsed');
      this.toggled(this.open);
    });
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, this.root);
    this.svg = svg;
    const edgeLayer = el('g', { class: 'edges' }, svg);
    const nodeLayer = el('g', { class: 'nodes' }, svg);
    for (const shape of shapes()) {
      const g = el('g', { class: 'edge', 'data-edge': shape.pair, 'data-s': `r:${shape.pair}` }, edgeLayer);
      // A wide invisible stroke under each, so that the pointer finds a thin edge.
      for (const s of shape.strokes) el('path', { d: s.d, class: 'hit' }, g);
      const strokes = shape.strokes.map((s) => el('path', { d: s.d, class: 'stroke', fill: 'none' }, g));
      const heads = [0, 1].map(() => el('path', { class: 'head', visibility: 'hidden' }, g));
      const junction = shape.junction ? el('circle', { cx: shape.junction[0], cy: shape.junction[1], r: 2.2 }, g) : undefined;
      const label = el('text', { x: shape.label.at[0], y: shape.label.at[1], 'text-anchor': shape.label.anchor, class: 'rate' }, g);
      this.edges.push({ shape, g, strokes, heads, label, junction });
    }
    this.edges.sort((a, b) => a.shape.pair - b.shape.pair);
    NODE_AT.forEach(([x, y], i) => {
      const g = el('g', { class: 'node', 'data-node': i, 'data-s': `x:${i}` }, nodeLayer);
      const circle = el('circle', { cx: x, cy: y, r: R, class: 'disc' }, g);
      el('text', { x, y: y + 3, 'text-anchor': 'middle', class: 'name' }, g).textContent = NODE_NAME[i];
      const mark = el('circle', { cx: x + R - 2, cy: y - R + 2, r: 2.6, class: 'mark', visibility: 'hidden' }, g);
      this.nodes.push({ g, circle, mark });
    });
    this.detail = document.createElement('div');
    this.detail.className = 'detail';
    this.root.appendChild(this.detail);
    svg.addEventListener('pointerover', (ev) => {
      const t = ev.target as Element;
      const e = t.closest('[data-edge]');
      const n = t.closest('[data-node]');
      this.hover = e ? { edge: +e.getAttribute('data-edge')! } : n ? { node: +n.getAttribute('data-node')! } : null;
      svg.querySelectorAll('.hl').forEach((h) => h.classList.remove('hl'));
      (e ?? n)?.classList.add('hl');
      this.showDetail();
    });
    svg.addEventListener('pointerleave', () => {
      this.hover = null;
      svg.querySelectorAll('.hl').forEach((h) => h.classList.remove('hl'));
      this.showDetail();
    });
  }

  /** The line under the graph: what the pointer is over, or the key. */
  private showDetail() {
    const s = this.last;
    if (!s) return;
    let text = 'Arrows: net flux, + α along the chain · width and color: rate, of the largest · outline: made (green) or used up (red) · dot: the network’s own step';
    const h = this.hover;
    if (h && 'edge' in h) {
      const e = s.cell.edges.find((x) => x.pair === h.edge);
      if (e) text = edgeText(e, s.cell);
    } else if (h && 'node' in h) {
      text = nodeText(h.node, s);
    }
    if (this.detail.textContent !== text) this.detail.textContent = text;
  }
}

/** A reaction as words: its nuclei, both ways' rates, the energy and how long its fuel lasts. */
function edgeText(e: ReactionEdge, cell: CellReactions): string {
  const name = (i: number) => NODE_NAME[i];
  const side = (sign: number) => [...e.nu].filter(([, n]) => n * sign > 0).map(([i, n]) => `${Math.abs(n) > 1 ? `${Math.abs(n)} ` : ''}${name(i)}`).join(' + ');
  const way = e.eq ? 'in equilibrium' : e.dir > 0 ? 'net forward' : e.dir < 0 ? 'net backward' : 'no flux';
  // The heavy fuel the net flux uses up, and how long it would last at this rate.
  const used = [...e.nu].filter(([i, n]) => i > 0 && n * e.dir < 0).sort((a, b) => b[0] - a[0])[0];
  const life = used && e.net > 0 ? cell.X[used[0]] / NET_DATA.A[used[0]] / (Math.abs(used[1]) * e.net) : NaN;
  return [
    `${side(-1)} → ${side(1)} (${e.label}; back: ${e.revLabel})`,
    `forward ${fmtRate(e.fwd)}, backward ${fmtRate(e.rev)} mol/g/s, ${way}`,
    `releases ${fmt3(e.energy)} erg/g/s`,
    ...(Number.isFinite(life) ? [`${name(used[0])} lasts ${fmt3(life)} s at this rate`] : []),
  ].join(' · ');
}

/** A species as words: its mass fraction and how fast the reference and the network change it. */
function nodeText(i: number, s: GraphState): string {
  const y = s.cell.dYdt[i];
  const how = (v: number) => (v > 0 ? 'made' : v < 0 ? 'used up' : 'unchanged');
  const net = s.network ? ` · the network’s step: ${how(s.network[i])}` : '';
  return `${NODE_NAME[i]}: X = ${fmt3(s.cell.X[i])} · reference dY/dt = ${fmt3(y)} mol/g/s (${how(y)})${net}`;
}
