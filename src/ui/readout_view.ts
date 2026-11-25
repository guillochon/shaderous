// The pinned query readout's view: its rows of values and its equations, each number with a
// sparkline of how it has gone since the readout was pinned, a popup of the whole series on
// hovering one, and the series as CSV.
//
// The rows and equations are kept as elements and updated in place: the readout is read again
// several times a second, and rebuilding it would lose the pointer's hover (the equations'
// breakdowns open on hover) and the sparklines' canvases.
import { drawFullLines, drawSparkLines, fmt3, varies } from './diag_plot';
import { importance, rampColor, type Equation } from './terms';

/** A row of values: a label, its text, and the code values it shows (with their display scale and unit). */
export interface Row {
  key: string;
  label: string;
  text: string;
  cls?: string;
  /** its name in the series (the popup, the CSV) where the label alone is ambiguous */
  name?: string;
  series?: { raw: number[]; scale: number; unit: string };
}

/** Past this many readings the older half is thinned to every other one. */
const MAX_READINGS = 20000;

interface Col {
  label: string;
  unit: string;
  scale: number;
  /** data[component][reading], NaN where the reading did not have it */
  data: number[][];
}

/** The readings since the readout was pinned: time, step, and every series. */
export class SeriesLog {
  t: number[] = [];
  steps: number[] = [];
  cols = new Map<string, Col>();
  /** the unit of the times (display units, as recorded) */
  tUnit = '';

  clear() {
    this.t = [];
    this.steps = [];
    this.cols.clear();
  }

  /** Add a reading. While the run is paused (the same step) the last reading is replaced. */
  push(t: number, step: number, values: Map<string, { label: string; raw: number[]; scale: number; unit: string }>) {
    const n = this.t.length;
    const replace = n > 0 && this.steps[n - 1] === step;
    if (n > 0 && step < this.steps[n - 1]) this.clear();
    const at = replace ? this.t.length - 1 : this.t.length;
    if (!replace) {
      this.t.push(t);
      this.steps.push(step);
      for (const c of this.cols.values()) for (const d of c.data) d.push(NaN);
    } else this.t[at] = t;
    for (const [key, v] of values) {
      let c = this.cols.get(key);
      if (!c || c.data.length !== v.raw.length) {
        c = { label: v.label, unit: v.unit, scale: v.scale, data: v.raw.map(() => new Array(this.t.length).fill(NaN)) };
        this.cols.set(key, c);
      }
      c.label = v.label;
      c.unit = v.unit;
      c.scale = v.scale;
      v.raw.forEach((r, k) => (c!.data[k][at] = r));
    }
    if (this.t.length > MAX_READINGS) {
      const half = this.t.length >> 1;
      const keep = (_: unknown, i: number) => i >= half || i % 2 === 0;
      this.t = this.t.filter(keep);
      this.steps = this.steps.filter(keep);
      for (const c of this.cols.values()) c.data = c.data.map((d) => d.filter(keep));
    }
  }

  /** The times, and each component of a series in its display units. */
  series(key: string): [number[], number[][]] | null {
    const c = this.cols.get(key);
    return c ? [this.t, c.data.map((d) => (c.scale === 1 ? d : d.map((v) => v * c.scale)))] : null;
  }

  /** Every series as CSV, in display units: time, step, then one column per component. */
  csv(): string {
    const cell = (v: number) => (Number.isFinite(v) ? String(+v.toPrecision(8)) : '');
    const cols = [...this.cols.values()];
    const head = [this.tUnit ? `t (${this.tUnit})` : 't', 'step', ...cols.flatMap((c) => c.data.map((_, k) => `${c.label}${c.data.length > 1 ? ` ${'xyz'[k]}` : ''}${c.unit ? ` (${c.unit})` : ''}`))];
    const quote = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
    const rows = this.t.map((t, i) => [cell(t), String(this.steps[i]), ...cols.flatMap((c) => c.data.map((d) => cell(d[i] * c.scale)))].join(','));
    return [head.map(quote).join(','), ...rows].join('\n') + '\n';
  }
}

/** An equation's name, for the series and the popup. */
const EQ_NAMES: Record<Equation['id'], string> = {
  mass: 'Mass', momentum: 'Momentum', energy: 'Energy', induction: 'Induction',
  species: 'Species', 'rad-energy': 'Radiation energy', 'rad-flux': 'Radiation flux',
};

const pctText = (v: number) => (v >= 1 || v === 0 ? v.toFixed(0) : v.toPrecision(1));
const escape = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
/** The equation's first term without its leading plus: = curl(v x B), not = + curl(v x B). */
const firstTerm = (math: string) => math.replace(/^<mrow><mo>\+<\/mo>/, '<mrow>');

/** One line of an equation's breakdown: a term's share of its equation. */
interface Line {
  /** the series key, and the term it is a share of (for the hover highlight) */
  key: string;
  term: string;
  label: string;
  size: number;
  largest: number;
}

/** One block of the equations: the equation, and the breakdown it opens. */
interface Block {
  key: string;
  name: string;
  math: string;
  /** the terms' sizes, for the colors of the terms in the equation */
  colors: { term: string; size: number; largest: number }[];
  lines: Line[];
  net: string;
}

export class ReadoutView {
  readonly log = new SeriesLog();
  /** Equations whose breakdown stays open (clicked), by block key. */
  private open = new Set<string>();
  private rowShape = '';
  private eqShape = '';
  private rows: Row[] = [];
  private blocks: Block[] = [];
  /** The series whose whole run the popup shows, and the element it is beside. */
  private popKey: string | null = null;
  private popAnchor: HTMLElement | null = null;

  constructor(
    private dl: HTMLElement,
    private eqs: HTMLElement,
    private pop: { el: HTMLElement; title: Element; canvas: HTMLCanvasElement },
    /** the readout itself, which the popup goes beside */
    private box: HTMLElement,
  ) {
    // A click on an equation keeps its breakdown open, or closes it again.
    eqs.addEventListener('click', (e) => {
      const eq = (e.target as Element).closest('.eq');
      const block = eq?.closest('.eqb') as HTMLElement | null;
      if (!block) return;
      const k = block.dataset.k!;
      if (this.open.has(k)) this.open.delete(k);
      else this.open.add(k);
      block.classList.toggle('open', this.open.has(k));
      this.drawSparks(block);
    });
    // Hovering a term of an equation bolds its line of the breakdown.
    eqs.addEventListener('pointerover', (e) => {
      const target = e.target as Element;
      const t = target.closest('math [data-term]');
      const block = target.closest('.eqb');
      eqs.querySelectorAll('.line.hl, math .hl').forEach((el) => el.classList.remove('hl'));
      // The term itself too: drawn heavier by a stroke, which leaves the layout as it is.
      t?.classList.add('hl');
      if (t && block) block.querySelectorAll(`.line[data-term="${t.getAttribute('data-term')}"]`).forEach((el) => el.classList.add('hl'));
    });
    eqs.addEventListener('pointerleave', () => eqs.querySelectorAll('.line.hl, math .hl').forEach((el) => el.classList.remove('hl')));
    // A breakdown opening on hover has had no room to draw its sparklines: draw them now.
    eqs.addEventListener('pointerover', (e) => {
      const block = (e.target as Element).closest('.eqb') as HTMLElement | null;
      if (block && !block.dataset.drawn) {
        block.dataset.drawn = '1';
        requestAnimationFrame(() => this.drawSparks(block));
      }
    });
    eqs.addEventListener('pointerout', (e) => {
      const block = (e.target as Element).closest('.eqb') as HTMLElement | null;
      if (block && !block.contains(e.relatedTarget as Node)) delete block.dataset.drawn;
    });
    // A sparkline hovered: its whole series beside the readout.
    for (const root of [dl, eqs]) {
      root.addEventListener('pointerover', (e) => {
        const s = (e.target as Element).closest('[data-s]') as HTMLElement | null;
        if (s) this.showPop(s.dataset.s!, s);
      });
      root.addEventListener('pointerout', (e) => {
        const s = (e.target as Element).closest('[data-s]');
        if (s && !s.contains(e.relatedTarget as Node)) this.hidePop();
      });
    }
  }

  /** Start the series again (a new pin), and close the breakdowns. */
  reset() {
    this.log.clear();
    this.open.clear();
    this.eqShape = '';
    this.hidePop();
  }

  /** Show these rows; with `sparks`, each numeric one with its sparkline. */
  setRows(rows: Row[], sparks: boolean) {
    this.rows = rows;
    const shape = `${sparks}|${rows.map((r) => `${r.key}:${r.cls ?? ''}`).join('|')}`;
    if (shape !== this.rowShape) {
      this.rowShape = shape;
      this.dl.classList.toggle('sparks', sparks);
      this.dl.innerHTML = rows
        .map((r) => {
          const spark = sparks ? (r.series ? `<canvas class="spark" data-s="row:${r.key}"></canvas>` : '<span></span>') : '';
          return `<dt${r.cls === 'head' ? ' class="head"' : ''}>${escape(r.label)}</dt><dd${r.cls ? ` class="${r.cls}"` : ''}></dd>${spark}`;
        })
        .join('');
    }
    const dds = this.dl.querySelectorAll('dd');
    rows.forEach((r, i) => {
      if (dds[i].textContent !== r.text) dds[i].textContent = r.text;
    });
  }

  /** Show the equations (none: an empty list), or a note in their place. */
  setEquations(list: Equation[] | null, note = '') {
    this.blocks = list ? blocksOf(list) : [];
    const shape = `${note}|${this.blocks.map((b) => `${b.key}:${b.colors.map((c) => c.term).join(',')}:${b.lines.map((l) => l.key).join(',')}`).join('|')}`;
    if (shape !== this.eqShape) {
      this.eqShape = shape;
      // A header with the ramp the terms are colored on.
      const head = this.blocks.length ? '<div class="head">Equations <span class="legend">term size, of the largest: <span class="ramp"></span></span></div>' : '';
      this.eqs.innerHTML =
        head +
        this.blocks
          .map(
            (b) =>
              `<div class="eqb${this.open.has(b.key) ? ' open' : ''}" data-k="${b.key}"><div class="eq" title="Click to keep the breakdown open">${b.math}</div><div class="bd">${b.lines
                .map((l) => `<div class="line" data-term="${l.term}"><span class="swatch"></span><span class="what">${escape(l.label)}</span><span class="size"></span><canvas class="spark" data-s="${l.key}"></canvas></div>`)
                .join('')}<div class="net"></div></div></div>`,
          )
          .join('') + (note ? `<div class="note">${escape(note)}</div>` : '');
    }
    this.eqs.hidden = !this.blocks.length && !note;
    // In place: each term's color in its equation, and its line's share.
    this.eqs.querySelectorAll<HTMLElement>('.eqb').forEach((el, i) => {
      const b = this.blocks[i];
      if (!b) return;
      const terms = el.querySelectorAll('math [data-term]');
      b.colors.forEach((c, k) => terms[k]?.setAttribute('style', `color: ${rampColor(importance(c.size, c.largest))}`));
      el.querySelectorAll<HTMLElement>('.line').forEach((line, k) => {
        const l = b.lines[k];
        const pct = l.largest > 0 ? (100 * l.size) / l.largest : 0;
        (line.children[0] as HTMLElement).style.background = rampColor(importance(l.size, l.largest));
        const text = `${pctText(pct)}%`;
        if (line.children[2].textContent !== text) line.children[2].textContent = text;
      });
      const net = el.querySelector('.net')!;
      if (net.textContent !== b.net) net.textContent = b.net;
    });
  }

  /** Take a reading of everything shown: at time t (in display units of `tUnit`), step `step`. Then redraw the sparklines. */
  record(t: number, step: number, tUnit = '') {
    this.log.tUnit = tUnit;
    const values = new Map<string, { label: string; raw: number[]; scale: number; unit: string }>();
    for (const r of this.rows) if (r.series) values.set(`row:${r.key}`, { label: r.name ?? r.label, ...r.series });
    for (const b of this.blocks) {
      for (const l of b.lines) {
        values.set(l.key, { label: `${b.name}: ${l.label}`, raw: [l.largest > 0 ? (100 * l.size) / l.largest : 0], scale: 1, unit: '%' });
      }
    }
    this.log.push(t, step, values);
    this.drawSparks(this.dl);
    this.eqs.querySelectorAll<HTMLElement>('.eqb').forEach((b) => this.drawSparks(b));
    this.drawPop();
  }

  /** Draw the sparklines under `root` that have room to (an open breakdown's, the rows'). */
  private drawSparks(root: HTMLElement) {
    root.querySelectorAll<HTMLCanvasElement>('canvas.spark').forEach((c) => {
      const s = this.log.series(c.dataset.s!);
      if (!s) return;
      const [xs, yss] = s;
      // Only a quantity that changes gets a line.
      if (varies(yss)) drawSparkLines(c, xs, yss);
      else drawSparkLines(c, [], []);
    });
  }

  private showPop(key: string, anchor: HTMLElement) {
    const s = this.log.series(key);
    if (!s || !varies(s[1])) return;
    this.popKey = key;
    this.popAnchor = anchor;
    this.pop.el.hidden = false;
    this.drawPop();
  }

  private hidePop() {
    this.popKey = null;
    this.popAnchor = null;
    this.pop.el.hidden = true;
  }

  private drawPop() {
    const key = this.popKey;
    if (!key || !this.popAnchor) return;
    const c = this.log.cols.get(key);
    const s = this.log.series(key);
    if (!c || !s) return this.hidePop();
    const [xs, yss] = s;
    const now = yss.map((d) => d[d.length - 1]);
    this.pop.title.textContent = `${c.label} · ${now.map((v) => fmt3(v)).join(', ')}${c.unit ? ` ${c.unit}` : ''}`;
    // Beside the readout, level with the sparkline.
    const r = this.popAnchor.getBoundingClientRect();
    const box = this.box.getBoundingClientRect();
    const pw = this.pop.el.offsetWidth;
    const ph = this.pop.el.offsetHeight;
    const x = box.right + 8 + pw <= window.innerWidth - 4 ? box.right + 8 : box.left - 8 - pw;
    this.pop.el.style.left = `${Math.round(Math.max(4, x))}px`;
    this.pop.el.style.top = `${Math.round(Math.min(window.innerHeight - ph - 4, Math.max(4, r.top + r.height / 2 - ph / 2)))}px`;
    drawFullLines(this.pop.canvas, xs, yss, this.log.tUnit ? `t (${this.log.tUnit})` : 't');
  }
}

/** The blocks of a list of equations: one per equation, and one for all the species together. */
function blocksOf(list: Equation[]): Block[] {
  const color = (e: Equation, i: number) => `<mrow data-term="${e.terms[i].id}">${i === 0 ? firstTerm(e.terms[i].math) : e.terms[i].math}</mrow>`;
  const netOf = (e: Equation) => {
    if (!(e.largest > 0)) return 'nothing changes here';
    const net = (100 * e.netSize) / e.largest;
    return `sum of the terms: ${pctText(net)}% of the largest term${net < 5 && e.terms.length > 1 ? ': in balance' : ''}`;
  };
  const out: Block[] = list
    .filter((e) => e.id !== 'species')
    .map((e) => ({
      key: e.id,
      name: EQ_NAMES[e.id],
      math: `<math><mrow><mrow class="lhs">${e.lhs}</mrow><mo>=</mo>${e.terms.map((_, i) => color(e, i)).join('')}</mrow></math>`,
      colors: e.terms.map((t) => ({ term: t.id, size: t.size, largest: e.largest })),
      lines: e.terms.map((t) => ({ key: `eq:${e.id}:${t.id}`, term: t.id, label: t.label, size: t.size, largest: e.largest })),
      net: netOf(e),
    }));
  // The species: one form, its terms colored by the largest share each has in any species'
  // equation, and a line per species and term.
  const species = list.filter((e) => e.id === 'species');
  if (species.length) {
    const form = species.reduce((a, b) => (b.terms.length > a.terms.length ? b : a));
    const share = (id: string) => Math.max(0, ...species.map((e) => (e.largest > 0 ? (e.terms.find((t) => t.id === id)?.size ?? 0) / e.largest : 0)));
    out.push({
      key: 'species',
      name: EQ_NAMES.species,
      math: `<math><mrow><mrow class="lhs">${form.lhs}</mrow><mo>=</mo>${form.terms.map((_, i) => color(form, i)).join('')}</mrow></math>`,
      colors: form.terms.map((t) => ({ term: t.id, size: share(t.id), largest: 1 })),
      lines: species.flatMap((e) =>
        e.terms.map((t) => ({ key: `eq:species:${e.name}:${t.id}`, term: t.id, label: `${e.name}: ${t.id === 'sp-burn' ? 'burning' : 'carried'}`, size: t.size, largest: e.largest })),
      ),
      net: '',
    });
  }
  return out;
}
