// Forms made from the option table (setup/options.ts): one control per kind of Control, built
// once and then kept in step with the document by update(), so a field being typed in is never
// replaced under the cursor. A form holds no values of its own: it reads and writes the
// document by path through the callbacks it is given.
//
// An option whose rules do not hold keeps its place: greyed, its reasons as its tooltip and,
// where there is no hover, as a line under it when tapped.
import { BC_CHOICES, flatten, PART_OPTIONS, unavailable, type Choice, type Control, type Option, type Value } from '../../setup/options';
import { partIds } from '../../setup/parts/bodies';
import { setPath, type SetupSpec } from '../../setup/spec';
import { displayLabel, displayScale, type UnitSystem } from '../../units/units';

/** A choice the page itself rules out (a grid that does not fit), with why. */
export interface FormChoice extends Choice {
  off?: string;
}

export interface FormContext {
  /** the document being edited */
  spec(): SetupSpec;
  /** a value by its whole path; undefined where the document has none */
  read(path: string): unknown;
  /** set a value by its whole path; undefined removes the key, and its default then applies */
  write(path: string, value: unknown): void;
  /** the facts these options are judged against: the setup's, or for a part's fields the setup's without that part */
  facts(): Record<string, Value>;
  /** the setup's physical units, where values are to be shown in them too */
  units?(): UnitSystem | undefined;
  /** whether a path is in the document's featured list, and adding or removing it */
  featured?(path: string): boolean;
  toggleFeatured?(path: string): void;
  /** another control for an option than the table's (the grids on offer as a menu) */
  control?(option: Option, path: string): Control | undefined;
  /** the place of the part these fields belong to: it cannot name itself */
  part?: number;
}

export interface Form {
  el: HTMLElement;
  /** Bring every field in step with the document: values, what is shown, what is available. */
  update(): void;
  /** Scroll to and focus the field a path leads to. False if this form has none. */
  reveal(path: string): boolean;
}

interface Pass {
  facts: Record<string, Value>;
  units: UnitSystem | undefined;
}
interface Field {
  el: HTMLElement;
  update(pass: Pass): void;
}

type Attrs = Record<string, string | number | boolean | undefined>;

/** An element with attributes and children; text is always set as text, never parsed. */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue;
    if (k === 'class') e.className = String(v);
    else if (k === 'text') e.textContent = String(v);
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  e.append(...children);
  return e;
}

let serial = 0;
const nextId = () => `sf${++serial}`;

/** A number as a field shows it: without the noise of its last binary digits. */
const shown = (v: number): string => String(+v.toPrecision(12));
const sig = (v: number, digits = 4): string => {
  const a = Math.abs(v);
  return a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(digits - 1) : String(+v.toPrecision(digits));
};
const clone = <T>(v: T): T => (v === undefined ? v : (JSON.parse(JSON.stringify(v)) as T));
const isSet = (v: unknown): boolean => v !== undefined && v !== null && v !== false && v !== 0;

const COST: Record<Option['apply'], { text: string; title: string }> = {
  live: { text: 'live', title: 'Takes effect at once: the run carries on' },
  shaders: { text: 'recompiles', title: 'Changing it rebuilds the shaders: a pause, then the run carries on' },
  restart: { text: 'restart', title: 'Changing it starts the run again from its initial conditions' },
};

/** Controls that can sit beside the preset menu as a featured field. */
const CAN_FEATURE = ['number', 'check', 'select'];

interface Build {
  ctx: FormContext;
  /** the options of the form, flat, for the defaults that `show` conditions fall back on */
  all: Option[];
  /** prefix of the form's own options: a `show` path is relative to it */
  root: string;
}

/**
 * Render options into an element. `prefix` is put before each option's path to make its whole
 * path in the document ('' for the general options, 'parts.2.' for a part's fields).
 * `siblings` are the other options of the same part, where only some of its fields are rendered
 * (the featured strip): a `show` condition may rest on one of their defaults.
 */
export function renderOptions(into: HTMLElement, options: Option[], prefix: string, ctx: FormContext, siblings?: Option[]): Form {
  const b: Build = { ctx, all: siblings ?? flatten(options), root: prefix };
  const fields = options.map((o) => buildField(o, prefix, b, false));
  into.append(...fields.map((f) => f.el));
  const units = () => {
    try {
      return ctx.units?.();
    } catch {
      return undefined;
    }
  };
  return {
    el: into,
    update() {
      const pass: Pass = { facts: ctx.facts(), units: units() };
      for (const f of fields) f.update(pass);
    },
    reveal(path) {
      let best: HTMLElement | null = null;
      for (const e of into.querySelectorAll<HTMLElement>('[data-path]')) {
        const p = e.dataset.path!;
        if ((path === p || path.startsWith(`${p}.`)) && (!best || p.length > best.dataset.path!.length)) best = e;
      }
      if (!best) return false;
      best.scrollIntoView({ block: 'center' });
      best.classList.add('sf-flash');
      setTimeout(() => best!.classList.remove('sf-flash'), 1200);
      best.querySelector<HTMLElement>('input:not(:disabled), select:not(:disabled), textarea:not(:disabled)')?.focus({ preventScroll: true });
      return true;
    },
  };
}

function buildField(o: Option, prefix: string, b: Build, inGroup: boolean): Field {
  const { ctx } = b;
  const path = prefix + o.path;
  const control = ctx.control?.(o, path) ?? o.control;
  const id = nextId();
  const wrap = h('div', { class: `sf sf-${control.type}`, 'data-path': path });
  const why = h('div', { class: 'sf-why', id: `${id}-why`, hidden: true });
  const whyText = h('span');
  const clear = h('button', { type: 'button', class: 'sf-clear', text: 'Remove', title: 'Take this setting out of the setup' });
  why.append(whyText, clear);
  // A disabled control swallows taps: a veil over it takes them and shows the reason.
  const veil = h('div', { class: 'sf-veil', hidden: true });
  veil.addEventListener('click', () => {
    why.hidden = !why.hidden;
  });

  const cost = h('span', { class: `sf-cost sf-${o.apply}`, text: COST[o.apply].text, title: COST[o.apply].title });
  const marks = h('span', { class: 'sf-marks' }, cost);
  let star: HTMLButtonElement | null = null;
  if (ctx.toggleFeatured && CAN_FEATURE.includes(control.type)) {
    star = h('button', { type: 'button', class: 'sf-star', 'aria-pressed': 'false', 'aria-label': `Feature ${o.label}` });
    star.addEventListener('click', () => ctx.toggleFeatured!(path));
    marks.append(star);
  }

  // Emptying a field takes its key out of the document. A group's object must stay whole, so
  // inside one the default is written instead.
  const put = (value: unknown) => ctx.write(path, value === undefined && inGroup && o.default !== undefined && o.default !== null ? clone(o.default) : value);
  clear.addEventListener('click', () => ctx.write(path, undefined));

  const body = makeControl(o, control, path, id, b, put);
  const body_ = h('div', { class: 'sf-body' }, ...body.nodes, veil);
  if (body.ownLabel) wrap.append(h('div', { class: 'sf-head' }, body.ownLabel, marks), body_, why);
  else wrap.append(h('div', { class: 'sf-head' }, h('label', { for: id, text: o.label }), marks), body_, why);
  if (body.below) wrap.append(body.below);

  const showOk = (): boolean => {
    if (!o.show) return true;
    const at = b.root + o.show.path;
    const v = ctx.read(at) ?? b.all.find((k) => b.root + k.path === at)?.default;
    return o.show.is.includes(v as Value);
  };

  return {
    el: wrap,
    update(pass) {
      const visible = showOk();
      wrap.hidden = !visible;
      if (!visible) return;
      const value = ctx.read(path);
      const reasons = unavailable(o, pass.facts);
      const off = reasons.length > 0;
      wrap.classList.toggle('off', off);
      wrap.title = off ? reasons.join('. ') : o.tip;
      veil.hidden = !off;
      veil.title = wrap.title;
      if (off) {
        whyText.textContent = reasons.join('. ');
        // A setting that is on where it cannot be is an error of the setup: say why, and let it go.
        clear.hidden = !isSet(value);
        if (isSet(value)) why.hidden = false;
      } else why.hidden = true;
      if (star) {
        const on = !!ctx.featured?.(path);
        star.setAttribute('aria-pressed', String(on));
        star.textContent = on ? '★' : '☆';
        star.title = on ? 'Featured: shown beside the preset menu. Click to take it out' : 'Feature: show this field beside the preset menu, and in the strip above the pages';
      }
      body.sync(value, off, pass);
    },
  };
}

interface Made {
  nodes: Node[];
  /** a label the control brings itself (a checkbox's), in place of the field's */
  ownLabel?: HTMLElement;
  /** what goes under the field: a group's own fields */
  below?: HTMLElement;
  sync(value: unknown, off: boolean, pass: Pass): void;
}

function numberInput(c: Extract<Control, { type: 'number' }> | undefined, attrs: Attrs): HTMLInputElement {
  const input = h('input', { type: 'number', step: c?.step ?? 'any', min: c?.min, max: c?.max, 'data-factor': c?.factor, inputmode: 'decimal', ...attrs });
  return input;
}

/** Put a number in a field, unless the field is being typed in and already says it. */
function setNumber(input: HTMLInputElement, v: unknown): void {
  const text = typeof v === 'number' && Number.isFinite(v) ? shown(v) : '';
  if (document.activeElement === input && input.value !== '' && text !== '' && Number(input.value) === Number(text)) return;
  if (input.value !== text) input.value = text;
}

/** A field's number, held to the option's range; undefined for an empty field, null for what is not a number. */
function readNumber(input: HTMLInputElement, c?: Extract<Control, { type: 'number' }>): number | undefined | null {
  const t = input.value.trim();
  if (t === '') return input.validity.badInput ? null : undefined;
  let v = Number(t);
  if (!Number.isFinite(v)) return null;
  if (c?.integer) v = Math.round(v);
  if (c?.min !== undefined) v = Math.max(c.min, v);
  if (c?.max !== undefined) v = Math.min(c.max, v);
  return v;
}

function makeControl(o: Option, control: Control, path: string, id: string, b: Build, put: (v: unknown) => void): Made {
  const { ctx } = b;
  const described = { 'aria-describedby': `${id}-why` };

  switch (control.type) {
    case 'number': {
      const input = numberInput(control, { id, ...described });
      if (typeof o.default === 'number') input.placeholder = shown(o.default);
      else if (control.nullable) input.placeholder = 'none';
      const phys = h('span', { class: 'sf-phys' });
      input.addEventListener('change', () => {
        const v = readNumber(input, control);
        if (v === null) return setNumber(input, ctx.read(path));
        const cur = ctx.read(path);
        // A value shown rounded and left as it was is not a change.
        if (typeof cur === 'number' && v !== undefined && v === Number(shown(cur))) return;
        put(v);
      });
      return {
        nodes: [input, phys],
        sync(value, off, pass) {
          input.disabled = off;
          setNumber(input, value);
          const v = typeof value === 'number' ? value : typeof o.default === 'number' ? o.default : undefined;
          const q = control.quantity;
          const text = q && pass.units && v !== undefined ? `${sig(v * displayScale(pass.units, q, 'cgs'))} ${displayLabel(pass.units, q, 'cgs')}` : '';
          phys.textContent = text;
          phys.hidden = !text;
          if (text) phys.title = `${o.label} in physical units`;
        },
      };
    }

    case 'select': {
      const select = h('select', { id, ...described });
      const choices = (): FormChoice[] => {
        const c = ctx.control?.(o, path) ?? o.control;
        return c.type === 'select' ? c.choices : [];
      };
      let drawn = '';
      select.addEventListener('change', () => {
        const c = choices().find((k) => String(k.value) === select.value);
        put(c ? c.value : undefined);
      });
      return {
        nodes: [select],
        sync(value, off, pass) {
          select.disabled = off;
          const list = choices();
          const cur = value ?? o.default;
          const rows = list.map((c) => ({ value: String(c.value), label: c.label, why: c.off ?? unavailable(c, pass.facts).join('. '), tip: c.tip ?? '' }));
          if (cur === undefined || cur === null) rows.unshift({ value: '', label: '—', why: '', tip: 'Not set' });
          else if (!rows.some((r) => r.value === String(cur))) rows.push({ value: String(cur), label: String(cur), why: '', tip: 'A value this menu does not offer' });
          const key = JSON.stringify(rows);
          if (key !== drawn) {
            drawn = key;
            select.replaceChildren(...rows.map((r) => {
              const opt = h('option', { value: r.value, text: r.label, title: r.why || r.tip });
              opt.disabled = !!r.why;
              return opt;
            }));
          }
          select.value = cur === undefined || cur === null ? '' : String(cur);
        },
      };
    }

    case 'check': {
      const input = h('input', { type: 'checkbox', id, ...described });
      input.addEventListener('change', () => put(input.checked));
      return {
        nodes: [],
        ownLabel: h('label', { class: 'check', for: id }, input, o.label),
        sync(value, off) {
          input.disabled = off;
          input.checked = (value ?? o.default) === true;
        },
      };
    }

    case 'text': {
      const input = control.long ? h('textarea', { id, rows: 3, maxlength: 4000, spellcheck: 'true', ...described }) : h('input', { type: 'text', id, maxlength: 200, spellcheck: 'false', ...described });
      if (typeof o.default === 'string') input.placeholder = o.default;
      input.addEventListener('change', () => put(input.value.trim() === '' ? undefined : input.value.trim()));
      return {
        nodes: [input],
        sync(value, off) {
          input.disabled = off;
          const text = typeof value === 'string' ? value : '';
          if (document.activeElement !== input && input.value !== text) input.value = text;
        },
      };
    }

    case 'vec3': {
      const def = Array.isArray(o.default) ? o.default : undefined;
      // A box's corner may be open on a side: an empty component there is no bound, not zero.
      const open = /(^|\.)shape\.(min|max)$/.test(path);
      const inputs = ['x', 'y', 'z'].map((ax, i) => {
        const input = numberInput(undefined, { 'aria-label': `${o.label}, ${ax}`, title: ax, ...(i === 0 ? { id } : {}) });
        input.placeholder = open ? `${ax}: open` : def ? shown(def[i]) : ax;
        return input;
      });
      const commit = () => {
        const vals = inputs.map((inp) => readNumber(inp));
        if (vals.some((v) => v === null)) return;
        if (vals.every((v) => v === undefined)) return put(undefined);
        put(vals.map((v, i) => (v !== undefined ? v : open ? null : (def?.[i] ?? 0))));
      };
      for (const inp of inputs) inp.addEventListener('change', commit);
      return {
        nodes: [h('div', { class: 'sf-vec', role: 'group', 'aria-label': o.label }, ...inputs)],
        sync(value, off, pass) {
          // The uniform acceleration may be given as a strength along the axis drawn downward.
          let v: unknown = value;
          if (v !== null && typeof v === 'object' && !Array.isArray(v) && typeof (v as { down?: unknown }).down === 'number') {
            const down = (v as { down: number }).down;
            v = pass.facts['domain.ndim'] === 2 ? [0, -down, 0] : [0, 0, -down];
          }
          inputs.forEach((inp, i) => {
            inp.disabled = off;
            setNumber(inp, Array.isArray(v) ? v[i] : undefined);
          });
        },
      };
    }

    case 'faces': {
      // A 2D run has faces of its own where the document keeps them apart.
      const target = () => (path === 'domain.bc' && ctx.facts()['domain.ndim'] === 2 && ctx.read('domain.bc2d') !== undefined ? 'domain.bc2d' : path);
      const faces = (): { lo: number[]; hi: number[] } => {
        const v = ctx.read(target()) as { lo?: number[]; hi?: number[] } | undefined;
        const side = (s: number[] | undefined) => [0, 1, 2].map((a) => (typeof s?.[a] === 'number' ? s[a] : 1));
        return { lo: side(v?.lo), hi: side(v?.hi) };
      };
      const selects: { side: 'lo' | 'hi'; axis: number; select: HTMLSelectElement; row: HTMLElement }[] = [];
      const grid = h('div', { class: 'sf-faces', role: 'group', 'aria-label': o.label });
      ['x', 'y', 'z'].forEach((ax, axis) => {
        for (const side of ['lo', 'hi'] as const) {
          const sid = nextId();
          const select = h('select', { id: sid });
          const row = h('label', { class: 'f', for: sid }, h('span', { text: `${ax} ${side === 'lo' ? 'lower' : 'upper'} face` }), select);
          select.addEventListener('change', () => {
            const f = faces();
            const v = +select.value;
            const other = side === 'lo' ? 'hi' : 'lo';
            f[side][axis] = v;
            // Periodic faces come in pairs: the opposite face follows, in and out of it.
            if (v === 0) f[other][axis] = 0;
            else if (f[other][axis] === 0) f[other][axis] = v;
            ctx.write(target(), f);
          });
          selects.push({ side, axis, select, row });
          grid.append(row);
        }
      });
      let drawn = '';
      return {
        nodes: [grid],
        sync(_value, off, pass) {
          const f = faces();
          const rows = BC_CHOICES.map((c) => ({ value: String(c.value), label: c.label, why: unavailable(c, pass.facts).join('. '), tip: c.tip ?? '' }));
          const key = JSON.stringify(rows);
          for (const s of selects) {
            if (key !== drawn) {
              s.select.replaceChildren(...rows.map((r) => {
                const opt = h('option', { value: r.value, text: r.label, title: r.why || r.tip });
                opt.disabled = !!r.why;
                return opt;
              }));
            }
            s.select.disabled = off;
            s.select.value = String(f[s.side][s.axis]);
            s.row.hidden = s.axis === 2 && pass.facts['domain.ndim'] === 2;
          }
          drawn = key;
        },
      };
    }

    case 'ref': {
      const candidates = (): { id: string; label: string }[] => {
        const parts = ctx.spec().parts ?? [];
        const ids = partIds(parts);
        return parts.flatMap((p, i) => (i !== ctx.part && control.kinds.includes(p.kind) ? [{ id: ids[i], label: `${ids[i]} (${PART_OPTIONS[p.kind].label.toLowerCase()})` }] : []));
      };
      const kinds = control.kinds.map((k) => PART_OPTIONS[k].label.toLowerCase()).join(' or ');
      if (control.count === 'any') {
        const box = h('div', { class: 'sf-refs', role: 'group', 'aria-label': o.label, id });
        let drawn = '';
        return {
          nodes: [box],
          sync(value, off) {
            const chosen = Array.isArray(value) ? value.map(String) : [];
            const list = candidates();
            for (const c of chosen) if (!list.some((k) => k.id === c)) list.push({ id: c, label: `${c} (not there)` });
            const key = JSON.stringify([list, chosen, off]);
            if (key === drawn) return;
            drawn = key;
            if (!list.length) return box.replaceChildren(h('span', { class: 'sf-none', text: `No ${kinds} to choose: add one first` }));
            box.replaceChildren(...list.map((c) => {
              const input = h('input', { type: 'checkbox' });
              input.checked = chosen.includes(c.id);
              input.disabled = off;
              input.addEventListener('change', () => {
                const now = [...box.querySelectorAll<HTMLInputElement>('input')].flatMap((k, i) => (k.checked ? [list[i].id] : []));
                put(now);
              });
              return h('label', { class: 'check' }, input, c.label);
            }));
          },
        };
      }
      const n = control.count;
      const selects = Array.from({ length: n }, (_, k) => h('select', { 'aria-label': n === 1 ? o.label : `${o.label}: ${k === 0 ? 'first' : 'second'}`, ...(k === 0 ? { id } : {}) }));
      const commit = () => put(n === 1 ? selects[0].value || undefined : selects.map((s) => s.value));
      for (const s of selects) s.addEventListener('change', commit);
      let drawn = '';
      return {
        nodes: [h('div', { class: n === 2 ? 'sf-pair' : '' }, ...selects)],
        sync(value, off) {
          const chosen = (n === 1 ? [value] : Array.isArray(value) ? value : []).map((v) => (typeof v === 'string' ? v : ''));
          const list = candidates();
          for (const c of chosen) if (c && !list.some((k) => k.id === c)) list.push({ id: c, label: `${c} (not there)` });
          const key = JSON.stringify(list);
          selects.forEach((s, k) => {
            if (key !== drawn) s.replaceChildren(h('option', { value: '', text: list.length ? '— choose —' : `no ${kinds} yet` }), ...list.map((c) => h('option', { value: c.id, text: c.label })));
            s.disabled = off;
            s.value = chosen[k] ?? '';
          });
          drawn = key;
        },
      };
    }

    case 'fractions': {
      // Rows being typed are the control's own until they have a name: the document holds only whole ones.
      let rows: { name: string; value: number | undefined }[] = [];
      const asObject = () => Object.fromEntries(rows.filter((r) => r.name && r.value !== undefined).map((r) => [r.name, r.value]));
      const list = h('div', { class: 'sf-rows' });
      const add = h('button', { type: 'button', class: 'sf-add', text: 'Add a species', title: 'Add a row: a species name and its mass fraction' });
      let disabled = false;
      const commit = () => {
        const obj = asObject();
        ctx.write(path, Object.keys(obj).length ? obj : undefined);
      };
      const draw = () => {
        list.replaceChildren(...rows.map((r, i) => {
          const name = h('input', { type: 'text', 'aria-label': `${o.label}: species ${i + 1}`, placeholder: 'he4', maxlength: 24, spellcheck: 'false' });
          name.value = r.name;
          const value = numberInput({ type: 'number', min: 0, max: 1, step: 0.05 }, { 'aria-label': `${o.label}: mass fraction of species ${i + 1}`, placeholder: '0' });
          if (r.value !== undefined) value.value = shown(r.value);
          const remove = h('button', { type: 'button', class: 'sf-x', text: '×', title: 'Remove this row', 'aria-label': `Remove species ${i + 1}` });
          name.disabled = value.disabled = remove.disabled = disabled;
          name.addEventListener('change', () => {
            r.name = name.value.trim();
            commit();
          });
          value.addEventListener('change', () => {
            const v = readNumber(value, { type: 'number', min: 0, max: 1 });
            r.value = v === null ? undefined : v;
            commit();
          });
          remove.addEventListener('click', () => {
            rows.splice(i, 1);
            draw();
            commit();
          });
          return h('div', { class: 'sf-row' }, name, value, remove);
        }));
      };
      add.addEventListener('click', () => {
        rows.push({ name: '', value: undefined });
        draw();
        list.querySelector<HTMLInputElement>('.sf-row:last-child input')?.focus();
      });
      return {
        nodes: [list, add],
        sync(value, off) {
          const doc = value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
          const theirs = Object.fromEntries(Object.entries(doc).filter(([, v]) => typeof v === 'number'));
          const stale = JSON.stringify(theirs) !== JSON.stringify(asObject());
          if (stale) rows = Object.entries(theirs).map(([name, v]) => ({ name, value: v as number }));
          if (stale || off !== disabled || (!list.childElementCount && rows.length)) {
            disabled = off;
            draw();
          }
          add.disabled = off;
        },
      };
    }

    case 'names': {
      const input = h('input', { type: 'text', id, maxlength: 600, spellcheck: 'false', placeholder: 'none', ...described });
      input.addEventListener('change', () => {
        const names = input.value.split(',').map((s) => s.trim()).filter(Boolean);
        // The burning network's thirteen go by one name.
        put(!names.length ? undefined : names.length === 1 && names[0] === 'aprox13' ? 'aprox13' : names);
      });
      return {
        nodes: [input],
        sync(value, off) {
          input.disabled = off;
          const text = typeof value === 'string' ? value : Array.isArray(value) ? value.join(', ') : '';
          if (document.activeElement !== input && input.value !== text) input.value = text;
        },
      };
    }

    case 'group': {
      const sub = h('div', { class: 'sf-sub' });
      const fields = control.fields.map((f) => buildField(f, `${path}.`, b, true));
      sub.append(...fields.map((f) => f.el));
      const defaults = () => {
        const obj = {};
        for (const f of flatten(control.fields)) if (f.control.type !== 'group' && f.default !== undefined && f.default !== null) setPath(obj, f.path, clone(f.default));
        return obj;
      };
      if (!control.optional) {
        return {
          nodes: [],
          ownLabel: h('span', { class: 'sf-title', text: o.label }),
          below: sub,
          sync(_value, _off, pass) {
            for (const f of fields) f.update(pass);
          },
        };
      }
      const input = h('input', { type: 'checkbox', id, ...described });
      input.addEventListener('change', () => ctx.write(path, input.checked ? defaults() : undefined));
      return {
        nodes: [],
        ownLabel: h('label', { class: 'check', for: id }, input, o.label),
        below: sub,
        sync(value, off, pass) {
          const on = value !== undefined && value !== null;
          input.checked = on;
          input.disabled = off;
          sub.hidden = !on;
          if (on) for (const f of fields) f.update(pass);
        },
      };
    }
  }
}

/**
 * Number fields with a factor (data-factor): a click of the arrows or an arrow key multiplies or
 * divides the value by it, rounded to three figures, as the side bar's fields do. The browser
 * takes its own step first; that tells the direction, and the value it started from is noted
 * on the press.
 */
export function installFactorSteps(root: HTMLElement): void {
  const from = new WeakMap<HTMLInputElement, number>();
  const factorOf = (t: EventTarget | null) => (t instanceof HTMLInputElement && t.dataset.factor ? +t.dataset.factor : 0);
  const note = (e: Event) => {
    if (factorOf(e.target)) from.set(e.target as HTMLInputElement, +(e.target as HTMLInputElement).value);
  };
  root.addEventListener('pointerdown', note, true);
  root.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') note(e);
  }, true);
  root.addEventListener('input', (e) => {
    const el = e.target as HTMLInputElement;
    const k = factorOf(el);
    // Typing comes with an input type; a step of the arrows does not.
    if (!k || (e as InputEvent).inputType) return;
    const v0 = from.get(el);
    // From zero or an empty field there is nothing to multiply: the browser's step stands.
    if (v0 === undefined || el.value === '' || !(v0 > 0)) return;
    const dir = Math.sign(+el.value - v0);
    if (!dir) return;
    let v = +(v0 * k ** dir).toPrecision(3);
    if (el.min !== '') v = Math.max(+el.min, v);
    if (el.max !== '') v = Math.min(+el.max, v);
    el.value = String(v);
    from.set(el, v);
  }, true);
}
