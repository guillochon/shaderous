// The setup menu's own popup: the native menu cannot hold a search box, so a click on it opens
// this list instead, with a box above it that filters the setups (and their variants) as the
// name is typed. The native <select> stays the record of what is picked; the keyboard still
// steps through it as before.

/** How well a query matches a text, and the characters of the text it matched. */
export interface Match {
  score: number;
  hits: number[];
}

const isWordStart = (s: string, i: number) => i === 0 || /[\s\-_:/(),.]/.test(s[i - 1]) || (/[a-z]/.test(s[i - 1]) && /[A-Z0-9]/.test(s[i]));

/** Edit distance with transpositions, given up past `max`. */
function editDistance(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const d: number[][] = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array<number>(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const c = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
    }
  }
  return d[a.length][b.length];
}

/** One word of the query against the text: a run of it, its letters in order, or a near miss. */
function matchToken(tok: string, text: string, low: string): Match | null {
  // A run of the word, best where a word of the text starts.
  let best: Match | null = null;
  for (let at = low.indexOf(tok); at >= 0; at = low.indexOf(tok, at + 1)) {
    // Inside a word it counts for less than the word's initials would ("kh" in "Chandrasekhar").
    const score = isWordStart(text, at) ? 12 * tok.length + 20 + (at === 0 ? 10 : 0) : 5 * tok.length;
    if (!best || score > best.score) best = { score, hits: Array.from({ length: tok.length }, (_, k) => at + k) };
  }
  if (best) return best;
  // Its letters in order: each one taken at the next start of a word if there is one before the
  // next plain match, so "kh" finds Kelvin-Helmholtz; letters together and word starts score.
  const hits: number[] = [];
  let from = 0;
  for (const ch of tok) {
    let at = -1;
    for (let i = from; i < low.length; i++) {
      if (low[i] !== ch) continue;
      if (at < 0) at = i;
      if (isWordStart(text, i)) {
        at = i;
        break;
      }
      // A plain match right after the last one is as good as a word start.
      if (hits.length && i === hits[hits.length - 1] + 1) break;
    }
    if (at < 0) {
      hits.length = 0;
      break;
    }
    hits.push(at);
    from = at + 1;
  }
  if (hits.length === tok.length) {
    let score = 0;
    hits.forEach((h, k) => {
      score += 2;
      if (isWordStart(text, h)) score += 6;
      else if (k && h === hits[k - 1] + 1) score += 4;
      else if (k) score -= Math.min(3, (h - hits[k - 1]) * 0.25);
    });
    return { score, hits };
  }
  // A misspelling: a word of the text within one edit (two for long words).
  if (tok.length < 4) return null;
  const max = tok.length >= 7 ? 2 : 1;
  const words = /[a-z0-9]+/g;
  let near: Match | null = null;
  for (let m = words.exec(low); m; m = words.exec(low)) {
    // Against the word and its start as long as the query word, so a typo in a prefix counts.
    const w = m[0];
    const dist = Math.min(editDistance(tok, w, max), w.length > tok.length ? editDistance(tok, w.slice(0, tok.length), max) : max + 1);
    if (dist > max) continue;
    const score = 6 * tok.length - 8 * dist;
    if (!near || score > near.score) near = { score, hits: Array.from({ length: Math.min(w.length, tok.length) }, (_, k) => m!.index + k) };
  }
  return near;
}

const tokens = (query: string) => query.toLowerCase().split(/\s+/).filter(Boolean);

/**
 * Score a query against a text: every word of the query must match somewhere in it. Null when
 * one does not. A higher score is a closer match.
 */
export function fuzzyMatch(query: string, text: string): Match | null {
  const m = matchParts(query, [text]);
  return m && { score: m.score, hits: m.hits[0] };
}

/**
 * A query against several texts (a variant's name and its setup's, say): each word of it matched
 * whole within one of them, never strewn across two. `used` says which texts a word matched in;
 * `penalty` is taken off a word's score for each text, and `whole` lets a text match only on
 * runs of a word (no initials, no typos).
 */
function matchParts(query: string, parts: string[], penalty: number[] = [], whole: boolean[] = []) {
  const toks = tokens(query);
  const lows = parts.map((t) => t.toLowerCase());
  const hits: number[][] = parts.map(() => []);
  const used = parts.map(() => false);
  let score = 0;
  for (const t of toks) {
    let best: { m: Match; k: number } | null = null;
    parts.forEach((text, k) => {
      if (whole[k] && !lows[k].includes(t)) return;
      const m = matchToken(t, text, lows[k]);
      if (m && (!best || m.score - (penalty[k] ?? 0) > best.m.score - (penalty[best.k] ?? 0))) best = { m, k };
    });
    if (!best) return null;
    const { m, k } = best as { m: Match; k: number };
    score += m.score - (penalty[k] ?? 0);
    hits[k].push(...m.hits);
    used[k] = true;
  }
  // The whole text asked for, or most of it, beats a long name that merely contains the words.
  const asked = toks.join(' ').length;
  score -= Math.max(0, lows.filter((_, k) => used[k]).join(' ').length - asked) * 0.05;
  return { score, hits: hits.map((h) => [...new Set(h)].sort((x, y) => x - y)), used };
}

/** A variant's row ranks a little below its setup's name matched as well. */
const VARIANT_PENALTY = 12;

/** An entry of the menu: a preset (with its variants) or one of the user's own setups. */
export interface SetupEntry {
  value: string;
  label: string;
  group: string;
  variants: { id: string; label: string }[];
}

/** A row of the list: a setup, or a variant of one (indented under it). */
export interface Row {
  value: string;
  variant?: string;
  label: string;
  hits: number[];
  group?: string;
  dim?: boolean;
  score?: number;
}

/**
 * The rows the list shows for a query. Empty, the menu as it stands, under its group headings.
 * Otherwise the setups that match, closest first, each followed by its variants that match;
 * a setup only some of whose variants match is shown dimmed above them.
 */
export function searchRows(entries: SetupEntry[], query: string): Row[] {
  if (!query.trim()) {
    const rows: Row[] = [];
    let group = '';
    for (const e of entries) {
      rows.push({ value: e.value, label: e.label, hits: [], group: e.group !== group ? e.group : undefined });
      group = e.group;
    }
    return rows;
  }
  const found: { score: number; order: number; rows: Row[] }[] = [];
  entries.forEach((e, order) => {
    // The setup's own name, helped a little by its group's ("stars" finds what is filed under
    // Stars and planets), but at least one word must be in the name unless all are in the group.
    const m = matchParts(query, [e.label, e.group], [0, 15], [false, true]);
    const self = m && { score: m.score, hits: m.hits[0] };
    const vrows: Row[] = [];
    // A variant matches with a word of its own name at least, the rest perhaps in the setup's
    // ("tidal full"); one matched only by the setup's name is the setup's match.
    if (e.variants.length > 1) {
      for (const v of e.variants) {
        const vm = matchParts(query, [v.label, e.label]);
        if (vm?.used[0]) vrows.push({ value: e.value, variant: v.id, label: v.label, hits: vm.hits[0], score: vm.score - VARIANT_PENALTY });
      }
      vrows.sort((a, b) => b.score! - a.score!);
    }
    if (!self && !vrows.length) return;
    found.push({
      score: Math.max(self?.score ?? -Infinity, vrows[0]?.score ?? -Infinity),
      order,
      rows: [{ value: e.value, label: e.label, hits: self?.hits ?? [], dim: !self, score: self?.score }, ...vrows],
    });
  });
  found.sort((a, b) => b.score - a.score || a.order - b.order);
  return found.flatMap((f) => f.rows);
}

/** The row a search starts on: the closest setup, or a variant of it that matched better. */
export function bestRow(rows: Row[], query: string): number {
  if (!query.trim()) return -1;
  const first = rows.findIndex((r) => !r.dim);
  if (first < 0 || rows[first].variant) return first;
  let best = first;
  for (let i = first + 1; i < rows.length && rows[i].variant && rows[i].value === rows[first].value; i++) {
    if ((rows[i].score ?? -Infinity) > (rows[best].score ?? -Infinity)) best = i;
  }
  return best;
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const marked = (label: string, hits: number[]) => {
  const on = new Set(hits);
  let out = '';
  let open = false;
  [...label].forEach((ch, i) => {
    if (on.has(i) !== open) {
      out += open ? '</b>' : '<b>';
      open = !open;
    }
    out += esc(ch);
  });
  return out + (open ? '</b>' : '');
};

/**
 * Give the setup menu its searchable popup. `entries` lists the menu as it stands when the popup
 * opens; `pick` takes the setup (and the variant, when one was picked) chosen in it.
 */
export function installSetupSearch(select: HTMLSelectElement, button: HTMLButtonElement, entries: () => SetupEntry[], pick: (value: string, variant?: string) => void) {
  const pop = document.createElement('div');
  pop.id = 'problem-pop';
  pop.hidden = true;
  pop.innerHTML = `<input type="search" placeholder="Search setups and variants" aria-label="Search setups" spellcheck="false" autocomplete="off" />
    <div class="list" role="listbox" tabindex="-1" aria-label="Setups"></div>`;
  document.body.append(pop);
  const input = pop.querySelector('input')!;
  const list = pop.querySelector<HTMLDivElement>('.list')!;
  let all: SetupEntry[] = [];
  let rows: Row[] = [];
  let active = -1;

  const place = () => {
    const r = select.getBoundingClientRect();
    const right = button.getBoundingClientRect().right;
    const width = Math.max(right - r.left, 240);
    const below = innerHeight - r.bottom - 8;
    const above = r.top - 8;
    // Under the menu, unless there is more room above it (the phone's sheet sits at the bottom).
    const down = below >= 260 || below >= above;
    const room = Math.max(160, Math.min(down ? below : above, 480));
    pop.style.left = `${Math.max(4, Math.min(r.left, innerWidth - width - 4))}px`;
    pop.style.width = `${Math.min(width, innerWidth - 8)}px`;
    pop.style.maxHeight = `${room}px`;
    pop.style.top = down ? `${r.bottom + 2}px` : '';
    pop.style.bottom = down ? '' : `${innerHeight - r.top + 2}px`;
  };

  const setActive = (i: number, scroll = true) => {
    list.querySelector('.active')?.classList.remove('active');
    active = i;
    const el = list.querySelector<HTMLElement>(`[data-i="${i}"]`);
    el?.classList.add('active');
    if (el) list.setAttribute('aria-activedescendant', el.id);
    if (el && scroll) el.scrollIntoView({ block: 'nearest' });
  };

  const current = () => {
    const variant = (document.getElementById('variant') as HTMLSelectElement | null)?.value;
    return rows.findIndex((r) => r.value === select.value && (!r.variant || r.variant === variant) && !r.dim);
  };

  const render = () => {
    const q = input.value;
    rows = searchRows(all, q);
    list.innerHTML = rows.length
      ? rows.map((r, i) =>
        (r.group ? `<div class="grp" role="presentation">${esc(r.group)}</div>` : '') +
        `<div class="opt${r.variant ? ' var' : ''}${r.dim ? ' dim' : ''}${r.value === select.value && !r.variant && !q.trim() ? ' cur' : ''}" role="option" id="pp-opt-${i}" data-i="${i}">${marked(r.label, r.hits)}</div>`).join('')
      : '<div class="none">No setup matches</div>';
    setActive(q.trim() ? bestRow(rows, q) : current());
    if (!q.trim()) list.querySelector<HTMLElement>('.cur')?.scrollIntoView({ block: 'center' });
    else list.scrollTop = 0;
  };

  const close = (refocus = true) => {
    if (pop.hidden) return;
    pop.hidden = true;
    select.setAttribute('aria-expanded', 'false');
    if (refocus) select.focus();
  };
  const open = (search: boolean) => {
    if (select.disabled) return;
    all = entries();
    input.value = '';
    pop.hidden = false;
    select.setAttribute('aria-expanded', 'true');
    place();
    render();
    (search ? input : list).focus({ preventScroll: true });
  };
  const choose = (i: number) => {
    const r = rows[i];
    if (!r) return;
    close();
    const variant = (document.getElementById('variant') as HTMLSelectElement | null)?.value;
    // Picking what is already picked changes nothing, as in the browser's menu.
    if (r.value !== select.value || (r.variant && r.variant !== variant)) pick(r.value, r.variant);
  };

  // A click on the menu opens this list instead of the browser's.
  select.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    if (pop.hidden) open(false);
    else close();
  });
  // Keys that open a menu open this one; the arrows still step through the setups as before.
  select.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'F4' || (e.altKey && (e.key === 'ArrowDown' || e.key === 'ArrowUp'))) {
      e.preventDefault();
      open(false);
    }
  });
  button.addEventListener('click', () => open(true));

  input.addEventListener('input', render);
  pop.addEventListener('keydown', (e) => {
    // Keys here are the list's, not the page's (space runs, Esc puts a tool away, ↑ ↓ move it).
    e.stopPropagation();
    const step = (d: number) => {
      if (!rows.length) return;
      let i = active;
      do i = (i + d + rows.length) % rows.length;
      while (rows[i].dim && i !== active);
      setActive(i);
    };
    if (e.key === 'ArrowDown') step(1);
    else if (e.key === 'ArrowUp') step(-1);
    else if (e.key === 'Home' && e.target !== input) setActive(rows.findIndex((r) => !r.dim));
    else if (e.key === 'End' && e.target !== input) setActive(rows.length - 1);
    else if (e.key === 'PageDown' || e.key === 'PageUp') {
      const n = Math.max(1, Math.floor(list.clientHeight / 26));
      setActive(Math.max(0, Math.min(rows.length - 1, active + (e.key === 'PageDown' ? n : -n))));
    } else if (e.key === 'Enter') choose(active);
    else if (e.key === 'Escape') close();
    else if (e.key === 'Tab') close(false);
    else if (e.target !== input && e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
      // Typing while the list has the keyboard types into the search box (a key's character
      // does not follow the focus moved while it is pressed, so it is put there here).
      input.focus();
      input.value += e.key;
      render();
    } else if (e.target !== input && e.key === 'Backspace') {
      input.focus();
      input.value = input.value.slice(0, -1);
      render();
    } else return;
    e.preventDefault();
  });
  list.addEventListener('mousemove', (e) => {
    const el = (e.target as HTMLElement).closest<HTMLElement>('.opt');
    if (el && +el.dataset.i! !== active) setActive(+el.dataset.i!, false);
  });
  list.addEventListener('click', (e) => {
    const el = (e.target as HTMLElement).closest<HTMLElement>('.opt');
    if (el) choose(+el.dataset.i!);
  });
  // A press anywhere else closes it; so does the page moving under it.
  document.addEventListener('pointerdown', (e) => {
    const t = e.target as Node;
    if (!pop.hidden && !pop.contains(t) && t !== select && t !== button && !button.contains(t)) close(false);
  }, true);
  addEventListener('resize', () => close(false));
  document.addEventListener('scroll', (e) => {
    if (!pop.hidden && !pop.contains(e.target as Node)) close(false);
  }, true);
}
