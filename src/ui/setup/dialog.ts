// The setup dialog: every setting of a run before it starts, edited as a copy of the running
// document and handed back to the page on Apply (ui/setup/host.ts). Left, the presets, the
// user's library and files; middle, the featured fields and the pages of options, generated
// from the option table (forms.ts); right, a preview drawn from the document alone
// (preview.ts); bottom, what is wrong with the setup, the memory its grid needs, and the
// buttons.
//
// A setting that costs a running simulation nothing (`live`) is applied as it is changed;
// everything else waits for Apply, whose label says what it will do.
import './setup.css';
import { checkSetup } from '../../setup/check';
import { formatSetup, readSetup } from '../../setup/codec';
import { factsOf, factsWithout, flatten, OPTIONS, optionAt, PART_OPTIONS, partFields, unavailable, type Control, type Issue, type Option, type Page, type Value } from '../../setup/options';
import { partIds } from '../../setup/parts/bodies';
import { changeBetween, sameSetup } from '../../setup/running';
import { cloneSetup, emptySetup, featuredOf, featuredPaths, getPath, setPath, type Featured, type Part, type PartKind, type SetupSpec, type Vec3 } from '../../setup/spec';
import type { UnitSystem } from '../../units/units';
import { h, installFactorSteps, renderOptions, type Form, type FormChoice, type FormContext } from './forms';
import type { SetupHost } from './host';
import { getSetup, listSetups, removeSetup, renameSetup, saveSetup } from './library';
import { createPreview, type Preview } from './preview';

const PAGES: { page: Page; label: string; tip: string }[] = [
  { page: 'domain', label: 'Domain and grid', tip: 'Dimensions, resolution, the size of the box and its physical units' },
  { page: 'gas', label: 'Gas', tip: 'Equation of state, composition, burning, cooling and diffusion' },
  { page: 'gravity', label: 'Gravity', tip: 'Uniform gravity, self-gravity and the frame' },
  { page: 'boundaries', label: 'Boundaries', tip: 'What the gas meets at the faces of the box' },
  { page: 'mesh', label: 'Mesh', tip: 'Where the adaptive mesh refines' },
  { page: 'solver', label: 'Solver', tip: 'Riemann solver, limiter and timestep' },
  { page: 'parts', label: 'Objects', tip: 'What is in the box: the ambient gas, point masses, stars, disks, regions, obstacles' },
  { page: 'run', label: 'Run', tip: 'The setup’s name, its clock and how it opens' },
];

const PRESETS_URL = 'https://github.com/guillochon/shaderous/tree/main/src/presets';
const MAX_FILE = 2_000_000;
const MAX_FEATURED = 32;

/** Where the document being edited came from: what "(modified)" is measured against. */
interface Origin {
  kind: 'preset' | 'library' | 'file' | 'new' | 'run';
  /** the library entry's name, for saving back over it */
  label: string;
  spec: SetupSpec;
}

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const safe = <T>(f: () => T, otherwise: T): T => {
  try {
    return f();
  } catch {
    return otherwise;
  }
};
const slug = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'setup';
const bytes = (b: number): string => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(b / 2 ** 20))} MB`);

function download(text: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = h('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A featured list with the parts' places changed: `to` gives a part's new place, or null if it is gone. */
function remapFeatured(spec: SetupSpec, to: (i: number) => number | null): void {
  if (!spec.featured) return;
  const move = (path: string): string | null => {
    const m = /^parts\.(\d+)(\..*)?$/.exec(path);
    if (!m) return path;
    const i = to(+m[1]);
    return i === null ? null : `parts.${i}${m[2] ?? ''}`;
  };
  const out: (string | Featured)[] = [];
  for (const f of spec.featured) {
    if (typeof f === 'string') {
      const p = move(f);
      if (p !== null) out.push(p);
      continue;
    }
    const paths = featuredPaths(f).map(move);
    if (paths.some((p) => p === null)) continue;
    out.push({ ...f, path: Array.isArray(f.path) ? (paths as string[]) : (paths[0] as string) });
  }
  spec.featured = out;
}

/** Every place a document names a part by its id, renamed. */
function renameRefs(spec: SetupSpec, from: string, to: string): void {
  for (const p of spec.parts ?? []) {
    if (p.kind === 'orbit' && Array.isArray(p.of)) p.of = p.of.map((id) => (id === from ? to : id)) as [string, string];
    if (p.kind === 'disk' && Array.isArray(p.about)) p.about = p.about.map((id) => (id === from ? to : id));
  }
  if (spec.background?.relativeTo === from) spec.background.relativeTo = to;
  const u = spec.domain?.units;
  for (const v of [u?.length, u?.mass, u?.density, u?.velocity]) {
    if (v && typeof v.per === 'string' && v.per.split('.')[0] === from) v.per = `${to}.${v.per.split('.').slice(1).join('.')}`;
  }
}

function build() {
  let host!: SetupHost;
  let draft: SetupSpec = emptySetup();
  let origin: Origin = { kind: 'new', label: '', spec: draft };
  let notes: string[] = [];
  let page: Page = 'domain';
  let issues: Issue[] = [];
  let unitsNow: UnitSystem | undefined;
  /** the forms on show, by page; `part` for a part's card */
  let forms: { page: Page; form: Form }[] = [];
  let strip: Form[] = [];
  let stripKey = '';
  let partsKey = '';
  const openParts = new Set<number>();
  let applying = false;
  let liveTimer = 0;
  /** the last document applied without being asked: never sent twice */
  let liveSent = '';
  let frame = 0;

  // ---- the frame ----
  const title = h('h2', { id: 'sd-title', text: 'Setup' });
  const nameEl = h('span', { class: 'sd-name' });
  const modEl = h('span', { class: 'sd-mod', text: '(modified)', title: 'It differs from the preset, library entry or file it came from', hidden: true });
  const closeX = h('button', { type: 'button', class: 'sd-x', text: '×', title: 'Close without applying what waits (Esc)', 'aria-label': 'Close' });
  const colTabs = h('div', { class: 'sd-cols', role: 'tablist', 'aria-label': 'Part of the dialog' });
  const left = h('aside', { class: 'sd-left', 'aria-label': 'Presets, library and files' });
  const mid = h('section', { class: 'sd-mid', 'aria-label': 'Settings' });
  const right = h('aside', { class: 'sd-right', 'aria-label': 'Preview' });
  const main = h('div', { class: 'sd-main', 'data-col': 'mid' }, left, mid, right);
  const issuesEl = h('ul', { class: 'sd-issues', 'aria-label': 'What is wrong with this setup' });
  const memEl = h('div', { class: 'sd-mem' });
  const statusEl = h('div', { class: 'sd-status', role: 'status', 'aria-live': 'polite' });
  const saveFile = h('button', { type: 'button', text: 'Save file', title: 'Download this setup as a .shaderous.json file: the whole document, to keep or to share' });
  const copyLink = h('button', { type: 'button', text: 'Copy link', title: 'Copy a link that opens the simulator with this setup' });
  const saveLib = h('button', { type: 'button', text: 'Save to library', title: 'Keep this setup under its name in this browser. The library lives in this browser only: a file is the durable copy' });
  const cancel = h('button', { type: 'button', text: 'Cancel', title: 'Close and discard what has not been applied. Live settings took effect as they were changed' });
  const apply = h('button', { type: 'button', class: 'sd-apply', text: 'Apply' });
  const el = h('dialog', { class: 'sd', 'aria-labelledby': 'sd-title' },
    h('header', { class: 'sd-top' }, title, nameEl, modEl, colTabs, closeX),
    main,
    h('footer', { class: 'sd-bottom' }, issuesEl, h('div', { class: 'sd-foot' }, memEl, statusEl, h('div', { class: 'sd-actions' }, saveFile, copyLink, saveLib, cancel, apply))),
  );
  document.body.append(el);
  installFactorSteps(el);

  const status = (text: string, bad = false) => {
    statusEl.textContent = text;
    statusEl.classList.toggle('bad', bad);
  };

  // On a narrow screen the three columns are tabs.
  const cols: [string, string, HTMLElement][] = [['left', 'Presets', left], ['mid', 'Settings', mid], ['right', 'Preview', right]];
  const colButtons = cols.map(([key, label]) => {
    const b = h('button', { type: 'button', role: 'tab', text: label, 'aria-selected': String(key === 'mid') });
    b.addEventListener('click', () => showCol(key));
    colTabs.append(b);
    return b;
  });
  function showCol(key: string) {
    main.dataset.col = key;
    cols.forEach(([k], i) => colButtons[i].setAttribute('aria-selected', String(k === key)));
    // A preview that was not on show had no size to draw at.
    if (key === 'right' && el.open) preview.update(draft);
  }

  // ---- left: presets, library, files ----
  const presetsEl = h('div', { class: 'sd-presets' });
  const libraryEl = h('div', { class: 'sd-library' });
  const fileInput = h('input', { type: 'file', accept: '.json,application/json', hidden: true, 'aria-label': 'Open a setup file' });
  const newBtn = h('button', { type: 'button', text: 'New', title: 'An empty setup on the grid of the run: a box of still gas, to build on' });
  const openBtn = h('button', { type: 'button', text: 'Open file…', title: 'Open a .shaderous.json file (or drop one on this dialog)' });
  left.append(
    h('h3', { text: 'Presets' }), presetsEl,
    h('a', { class: 'sd-all', href: PRESETS_URL, target: '_blank', rel: 'noopener', text: 'All preset files on GitHub', title: 'The folder of preset files in the repository, to download them all' }),
    h('h3', { text: 'Your library' }), libraryEl,
    h('p', { class: 'sd-hint', text: 'The library lives in this browser. Files are the durable copy.' }),
    h('h3', { text: 'Files' }), h('div', { class: 'sd-files' }, newBtn, openBtn, fileInput),
  );

  function drawPresets() {
    const groups = safe(() => host.presets(), []);
    presetsEl.replaceChildren(...groups.map((g) =>
      h('div', { class: 'sd-group' }, h('h4', { text: g.group }), ...g.entries.map((p) => {
        const load = h('button', { type: 'button', class: 'sd-entry', text: p.label, title: 'Load this preset into the editor (the run is untouched until Apply)' });
        load.addEventListener('click', () => {
          const spec = safe(() => host.presetSpec(p.id), null);
          if (!spec) return status(`The preset "${p.label}" could not be loaded`, true);
          setDraft(spec, { kind: 'preset', label: p.id, spec }, []);
          showCol('mid');
        });
        const dl = h('button', { type: 'button', class: 'sd-dl', text: '↓', title: `Download ${p.id}.shaderous.json, as it is in the repository`, 'aria-label': `Download the preset ${p.label}` });
        dl.addEventListener('click', () => {
          try {
            download(host.presetFile(p.id), `${p.id}.shaderous.json`);
          } catch (e) {
            status(`Could not download: ${message(e)}`, true);
          }
        });
        return h('div', { class: 'sd-row' }, load, dl);
      })),
    ));
  }

  function drawLibrary() {
    const names = listSetups();
    if (!names.length) return libraryEl.replaceChildren(h('p', { class: 'sd-hint', text: 'Nothing saved yet: Save to library keeps the setup here under its name.' }));
    libraryEl.replaceChildren(...names.map((name) => {
      const load = h('button', { type: 'button', class: 'sd-entry', text: name, title: 'Load this setup into the editor' });
      load.addEventListener('click', () => {
        const got = getSetup(name);
        if (!got) return status(`"${name}" could not be read from the library`, true);
        setDraft(got.spec, { kind: 'library', label: name, spec: got.spec }, got.notes);
        showCol('mid');
      });
      const row = h('div', { class: 'sd-row' }, load);
      const rename = h('button', { type: 'button', class: 'sd-dl', text: '✎', title: 'Rename', 'aria-label': `Rename ${name}` });
      rename.addEventListener('click', () => {
        const input = h('input', { type: 'text', maxlength: 80, 'aria-label': `New name for ${name}` });
        input.value = name;
        let done = false;
        const finish = (keep: boolean) => {
          if (done) return;
          done = true;
          const used = keep ? renameSetup(name, input.value) : null;
          if (used && origin.kind === 'library' && origin.label === name) origin.label = used;
          drawLibrary();
        };
        input.addEventListener('blur', () => finish(true));
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') finish(true);
          if (e.key === 'Escape') {
            // Esc here ends the renaming, not the dialog.
            e.preventDefault();
            e.stopPropagation();
            finish(false);
          }
        });
        load.replaceWith(input);
        input.focus();
        input.select();
      });
      const exp = h('button', { type: 'button', class: 'sd-dl', text: '↓', title: 'Export as a file', 'aria-label': `Export ${name} as a file` });
      exp.addEventListener('click', () => {
        const got = getSetup(name);
        if (got) download(formatSetup(got.spec), `${slug(name)}.shaderous.json`);
        else status(`"${name}" could not be read from the library`, true);
      });
      // Deleting cannot be undone: the button asks once more before it does.
      const del = h('button', { type: 'button', class: 'sd-dl', text: '×', title: 'Delete from the library', 'aria-label': `Delete ${name} from the library` });
      let armed = 0;
      del.addEventListener('click', () => {
        if (!armed) {
          del.textContent = 'Delete?';
          del.classList.add('armed');
          del.title = 'Press again to delete it for good';
          armed = window.setTimeout(() => {
            armed = 0;
            del.textContent = '×';
            del.classList.remove('armed');
            del.title = 'Delete from the library';
          }, 4000);
          return;
        }
        clearTimeout(armed);
        removeSetup(name);
        if (origin.kind === 'library' && origin.label === name) origin = { ...origin, kind: 'file' };
        drawLibrary();
        status(`"${name}" was deleted from the library`);
      });
      row.append(rename, exp, del);
      return row;
    }));
  }

  /** The grid of the run, for a document that names none. */
  function withGrid(spec: SetupSpec): SetupSpec {
    const out = cloneSetup(spec);
    const run = safe(() => host.running().domain, undefined) ?? {};
    const d = (out.domain ??= {});
    if (d.ndim === undefined) d.ndim = d.supports2d === false ? 3 : (run.ndim ?? 3);
    if (d.n === undefined) d.n = d.ndim === run.ndim && run.n ? run.n : d.ndim === 2 ? 512 : 128;
    if (d.amr === undefined) d.amr = d.ndim === run.ndim && d.n === run.n ? (run.amr ?? 0) : 0;
    return out;
  }

  newBtn.addEventListener('click', () => {
    const spec = withGrid({ ...emptySetup(), name: 'New setup' });
    setDraft(spec, { kind: 'new', label: '', spec }, []);
    showCol('mid');
  });
  openBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    fileInput.value = '';
    if (f) void openFile(f);
  });

  async function openFile(file: File) {
    try {
      if (file.size > MAX_FILE) return status(`${file.name} is too large to be a setup`, true);
      const { spec, report } = readSetup(await file.text(), (id) => safe(() => host.presetSpec(id), undefined));
      if (!spec) return status(`${file.name} could not be opened: ${report.error ?? 'it is not a setup'}`, true);
      setDraft(spec, { kind: 'file', label: file.name, spec }, report.notes);
      status(`Opened ${file.name}`);
      showCol('mid');
    } catch (e) {
      status(`${file.name} could not be opened: ${message(e)}`, true);
    }
  }

  // A setup file dropped anywhere on the dialog opens in it.
  el.addEventListener('dragover', (e) => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  el.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    e.preventDefault();
    e.stopPropagation();
    void openFile(f);
  });

  // ---- middle: notes, featured strip, pages ----
  const notesEl = h('div', { class: 'sd-notes', hidden: true });
  const stripEl = h('div', { class: 'sd-strip', 'aria-label': 'Featured fields' });
  const tabsEl = h('div', { class: 'sd-tabs', role: 'tablist', 'aria-label': 'Pages of settings' });
  const panels = new Map<Page, HTMLElement>();
  const tabs = new Map<Page, HTMLButtonElement>();
  const pagesEl = h('div', { class: 'sd-pages' });
  for (const p of PAGES) {
    const tab = h('button', { type: 'button', role: 'tab', id: `sd-tab-${p.page}`, 'aria-controls': `sd-page-${p.page}`, text: p.label, title: p.tip });
    tab.addEventListener('click', () => showPage(p.page));
    tabs.set(p.page, tab);
    tabsEl.append(tab);
    const panel = h('div', { class: 'sd-page', role: 'tabpanel', id: `sd-page-${p.page}`, 'aria-labelledby': `sd-tab-${p.page}`, hidden: true });
    panels.set(p.page, panel);
    pagesEl.append(panel);
  }
  // Left and right move among the tabs, as a tab list does.
  tabsEl.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const at = PAGES.findIndex((p) => p.page === page);
    const next = PAGES[(at + (e.key === 'ArrowRight' ? 1 : PAGES.length - 1)) % PAGES.length].page;
    showPage(next);
    tabs.get(next)!.focus();
    e.preventDefault();
  });
  mid.append(notesEl, stripEl, tabsEl, pagesEl);

  function showPage(p: Page) {
    page = p;
    for (const [k, panel] of panels) panel.hidden = k !== p;
    for (const [k, tab] of tabs) {
      tab.setAttribute('aria-selected', String(k === p));
      tab.tabIndex = k === p ? 0 : -1;
    }
  }

  // ---- right: the preview ----
  const preview: Preview = createPreview({
    onMove(part: number, pos: Vec3) {
      setPath(draft, `parts.${part}.pos`, pos);
      // The picture follows the pointer; the forms and the checks catch up once a frame.
      preview.update(draft);
      if (!frame) frame = requestAnimationFrame(() => {
        frame = 0;
        refresh(false);
      });
    },
  });
  right.append(h('h3', { text: 'Preview' }), preview.el);

  // ---- the document ----
  const isFeatured = (path: string): boolean => (draft.featured ?? []).some((f) => featuredPaths(featuredOf(f)).includes(path));
  function toggleFeatured(path: string) {
    const list = draft.featured ?? [];
    if (isFeatured(path)) draft.featured = list.filter((f) => !featuredPaths(featuredOf(f)).includes(path));
    else if (list.length >= MAX_FEATURED) return status(`A setup features at most ${MAX_FEATURED} fields`, true);
    else draft.featured = [...list, path];
    if (!draft.featured.length) delete draft.featured;
    refresh();
  }

  /** The grids on offer as menus, as the side bar's Resolution and Refinement are. */
  function gridControl(_o: Option, path: string): Control | undefined {
    if (path !== 'domain.n' && path !== 'domain.amr') return undefined;
    const g = safe(() => host.grids(), null);
    if (!g) return undefined;
    const ndim = draft.domain?.ndim === 2 ? 2 : 3;
    const amr = draft.domain?.amr ?? 0;
    const maxLevels = (n: number) => safe(() => g.maxLevels(n), 0);
    if (path === 'domain.n') {
      const uniform = ndim === 2 ? g.uniformMax2d : g.uniformMax3d;
      const choices: FormChoice[] = (ndim === 2 ? g.res2d : g.res3d).map((n) => ({
        value: n, label: `${n}${ndim === 2 ? '²' : '³'}`,
        ...(n > uniform && (amr === 0 || maxLevels(n) === 0) ? { off: 'Too large for the GPU memory as a uniform grid: choose a level of refinement first' } : {}),
      }));
      return { type: 'select', choices };
    }
    const n = draft.domain?.n;
    const most = n ? maxLevels(n) : 5;
    const choices: FormChoice[] = [0, 1, 2, 3, 4, 5].map((l) => ({
      value: l, label: l === 0 ? 'Uniform' : `${l} level${l > 1 ? 's' : ''}`,
      ...(l > most ? { off: `This resolution allows at most ${most} level${most === 1 ? '' : 's'} of refinement` } : {}),
    }));
    return { type: 'select', choices };
  }

  /** After the dimensions or the resolution change: a resolution that grid has, and no more levels than it allows. */
  function fixGrid() {
    const g = safe(() => host.grids(), null);
    const d = draft.domain;
    if (!g || !d) return;
    const ndim = d.ndim === 2 ? 2 : 3;
    const list = ndim === 2 ? g.res2d : g.res3d;
    if (d.n !== undefined && list.length && !list.includes(d.n)) d.n = list.includes(ndim === 2 ? 512 : 128) ? (ndim === 2 ? 512 : 128) : list[0];
    if (d.n !== undefined && d.amr) d.amr = Math.min(d.amr, Math.max(0, safe(() => g.maxLevels(d.n!), 0)));
  }

  function write(path: string, value: unknown) {
    setPath(draft, path, value);
    if (path === 'domain.ndim' || path === 'domain.n') fixGrid();
    // A part's choice that brings fields of its own (a shape's kind, an orbit's) brings their
    // defaults with it: the compiler reads a box's corners, it does not make them up.
    const m = /^parts\.(\d+)\.(.+)$/.exec(path);
    const part = m ? draft.parts?.[+m[1]] : undefined;
    if (m && part) {
      for (const o of flatten(partFields(part))) {
        if (o.show?.path !== m[2] || !o.show.is.includes(value as Value) || o.default === undefined || o.default === null) continue;
        const at = `parts.${m[1]}.${o.path}`;
        if (getPath(draft, at) === undefined) setPath(draft, at, cloneSetup(o.default));
      }
    }
    refresh();
  }

  const context = (part?: number): FormContext => ({
    spec: () => draft,
    read: (p) => getPath(draft, p),
    write,
    facts: () => (part === undefined ? factsOf(draft) : factsWithout(draft, part)),
    units: () => unitsNow,
    featured: isFeatured,
    toggleFeatured,
    control: gridControl,
    part,
  });

  /** Give every part the id it goes by, before the parts change places: ids made from a part's place would shift. */
  function pinIds() {
    const parts = draft.parts ?? [];
    const ids = partIds(parts);
    draft.parts = parts.map((p, i) => {
      const { kind, id: _id, ...rest } = p as Part & Record<string, unknown>;
      return { kind, id: ids[i], ...rest } as Part;
    });
  }

  function freeId(kind: string): string {
    const ids = new Set(partIds(draft.parts ?? []));
    for (let k = 1; ; k++) if (!ids.has(`${kind}${k}`)) return `${kind}${k}`;
  }

  function addPart(kind: PartKind) {
    pinIds();
    const part = PART_OPTIONS[kind].make();
    part.id = freeId(kind);
    const parts = (draft.parts ??= []);
    const ids = partIds(parts);
    const of = (k: PartKind) => parts.flatMap((p, i) => (p.kind === k ? [ids[i]] : []));
    // A new orbit or disk starts about the bodies that are there, where it can.
    if (part.kind === 'orbit' && of('mass').length >= 2) part.of = [of('mass')[0], of('mass')[1]];
    if (part.kind === 'disk') part.about = of('mass').slice(0, 2);
    parts.push(part);
    openParts.add(parts.length - 1);
    refresh();
    cards[parts.length - 1]?.scrollIntoView({ block: 'nearest' });
  }

  function removePart(i: number) {
    pinIds();
    draft.parts!.splice(i, 1);
    if (!draft.parts!.length) delete draft.parts;
    remapFeatured(draft, (k) => (k === i ? null : k > i ? k - 1 : k));
    const open = [...openParts];
    openParts.clear();
    for (const k of open) if (k !== i) openParts.add(k > i ? k - 1 : k);
    refresh();
  }

  function movePart(i: number, by: number) {
    const j = i + by;
    const parts = draft.parts ?? [];
    if (j < 0 || j >= parts.length) return;
    pinIds();
    const list = draft.parts!;
    [list[i], list[j]] = [list[j], list[i]];
    const swap = (k: number) => (k === i ? j : k === j ? i : k);
    remapFeatured(draft, swap);
    const open = [...openParts].map(swap);
    openParts.clear();
    for (const k of open) openParts.add(k);
    refresh();
    cards[j]?.querySelector<HTMLElement>(by < 0 ? '.sd-up' : '.sd-down')?.focus();
  }

  function renamePart(i: number, wanted: string): boolean {
    const id = wanted.trim();
    const ids = partIds(draft.parts ?? []);
    if (id === ids[i]) return true;
    // An id is a word: units and orbits name parts as "id.mass" and "a+b".
    if (!/^[A-Za-z0-9_-]{1,40}$/.test(id)) return status('An id is letters, digits, - and _, without spaces', true), false;
    if (ids.includes(id)) return status(`Another part is already called "${id}"`, true), false;
    pinIds();
    renameRefs(draft, ids[i], id);
    draft.parts![i].id = id;
    status('');
    refresh();
    return true;
  }

  // ---- building the forms ----
  let cards: HTMLElement[] = [];
  const partsList = h('div', { class: 'sd-parts' });
  const addMenu = h('details', { class: 'sd-add' });
  const addWhy = h('div', { class: 'sf-why', hidden: true });
  let generalBuilt = false;
  const general: { page: Page; form: Form }[] = [];

  function buildGeneral() {
    if (generalBuilt) return;
    generalBuilt = true;
    for (const p of PAGES) {
      const panel = panels.get(p.page)!;
      const box = h('div', { class: 'sd-fields' });
      if (p.page === 'parts') panel.append(h('h4', { text: 'The gas the objects sit in' }));
      panel.append(box);
      general.push({ page: p.page, form: renderOptions(box, OPTIONS.filter((o) => o.page === p.page), '', context()) });
    }
    const panel = panels.get('parts')!;
    addMenu.append(h('summary', { text: 'Add an object', title: 'Add a point mass, a star, a disk, a region of gas, an obstacle, an inflow, a magnetic field, an orbit or a built-in flow' }));
    const list = h('div', { class: 'sd-kinds' });
    for (const k of Object.values(PART_OPTIONS)) {
      const b = h('button', { type: 'button', text: k.label, 'data-kind': k.kind });
      b.addEventListener('click', () => {
        // Greyed but not disabled, so a tap can still say why.
        const why = b.dataset.why;
        if (why) {
          addWhy.textContent = `${k.label}: ${why}`;
          addWhy.hidden = false;
          return;
        }
        addWhy.hidden = true;
        addMenu.open = false;
        addPart(k.kind);
      });
      list.append(b);
    }
    addMenu.append(list, addWhy);
    panel.append(h('h4', { text: 'Objects' }), partsList, addMenu);
  }

  function updateAddMenu() {
    const facts = factsOf(draft);
    for (const b of addMenu.querySelectorAll<HTMLButtonElement>('button[data-kind]')) {
      const k = PART_OPTIONS[b.dataset.kind as PartKind];
      const reasons = unavailable(k, facts);
      const have = (draft.parts ?? []).filter((p) => p.kind === k.kind).length;
      if (k.max !== undefined && have >= k.max) reasons.push(`A setup holds at most ${k.max}`);
      const why = reasons.join('. ');
      b.setAttribute('aria-disabled', String(!!why));
      b.title = why || k.tip;
      if (why) b.dataset.why = why;
      else delete b.dataset.why;
    }
  }

  /** The parts' cards, built again when the list itself changed: a part added, removed, moved, renamed, or a flow changed for another. */
  function buildParts() {
    const parts = draft.parts ?? [];
    const ids = partIds(parts);
    forms = forms.filter((f) => general.includes(f));
    cards = parts.map((part, i) => {
      const k = PART_OPTIONS[part.kind];
      const bodyId = `sd-part-${i}`;
      const toggle = h('button', { type: 'button', class: 'sd-toggle', 'aria-expanded': String(openParts.has(i)), 'aria-controls': bodyId, title: k.tip },
        h('span', { class: 'sd-kind', text: k.label }), h('span', { class: 'sd-id', text: ids[i] }));
      const up = h('button', { type: 'button', class: 'sd-up', text: '↑', title: 'Move up: parts are painted in order, later ones over earlier ones', 'aria-label': `Move ${ids[i]} up` });
      const down = h('button', { type: 'button', class: 'sd-down', text: '↓', title: 'Move down: parts are painted in order, later ones over earlier ones', 'aria-label': `Move ${ids[i]} down` });
      const remove = h('button', { type: 'button', class: 'sd-remove', text: 'Remove', title: 'Take this part out of the setup', 'aria-label': `Remove ${ids[i]}` });
      up.disabled = i === 0;
      down.disabled = i === parts.length - 1;
      up.addEventListener('click', () => movePart(i, -1));
      down.addEventListener('click', () => movePart(i, 1));
      remove.addEventListener('click', () => removePart(i));
      const idInput = h('input', { type: 'text', id: `sd-part-id-${i}`, maxlength: 40, spellcheck: 'false' });
      idInput.value = ids[i];
      idInput.addEventListener('change', () => {
        if (!renamePart(i, idInput.value)) idInput.value = partIds(draft.parts ?? [])[i] ?? '';
      });
      const fields = h('div', { class: 'sd-fields' });
      const body = h('div', { class: 'sd-part-body', id: bodyId, hidden: !openParts.has(i) },
        h('label', { class: 'f sd-idrow', for: `sd-part-id-${i}`, title: 'The name orbits, disks and units know this part by. Unique in the setup' }, h('span', { text: 'Id' }), idInput),
        fields);
      toggle.addEventListener('click', () => {
        if (openParts.has(i)) openParts.delete(i);
        else openParts.add(i);
        body.hidden = !openParts.has(i);
        toggle.setAttribute('aria-expanded', String(openParts.has(i)));
      });
      forms.push({ page: 'parts', form: renderOptions(fields, partFields(part), `parts.${i}.`, context(i)) });
      return h('div', { class: 'sd-part', 'data-path': `parts.${i}` }, h('div', { class: 'sd-part-head' }, toggle, up, down, remove), body);
    });
    partsList.replaceChildren(...(cards.length ? cards : [h('p', { class: 'sd-hint', text: 'Nothing but the ambient gas. Add an object below.' })]));
  }

  /** The featured fields: the same controls as on their pages, from the same table. */
  function buildStrip() {
    strip = [];
    const boxes: HTMLElement[] = [];
    for (const entry of draft.featured ?? []) {
      const f = featuredOf(entry);
      const paths = featuredPaths(f);
      const option = optionAt(draft, paths[0]);
      if (!option) continue;
      const m = /^parts\.(\d+)\./.exec(paths[0]);
      const prefix = m ? `parts.${m[1]}.` : '';
      const ctx = context(m ? +m[1] : undefined);
      // Several paths that take the same value are written together.
      if (paths.length > 1) {
        ctx.write = (_p, value) => {
          for (const p of paths) setPath(draft, p, value);
          refresh();
        };
      }
      const box = h('div', { class: 'sd-fields' });
      const siblings = m ? flatten(partFields(draft.parts![+m[1]])) : undefined;
      strip.push(renderOptions(box, [{ ...option, label: f.label ?? option.label, tip: f.tip ?? option.tip }], prefix, ctx, siblings));
      boxes.push(box);
    }
    stripEl.replaceChildren(...(boxes.length ? [h('h4', { text: 'Featured', title: 'The fields this setup shows beside the preset menu. The star on a field adds or removes it' }), h('div', { class: 'sd-strip-in' }, ...boxes)] : []));
    stripEl.hidden = !boxes.length;
  }

  // ---- keeping everything in step ----
  function change() {
    try {
      return changeBetween(host.running(), draft);
    } catch {
      // Two documents that cannot be compared are taken to differ in everything.
      return { cost: 'restart' as const };
    }
  }

  function updateApply() {
    const cost = change().cost;
    const errors = issues.some((i) => i.level === 'error');
    apply.textContent = applying ? 'Applying…' : cost === 'shaders' ? 'Apply and recompile' : cost === 'restart' ? 'Apply and restart' : 'Apply';
    apply.disabled = applying || cost === 'none' || errors;
    apply.title = errors ? 'The setup has errors: see the list' : cost === 'none' ? 'Nothing differs from the run' : cost === 'live' ? 'The run carries on with the new values' : cost === 'shaders' ? 'The shaders are rebuilt and the run carries on' : 'The run starts again from this setup’s initial conditions';
    return { cost, errors };
  }

  function drawIssues() {
    issuesEl.replaceChildren(...issues.map((issue) => {
      const mark = issue.level === 'error' ? 'Error' : 'Warning';
      if (!issue.path) return h('li', { class: issue.level }, h('span', { text: `${mark}: ${issue.text}` }));
      const b = h('button', { type: 'button', text: `${mark}: ${issue.text}`, title: 'Go to the field this is about' });
      b.addEventListener('click', () => goTo(issue.path!));
      return h('li', { class: issue.level }, b);
    }));
    issuesEl.hidden = !issues.length;
  }

  function goTo(path: string) {
    const part = /^parts\.(\d+)/.exec(path);
    let target: Page | undefined = part ? 'parts' : optionAt(draft, path)?.page;
    if (!target) target = flatten(OPTIONS).filter((o) => path.startsWith(o.path) || o.path.startsWith(path)).sort((a, b) => b.path.length - a.path.length)[0]?.page;
    showPage(target ?? page);
    showCol('mid');
    if (part) {
      const i = +part[1];
      if (!openParts.has(i)) cards[i]?.querySelector<HTMLElement>('.sd-toggle')?.click();
    }
    for (const f of forms) if (f.page === page && f.form.reveal(path)) return;
    if (part) {
      cards[+part[1]]?.scrollIntoView({ block: 'center' });
      cards[+part[1]]?.querySelector<HTMLElement>('.sd-toggle')?.focus();
    }
  }

  /** Everything on show brought in step with the document. `draw` false leaves the preview, which is already there. */
  function refresh(draw = true) {
    // A card per part, and each part's fields its kind's: built again only when that changed.
    const parts = draft.parts ?? [];
    const key = JSON.stringify(parts.map((p, i) => [p.kind, partIds(parts)[i], p.kind === 'flow' ? p.name : 0]));
    if (key !== partsKey) {
      partsKey = key;
      buildParts();
    }
    const sKey = JSON.stringify([draft.featured ?? [], key]);
    if (sKey !== stripKey) {
      stripKey = sKey;
      buildStrip();
    }
    unitsNow = safe(() => (host.physical() ? host.units(draft) : undefined), undefined);
    for (const f of forms) f.form.update();
    for (const f of strip) f.update();
    updateAddMenu();
    if (draw) preview.update(draft);

    const memory = safe(() => host.memory(draft), null);
    issues = checkSetup(draft, { memory });
    drawIssues();
    memEl.textContent = memory ? `Grid: ${bytes(memory.need)} of ${bytes(memory.budget)} GPU memory` : '';
    memEl.classList.toggle('bad', !!memory && memory.need > memory.budget);
    memEl.title = 'GPU memory the grid of this setup needs, and what the page may take (Performance, in the side bar)';

    nameEl.textContent = draft.name ?? 'Custom setup';
    modEl.hidden = safe(() => sameSetup(draft, origin.spec), false);
    notesEl.hidden = !notes.length;
    const { cost, errors } = updateApply();
    // A setting that costs the run nothing takes effect as it is changed.
    clearTimeout(liveTimer);
    if (el.open && cost === 'live' && !errors && !applying && JSON.stringify(draft) !== liveSent) liveTimer = window.setTimeout(() => void send(false), 300);
  }

  /** Hand the document to the page. Asked for (the button), the dialog then starts again from what runs. */
  async function send(asked: boolean) {
    if (applying) return;
    clearTimeout(liveTimer);
    applying = true;
    const sent = cloneSetup(draft);
    const text = JSON.stringify(draft);
    if (!asked) liveSent = text;
    updateApply();
    try {
      await host.apply(sent);
      if (asked) {
        status('Applied');
        // What runs is the page's to say: the grid it settled on, the values it held.
        if (JSON.stringify(draft) === text) setDraft(host.running(), origin, notes, true, true);
      }
    } catch (e) {
      status(`Could not apply: ${message(e)}`, true);
    } finally {
      applying = false;
      refresh(false);
    }
  }

  /**
   * Start editing a document. `asIs` takes it as it stands (what runs has its grid); otherwise a
   * document that names no grid gets the run's. `keep` leaves the cards and the status line as
   * they are (after Apply).
   */
  function setDraft(spec: SetupSpec, from: Origin, dropped: string[], asIs = false, keep = false) {
    draft = asIs ? cloneSetup(spec) : withGrid(spec);
    origin = from;
    notes = dropped;
    notesEl.replaceChildren(h('strong', { text: 'Reading the file changed it:' }), h('ul', {}, ...dropped.map((n) => h('li', { text: n }))));
    if (!keep) {
      openParts.clear();
      // A short list opens whole; a long one stays folded until a part is asked for.
      if ((draft.parts ?? []).length <= 2) (draft.parts ?? []).forEach((_, i) => openParts.add(i));
      status('');
    }
    liveSent = '';
    partsKey = stripKey = '';
    buildGeneral();
    forms = [...general];
    refresh();
  }

  // ---- the buttons ----
  closeX.addEventListener('click', () => el.close());
  cancel.addEventListener('click', () => el.close());
  el.addEventListener('close', () => clearTimeout(liveTimer));
  apply.addEventListener('click', () => void send(true));
  saveFile.addEventListener('click', () => {
    try {
      download(formatSetup(draft), `${slug(draft.name ?? 'setup')}.shaderous.json`);
      status('Saved as a file');
    } catch (e) {
      status(`Could not save: ${message(e)}`, true);
    }
  });
  copyLink.addEventListener('click', async () => {
    try {
      const { url, tooLong } = await host.link(draft);
      if (tooLong) {
        status(`This setup makes a link of ${url.length} characters, too long for most places a link goes. Save file keeps all of it`, true);
        saveFile.focus();
        return;
      }
      await navigator.clipboard.writeText(url);
      status('Link copied');
    } catch (e) {
      status(`Could not copy a link (${message(e).replace(/\.$/, '')}). Save file keeps the setup`, true);
    }
  });
  saveLib.addEventListener('click', () => {
    const name = draft.name ?? 'Untitled setup';
    const used = saveSetup(name, draft, { replace: origin.kind === 'library' && origin.label === name });
    if (!used) return status('This browser would not store the setup. Save file keeps it', true);
    draft.name = used;
    origin = { kind: 'library', label: used, spec: cloneSetup(draft) };
    drawLibrary();
    refresh(false);
    status(`Saved to the library as "${used}". The library lives in this browser: a file is the durable copy`);
  });

  return {
    el,
    open(to: SetupHost, opts: { file?: SetupSpec; notes?: string[] } = {}) {
      host = to;
      drawPresets();
      drawLibrary();
      if (opts.file) setDraft(opts.file, { kind: 'file', label: opts.file.name ?? '', spec: opts.file }, opts.notes ?? []);
      else {
        // What runs came from a preset if it still carries one's id: "(modified)" is measured against that.
        const running = host.running();
        const preset = running.id ? safe(() => host.presetSpec(running.id!), null) : null;
        setDraft(running, { kind: preset ? 'preset' : 'run', label: running.id ?? '', spec: preset ?? running }, [], true);
      }
      showPage(page);
      if (!el.open) {
        el.showModal();
        // Drawn again now that the canvases have a size.
        preview.update(draft);
      }
    },
    close() {
      if (el.open) el.close();
    },
  };
}

let dialog: ReturnType<typeof build> | null = null;

/**
 * Open the setup dialog on the running document, or on a file just read (with what reading it
 * dropped or held, to show). Made once and used again; opening it while it is open loads the
 * file into it.
 */
export function openSetupDialog(host: SetupHost, opts: { file?: SetupSpec; notes?: string[] } = {}): void {
  (dialog ??= build()).open(host, opts);
}

/** Close the dialog, discarding what was not applied. */
export function closeSetupDialog(): void {
  dialog?.close();
}
