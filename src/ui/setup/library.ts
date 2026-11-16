// The user's own setups, by name, in the browser's localStorage: one JSON object, name ->
// document, under one key. The store is somebody's browser: it may be missing, full, switched
// off (reading it then throws) or hold anything at all, so nothing here throws and every
// document read back goes through readSetup like a file from a stranger. Files are the durable
// copy; the dialog says so.
import { readSetup } from '../../setup/codec';
import { cloneSetup, type SetupSpec } from '../../setup/spec';

export const LIBRARY_KEY = 'shaderous.setups';
const MAX_NAME = 80;

/** The part of Storage the library uses, so a test can pass its own. */
export type SetupStore = Pick<Storage, 'getItem' | 'setItem'>;

/** The browser's store, or null where there is none or it may not be touched. */
function browserStore(): SetupStore | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage;
  } catch {
    return null;
  }
}

/** Everything stored, in the order it was saved; empty for a missing or corrupt store. */
function load(store: SetupStore | null): Map<string, unknown> {
  const out = new Map<string, unknown>();
  if (!store) return out;
  try {
    const raw = store.getItem(LIBRARY_KEY);
    if (!raw) return out;
    const all: unknown = JSON.parse(raw);
    if (all === null || typeof all !== 'object' || Array.isArray(all)) return out;
    for (const [name, doc] of Object.entries(all)) if (doc !== null && typeof doc === 'object') out.set(name, doc);
  } catch {
    // Not JSON, or a store that refuses: an empty library.
  }
  return out;
}

function store_(all: Map<string, unknown>, store: SetupStore | null): boolean {
  if (!store) return false;
  try {
    store.setItem(LIBRARY_KEY, JSON.stringify(Object.fromEntries(all)));
    return true;
  } catch {
    // Full, or private browsing.
    return false;
  }
}

const tidy = (name: string): string => name.replace(/\s+/g, ' ').trim().slice(0, MAX_NAME);

/** A name not yet taken: the one asked for, or it with the first free number after it. */
function free(name: string, all: Map<string, unknown>): string {
  if (!all.has(name)) return name;
  for (let k = 2; ; k++) {
    const next = `${name} (${k})`;
    if (!all.has(next)) return next;
  }
}

/** The names of the stored setups, in the order they were saved. */
export function listSetups(store: SetupStore | null = browserStore()): string[] {
  return [...load(store).keys()];
}

/** A stored setup, read as a file would be: the document and what reading it dropped or held; null if there is none or it cannot be read. */
export function getSetup(name: string, store: SetupStore | null = browserStore()): { spec: SetupSpec; notes: string[] } | null {
  const doc = load(store).get(name);
  if (doc === undefined) return null;
  try {
    const { spec, report } = readSetup(doc);
    return spec ? { spec, notes: report.notes } : null;
  } catch {
    return null;
  }
}

/**
 * Store a setup under a name and return the name used: the one given, tidied ("Untitled setup"
 * for an empty one), with a number after it if it is taken and `replace` is not set. Null if
 * the browser would not store it.
 */
export function saveSetup(name: string, spec: SetupSpec, opts: { replace?: boolean; store?: SetupStore | null } = {}): string | null {
  const store = opts.store === undefined ? browserStore() : opts.store;
  const all = load(store);
  const wanted = tidy(name) || 'Untitled setup';
  const used = opts.replace ? wanted : free(wanted, all);
  // The library's name for it is the document's own, and a library entry is no preset.
  const doc = cloneSetup(spec);
  doc.name = used;
  delete doc.id;
  delete doc.extends;
  all.set(used, doc);
  return store_(all, store) ? used : null;
}

/** Give a stored setup another name, keeping its place in the list. Returns the name it now has, or null if nothing changed. */
export function renameSetup(from: string, to: string, store: SetupStore | null = browserStore()): string | null {
  const all = load(store);
  const wanted = tidy(to);
  if (!all.has(from) || !wanted || wanted === from) return null;
  const used = free(wanted, all);
  const next = new Map<string, unknown>();
  for (const [name, doc] of all) {
    if (name !== from) next.set(name, doc);
    else next.set(used, doc !== null && typeof doc === 'object' ? { ...doc, name: used } : doc);
  }
  return store_(next, store) ? used : null;
}

/** Remove a stored setup. False if it was not there or the browser would not store the rest. */
export function removeSetup(name: string, store: SetupStore | null = browserStore()): boolean {
  const all = load(store);
  if (!all.delete(name)) return false;
  return store_(all, store);
}
