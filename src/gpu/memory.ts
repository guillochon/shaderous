/**
 * How much GPU memory a simulation may take.
 *
 * No browser says how much video memory there is. A desktop GPU reports running out (an
 * out-of-memory error, which AmrSim.create answers with a smaller atlas). iOS does not: memory is
 * shared with the CPU and counts against the page's limit, and the system kills the page before
 * WebGPU reports anything. Safari then reloads it, and after a second crash shows "A problem
 * repeatedly occurred".
 *
 * So on a phone or tablet the simulation stays under PHONE_BUDGET. On every device, each large
 * allocation leaves a note in localStorage until it has survived a few seconds: a note still there
 * when the page loads means the page died holding that much, and the budget drops below it for
 * good. The reload Safari does after a crash then builds something smaller instead of crashing again.
 */

const GB = 1024 ** 3;
const MB = 1024 ** 2;
/** Most a phone or tablet run may take. */
export const PHONE_BUDGET = 1 * GB;
/** The learned budget never drops below this, so something can always run. */
const FLOOR = 128 * MB;
/** After a crash, the budget becomes this fraction of what the page died holding. */
const BACKOFF = 0.5;

const TRY_KEY = 'shaderous.gpuTry';
const CEILING_KEY = 'shaderous.gpuCeiling';

function get(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function set(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // Storage blocked (private browsing): no memory of crashes, the fixed budget still applies.
  }
}

/** iPhone, iPad (which may call itself a Mac), or an Android phone or tablet. */
export function isMobile(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  return /iPhone|iPad|iPod|Android/i.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
}

// A note left by the last page means it died before the allocation proved safe.
let crashedAt = 0;
{
  const tried = Number(get(TRY_KEY));
  if (tried > 0) {
    crashedAt = tried;
    const old = Number(get(CEILING_KEY)) || Infinity;
    const lowered = Math.max(FLOOR, Math.floor(Math.min(old, tried) * BACKOFF));
    set(CEILING_KEY, String(lowered));
    console.warn(`the page died last time holding about ${Math.round(tried / MB)} MB of GPU memory: the budget is now ${Math.round(lowered / MB)} MB`);
  }
  set(TRY_KEY, null);
}
// Leaving the page normally (a reload, a link, closing the tab) is not a crash.
if (typeof addEventListener === 'function') addEventListener('pagehide', () => set(TRY_KEY, null));

/** Bytes a simulation may take on this device: Infinity on a desktop that has never crashed. */
export function memoryBudget(): number {
  const learned = Number(get(CEILING_KEY)) || Infinity;
  return Math.min(isMobile() ? PHONE_BUDGET : Infinity, learned);
}

/** The page died last time, holding this many bytes (0 if it did not). */
export function lastCrash(): number {
  return crashedAt;
}

/** About to allocate this many bytes: leave a note in case the page dies doing it. */
export function noteAllocation(bytes: number) {
  set(TRY_KEY, String(Math.round(bytes)));
}

/** The last noted allocation has run long enough to count as safe. */
export function allocationSurvived() {
  set(TRY_KEY, null);
}
