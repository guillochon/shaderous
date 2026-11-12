/**
 * How much GPU memory a simulation may take.
 *
 * No browser says how much video memory there is. A desktop GPU reports running out (an
 * out-of-memory error, which AmrSim.create answers with a smaller atlas). iOS does not: memory is
 * shared with the CPU and counts against the page's limit, and the system kills the page before
 * WebGPU reports anything. Safari then reloads it, and after a second crash shows "A problem
 * repeatedly occurred".
 *
 * So the budget is guessed from what the browser does say (autoBudget), and the user may set
 * their own (the Performance panel). On every device, each large allocation leaves a note in
 * localStorage until it has survived a few seconds: a note still there when the page loads means
 * the page died holding that much, and the automatic budget drops below it for good. A budget the
 * user set is dropped after such a crash. The reload Safari does after a crash then builds
 * something smaller instead of crashing again.
 */

export const GB = 1024 ** 3;
const MB = 1024 ** 2;
/** A desktop with its own GPU memory: the adaptive mesh's atlas as it was before budgets (13^3 blocks in 3D). */
const DESKTOP_BUDGET = 2.5 * GB;
/** A desktop whose GPU shares the system's memory (Apple silicon, most Intel GPUs). */
const SHARED_BUDGET = 2 * GB;
/** A phone or tablet. */
const PHONE_BUDGET = 1 * GB;
/** The learned budget never drops below this, so something can always run. */
const FLOOR = 128 * MB;
/** After a crash, the budget becomes this fraction of what the page died holding. */
const BACKOFF = 0.5;

const TRY_KEY = 'shaderous.gpuTry';
const CEILING_KEY = 'shaderous.gpuCeiling';
const USER_KEY = 'shaderous.gpuBudget';

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
    // Storage blocked (private browsing): no memory of crashes or of the user's budget.
  }
}

/** iPhone, iPad (which may call itself a Mac), or an Android phone or tablet. */
export function isMobile(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  return /iPhone|iPad|iPod|Android/i.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
}

/** What the adapter says about itself (gpu/device.ts), for telling shared memory from its own. */
let adapter = { vendor: '', architecture: '', description: '' };
export function setAdapter(info: { vendor: string; architecture: string; description: string }) {
  adapter = info;
}

/** A GPU without memory of its own: Apple's, and Intel's other than the discrete Arc cards. */
function sharedMemory(): boolean {
  const v = adapter.vendor.toLowerCase();
  if (v.includes('apple')) return true;
  if (v.includes('intel')) return !/hpg|arc/i.test(`${adapter.architecture} ${adapter.description}`);
  return false;
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
    // The user's own budget was too much for this device.
    set(USER_KEY, null);
    console.warn(`the page died last time holding about ${Math.round(tried / MB)} MB of GPU memory: the budget is now ${Math.round(lowered / MB)} MB`);
  }
  set(TRY_KEY, null);
}
// Leaving the page normally (a reload, a link, closing the tab) is not a crash.
if (typeof addEventListener === 'function') addEventListener('pagehide', () => set(TRY_KEY, null));

/** The budget this device gets unless the user sets one, and why. */
export function autoBudget(): { bytes: number; reason: string } {
  let bytes: number;
  let reason: string;
  if (isMobile()) {
    bytes = PHONE_BUDGET;
    reason = 'phone or tablet';
  } else if (sharedMemory()) {
    bytes = SHARED_BUDGET;
    reason = 'GPU sharing system memory';
  } else {
    bytes = DESKTOP_BUDGET;
    reason = 'desktop';
  }
  // A third of the system's memory, where the browser says how much there is (Chromium only;
  // it reports at most 8 GB, which says nothing about a machine with more).
  const ram = typeof navigator !== 'undefined' ? (navigator as { deviceMemory?: number }).deviceMemory : undefined;
  if (ram && ram < 8 && (ram * GB) / 3 < bytes) {
    bytes = (ram * GB) / 3;
    reason = `a third of ${ram} GB of memory`;
  }
  const learned = Number(get(CEILING_KEY));
  if (learned && learned < bytes) {
    bytes = learned;
    reason = 'lowered after a crash';
  }
  return { bytes, reason };
}

/** The budget the user set (bytes), if any. */
export function userBudget(): number | null {
  const v = Number(get(USER_KEY));
  return v > 0 ? v : null;
}

/** Set the user's budget, or go back to the automatic one (null). */
export function setUserBudget(bytes: number | null) {
  set(USER_KEY, bytes && bytes > 0 ? String(Math.round(bytes)) : null);
}

/** Bytes a simulation may take on this device. */
export function memoryBudget(): number {
  return userBudget() ?? autoBudget().bytes;
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
