// What a load is doing, for the page's loading notice: the simulations and the page say which
// stage they are at, pipelines count themselves as they compile, and the page reads both. Nothing
// here touches the page, so the simulations report the same way where there is none (tests).

let stage = '';
let done = 0;
let total = 0;
let listener: (() => void) | null = null;
let painter: (() => Promise<void>) | null = null;

/** The page's hooks: `changed` when there is something new to show, `painted` resolves once it is on screen. */
export function watchLoading(changed: () => void, painted: () => Promise<void>) {
  listener = changed;
  painter = painted;
}

/** The stage a load is at, as a phrase without a full stop ('' for none). */
export const loadStage = () => stage;

/** Name the stage a load has reached ('' once it is over). */
export function setLoadStage(text: string) {
  if (text === stage) return;
  stage = text;
  listener?.();
}

/** Leave a stage, unless another has been named since. */
export function endLoadStage(text: string) {
  if (stage === text) setLoadStage('');
}

/**
 * Name a stage that is about to block the page (initial conditions worked out on the CPU), and
 * wait for the notice to show it first.
 */
export async function showLoadStage(text: string) {
  setLoadStage(text);
  await painter?.();
}

// A load asks for its pipelines in rounds (later ones once earlier ones are in), so the count
// alone would show a total that keeps growing. Each setup's total is remembered instead, from the
// last time it loaded, and shown from the start. Pipelines found already compiled count too, as
// done at once, so a setup asks for the same number however many of them are cached.
const EXPECTED_KEY = 'shaderous.pipelineCounts';
const EXPECTED_MAX = 200;
/** The setup being compiled ('' for none named), and what it asked for since the count started over. */
let setup = '';
let asked = 0;
/** Pipelines still compiling from before the count started over. */
let carried = 0;
let expected = 0;

function readExpected(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(EXPECTED_KEY) ?? '{}') ?? {};
  } catch {
    return {};
  }
}

function rememberSetup() {
  if (!setup || !asked) return;
  try {
    const all = readExpected();
    delete all[setup]; // most recent last, so the oldest is the first one dropped
    all[setup] = asked;
    const keys = Object.keys(all);
    for (const k of keys.slice(0, Math.max(0, keys.length - EXPECTED_MAX))) delete all[k];
    localStorage.setItem(EXPECTED_KEY, JSON.stringify(all));
  } catch {
    // No storage (tests, a private window): the total grows as it is found.
  }
}

/** A short name for a setup's description. */
function hash(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(36) + s.length.toString(36);
}

/**
 * Start the count of pipelines over, at the start of a load. A load asks for its pipelines in
 * several rounds, so the count runs across all of them rather than starting over whenever none
 * are left. The setup the last count compiled keeps its total for next time.
 */
export function resetCompileCount() {
  rememberSetup();
  setup = '';
  asked = 0;
  expected = 0;
  total -= done; // those still compiling stay counted
  carried = total;
  done = 0;
  listener?.();
}

/** The load failed: what it asked for is no setup's total. */
export function forgetCompileSetup() {
  setup = '';
  asked = 0;
}

/**
 * Name the setup being compiled, by whatever decides its pipelines (its configuration), so the
 * total is the one it had the last time it loaded.
 */
export function compileSetup(description: unknown) {
  let s: string;
  try {
    // Arrays of data (a table, a loaded volume) stand in by their length: spelling them out costs.
    s = hash(JSON.stringify(description, (_k, v) => (ArrayBuffer.isView(v) || v instanceof ArrayBuffer ? `bytes ${v.byteLength}` : v)));
  } catch {
    return;
  }
  // What was asked for before (another setup that failed to fit, the renderer) is not its own.
  setup = s;
  asked = 0;
  carried = total;
  expected = readExpected()[s] ?? 0;
  listener?.();
}

/** Count a pipeline asked for: compiling, or (`p` settled already) found in the cache. */
export function countCompile(p: Promise<unknown>) {
  total++;
  asked++;
  listener?.();
  const end = () => {
    done++;
    // Kept whenever the count catches up, as well as when the next load starts: the last load
    // before the page closes has no next.
    if (done === total) rememberSetup();
    listener?.();
  };
  p.then(end, end);
}

/**
 * Pipelines compiled, of those the load will ask for since the count last started over (as many
 * as the setup asked for last time, or as many as it has asked for so far, if more).
 */
export const compileProgress = () => ({ done, total: Math.max(total, carried + expected), pending: total - done });

/** Stages the simulations name, shared so the page can tell them apart. */
export const COMPILE_STAGE = 'Compiling shaders';
export const INIT_STAGE = 'Setting up the initial conditions';
/** A star being settled. The rounds stop once it is in balance, so there is no total to show. */
export const settleStage = (round: number) => `Settling the star into balance (round ${round + 1})`;
