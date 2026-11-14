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

/**
 * Start the count of pipelines over, at the start of a load. A load asks for its pipelines in
 * several rounds, so the count runs across all of them rather than starting over whenever none
 * are left.
 */
export function resetCompileCount() {
  total -= done; // those still compiling stay counted
  done = 0;
  listener?.();
}

/** Count a pipeline that has started compiling. */
export function countCompile(p: Promise<unknown>) {
  total++;
  listener?.();
  const end = () => {
    done++;
    listener?.();
  };
  p.then(end, end);
}

/** Pipelines compiled, of those asked for since the count last started over. */
export const compileProgress = () => ({ done, total, pending: total - done });

/** Stages the simulations name, shared so the page can tell them apart. */
export const COMPILE_STAGE = 'Compiling shaders';
export const INIT_STAGE = 'Setting up the initial conditions';
/** A star being settled. The rounds stop once it is in balance, so there is no total to show. */
export const settleStage = (round: number) => `Settling the star into balance (round ${round + 1})`;
