// Close a Playwright browser so that Chrome's crash handler cannot hold the caller.
//
// On macOS Chrome starts chrome_crashpad_handler detached (its parent is launchd), with Chrome's
// stderr as its own. Playwright reads that stderr through a pipe and its Browser.close waits for
// the pipe to end, so it waits for the handler too. The handler normally goes when Chrome does.
// After some runs it stays (Chrome 157: the validation test mhd-near-vacuum-bins alone does it,
// 40 s, and so every full run of the suite): Chrome has exited with code 0, and Browser.close
// never returns. A test run then printed no summary and waited for its global timeout, and a
// headless run would not exit. So: close, and end any handler that still holds the pipe five
// seconds later. Used by the headless session (session.mjs) and the GPU tests' browser fixture
// (tests/gpu/fixtures.ts).
//
// To see the hang: run that test from a spec that imports `test` from '@playwright/test'.
import { execFileSync } from 'node:child_process';

const lines = (cmd, args) => {
  try {
    return execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).split('\n');
  } catch (e) {
    // pgrep and lsof exit with 1 when nothing matches; what they printed still counts.
    return String(e.stdout ?? '').split('\n');
  }
};

/** The crash handlers whose stderr is a socket of this process: those of the Chrome it launched. */
export function crashHandlers() {
  if (process.platform !== 'darwin') return [];
  // lsof prints a unix socket as "<its address> 0t0 ->(its peer's address)".
  const mine = new Set(lines('lsof', ['-a', '-U', '-p', String(process.pid)]).map((l) => /->(0x[0-9a-f]+)/.exec(l)?.[1]).filter(Boolean));
  if (mine.size === 0) return [];
  return lines('pgrep', ['-f', 'chrome_crashpad_handler'])
    .map(Number)
    .filter((pid) => pid > 0)
    .filter((pid) => lines('lsof', ['-a', '-U', '-d', '2', '-p', String(pid)]).some((l) => mine.has(/(0x[0-9a-f]+)\s+0t0/.exec(l)?.[1])));
}

const alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

/**
 * browser.close(), not held by a crash handler that outlives Chrome. `log` is told of each
 * handler ended. Errors of the close itself are swallowed, as a closing caller wants.
 */
export async function closeBrowser(browser, log = () => {}) {
  let handlers = [];
  try {
    handlers = crashHandlers();
  } catch {
    // No lsof or pgrep: close as Playwright does.
  }
  const closed = browser.close().catch(() => {});
  if (handlers.length === 0) return closed;
  let timer;
  const inTime = await Promise.race([closed.then(() => true), new Promise((resolve) => (timer = setTimeout(() => resolve(false), 5000)))]);
  clearTimeout(timer);
  if (!inTime) {
    for (const pid of handlers) {
      if (!alive(pid)) continue;
      log(`ending Chrome's crash handler ${pid}, which outlived Chrome and held Browser.close`);
      try {
        process.kill(pid);
      } catch {
        // Gone meanwhile.
      }
    }
  }
  await closed;
}
