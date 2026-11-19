// The GPU specs' `test`: Playwright's, with the browser closed by a fixture that cannot be held
// by Chrome's crash handler.
//
// On macOS Chrome starts chrome_crashpad_handler detached (its parent is launchd), with Chrome's
// stderr as its own. Playwright reads that stderr through a pipe and its Browser.close waits for
// the pipe to end, so it waits for the handler too. The handler normally goes when Chrome does.
// After some tests it stays (Chrome 157: mhd-near-vacuum-bins alone does it, 40 s, and so every
// full run of the validation suite): the tests passed, Chrome had exited with code 0, and the
// worker waited on Browser.close for good, the runner on the worker, with no summary printed and
// the global timeout the only way out. So this fixture closes the browser itself and ends any
// handler that still holds the pipe five seconds later.
//
// To see the hang: run that test from a spec that imports `test` from '@playwright/test'.
import { test as base } from '@playwright/test';

export { expect, type Page } from '@playwright/test';

// The project is typed for the browser (tsconfig.json has no Node types): what this file uses of
// Node, declared here.
declare const process: { platform: string; pid: number; kill(pid: number, signal?: number): boolean };
type Exec = (cmd: string, args: string[], opts: object) => string;
const nodeChildProcess = 'node:child_process';
let execFileSync: Exec; // loaded by the fixture, before anything below runs

const lines = (cmd: string, args: string[]): string[] => {
  try {
    return execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).split('\n');
  } catch (e) {
    // pgrep and lsof exit with 1 when nothing matches; what they printed still counts.
    return String((e as { stdout?: string }).stdout ?? '').split('\n');
  }
};

/** The far ends of this process's unix sockets (lsof prints a socket as "0xA ... ->0xB"). */
const socketPeers = (pid: number): Set<string> =>
  new Set(lines('lsof', ['-a', '-U', '-p', String(pid)]).map((l) => /->(0x[0-9a-f]+)/.exec(l)?.[1]).filter((x): x is string => !!x));

/** The crash handlers whose stderr is a socket of this process: those of the Chrome it launched. */
function crashHandlers(): number[] {
  if (process.platform !== 'darwin') return [];
  const mine = socketPeers(process.pid);
  if (mine.size === 0) return [];
  return lines('pgrep', ['-f', 'chrome_crashpad_handler'])
    .map(Number)
    .filter((pid) => pid > 0)
    .filter((pid) => lines('lsof', ['-a', '-U', '-d', '2', '-p', String(pid)]).some((l) => mine.has(/(0x[0-9a-f]+)\s+0t0/.exec(l)?.[1] ?? '')));
}

const alive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const test = base.extend<object, { closeBrowser: void }>({
  closeBrowser: [
    async ({ browser }, use) => {
      await use();
      ({ execFileSync } = (await import(nodeChildProcess)) as { execFileSync: Exec });
      const handlers = crashHandlers();
      const closed = browser.close().catch(() => {});
      const done = Promise.race([closed.then(() => true), sleep(5000).then(() => false)]);
      // Give the handler five seconds to leave on its own, as it does after a short run.
      if (!(await done)) {
        for (const pid of handlers) {
          if (!alive(pid)) continue;
          console.log(`ending Chrome's crash handler ${pid}, which outlived Chrome and held Browser.close`);
          try {
            process.kill(pid);
          } catch {
            // Gone meanwhile.
          }
        }
      }
      await closed;
    },
    { scope: 'worker', auto: true },
  ],
});
