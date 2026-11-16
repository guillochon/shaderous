#!/usr/bin/env node
// Data for the paper's figures that the scripting API cannot reach (the validation suite's own
// setups, and readbacks of the simulator's internals): runs src/validation/figure_data.ts inside
// the headless page that tools/headless/session.mjs opens, and writes what it returns as JSON.
//
//   node paper/figures/collect.mjs NAME OUT.json [ARGS_JSON] [--normal] [--firefox]
//
// NAME is a function exported by src/validation/figure_data.ts; ARGS_JSON a JSON list of its
// arguments. Run from the code's repository, which Vite serves to the page. The run is at
// below-normal CPU priority, except with --normal (for benchmarks).
//
// --firefox runs the function in the installed Firefox instead (FIREFOX names its executable).
// Playwright drives only its own patched Firefox, so the page is opened in a window of the stock
// one, on a fresh profile, and posts its result back to the server that serves it.
import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { lowerPriority, openSession, REPO } from '../../../tools/headless/session.mjs';

const argv = process.argv.slice(2);
// --normal: stay at normal priority, for benchmarks; everything else runs below normal.
const normal = argv.includes('--normal');
const firefox = argv.includes('--firefox');
const [name, out, args] = argv.filter((a) => !a.startsWith('--'));
if (!name || !out) {
  process.stderr.write('usage: node paper/figures/collect.mjs NAME OUT.json [ARGS_JSON] [--normal] [--firefox]\n');
  process.exit(1);
}
if (!normal) lowerPriority();

/** Run the function in a window of the installed Firefox; what it returns, or an Error. */
async function inFirefox() {
  const { createServer } = await import('vite');
  let done;
  const result = new Promise((resolve, reject) => (done = { resolve, reject }));
  const page = `<!doctype html><meta charset="utf-8"><title>collect</title><script type="module">
    const post = (body) => fetch('/__collect_result', { method: 'POST', body: JSON.stringify(body) });
    try {
      const m = await import('/src/validation/figure_data.ts');
      post({ data: await m[${JSON.stringify(name)}](...${args ?? '[]'}) });
    } catch (e) {
      post({ error: String(e?.message ?? e) });
    }
  </script>`;
  const plugin = {
    name: 'collect',
    configureServer(server) {
      server.middlewares.use('/__collect.html', (_req, res) => res.setHeader('Content-Type', 'text/html').end(page));
      server.middlewares.use('/__collect_result', (req, res) => {
        let body = '';
        req.on('data', (c) => (body += c));
        req.on('end', () => {
          res.end('ok');
          const r = JSON.parse(body);
          if (r.error) done.reject(new Error(r.error));
          else done.resolve(r.data);
        });
      });
    },
  };
  // No reload when a source file changes: the run keeps the code it started with.
  const server = await createServer({
    root: REPO, configFile: path.join(REPO, 'vite.config.ts'), plugins: [plugin],
    server: { port: 0, strictPort: false, host: '127.0.0.1', hmr: false, watch: null }, logLevel: 'error', clearScreen: false,
  });
  await server.listen();
  const profile = mkdtempSync(path.join(os.tmpdir(), 'shaderous-firefox-'));
  // A fresh profile that opens straight on the page: no welcome or terms window, no first-run
  // tabs, no default-browser question, no telemetry notice.
  const prefs = {
    'dom.webgpu.enabled': true,
    'browser.preonboarding.enabled': false, 'browser.aboutwelcome.enabled': false, 'termsofuse.bypassNotification': true,
    'trailhead.firstrun.didSeeAboutWelcome': true, 'browser.startup.homepage_override.mstone': 'ignore',
    'startup.homepage_welcome_url': '', 'startup.homepage_welcome_url.additional': '',
    'browser.shell.checkDefaultBrowser': false, 'browser.shell.skipDefaultBrowserCheckOnFirstRun': true,
    'datareporting.policy.dataSubmissionEnabled': false, 'datareporting.policy.dataSubmissionPolicyBypassNotification': true,
    'toolkit.telemetry.reportingpolicy.firstRun': false, 'browser.messaging-system.whatsNewPanel.enabled': false,
  };
  writeFileSync(path.join(profile, 'user.js'), Object.entries(prefs).map(([k, v]) => `user_pref("${k}", ${JSON.stringify(v)});
`).join(''));
  const exe = process.env.FIREFOX ?? (process.platform === 'win32' ? 'C:/Program Files/Mozilla Firefox/firefox.exe' : 'firefox');
  const child = spawn(exe, ['-no-remote', '-new-instance', '-profile', profile, `${server.resolvedUrls.local[0]}__collect.html`], { stdio: 'ignore' });
  try {
    return await result;
  } finally {
    // Firefox's launcher starts the browser as a process of its own, which outlives the launcher:
    // close every process that was started on this profile, or its window stays open and keeps
    // its GPU memory.
    child.kill();
    if (process.platform === 'win32') {
      const find = `Get-CimInstance Win32_Process -Filter "name='firefox.exe'" | Where-Object { $_.CommandLine -like '*${path.basename(profile)}*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`;
      try {
        execFileSync('powershell', ['-NoProfile', '-Command', find], { stdio: 'ignore' });
      } catch {
        // Nothing left to close.
      }
    }
    await server.close().catch(() => {});
    // Firefox lets go of its profile a moment after it exits.
    await new Promise((r) => setTimeout(r, 2000));
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {
      // Still held: the system's cleanup of its temporary folder takes it.
    }
  }
}

if (firefox) {
  const t0 = Date.now();
  writeFileSync(out, JSON.stringify(await inFirefox()) + '\n');
  process.stderr.write(`${name} in Firefox: ${((Date.now() - t0) / 1000).toFixed(0)} s -> ${out}\n`);
  process.exit(0);
}

// A small, idle run: the page's own simulator stays out of the way of the one the data needs.
const s = await openSession({ setup: 'problem=sedov&dim=2&n=128', log: () => {} });
try {
  const t0 = Date.now();
  const data = await s.page.evaluate(async ([n, a]) => {
    const m = await import('/src/validation/figure_data.ts');
    return m[n](...a);
  }, [name, JSON.parse(args ?? '[]')]);
  writeFileSync(out, JSON.stringify(data) + '\n');
  process.stderr.write(`${name}: ${((Date.now() - t0) / 1000).toFixed(0)} s -> ${out}\n`);
} finally {
  await s.close();
}
