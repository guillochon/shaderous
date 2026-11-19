import { defineConfig } from '@playwright/test';

// GPU tests drive the installed Chrome (no browser download) with WebGPU enabled.
// Set PW_CHANNEL=msedge to use Edge, or PW_HEADED=1 if headless Chrome has no GPU adapter.
// PW_PORT serves this checkout on another port, when another checkout's server holds 5183.
const port = process.env.PW_PORT ?? '5183';
// A failed test's error context (error-context.md) takes an accessibility snapshot of its page
// after the test body; the message already holds what a validation test measured, so skip it.
process.env.PLAYWRIGHT_NO_COPY_PROMPT ??= '1';
export default defineConfig({
  testDir: 'tests/gpu',
  timeout: 300_000,
  // The whole run stops after this (the full suite takes 15 to 90 minutes): the runner process
  // keeps this clock, so it holds even where a worker stops answering. PW_GLOBAL_TIMEOUT in
  // minutes changes it.
  globalTimeout: Number(process.env.PW_GLOBAL_TIMEOUT ?? 180) * 60_000,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: `http://localhost:${port}`,
    channel: process.env.PW_CHANNEL ?? 'chrome',
    headless: !process.env.PW_HEADED,
    launchOptions: {
      // Direct3D on Windows only (tools/headless/session.mjs does the same): elsewhere it leaves
      // Chrome without a WebGPU adapter.
      args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', ...(process.platform === 'win32' ? ['--use-angle=d3d11'] : [])],
    },
  },
  webServer: {
    command: `npx vite --port ${port}`,
    url: `http://localhost:${port}`,
    reuseExistingServer: true,
  },
});
