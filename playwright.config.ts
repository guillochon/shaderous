import { defineConfig } from '@playwright/test';

// GPU tests drive the installed Chrome (no browser download) with WebGPU enabled.
// Set PW_CHANNEL=msedge to use Edge, or PW_HEADED=1 if headless Chrome has no GPU adapter.
// PW_PORT serves this checkout on another port, when another checkout's server holds 5183.
const port = process.env.PW_PORT ?? '5183';
export default defineConfig({
  testDir: 'tests/gpu',
  timeout: 300_000,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: `http://localhost:${port}`,
    channel: process.env.PW_CHANNEL ?? 'chrome',
    headless: !process.env.PW_HEADED,
    launchOptions: {
      args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=d3d11'],
    },
  },
  webServer: {
    command: `npx vite --port ${port}`,
    url: `http://localhost:${port}`,
    reuseExistingServer: true,
  },
});
