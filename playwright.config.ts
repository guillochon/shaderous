import { defineConfig } from '@playwright/test';

// GPU tests drive the installed Chrome (no browser download) with WebGPU enabled.
// Set PW_CHANNEL=msedge to use Edge, or PW_HEADED=1 if headless Chrome has no GPU adapter.
export default defineConfig({
  testDir: 'tests/gpu',
  timeout: 300_000,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5183',
    channel: process.env.PW_CHANNEL ?? 'chrome',
    headless: !process.env.PW_HEADED,
    launchOptions: {
      args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=d3d11'],
    },
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5183',
    reuseExistingServer: true,
  },
});
