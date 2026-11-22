// The GPU specs (tests/gpu) on a Linux GPU node, for tools/headless/slurm/tests.sbatch: the
// repository's playwright.config.ts with Playwright's own headless Chromium on Vulkan, launched
// as tools/headless/session.mjs launches it there. (The repository's config asks for the
// installed Chrome, with Direct3D on Windows: in the Playwright image that finds no adapter.)
import { fileURLToPath } from 'node:url';
import { defineConfig } from '@playwright/test';
import base from '../../../playwright.config';

const args = ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'];
// (As in session.mjs: Chrome's own drawing off the GPU, which H100 and H200 nodes need.)
if (process.env.SHADEROUS_NO_GPU_COMPOSITING) args.push('--disable-gpu-compositing', '--disable-gpu-rasterization', '--disable-accelerated-2d-canvas');

export default defineConfig({
  ...base,
  testDir: fileURLToPath(new URL('../../../tests/gpu', import.meta.url)),
  use: { ...base.use, channel: undefined, launchOptions: { args } },
});
