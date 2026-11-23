// The GPU specs' `test`: Playwright's, with the browser closed by a fixture that Chrome's crash
// handler cannot hold (tools/headless/close.mjs says how it did: every full run of the validation
// suite passed its tests and then waited for good).
import { test as base } from '@playwright/test';
import { closeBrowser } from '../../tools/headless/close.mjs';

export { expect, type Page } from '@playwright/test';

declare const process: { env: Record<string, string | undefined> };

export const test = base.extend<object, { closeBrowser: void }>({
  // With Chrome's drawing off the GPU (SHADEROUS_NO_GPU_COMPOSITING, which data-center GPUs need:
  // tools/headless/slurm/playwright.config.ts), a WebGPU canvas cannot be read, so screenshots
  // must be read back from the GPU, as tools/headless/session.mjs asks: without this they came
  // out blank (a 512-wide screenshot of 9,498 characters, field lines that changed nothing).
  context: async ({ context }, use) => {
    if (process.env.SHADEROUS_NO_GPU_COMPOSITING) await context.addInitScript(() => ((globalThis as { shaderousReadBack?: boolean }).shaderousReadBack = true));
    await use(context);
  },
  closeBrowser: [
    async ({ browser }, use) => {
      await use();
      await closeBrowser(browser, (message) => console.log(message));
    },
    { scope: 'worker', auto: true },
  ],
});
