// The GPU specs' `test`: Playwright's, with the browser closed by a fixture that Chrome's crash
// handler cannot hold (tools/headless/close.mjs says how it did: every full run of the validation
// suite passed its tests and then waited for good).
import { test as base } from '@playwright/test';
import { closeBrowser } from '../../tools/headless/close.mjs';

export { expect, type Page } from '@playwright/test';

export const test = base.extend<object, { closeBrowser: void }>({
  closeBrowser: [
    async ({ browser }, use) => {
      await use();
      await closeBrowser(browser, (message) => console.log(message));
    },
    { scope: 'worker', auto: true },
  ],
});
