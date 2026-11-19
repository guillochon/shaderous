/** The pids of Chrome crash handlers holding this process's pipe to a Chrome it launched (macOS). */
export function crashHandlers(): number[];
/** browser.close(), not held by a crash handler that outlives Chrome (tools/headless/close.mjs). */
export function closeBrowser(browser: { close(): Promise<void> }, log?: (message: string) => void): Promise<void>;
