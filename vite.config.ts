import { execSync } from 'node:child_process';
import { defineConfig } from 'vitest/config';

// Shown in the page footer and in error reports, so a stale cached page is recognisable.
function build(): string {
  try {
    const hash = execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
    const date = execSync('git log -1 --format=%cs', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
    return `${hash} (${date})`;
  } catch {
    return 'unknown';
  }
}

export default defineConfig({
  define: { __BUILD__: JSON.stringify(build()) },
  base: './', // relative asset paths, so the build works under a GitHub Pages subpath
  server: { port: 5183, strictPort: true },
  build: {
    target: 'es2022',
    rollupOptions: { input: { main: 'index.html', validate: 'validate.html' } },
  },
  test: { include: ['tests/unit/**/*.test.ts'] },
});
