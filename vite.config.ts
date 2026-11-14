import { execSync } from 'node:child_process';
import { defineConfig } from 'vitest/config';

// Shown in the page footer and in error reports, so a stale cached page is recognizable.
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
  server: {
    port: 5183,
    strictPort: true,
    // Python environments, training logs and the paper are not the app's: watching them (tens
    // of thousands of files under tools/burn alone) held a fresh server's first page for 50 s.
    watch: { ignored: ['**/.venv/**', '**/tools/**', '**/paper/**', '**/.claude/**', '**/test-results/**', '**/dist/**'] },
  },
  // The app's own pages only: by default Vite crawls every .html under the root for imports,
  // including those inside Python environments' packages.
  optimizeDeps: { entries: ['index.html', 'validate.html'] },
  build: {
    target: 'es2022',
    rollupOptions: { input: { main: 'index.html', validate: 'validate.html' } },
  },
  test: { include: ['tests/unit/**/*.test.ts'] },
});
