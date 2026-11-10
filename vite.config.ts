import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './', // relative asset paths, so the build works under a GitHub Pages subpath
  server: { port: 5183, strictPort: true },
  build: {
    target: 'es2022',
    rollupOptions: { input: { main: 'index.html', validate: 'validate.html' } },
  },
  test: { include: ['tests/unit/**/*.test.ts'] },
});
