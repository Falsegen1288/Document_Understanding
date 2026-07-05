import { defineConfig, mergeConfig } from 'vitest/config';
import viteConfigFunc from './vite.config';

export default defineConfig((env) => {
  const baseConfig = typeof viteConfigFunc === 'function' ? viteConfigFunc(env) : viteConfigFunc;
  return mergeConfig(
    baseConfig,
    {
      test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: ['./vitest.setup.ts'],
        include: ['src/**/*.{test,spec}.{ts,tsx}'],
      },
    }
  );
});
