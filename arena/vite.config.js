import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1', port: Number(process.env.VIEWER_PORT ?? 8820), strictPort: true,
    proxy: { '/sim': process.env.SIM_URL ?? 'http://127.0.0.1:8818' },
  },
});
