import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1', port: 8816, strictPort: true,
    proxy: {
      '/api': 'http://127.0.0.1:8817',
      '/socket': { target: 'ws://127.0.0.1:8817', ws: true },
    },
  },
});
