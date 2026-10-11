import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1', port: 8820, strictPort: true,
    proxy: { '/sim': 'http://127.0.0.1:8818' },
  },
});
