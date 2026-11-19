import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({

  base: process.env.BASE_PATH || '/',
  server: {
    port: 5173,
    open: '/index.html',
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        login: resolve(__dirname, 'pages/login.html'),
        signup: resolve(__dirname, 'pages/signup.html'),
        app: resolve(__dirname, 'pages/app.html'),
      },
    },
  },
});