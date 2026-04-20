import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // Proxy images
      '/images': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy PDFs - THIS WAS MISSING OR INCORRECT
      '/pdfs': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})