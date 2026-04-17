import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
      },
      // Images served as static files from FastAPI — proxy them too
      '/images': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})