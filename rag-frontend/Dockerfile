# rag-frontend/Dockerfile

# ── Stage 1: Builder ──────────────────────────────────────────
# Build the React app into static files.
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for layer caching
COPY package.json package-lock.json* ./

RUN npm ci --silent

# Copy source and build
COPY . .
RUN npm run build
# Output is in /app/dist


# ── Stage 2: Runtime (nginx) ──────────────────────────────────
# Serve the built static files with nginx.
# nginx is tiny (~20MB) vs running a full Node server.
FROM nginx:alpine

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config:
#   - Serves static files
#   - Proxies /api/* and /images/* to the backend container
#   - Handles React Router (SPA fallback to index.html)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]