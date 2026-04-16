# rag-backend/Dockerfile

# ── Stage 1: Builder ──────────────────────────────────────────
# Install all dependencies in a separate stage so the final image
# doesn't carry pip's cache or build tools.
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps needed for some packages (bcrypt, qdrant-client, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer.
# Only re-runs pip install if requirements.txt changes.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create data directories that the app expects
# These will be overridden by the Docker volume mount in production,
# but need to exist for the container to start cleanly.
RUN mkdir -p data/qdrant data/images

# The embedding model (BGE) is downloaded from HuggingFace on first run
# and cached inside the container at this path.
# Mount a volume here too if you want the model to persist across rebuilds.
ENV HF_HOME=/app/data/hf_cache
RUN mkdir -p data/hf_cache

# Expose FastAPI port
EXPOSE 8000

# Run uvicorn — host 0.0.0.0 so it's reachable from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]