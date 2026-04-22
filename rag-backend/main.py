# rag-backend/main.py
#
# CHANGE: CORS allow_origins updated to ["*"] so that mobile clients on LAN
# (e.g. phone at 192.168.x.x) are not blocked by origin checking.
#
# WHY: The original config only allowed http://localhost:5173 (the Vite dev
# server). Any request from a React Native app running on a real device or
# Expo Go had a different Origin header and was rejected with a CORS error
# before it even reached any route handler.
#
# SECURITY NOTE: allow_origins=["*"] is safe here because this server runs
# on a closed ship LAN with no external exposure. If you later expose the
# backend publicly, lock this down to specific origins.
#
# NOTE: allow_credentials MUST be False when allow_origins=["*"].
# Browsers reject credentialed requests to wildcard origins.
#
# CHANGE: Admin router registered under /admin prefix.
# All write operations are available at /admin/* (protected by ADMIN_TOKEN).
# The original /ingest, /kb, /sync routers are kept for backward compatibility.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from pathlib    import Path
from fastapi    import FastAPI
from fastapi.middleware.cors    import CORSMiddleware
from fastapi.staticfiles        import StaticFiles

from services.rag_service import startup
from routers              import chat, ingest, kb
from routers              import sync as sync_router
from routers              import admin as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield


app = FastAPI(
    title   = "RAG Chatbot API",
    version = "3.1.0",
    lifespan= lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_origins=["*"] — permits mobile clients on any LAN IP.
# allow_credentials must be False when using wildcard origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
images_dir = Path(__file__).parent / "data" / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

pdfs_dir = Path(__file__).parent / "data" / "pdfs"
pdfs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/pdfs", StaticFiles(directory=str(pdfs_dir)), name="pdfs")

# ── Routers ───────────────────────────────────────────────────────────────────
# Admin router — all write operations under /admin/* (requires ADMIN_TOKEN).
app.include_router(admin_router.router)

# Existing routers — kept for backward compatibility.
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(kb.router)
app.include_router(sync_router.router)