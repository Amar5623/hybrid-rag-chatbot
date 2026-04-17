# main.py
#
# CHANGES vs previous version (Day 2 — A5):
#   - /pdfs static mount added — serves PDF files from data/pdfs/.
#     PDFs are copied there at ingest time (see routers/ingest.py).
#     Frontend PDF viewer (Person B, Task B1) fetches from /pdfs/{filename}.
#
# Everything else unchanged.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.rag_service import startup
from routers import chat, ingest, kb
from routers import sync as sync_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise RAG singletons
    await startup()
    yield


app = FastAPI(
    title="RAG Chatbot API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────

# Serve extracted images at /images/{filename}
images_dir = Path(__file__).parent / "data" / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

# ── NEW (A5): Serve original PDF files at /pdfs/{filename} ────
# When a PDF is ingested, routers/ingest.py copies it to this directory.
# The frontend PDF viewer (Person B, Task B1) fetches:
#     GET /pdfs/engine_manual.pdf
# and renders the page with PDF.js, drawing a highlight bbox over the matched text.
pdfs_dir = Path(__file__).parent / "data" / "pdfs"
pdfs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/pdfs", StaticFiles(directory=str(pdfs_dir)), name="pdfs")

# ── Routers ───────────────────────────────────────────────────

app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(kb.router)
app.include_router(sync_router.router)