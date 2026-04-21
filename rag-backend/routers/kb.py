# rag-backend/routers/kb.py
#
# CHANGES vs previous version:
#
#   NEW ENDPOINT: GET /kb/export
#     Returns all indexed chunks in a JSON format the mobile app can store
#     in local SQLite for offline (Mode 3) use.
#
#     Uses BM25Store._chunks which is already loaded in memory — no extra
#     DB query needed. Each chunk dict has at minimum: source, content,
#     parent_content, page, type.
#
#     The mobile sync queue calls this once per sync cycle and replaces
#     its local SQLite database with the response.
#
#   KEPT: FORCE_OFFLINE_MODE override on /health (from previous change)

import os

from fastapi import APIRouter
from schemas import (
    DocumentsResponse, HealthResponse,
    StatsResponse, WipeResponse,
)
from services    import rag_service
from routers.ingest import _wipe_hashes
from config      import settings

router = APIRouter(tags=["kb"])


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Public — used by the mobile app to determine its network mode.
    FORCE_OFFLINE_MODE=true in environment → always returns is_online=false.
    """
    force_offline = os.getenv("FORCE_OFFLINE_MODE", "false").strip().lower() == "true"

    if force_offline:
        return {
            "status":          "ok",
            "is_online":       False,
            "groq_configured": bool(settings.groq_api_key),
            "forced":          True,
        }

    return HealthResponse(
        status          = "ok",
        groq_configured = bool(settings.groq_api_key),
        is_online       = rag_service.is_online(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK EXPORT — for mobile offline sync
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/kb/export")
async def export_chunks():
    """
    Export all indexed chunks for mobile offline sync.

    The mobile app's syncQueue.syncFromServer() calls this endpoint and stores
    the result in local SQLite (expo-sqlite) for Mode 3 (deep offline) use.

    Returns the BM25Store chunk list which is already in memory.
    No extra vector DB query needed.

    Each chunk includes:
      id             : stable identifier (source + index if no id field)
      source         : original filename
      content        : child chunk text (300 chars — used for indexing)
      parent_content : full parent passage (1500 chars — shown to user)
      page           : page number in source PDF
      chunk_type     : 'text' | 'table' | 'image'

    NOTE: This does NOT include embedding vectors. The mobile app uses
    SQLite FTS5 (BM25 full-text search) for local retrieval, so embeddings
    are not needed for Phase 2.1. Phase 2.2 will add vector export.
    """
    bm25   = rag_service.get_bm25_store()
    raw    = bm25._chunks  # list of dicts already in memory

    chunks = []
    for i, c in enumerate(raw):
        source  = c.get("source", "")
        page    = c.get("page",   0)
        content = c.get("content", "")

        chunks.append({
            # Stable ID: prefer stored id, fall back to source+index
            "id":             c.get("id") or f"{source}_{page}_{i}",
            "source":         source,
            "content":        content,
            # parent_content is the full readable passage (1500 chars)
            # Fall back to content for atomic chunks (tables, images)
            "parent_content": c.get("parent_content") or content,
            "page":           page,
            "chunk_type":     c.get("type", "text"),
        })

    return {
        "chunks": chunks,
        "total":  len(chunks),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING ENDPOINTS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
async def stats():
    """Public — frontend polls this to show KB status."""
    vs   = rag_service.get_vector_store()
    bm25 = rag_service.get_bm25_store()
    return StatsResponse(
        total_vectors   = vs.count(),
        bm25_docs       = len(bm25),
        parent_count    = 0,
        indexed_files   = vs.list_sources(),
        embedding_model = settings.embedding_model,
        llm_model       = settings.groq_model,
        collection      = settings.qdrant_collection,
    )


@router.get("/documents", response_model=DocumentsResponse)
async def documents():
    files = rag_service.get_vector_store().list_sources()
    return DocumentsResponse(files=files, total_files=len(files))


@router.delete("/collection", response_model=WipeResponse)
async def wipe():
    """Wipe the entire knowledge base."""
    rag_service.get_vector_store().reset_collection()
    rag_service.get_bm25_store().reset()
    _wipe_hashes()
    return WipeResponse(status="ok", message="Knowledge base wiped.")