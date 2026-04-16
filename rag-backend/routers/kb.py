# routers/kb.py
#
# CHANGES vs original:
#   - JWT auth dependency REMOVED from all routes.
#   - /health now returns is_online: bool from the network monitor.
#   - parent_count stays 0 (inline parents, no separate store).

from fastapi import APIRouter
from schemas import (
    DocumentsResponse, HealthResponse,
    StatsResponse, WipeResponse,
)
from services import rag_service
from routers.ingest import _wipe_hashes
from config import settings

router = APIRouter(tags=["kb"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Public — used by frontend to check if backend is up and if we're online."""
    return HealthResponse(
        status          = "ok",
        groq_configured = bool(settings.groq_api_key),
        is_online       = rag_service.is_online(),
    )


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
    """Wipe the entire knowledge base. No auth required (single-user deployment)."""
    rag_service.get_vector_store().reset_collection()
    rag_service.get_bm25_store().reset()
    _wipe_hashes()
    return WipeResponse(status="ok", message="Knowledge base wiped.")