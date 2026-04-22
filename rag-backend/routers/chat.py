# routers/chat.py
#
# CHANGES vs previous version:
#   - JWT auth dependency REMOVED from all routes — no auth needed.
#   - get_or_create_session() replaced with get_chain() (single shared chain).
#   - Online path  → SSE stream (same as before).
#   - Offline path → run retrieval only, return OfflineQueryResponse as normal
#     JSON response. No SSE needed since there is no LLM streaming.
#   - /session/pin and /session/clear still work, operate on shared chain.
#
# B-Phase 3: Offline Reranker
#   - When ENABLE_OFFLINE_RERANKER=true in .env, the offline path now applies
#     the cross-encoder reranker between retrieval and response building.
#   - Retrieval always fetches top_k=settings.top_k candidates (default 20).
#   - With reranker ON : cross-encoder rescores + keeps reranker_top_k (default 5)
#   - With reranker OFF: simple slice to offline_top_k (default 5)
#   - get_reranker() accessor used (already added to rag_service.py).
#
# CHANGE — Add /chat/offline endpoint (Mode 2 fix):
#   - Mobile app calls POST /chat/offline which previously returned 404.
#   - This endpoint accepts the same ChatRequest body and forces offline mode,
#     bypassing rag_service.is_online() entirely.
#   - This lets the mobile app explicitly request offline retrieval even if
#     the server's NetworkMonitor hasn't detected the state change yet.
#   - The offline retrieval + reranker logic is identical to the offline branch
#     in /chat/stream — both paths use the same shared helpers.

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from schemas import ChatRequest, ClearRequest, OfflineQueryResponse, OfflineChunk
from services import rag_service
from config import settings

router = APIRouter(tags=["chat"])


class PinRequest(BaseModel):
    filename: str


# ── Shared offline retrieval helper ──────────────────────────────────────────
#
# Both /chat/stream (offline branch) and /chat/offline use the exact same
# retrieval + reranker logic.  Extracted here as a plain function to avoid
# duplication.  The caller passes in the already-resolved chain and store so
# this function stays pure (no rag_service calls inside).

def _run_offline_retrieval(question: str, chain, active_store) -> list:
    """
    Run hybrid retrieval (+ optional reranker) and return a list of chunk dicts.

    Parameters
    ----------
    question     : user query string
    chain        : the active RAGChain instance (for retriever + source filter)
    active_store : the local vector store instance

    Returns
    -------
    list[dict]   : final ranked chunks ready to be serialised as OfflineChunk
    """
    retriever = chain.retriever

    retrieval = retriever.retrieve(
        query        = question,
        top_k        = settings.top_k,          # always fetch full candidate pool (default 20)
        filter_field = "source" if chain.get_source_filter() else None,
        filter_value = chain.get_source_filter(),
        is_offline   = True,
        store        = active_store,
    )

    if settings.enable_offline_reranker:
        # Apply cross-encoder reranker: rescores all candidates → keeps top reranker_top_k
        reranker     = rag_service.get_reranker()
        reranked     = reranker.rerank(
            query     = question,
            retrieval = retrieval,
            top_k     = settings.reranker_top_k,
        )
        final_chunks = reranked.get_chunks()
        print(
            f"  [CHAT/OFFLINE] Reranker ON — "
            f"{len(retrieval)} candidates → {len(final_chunks)} reranked chunks"
        )
    else:
        # No reranker: just slice to offline_top_k
        final_chunks = retrieval.get_chunks()[:settings.offline_top_k]
        print(
            f"  [CHAT/OFFLINE] Reranker OFF — "
            f"returning top {len(final_chunks)} chunks from MMR"
        )

    # Log chunk ordering for comparison (toggle ENABLE_OFFLINE_RERANKER to see the difference)
    for i, c in enumerate(final_chunks):
        score_label = (
            f"rerank={c.get('rerank_score', '?'):.4f}"
            if settings.enable_offline_reranker
            else f"rrf={c.get('rrf_score', c.get('score', '?'))}"
        )
        print(
            f"  [CHAT/OFFLINE] chunk[{i}] {score_label} "
            f"src={c.get('source','?')} p={c.get('page','?')} "
            f"| {c.get('content','')[:80].replace(chr(10),' ')!r}"
        )

    return final_chunks


def _build_offline_response(question: str, final_chunks: list) -> OfflineQueryResponse:
    """
    Convert a list of raw chunk dicts into a fully populated OfflineQueryResponse.
    """
    offline_chunks = [
        OfflineChunk(
            source       = c.get("source", "unknown"),
            page         = c.get("page"),
            heading      = c.get("heading", ""),
            section_path = c.get("section_path", ""),
            content      = c.get("parent_content") or c.get("content", ""),
            score        = round(float(c.get("score", 0.0)), 4),
            chunk_type   = c.get("type", "text"),
            bbox         = c.get("bbox"),
            page_width   = c.get("page_width"),
            page_height  = c.get("page_height"),
        )
        for c in final_chunks
    ]

    return OfflineQueryResponse(
        query      = question,
        chunks     = offline_chunks,
        total      = len(offline_chunks),
        is_offline = True,
    )


# ── Chat stream (online) / chunk response (offline) ──────────

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Online  → SSE stream of tokens then a done event with citations.
    Offline → Normal JSON response (OfflineQueryResponse) with manual sections.
              No SSE needed — there is no LLM streaming in offline mode.
    """
    vector_store = rag_service.get_vector_store()
    has_kb       = vector_store.count() > 0
    chain        = rag_service.get_chain()
    online       = rag_service.is_online()

    # ── OFFLINE ───────────────────────────────────────────
    if not online:
        if not has_kb:
            result = OfflineQueryResponse(
                query      = req.question,
                chunks     = [],
                total      = 0,
                is_offline = True,
            )
            return JSONResponse(content=result.model_dump())

        # Import active store so offline path uses local store (same as chain)
        active_store = rag_service.get_local_store()

        final_chunks = _run_offline_retrieval(req.question, chain, active_store)
        result       = _build_offline_response(req.question, final_chunks)
        return JSONResponse(content=result.model_dump())

    # ── ONLINE (SSE stream) ────────────────────────────────
    async def event_generator():
        try:
            for chunk in chain.stream(req.question, has_kb=has_kb, is_online=True):
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                else:
                    # Final ChainResponse
                    is_document = chunk.query_type == "document"
                    citations   = []
                    image_urls  = []

                    if is_document:
                        citations = [
                            {
                                "source"      : c.get("source", ""),
                                "page"        : c.get("page"),
                                "heading"     : c.get("heading", ""),
                                "section_path": c.get("section_path", ""),
                                "chunk_type"  : c.get("type", "text"),
                            }
                            for c in chunk.get_citations()
                        ]
                        image_urls = [
                            f"/images/{Path(p).name}"
                            for p in chunk.get_images()
                        ]

                    yield f"data: {json.dumps({'done': True, 'citations': citations, 'image_urls': image_urls, 'query_type': chunk.query_type, 'usage': chunk.usage})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Intranet-only / forced offline endpoint ───────────────────────────────────

@router.post("/chat/offline")
async def chat_offline(req: ChatRequest):
    """
    Intranet-only mode (Mode 2):
    Server is reachable via LAN but there is no internet for Groq.
    Runs retrieval-only pipeline and returns OfflineQueryResponse as plain JSON.
    No SSE, no LLM call.

    This endpoint exists as an explicit alternative to /chat/stream so the
    mobile app can force offline retrieval regardless of what the server's
    NetworkMonitor reports.  The mobile app always calls /chat/offline when
    it detects Mode 2 (server reachable, no internet).

    The retrieval + reranker logic is identical to the offline branch inside
    /chat/stream — both share the same _run_offline_retrieval() helper above.
    """
    chain        = rag_service.get_chain()
    active_store = rag_service.get_local_store()

    has_kb = active_store.count() > 0

    if not has_kb:
        result = OfflineQueryResponse(
            query      = req.question,
            chunks     = [],
            total      = 0,
            is_offline = True,
        )
        return JSONResponse(content=result.model_dump())

    final_chunks = _run_offline_retrieval(req.question, chain, active_store)
    result       = _build_offline_response(req.question, final_chunks)
    return JSONResponse(content=result.model_dump())


# ── Session management ────────────────────────────────────────

@router.post("/session/clear")
async def clear_session(req: ClearRequest):
    rag_service.clear_chain_memory()
    return {"status": "ok"}


# ── Pin / unpin ───────────────────────────────────────────────

@router.post("/session/pin")
async def pin_source(req: PinRequest):
    """Pin the chain to a single source file."""
    chain   = rag_service.get_chain()
    sources = rag_service.get_vector_store().list_sources()

    if req.filename not in sources:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"File '{req.filename}' not found in the knowledge base.",
        )

    chain.set_source_filter(req.filename)
    return {"status": "ok", "pinned": req.filename}


@router.delete("/session/pin")
async def unpin_source():
    """Remove the source pin."""
    rag_service.get_chain().clear_source_filter()
    return {"status": "ok", "pinned": None}


@router.get("/session/pin")
async def get_pin():
    """Return the currently pinned filename, or null."""
    return {"pinned": rag_service.get_chain().get_source_filter()}