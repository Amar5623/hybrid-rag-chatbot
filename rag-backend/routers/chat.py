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

        # ── B-Phase 3: Offline Reranker ───────────────────
        # Always retrieve the full candidate pool (top_k=settings.top_k, default 20)
        # so we have enough candidates for the reranker to work with.
        # The decision of how many to *return* happens after retrieval.
        retriever = chain.retriever

        # Import active store so offline path uses local store (same as chain)
        active_store = rag_service.get_local_store()

        retrieval = retriever.retrieve(
            query        = req.question,
            top_k        = settings.top_k,          # always fetch 20 candidates
            filter_field = "source" if chain.get_source_filter() else None,
            filter_value = chain.get_source_filter(),
            is_offline   = True,
            store        = active_store,
        )

        if settings.enable_offline_reranker:
            # Apply cross-encoder reranker: rescores all 20 candidates → keeps top reranker_top_k
            reranker    = rag_service.get_reranker()
            reranked    = reranker.rerank(
                query   = req.question,
                retrieval = retrieval,
                top_k   = settings.reranker_top_k,
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

        result = OfflineQueryResponse(
            query      = req.question,
            chunks     = offline_chunks,
            total      = len(offline_chunks),
            is_offline = True,
        )
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