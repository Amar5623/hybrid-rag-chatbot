# routers/chat.py
#
# CHANGES vs original:
#   - JWT auth dependency REMOVED from all routes — no auth needed.
#   - get_or_create_session() replaced with get_chain() (single shared chain).
#   - Online path  → SSE stream (same as before).
#   - Offline path → run retrieval only, return OfflineQueryResponse as normal
#     JSON response. No SSE needed since there is no LLM streaming.
#   - /session/pin and /session/clear still work, operate on shared chain.

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from schemas import ChatRequest, ClearRequest, OfflineQueryResponse
from services import rag_service

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
        # stream() with is_online=False yields a single OfflineQueryResponse
        result = None
        for item in chain.stream(req.question, has_kb=has_kb, is_online=False):
            result = item  # only one item yielded in offline mode

        if result is None:
            result = OfflineQueryResponse(query=req.question, chunks=[], total=0)

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