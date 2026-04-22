# routers/admin.py
#
# Admin-only router — all routes are protected by require_admin dependency.
#
# All write operations (ingest, delete, wipe) are grouped here under the
# /admin prefix.  The original /ingest endpoints in routers/ingest.py are
# kept for backward compatibility — they remain accessible as before.
#
# Routes exposed here:
#   POST   /admin/ingest           — Upload + index one or more PDFs
#   DELETE /admin/file/{filename}  — Delete a file from the knowledge base
#   DELETE /admin/collection       — Wipe the entire knowledge base
#   GET    /admin/files            — List all indexed files
#   GET    /admin/stats            — Full system stats
#
# Auth:
#   All routes require a valid Bearer token matching ADMIN_TOKEN in .env.
#   If ADMIN_TOKEN is empty, auth is disabled (dev mode).

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from middleware.admin_auth import require_admin
from services import rag_service
from schemas import IngestResponse, DeleteFileResponse, WipeResponse
from config import settings

# Re-use all the existing ingest helpers from routers/ingest.py
# so there is no logic duplication — admin routes call the same
# internal functions that the original /ingest endpoints use.
from routers.ingest import (
    _ingest_files_sync,
    _store_pdf_file,
    _delete_pdf_file,
    _remove_hash_for_file,
    _wipe_hashes,
)

router = APIRouter(
    prefix       = "/admin",
    tags         = ["admin"],
    dependencies = [Depends(require_admin)],   # ALL routes in this router require admin token
)


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def admin_ingest(files: list[UploadFile] = File(...)):
    """
    Upload and index one or more PDFs into the knowledge base.
    Admin only — requires Authorization: Bearer <ADMIN_TOKEN> header.

    Behaviour is identical to POST /ingest.
    """
    # Save original PDFs for the viewer before processing (same as /ingest)
    pdfs_dir = Path(settings.qdrant_path).parent / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.filename.lower().endswith(".pdf"):
            dest_path = pdfs_dir / file.filename
            content = await file.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            await file.seek(0)

    if not files:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No files provided.")

    tmp_dir    = Path("/tmp") / f"rag_admin_ingest_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_paths : list[tuple[str, str]] = []

    try:
        for upload in files:
            tmp_path = tmp_dir / upload.filename
            content  = await upload.read()
            tmp_path.write_bytes(content)
            file_paths.append((str(tmp_path), upload.filename))

        result = await run_in_threadpool(_ingest_files_sync, file_paths)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IngestResponse(
        status        = "ok",
        files_indexed = result["files_indexed"],
        total_chunks  = result["total_chunks"],
        total_parents = result["total_parents"],
        message       = (
            f"Indexed {len(result['files_indexed'])} file(s). "
            f"Skipped {len(result['skipped'])} duplicate(s)."
        ),
    )


# ── Delete file ───────────────────────────────────────────────────────────────

@router.delete("/file/{filename}", response_model=DeleteFileResponse)
async def admin_delete_file(filename: str):
    """
    Delete a file from the knowledge base.
    Admin only — requires Authorization: Bearer <ADMIN_TOKEN> header.

    Behaviour is identical to DELETE /ingest/{filename}.
    """
    from fastapi import HTTPException, status as http_status

    # ── Guard 1: must be online ───────────────────────────
    if not rag_service.is_online():
        raise HTTPException(
            status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                f"Cannot delete '{filename}' while offline. "
                "Deletion requires a cloud connection so the change is "
                "applied to the authoritative store. Please reconnect and try again."
            ),
        )

    # ── Guard 2: cloud store must be configured ───────────
    cloud_store = rag_service.get_cloud_store()
    if cloud_store is None:
        raise HTTPException(
            status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                f"Cannot delete '{filename}': no cloud store is configured. "
                "Set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY in .env to enable deletion."
            ),
        )

    # ── Guard 3: file must exist in cloud ─────────────────
    cloud_sources = cloud_store.list_sources()
    if filename not in cloud_sources:
        raise HTTPException(
            status_code = http_status.HTTP_404_NOT_FOUND,
            detail      = f"File '{filename}' not found in the cloud knowledge base.",
        )

    # ── Delete from cloud ─────────────────────────────────
    result = await run_in_threadpool(rag_service.delete_file_from_cloud, filename)

    # ── Clean up local side effects immediately ───────────
    _remove_hash_for_file(filename)
    _delete_pdf_file(filename)

    return DeleteFileResponse(
        status          = "ok",
        filename        = filename,
        vectors_deleted = result["vectors_deleted"],
        message         = (
            f"Deleted '{filename}' from cloud: "
            f"{result['vectors_deleted']} vectors removed. "
            f"Local vectors will be cleaned up on next sync."
        ),
    )


# ── Wipe collection ───────────────────────────────────────────────────────────

@router.delete("/collection", response_model=WipeResponse)
async def admin_wipe():
    """
    Wipe the entire knowledge base (vectors + BM25 + hash registry).
    Admin only — requires Authorization: Bearer <ADMIN_TOKEN> header.
    """
    rag_service.get_vector_store().reset_collection()
    rag_service.get_bm25_store().reset()
    _wipe_hashes()
    return WipeResponse(status="ok", message="Knowledge base wiped.")


# ── List files ────────────────────────────────────────────────────────────────

@router.get("/files")
async def admin_list_files():
    """
    List all indexed files in the knowledge base.
    Admin only — requires Authorization: Bearer <ADMIN_TOKEN> header.
    """
    vs = rag_service.get_vector_store()
    return {"files": vs.list_sources()}


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def admin_stats():
    """
    Full system stats.
    Admin only — requires Authorization: Bearer <ADMIN_TOKEN> header.
    """
    vs   = rag_service.get_vector_store()
    bm25 = rag_service.get_bm25_store()
    return {
        "total_vectors"  : vs.count(),
        "bm25_docs"      : len(bm25),
        "indexed_files"  : vs.list_sources(),
        "embedding_model": settings.embedding_model,
        "llm_model"      : settings.groq_model,
        "collection"     : settings.qdrant_collection,
    }