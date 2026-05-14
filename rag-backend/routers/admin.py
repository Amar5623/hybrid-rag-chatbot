# routers/admin.py
#
# Phase 2 — Authentication System
#
# CHANGES vs previous version:
#   - Auth:  Replaced static ADMIN_TOKEN dependency (require_admin) with
#            Supabase JWT-based auth (resolve_tenant + require_admin_role).
#   - Stores: All vector store / BM25 calls now use get_tenant_stores()
#             scoped to request.state.tenant_slug instead of global singletons.
#   - NEW endpoints:
#       GET  /admin/join-code             — return current join code
#       POST /admin/join-code/regenerate  — rotate join code (old one invalid)
#   - All existing routes (ingest, delete, wipe, files, stats) are unchanged
#     in behaviour — only the auth layer and store scope have changed.
#
# Auth:
#   All routes require a valid Supabase JWT with role == 'admin' or 'super_admin'.
#   Obtain a JWT via POST /auth/admin/login.
#   Include in every request: Authorization: Bearer <access_token>

import random
import shutil
import string
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from middleware.tenant_resolver import require_admin_role, resolve_tenant
from services                  import rag_service
from services.rag_service      import get_tenant_stores
from services.supabase_client  import get_supabase_admin
from schemas                   import IngestResponse, DeleteFileResponse, WipeResponse
from config                    import settings
from utils.logger              import get_logger

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

logger = get_logger(__name__)

router = APIRouter(
    prefix       = "/admin",
    tags         = ["admin"],
    dependencies = [
        Depends(resolve_tenant),      # validates JWT, sets request.state.*
        Depends(require_admin_role),  # enforces role == 'admin' | 'super_admin'
    ],
)


# ── Join code helpers ──────────────────────────────────────────────────────────

_JOIN_WORDS: list[str] = [
    "SHIP", "DOCK", "CREW", "MAST", "SAIL", "PORT", "HULL", "DECK",
    "KEEL", "HELM", "TIDE", "WAVE", "REEF", "BUOY", "LANE", "WIND",
    "BOLT", "CRANE", "LOCK", "PIER", "ROPE", "TANK", "YARD", "STAR",
    "GULF", "CAPE", "COVE", "ISLE", "QUAY", "BRIG",
]


def _gen_unique_join_code(sb) -> str:
    """Generate a join code that doesn't already exist in the DB."""
    for _ in range(20):
        word   = random.choice(_JOIN_WORDS)
        digits = random.randint(1000, 9999)
        code   = f"{word}-{digits}"
        result = sb.table("tenants").select("id").eq("join_code", code).execute()
        if not result.data:
            return code
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


# ── GET /admin/join-code ──────────────────────────────────────────────────────

@router.get("/join-code")
async def get_join_code(request: Request):
    """
    Return the current join code for this tenant.

    Employees use this code when signing up via POST /auth/mobile/signup.
    Only tenant admins can view the join code.
    """
    tenant_id = request.state.tenant_id
    sb        = get_supabase_admin()

    result = (
        sb.table("tenants")
        .select("join_code, slug, display_name")
        .eq("id", tenant_id)
        .single()
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = "Tenant not found.",
        )

    data = result.data
    logger.info(
        "[ADMIN/JOIN-CODE] GET — tenant=%s  code=%s",
        data.get("slug"), data.get("join_code"),
    )
    return {
        "join_code"   : data["join_code"],
        "slug"        : data["slug"],
        "display_name": data["display_name"],
        "hint"        : "Share this code with employees so they can sign up.",
    }


# ── POST /admin/join-code/regenerate ─────────────────────────────────────────

@router.post("/join-code/regenerate")
async def regenerate_join_code(request: Request):
    """
    Generate a new join code for this tenant, replacing the old one.

    The old join code is immediately invalidated.
    Employees who already signed up are unaffected — their JWTs carry
    tenant_id directly and do not depend on the join code.
    """
    tenant_id = request.state.tenant_id
    sb        = get_supabase_admin()

    new_code = _gen_unique_join_code(sb)

    result = (
        sb.table("tenants")
        .update({"join_code": new_code})
        .eq("id", tenant_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Failed to update join code. Please try again.",
        )

    logger.info(
        "[ADMIN/JOIN-CODE] Regenerated — tenant=%s  new_code=%s",
        request.state.tenant_slug, new_code,
    )
    return {
        "join_code": new_code,
        "message"  : "Join code rotated. Share the new code with new employees.",
    }


# ── POST /admin/ingest ────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def admin_ingest(request: Request, files: list[UploadFile] = File(...)):
    """
    Upload and index one or more PDFs into the knowledge base.
    Admin only — requires a valid admin JWT.

    Documents are stored in the tenant-scoped vector collection and BM25 index.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    tenant_slug = request.state.tenant_slug

    # Save original PDFs for the viewer before processing
    pdfs_dir = Path(settings.qdrant_path).parent / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.filename.lower().endswith(".pdf"):
            dest_path = pdfs_dir / file.filename
            content   = await file.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            await file.seek(0)

    tmp_dir    = Path("/tmp") / f"rag_admin_ingest_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_paths : list[tuple[str, str]] = []

    try:
        for upload in files:
            tmp_path = tmp_dir / upload.filename
            content  = await upload.read()
            tmp_path.write_bytes(content)
            file_paths.append((str(tmp_path), upload.filename))

        # Pass tenant_slug so _ingest_files_sync uses tenant-scoped stores
        result = await run_in_threadpool(
            _ingest_files_sync, file_paths, tenant_slug
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "[ADMIN/INGEST] ✅ tenant=%s  files=%s  chunks=%d",
        tenant_slug,
        result["files_indexed"],
        result["total_chunks"],
    )

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


# ── DELETE /admin/file/{filename} ─────────────────────────────────────────────

@router.delete("/file/{filename}", response_model=DeleteFileResponse)
async def admin_delete_file(request: Request, filename: str):
    """
    Delete a file from the knowledge base.
    Admin only — requires a valid admin JWT.
    """
    tenant_slug = request.state.tenant_slug
    vs, bm25    = get_tenant_stores(tenant_slug)

    # ── Guard 1: must be online ───────────────────────────────────────────────
    if not rag_service.is_online():
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                f"Cannot delete '{filename}' while offline. "
                "Deletion requires a cloud connection. Please reconnect and try again."
            ),
        )

    # ── Guard 2: cloud store must be configured ───────────────────────────────
    cloud_store = rag_service.get_cloud_store()
    if cloud_store is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                f"Cannot delete '{filename}': no cloud store is configured. "
                "Set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY in .env."
            ),
        )

    # ── Guard 3: file must exist in tenant store ──────────────────────────────
    tenant_sources = vs.list_sources()
    if filename not in tenant_sources:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"File '{filename}' not found in the knowledge base.",
        )

    # ── Delete from tenant vector store ──────────────────────────────────────
    vectors_deleted = await run_in_threadpool(vs.delete_by_source, filename)

    # ── Clean up tenant BM25 immediately ─────────────────────────────────────
    bm25.delete_by_source(filename)

    # ── Also clean up global cloud store if it mirrors this tenant's data ─────
    # (belt-and-suspenders — harmless no-op if cloud has no matching docs)
    try:
        cloud_sources = cloud_store.list_sources()
        if filename in cloud_sources:
            await run_in_threadpool(rag_service.delete_file_from_cloud, filename)
    except Exception as exc:
        logger.warning(
            "[ADMIN/DELETE] Cloud store cleanup skipped: %s", exc
        )

    _remove_hash_for_file(filename)
    _delete_pdf_file(filename)

    logger.info(
        "[ADMIN/DELETE] ✅ tenant=%s  file=%s  vectors_deleted=%d",
        tenant_slug, filename, vectors_deleted,
    )

    return DeleteFileResponse(
        status          = "ok",
        filename        = filename,
        vectors_deleted = vectors_deleted,
        message         = (
            f"Deleted '{filename}': {vectors_deleted} vectors removed."
        ),
    )


# ── DELETE /admin/collection ──────────────────────────────────────────────────

@router.delete("/collection", response_model=WipeResponse)
async def admin_wipe(request: Request):
    """
    Wipe the entire knowledge base (vectors + BM25 + hash registry) for this tenant.
    Admin only — irreversible.
    """
    tenant_slug = request.state.tenant_slug
    vs, bm25    = get_tenant_stores(tenant_slug)

    logger.warning(
        "[ADMIN/WIPE] ⚠  Wipe requested — tenant=%s  THIS IS IRREVERSIBLE",
        tenant_slug,
    )

    vs.reset_collection()
    bm25.reset()
    _wipe_hashes()

    logger.info("[ADMIN/WIPE] ✅ Knowledge base wiped — tenant=%s", tenant_slug)
    return WipeResponse(status="ok", message="Knowledge base wiped.")


# ── GET /admin/files ──────────────────────────────────────────────────────────

@router.get("/files")
async def admin_list_files(request: Request):
    """
    List all indexed files in this tenant's knowledge base.
    Admin only — requires a valid admin JWT.
    """
    tenant_slug = request.state.tenant_slug
    vs, _       = get_tenant_stores(tenant_slug)
    files       = vs.list_sources()

    logger.debug(
        "[ADMIN/FILES] tenant=%s  count=%d", tenant_slug, len(files)
    )
    return {"files": files}


# ── GET /admin/stats ──────────────────────────────────────────────────────────

@router.get("/stats")
async def admin_stats(request: Request):
    """
    Full system stats for this tenant's knowledge base.
    Admin only — requires a valid admin JWT.
    """
    tenant_slug = request.state.tenant_slug
    vs, bm25    = get_tenant_stores(tenant_slug)

    stats = {
        "tenant_slug"    : tenant_slug,
        "total_vectors"  : vs.count(),
        "bm25_docs"      : len(bm25),
        "indexed_files"  : vs.list_sources(),
        "embedding_model": settings.embedding_model,
        "llm_model"      : settings.groq_model,
        "collection"     : f"rag_docs_{tenant_slug}",
    }

    logger.debug(
        "[ADMIN/STATS] tenant=%s  vectors=%d  bm25=%d",
        tenant_slug, stats["total_vectors"], stats["bm25_docs"],
    )
    return stats


__all__ = ["router"]