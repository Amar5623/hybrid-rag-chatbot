# rag-backend/routers/ingest.py
#
# Phase 2 — Authentication System
#
# CHANGES vs previous version:
#   - POST   /ingest             — now requires JWT auth (resolve_tenant + require_admin_role)
#   - DELETE /ingest/{filename}  — now requires JWT auth (resolve_tenant + require_admin_role)
#   - _ingest_files_sync()       — now accepts optional tenant_slug parameter.
#     When provided, uses get_tenant_stores(tenant_slug) instead of global singletons.
#     Backward compatible: tenant_slug=None → falls back to global stores (dev mode).
#   - Supabase upload path scoped per-tenant: pdfs/{tenant_slug}/{filename}
#
# Read-only routes (GET /ingest/status, POST /ingest/sync) remain open —
# they carry no write risk and are used by the sync engine.
#
# All existing ingest logic (hash dedup, chunking, BM25, vector store) is UNCHANGED.
# Only the stores passed in and the auth gate change.

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from config        import settings, PDFS_DIR
from services      import rag_service as _svc
from services      import rag_service
from ingestion.pdf_loader import PDFLoader
from schemas       import DeleteFileResponse, IngestResponse, IngestStatusResponse
from utils.logger  import get_logger

# Phase 2 — JWT auth dependencies
from middleware.tenant_resolver import resolve_tenant, require_admin_role
from services.rag_service       import get_tenant_stores

# NOTE: kb.py caches are imported locally inside functions to avoid circular deps

logger = get_logger(__name__)

router = APIRouter(tags=["ingest"])


# ── Hash registry ─────────────────────────────────────────────────────────────

_HASH_FILE = Path(settings.qdrant_path).parent / "file_hashes.json"

# ── PDFs directory ────────────────────────────────────────────────────────────

_PDFS_DIR = Path(PDFS_DIR)


def _load_hashes() -> dict:
    if _HASH_FILE.exists():
        try:
            return json.loads(_HASH_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_hashes(hashes: dict) -> None:
    _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HASH_FILE.write_text(json.dumps(hashes, indent=2))


def _wipe_hashes() -> None:
    if _HASH_FILE.exists():
        _HASH_FILE.unlink()


def _remove_hash_for_file(filename: str) -> None:
    hashes  = _load_hashes()
    updated = {h: f for h, f in hashes.items() if f != filename}
    _save_hashes(updated)


# ── PDF file management ───────────────────────────────────────────────────────

def _store_pdf_file(tmp_path: str, filename: str) -> Path | None:
    _PDFS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _PDFS_DIR / filename
    try:
        shutil.copy2(tmp_path, dest)
        print(f"  [INGEST] PDF stored for viewer: data/pdfs/{filename}")
        return dest
    except Exception as e:
        print(f"  [INGEST] Warning: could not store PDF for viewer: {e}")
        return None


def _delete_pdf_file(filename: str) -> None:
    pdf_path = _PDFS_DIR / filename
    if pdf_path.exists():
        try:
            pdf_path.unlink()
            print(f"  [INGEST] PDF deleted from viewer store: data/pdfs/{filename}")
        except Exception as e:
            print(f"  [INGEST] Warning: could not delete PDF from viewer store: {e}")


# ── Loader dispatch ───────────────────────────────────────────────────────────

def _get_loader(tmp_path: str, filename: str):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return PDFLoader(tmp_path)
    return None


# ── Core ingest logic (runs in threadpool) ────────────────────────────────────

def _ingest_files_sync(
    file_paths  : list[tuple[str, str]],
    tenant_slug : str = None,           # Phase 2 — thread tenant scope through ingest
) -> dict:
    """
    Ingest one or more files into the vector store and BM25 index.

    Phase 2 change:
      When tenant_slug is provided, uses get_tenant_stores(tenant_slug) to get
      the tenant-scoped vector store and BM25 index. This ensures documents are
      stored in the correct per-tenant Qdrant collection (rag_docs_{tenant_slug})
      and the correct per-tenant BM25 file (bm25_{tenant_slug}.pkl).

      When tenant_slug is None (legacy / single-tenant dev mode), falls back to
      the global singletons from rag_service — backward compatible.

    For each PDF:
      1. Remove existing vectors/BM25 chunks for this filename (overwrite support).
      2. Duplicate check (SHA-256) — guards against same file twice in one batch.
      3. Load blocks via PDFLoader.
      4. Chunk (hierarchical or other strategy).
      5. [Supabase] Upload to Supabase Storage → get public_url (skipped if not configured).
      6. [Supabase] Inject source_url into every chunk dict.
      7. Add chunks to vector store + BM25.
      8. Copy PDF to data/pdfs/ for the local viewer.
    """
    try:
        from services.supabase_storage import upload_pdf_to_supabase
        _supabase_import_ok = True
    except ImportError:
        _supabase_import_ok = False
        print("  [INGEST] ⚠  supabase_storage import failed — Supabase upload disabled")

    # ── Resolve stores: tenant-scoped or global fallback ─────────────────────
    if tenant_slug:
        vector_store, bm25_store = get_tenant_stores(tenant_slug)
        logger.info(
            "[INGEST] Using tenant-scoped stores — slug=%s", tenant_slug
        )
    else:
        vector_store = rag_service.get_vector_store()
        bm25_store   = rag_service.get_bm25_store()
        logger.info("[INGEST] Using global stores (single-tenant / dev mode)")

    hashes       = _load_hashes()
    chunker      = _svc.get_chunker()

    files_indexed : list[str] = []
    skipped       : list[str] = []
    all_children  : list[dict] = []

    # Save original PDFs for viewer (early pass — keeps existing behaviour)
    pdfs_dir = Path(settings.qdrant_path).parent / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    for tmp_path_str, filename in file_paths:
        tmp_path = Path(tmp_path_str)
        if tmp_path.exists():
            shutil.copy2(tmp_path, pdfs_dir / filename)

    # Track sha256s seen in THIS batch only
    batch_hashes: set[str] = set()

    for tmp_path, filename in file_paths:

        # ── 1. Remove existing data for this filename before hash check ───────
        existing_sources = vector_store.list_sources()
        if filename in existing_sources:
            print(f"  [INGEST] Overwrite detected for '{filename}' — removing old data")
            vector_store.delete_by_source(filename)
            bm25_store.delete_by_source(filename)
            _remove_hash_for_file(filename)
            hashes = _load_hashes()
            print(f"  [INGEST] Old data cleared for '{filename}' — re-indexing fresh")

        # ── 2. Duplicate check (within THIS batch only) ───────────────────────
        raw   = Path(tmp_path).read_bytes()
        fhash = hashlib.sha256(raw).hexdigest()

        if fhash in batch_hashes:
            print(f"  [INGEST] Skipping duplicate in batch: {filename}")
            skipped.append(filename)
            continue

        batch_hashes.add(fhash)

        # ── 3. Load ───────────────────────────────────────────────────────────
        loader = _get_loader(tmp_path, filename)
        if not loader:
            print(f"  [INGEST] Unsupported file type: {filename}")
            skipped.append(filename)
            continue

        try:
            blocks = loader.load()
        except Exception as e:
            print(f"  [INGEST] Load failed for {filename}: {e}")
            skipped.append(filename)
            continue

        if not blocks:
            skipped.append(filename)
            continue

        for b in blocks:
            b["source"] = filename

        # ── 4. Chunk ──────────────────────────────────────────────────────────
        from ingestion.chunker import HierarchicalChunker
        if isinstance(chunker, HierarchicalChunker):
            children = chunker.chunk_hierarchical(blocks)
        else:
            children = chunker.chunk_documents(blocks)

        # ── 5. Upload to Supabase Storage (tenant-scoped path) ────────────────
        source_url = ""
        if _supabase_import_ok and Path(filename).suffix.lower() == ".pdf":
            try:
                # Phase 2: Pass tenant_slug so upload path is pdfs/{slug}/{filename}
                public_url = upload_pdf_to_supabase(
                    tmp_path,
                    tenant_slug=tenant_slug or "",
                )
                if public_url:
                    source_url = public_url
                    print(f"  [INGEST] [SUPABASE] source_url set: {source_url}")
                else:
                    print(f"  [INGEST] [SUPABASE] Upload returned None — source_url left empty")
            except Exception as exc:
                print(f"  [INGEST] [SUPABASE] Upload exception: {exc} — continuing")

        # ── 6. Inject source_url into every chunk ─────────────────────────────
        for child in children:
            child["source_url"] = source_url

        all_children.extend(children)
        files_indexed.append(filename)
        hashes[fhash] = filename

        # ── 7. Copy PDF to local viewer store ─────────────────────────────────
        if Path(filename).suffix.lower() == ".pdf":
            _store_pdf_file(tmp_path, filename)

    # ── 8. Index all new chunks ───────────────────────────────────────────────
    if all_children:
        vector_store.add_documents(all_children)
        bm25_store.add(all_children)

    _save_hashes(hashes)

    logger.info(
        "[INGEST] Complete — tenant=%s  indexed=%s  chunks=%d  skipped=%s",
        tenant_slug or "(global)",
        files_indexed,
        len(all_children),
        skipped,
    )

    return {
        "files_indexed": files_indexed,
        "skipped"      : skipped,
        "total_chunks" : len(all_children),
        "total_parents": len(all_children),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model = IngestResponse,
    dependencies   = [Depends(resolve_tenant), Depends(require_admin_role)],
)
async def ingest(request: Request, files: list[UploadFile] = File(...)):
    """
    Upload and index one or more PDFs into the knowledge base.

    Phase 2: Requires a valid admin JWT (Authorization: Bearer <access_token>).
    Documents are stored in the tenant-scoped vector collection and BM25 index.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    tenant_slug = request.state.tenant_slug

    # Save PDFs for the viewer
    pdfs_dir = Path(settings.qdrant_path).parent / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.filename.lower().endswith(".pdf"):
            dest_path = pdfs_dir / file.filename
            content   = await file.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            await file.seek(0)

    tmp_dir    = Path("/tmp") / f"rag_ingest_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_paths : list[tuple[str, str]] = []

    try:
        for upload in files:
            tmp_path = tmp_dir / upload.filename
            content  = await upload.read()
            tmp_path.write_bytes(content)
            file_paths.append((str(tmp_path), upload.filename))

        result = await run_in_threadpool(
            _ingest_files_sync, file_paths, tenant_slug
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Invalidate kb.py vector/source caches so next export sees fresh data
    try:
        from routers.kb import _vec_cache, _source_hash_cache
        _vec_cache.clear()
        _source_hash_cache.clear()
    except Exception:
        pass

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


@router.delete(
    "/ingest/{filename}",
    response_model = DeleteFileResponse,
    dependencies   = [Depends(resolve_tenant), Depends(require_admin_role)],
)
async def delete_file(request: Request, filename: str):
    """
    Delete a file from the knowledge base.

    Phase 2: Requires a valid admin JWT.

    RULES:
      - Must be ONLINE to delete (cloud is authoritative).
      - Must have a cloud store configured (QDRANT_CLOUD_URL set).
      - Deletes from the tenant's vector store + BM25 immediately.
      - Also cleans up from global cloud store if present.
    """
    tenant_slug = request.state.tenant_slug
    vs, bm25    = get_tenant_stores(tenant_slug)

    if not rag_service.is_online():
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                f"Cannot delete '{filename}' while offline. "
                "Deletion requires a cloud connection. Please reconnect and try again."
            ),
        )

    cloud_store = rag_service.get_cloud_store()
    if cloud_store is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                f"Cannot delete '{filename}': no cloud store is configured. "
                "Set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY in .env."
            ),
        )

    # Check existence in the tenant's store
    tenant_sources = vs.list_sources()
    if filename not in tenant_sources:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"File '{filename}' not found in the knowledge base.",
        )

    # Delete from tenant vector store
    vectors_deleted = await run_in_threadpool(vs.delete_by_source, filename)

    # Clean up tenant BM25 immediately
    bm25.delete_by_source(filename)

    # Best-effort cloud store cleanup
    try:
        cloud_sources = cloud_store.list_sources()
        if filename in cloud_sources:
            await run_in_threadpool(rag_service.delete_file_from_cloud, filename)
    except Exception as exc:
        logger.warning("[INGEST/DELETE] Cloud cleanup skipped: %s", exc)

    _remove_hash_for_file(filename)
    _delete_pdf_file(filename)

    # Invalidate kb.py caches
    try:
        from routers.kb import _vec_cache, _source_hash_cache
        _vec_cache.clear()
        _source_hash_cache.clear()
    except Exception:
        pass

    logger.info(
        "[INGEST/DELETE] ✅ tenant=%s  file=%s  vectors_deleted=%d",
        tenant_slug, filename, vectors_deleted,
    )

    return DeleteFileResponse(
        status          = "ok",
        filename        = filename,
        vectors_deleted = vectors_deleted,
        message         = (
            f"Deleted '{filename}': {vectors_deleted} vectors removed. "
            "Local vectors will be cleaned up on next sync."
        ),
    )


@router.get("/ingest/status/{task_id}", response_model=IngestStatusResponse)
async def ingest_status(task_id: str):
    """Return status of an async ingest task by ID."""
    task = rag_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return IngestStatusResponse(**task)


@router.post("/ingest/sync")
async def trigger_sync():
    """
    Manually trigger a sync check.
    Called automatically by NetworkMonitor when internet is detected.

    Note: This route intentionally does NOT require auth — it is called by
    internal services (network monitor, mobile app reconnect logic).
    """
    from services.sync_service import SyncService
    sync = SyncService()

    if rag_service.get_cloud_store() is None:
        return {"status": "skipped", "message": "Cloud store not configured."}

    import asyncio
    asyncio.create_task(run_in_threadpool(sync.run))
    return {"status": "triggered", "message": "Sync started in background."}


__all__ = ["router", "_ingest_files_sync", "_store_pdf_file", "_delete_pdf_file",
           "_remove_hash_for_file", "_wipe_hashes"]