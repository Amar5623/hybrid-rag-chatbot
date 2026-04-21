# routers/ingest.py
#
# CHANGES vs previous version:
#
#   Supabase Storage integration (new — backward compatible):
#     After a PDF is successfully chunked and indexed, it is uploaded to the
#     configured Supabase public bucket.  The permanent public URL is then
#     injected into every chunk's metadata as "source_url" before the chunks
#     are written to the vector store.
#
#     If Supabase is NOT configured (SUPABASE_URL / SUPABASE_SERVICE_KEY empty),
#     the upload step is skipped silently and "source_url" is set to "" so
#     existing behaviour is completely unchanged.
#
#   Key changes inside _ingest_files_sync():
#     1. After chunking, call upload_pdf_to_supabase(tmp_path) → public_url.
#     2. Inject "source_url": public_url into every chunk dict.
#     3. Then index as before.
#
# Everything else (duplicate detection, PDF viewer copy, delete logic,
# hash registry, etc.) is UNCHANGED.

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from config import settings, PDFS_DIR
from services import rag_service as _svc
from ingestion.pdf_loader  import PDFLoader
from schemas import DeleteFileResponse, IngestResponse, IngestStatusResponse
from services import rag_service

router = APIRouter(tags=["ingest"])

# ── Hash registry ─────────────────────────────────────────────
_HASH_FILE = Path(settings.qdrant_path).parent / "file_hashes.json"

# ── PDFs directory (A5) ───────────────────────────────────────
_PDFS_DIR  = Path(PDFS_DIR)


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


# ── PDF file management (A5) ──────────────────────────────────

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


# ── Loader dispatch ───────────────────────────────────────────

def _get_loader(tmp_path: str, filename: str):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return PDFLoader(tmp_path)
    return None


# ── Core ingest logic (runs in threadpool) ────────────────────

def _ingest_files_sync(file_paths: list[tuple[str, str]]) -> dict:
    """
    Ingest one or more files into the vector store.

    For each PDF:
      1. Duplicate check (SHA-256).
      2. Load blocks via PDFLoader.
      3. Chunk (hierarchical or other strategy).
      4. [NEW] Upload to Supabase Storage → get public_url (skipped if not configured).
      5. [NEW] Inject source_url into every chunk dict.
      6. Add chunks to vector store + BM25.
      7. Copy PDF to data/pdfs/ for the local viewer.

    The source_url field is "" when Supabase is not configured so downstream
    code (sync engine, frontend) can safely check its truthiness.
    """
    # ── Import Supabase helper (lazy — avoids import errors if requests not installed)
    try:
        from services.supabase_storage import upload_pdf_to_supabase
        _supabase_import_ok = True
    except ImportError:
        _supabase_import_ok = False
        print("  [INGEST] ⚠  supabase_storage import failed — Supabase upload disabled")

    hashes       = _load_hashes()
    chunker      = _svc.get_chunker()
    vector_store = rag_service.get_vector_store()
    bm25_store   = rag_service.get_bm25_store()

    files_indexed : list[str] = []
    skipped       : list[str] = []
    all_children  : list[dict] = []

    # Save original PDF for viewer (early pass — keeps existing behaviour)
    pdfs_dir = Path(settings.qdrant_path).parent / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    for tmp_path_str, filename in file_paths:
        tmp_path = Path(tmp_path_str)
        if tmp_path.exists():
            shutil.copy2(tmp_path, pdfs_dir / filename)

    for tmp_path, filename in file_paths:
        # ── 1. Duplicate check ────────────────────────────
        raw   = Path(tmp_path).read_bytes()
        fhash = hashlib.sha256(raw).hexdigest()

        if fhash in hashes:
            print(f"  [INGEST] Skipping duplicate: {filename}")
            skipped.append(filename)
            continue

        # ── 2. Load ───────────────────────────────────────
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

        # ── 3. Chunk ──────────────────────────────────────
        from ingestion.chunker import HierarchicalChunker
        if isinstance(chunker, HierarchicalChunker):
            children = chunker.chunk_hierarchical(blocks)
        else:
            children = chunker.chunk_documents(blocks)

        # ── 4. Upload to Supabase Storage ─────────────────
        # Attempt upload only for PDFs and only when Supabase is configured.
        # On failure we log and continue — ingestion is not blocked.
        source_url = ""
        if _supabase_import_ok and Path(filename).suffix.lower() == ".pdf":
            try:
                public_url = upload_pdf_to_supabase(tmp_path)
                if public_url:
                    source_url = public_url
                    # Log once per file (not per chunk)
                    print(
                        f"  [INGEST] [SUPABASE] source_url set for "
                        f"'{filename}': {source_url}"
                    )
                else:
                    print(
                        f"  [INGEST] [SUPABASE] Upload returned None for "
                        f"'{filename}' — source_url left empty"
                    )
            except Exception as exc:
                print(
                    f"  [INGEST] [SUPABASE] Upload exception for "
                    f"'{filename}': {exc} — continuing without source_url"
                )

        # ── 5. Inject source_url into every chunk ─────────
        # This happens regardless of whether upload succeeded:
        #   - source_url = "<public_url>"  → Supabase upload succeeded
        #   - source_url = ""              → Supabase not configured or failed
        # Downstream code can always do: chunk.get("source_url") or ""
        for child in children:
            child["source_url"] = source_url

        all_children.extend(children)
        files_indexed.append(filename)
        hashes[fhash] = filename

        # ── 6. Copy PDF to local viewer store ────────────
        if Path(filename).suffix.lower() == ".pdf":
            _store_pdf_file(tmp_path, filename)

    # ── 7. Index all chunks ───────────────────────────────
    if all_children:
        vector_store.add_documents(all_children)
        bm25_store.add(all_children)

    _save_hashes(hashes)

    return {
        "files_indexed": files_indexed,
        "skipped"      : skipped,
        "total_chunks" : len(all_children),
        "total_parents": len(all_children),
    }


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)):
    # Save original PDFs for the viewer before processing
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
        raise HTTPException(status_code=400, detail="No files provided.")

    tmp_dir    = Path("/tmp") / f"rag_ingest_{uuid.uuid4().hex}"
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


@router.delete("/ingest/{filename}", response_model=DeleteFileResponse)
async def delete_file(filename: str):
    """
    Delete a file from the knowledge base.

    RULES:
      - Must be ONLINE to delete (cloud is authoritative).
      - Must have a cloud store configured (QDRANT_CLOUD_URL set).
      - Deletes from CLOUD only — local vectors are cleaned up by the
        next sync run (sync engine diffs cloud vs local and removes stale points).
      - BM25 index and local PDF file are cleaned up immediately.

    BLOCKED when:
      - User is offline (network monitor says no connectivity).
      - No cloud store is configured (pure local deployment has no sync,
        so deleting locally would cause permanent data divergence on reconnect).
    """

    # ── Guard 1: must be online ───────────────────────────
    if not rag_service.is_online():
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
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
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                f"Cannot delete '{filename}': no cloud store is configured. "
                "Set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY in .env to enable deletion."
            ),
        )

    # ── Guard 3: file must exist in cloud ─────────────────
    cloud_sources = cloud_store.list_sources()
    if filename not in cloud_sources:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
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


@router.get("/ingest/status/{task_id}", response_model=IngestStatusResponse)
async def ingest_status(task_id: str):
    task = rag_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return IngestStatusResponse(**task)


@router.post("/ingest/sync")
async def trigger_sync():
    """
    Manually trigger a sync check.
    Called automatically by NetworkMonitor when internet is detected.
    """
    from services.sync_service import SyncService
    sync = SyncService()

    if rag_service.get_cloud_store() is None:
        return {"status": "skipped", "message": "Cloud store not configured."}

    import asyncio
    asyncio.create_task(run_in_threadpool(sync.run))
    return {"status": "triggered", "message": "Sync started in background."}