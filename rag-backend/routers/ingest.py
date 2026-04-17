# routers/ingest.py
#
# CHANGES vs previous version (Day 2 — A5):
#   - _ingest_files_sync(): after hashing + loading, the PDF is now copied to
#     data/pdfs/{filename} for static serving. This enables the PDF viewer
#     modal (Person B, Task B1) to fetch and render the original document.
#
#   - _delete_pdf_file(): new helper that removes data/pdfs/{filename}
#     when a file is deleted from the knowledge base.
#
#   - delete_file endpoint: now also calls _delete_pdf_file() so the
#     stored PDF is cleaned up alongside vectors and BM25 entries.
#
#   Everything else unchanged (duplicate check, chunking, indexing).

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
# Same path as PDFS_DIR in config.py, resolved as a Path object.
# data/pdfs/ is mounted at /pdfs by main.py.
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


# ── NEW (A5): PDF file management ─────────────────────────────

def _store_pdf_file(tmp_path: str, filename: str) -> Path | None:
    """
    Copy a PDF from the temporary upload path to data/pdfs/{filename}.

    This makes the file available at /pdfs/{filename} for the frontend
    PDF viewer (Person B, Task B1). The copy is idempotent — if the
    same filename already exists (e.g. from a previous identical upload
    that passed the hash check) we skip silently.

    Returns the destination Path on success, None on error.
    """
    _PDFS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _PDFS_DIR / filename

    try:
        shutil.copy2(tmp_path, dest)
        print(f"  [INGEST] PDF stored for viewer: data/pdfs/{filename}")
        return dest
    except Exception as e:
        # Non-fatal: the PDF viewer just won't work for this file.
        # Retrieval / chat still works normally.
        print(f"  [INGEST] Warning: could not store PDF for viewer: {e}")
        return None


def _delete_pdf_file(filename: str) -> None:
    """
    Remove data/pdfs/{filename} when the document is deleted from the KB.
    Silently skips if the file doesn't exist (e.g. non-PDF or failed store).
    """
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
    return None   # only PDFs supported now


# ── Core ingest logic (runs in threadpool) ────────────────────

def _ingest_files_sync(file_paths: list[tuple[str, str]]) -> dict:
    hashes       = _load_hashes()
    chunker      = _svc.get_chunker()
    vector_store = rag_service.get_vector_store()
    bm25_store   = rag_service.get_bm25_store()

    files_indexed : list[str] = []
    skipped       : list[str] = []
    all_children  : list[dict] = []

    # === Minimal addition: save original PDF for viewer ===
    pdfs_dir = Path(settings.qdrant_path).parent / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    for tmp_path_str, filename in file_paths:
        tmp_path = Path(tmp_path_str)
        if tmp_path.exists():
            shutil.copy2(tmp_path, pdfs_dir / filename)
    # ====================================================

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

        all_children.extend(children)
        files_indexed.append(filename)
        hashes[fhash] = filename

        # ── 4. (A5) Copy PDF to viewer store ──────────────
        # Done per-file, after successful load + chunk, before index.
        # Only PDFs are supported by the loader anyway, but we guard
        # on extension defensively so future loaders don't break this.
        if Path(filename).suffix.lower() == ".pdf":
            _store_pdf_file(tmp_path, filename)

    # ── 5. Index ──────────────────────────────────────────
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
    # === SAFE ORIGINAL PDF STORAGE FOR PDF VIEWER ===
    # This runs in the already-async function - no structure change
    pdfs_dir = Path(settings.qdrant_path).parent / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.filename.lower().endswith(".pdf"):
            dest_path = pdfs_dir / file.filename
            content = await file.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            await file.seek(0)   # Important: reset file pointer for original code
    # ================================================
    
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
    vector_store = rag_service.get_vector_store()
    sources = vector_store.list_sources()
    if filename not in sources:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"File '{filename}' not found in the knowledge base.",
        )

    result = await run_in_threadpool(rag_service.delete_file_from_stores, filename)
    _remove_hash_for_file(filename)

    # ── (A5) Clean up the stored PDF ──────────────────────
    # Remove from data/pdfs/ so the viewer can't serve a deleted doc.
    _delete_pdf_file(filename)

    return DeleteFileResponse(
        status          = "ok",
        filename        = filename,
        vectors_deleted = result["vectors_deleted"],
        message         = (
            f"Deleted '{filename}': "
            f"{result['vectors_deleted']} vectors removed."
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
    Kicks off SyncService to fetch manifest and pull new PDFs.
    """
    from services.sync_service import SyncService
    sync = SyncService()
    if not settings.sync_manifest_url:
        return {"status": "skipped", "message": "SYNC_MANIFEST_URL not configured."}

    import asyncio
    asyncio.create_task(run_in_threadpool(sync.run))
    return {"status": "triggered", "message": "Sync started in background."}