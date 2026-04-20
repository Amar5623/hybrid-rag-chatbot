# services/sync_service.py
#
# Person A — Phase 5 (Day 6-8)
#
# FULL REWRITE — replaces the original PDF-download-and-re-embed approach
# with a vector-pull sync that copies points directly from the cloud store.
#
# OLD approach (original SyncService):
#   1. Fetch manifest JSON from central server
#   2. Download PDFs that are new/changed
#   3. Run the full ingest pipeline (chunk → embed → upsert)
#   Problem: expensive, slow, requires internet for the full embedding run.
#
# NEW approach (VectorPullSyncService):
#   STEP 1 — Vector sync
#     a. Scroll cloud store → all point IDs + source + sha256 fields
#     b. Scroll local store → same
#     c. Diff: points in cloud but not local → fetch + upsert (no embed)
#              points in local but not cloud → delete (admin removed a doc)
#
#   STEP 2 — PDF sync
#     For every source now in local store, check if the PDF exists in
#     data/pdfs/. Download any missing ones from a manifest URL.
#     (The manifest still provides download URLs — we just no longer use
#     it to drive re-embedding.)
#
#   STEP 3 — BM25 rebuild
#     Trigger rag_service.rebuild_bm25_async() after vector sync.
#     Non-blocking — runs in a background thread so reconnect is instant.
#
# PREREQUISITES:
#   - The cloud Qdrant collection must have been populated by the same admin
#     pipeline that uses the same payload schema as add_documents().
#   - Cloud store creds must be set (QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY
#     for Qdrant, or equivalent for LanceDB/Chroma).
#   - SYNC_MANIFEST_URL must point to a JSON manifest with PDF download URLs.
#     Format:  { "docs": [{"filename": "x.pdf", "sha256": "abc", "url": "https://..."}] }
#
# BACKWARD COMPATIBILITY:
#   The old SyncService is preserved as a class alias so existing imports
#   don't break. Swap is silent.

import json
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "sync_manifest.json"
_SYNC_LOG_PATH  = Path(__file__).parent.parent / "data" / "sync_log.json"
_SYNC_LOCK      = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR-PULL SYNC SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class VectorPullSyncService:
    """
    Sync the local vector store from the cloud store without re-embedding.

    Three-step process on each sync trigger:
      1. Vector sync  — copy new/changed points from cloud to local
      2. PDF sync     — download any missing PDF files
      3. BM25 rebuild — rebuild sparse index from local store (async)

    Thread-safe: only one sync runs at a time.
    """

    def __init__(self):
        from config import settings
        self.manifest_url = settings.sync_manifest_url
        self.timeout      = 30   # seconds for HTTP requests

    # ── PUBLIC ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Main sync method. Thread-safe — only one sync runs at a time.

        Returns:
            Status dict with keys: status, vectors_added, vectors_deleted,
            pdfs_downloaded, errors, duration_s
        """
        if not _SYNC_LOCK.acquire(blocking=False):
            return {"status": "skipped", "message": "Sync already in progress"}

        import time
        t0 = time.time()

        try:
            print("  [SYNC] Starting vector-pull sync...")
            result = self._run_sync()
            result["duration_s"] = round(time.time() - t0, 2)
            self._log_sync(result)
            print(
                f"  [SYNC] Done in {result['duration_s']}s — "
                f"added: {result.get('vectors_added', 0)}, "
                f"deleted: {result.get('vectors_deleted', 0)}, "
                f"pdfs: {result.get('pdfs_downloaded', 0)}, "
                f"errors: {len(result.get('errors', []))}"
            )
            return result
        finally:
            _SYNC_LOCK.release()

    def get_status(self) -> dict:
        """Return sync status for the /sync/status endpoint."""
        is_syncing = not _SYNC_LOCK.acquire(blocking=False)
        if not is_syncing:
            _SYNC_LOCK.release()

        last_synced = None
        if _SYNC_LOG_PATH.exists():
            try:
                log = json.loads(_SYNC_LOG_PATH.read_text())
                if log:
                    last_synced = log[-1].get("timestamp")
            except Exception:
                pass

        # Estimate pending count from cloud vs local diff (fast path — IDs only)
        pending_count = 0
        try:
            import services.rag_service as rag_svc
            cloud = rag_svc.get_cloud_store()
            local = rag_svc.get_local_store()
            if cloud and local:
                cloud_ids = {p["id"] for p in cloud.get_all_ids()}
                local_ids = {p["id"] for p in local.get_all_ids()}
                pending_count = len(cloud_ids - local_ids)
        except Exception:
            pass

        return {
            "last_synced"  : last_synced,
            "is_syncing"   : is_syncing,
            "pending_count": pending_count,
            "message"      : "Sync service ready" if self.manifest_url else "SYNC_MANIFEST_URL not configured",
        }

    # ── INTERNAL ──────────────────────────────────────────────────────────

    def _run_sync(self) -> dict:
        errors: list[str] = []

        import services.rag_service as rag_svc
        cloud = rag_svc.get_cloud_store()
        local = rag_svc.get_local_store()

        if cloud is None:
            return {
                "status" : "skipped",
                "message": "Cloud store not configured — no cloud creds set",
                "vectors_added": 0, "vectors_deleted": 0,
                "pdfs_downloaded": 0, "errors": [],
            }

        # ── STEP 1: Vector sync ────────────────────────────────────────────
        vectors_added   = 0
        vectors_deleted = 0

        try:
            vectors_added, vectors_deleted = self._sync_vectors(cloud, local, errors)
        except Exception as e:
            err = f"Vector sync failed: {e}"
            print(f"  [SYNC] ❌ {err}")
            errors.append(err)

        # ── STEP 2: PDF sync ───────────────────────────────────────────────
        pdfs_downloaded = 0
        try:
            pdfs_downloaded = self._sync_pdfs(local, errors)
        except Exception as e:
            err = f"PDF sync failed: {e}"
            print(f"  [SYNC] ⚠  {err}")
            errors.append(err)

        # ── STEP 3: BM25 rebuild (async, non-blocking) ─────────────────────
        if vectors_added > 0 or vectors_deleted > 0:
            try:
                rag_svc.rebuild_bm25_async()
            except Exception as e:
                errors.append(f"BM25 rebuild trigger failed: {e}")

        status = "error" if errors and vectors_added == 0 else "ok"
        return {
            "status"          : status,
            "vectors_added"   : vectors_added,
            "vectors_deleted" : vectors_deleted,
            "pdfs_downloaded" : pdfs_downloaded,
            "errors"          : errors,
        }

    def _sync_vectors(
        self,
        cloud,
        local,
        errors: list[str],
    ) -> tuple[int, int]:
        """
        Diff cloud vs local by point IDs and sync the delta.

        Returns (vectors_added, vectors_deleted).
        """
        print("  [SYNC] Diffing cloud vs local point IDs...")

        # Fetch IDs from both stores (no vectors — fast path)
        cloud_entries = cloud.get_all_ids(with_payload_fields=["source"])
        local_entries = local.get_all_ids(with_payload_fields=["source"])

        cloud_ids = {e["id"] for e in cloud_entries}
        local_ids = {e["id"] for e in local_entries}

        to_pull   = list(cloud_ids - local_ids)   # in cloud but not local
        to_delete = list(local_ids - cloud_ids)   # in local but not cloud (deleted upstream)

        print(f"  [SYNC] Cloud: {len(cloud_ids)} | Local: {len(local_ids)} | "
              f"To pull: {len(to_pull)} | To delete: {len(to_delete)}")

        # ── Pull missing points from cloud ─────────────────────────────────
        vectors_added = 0
        if to_pull:
            batch_size = 100
            for i in range(0, len(to_pull), batch_size):
                batch = to_pull[i : i + batch_size]
                try:
                    points = cloud.get_points_by_ids(batch)
                    if points:
                        local.upsert_from_points(points)
                        vectors_added += len(points)
                        print(f"  [SYNC] Pulled batch {i//batch_size + 1}: {len(points)} points")
                except Exception as e:
                    err = f"Pull batch {i//batch_size + 1} failed: {e}"
                    print(f"  [SYNC] ⚠  {err}")
                    errors.append(err)

        # ── Delete removed points from local ────────────────────────────────
        vectors_deleted = 0
        if to_delete:
            try:
                vectors_deleted = local.delete_by_ids(to_delete)
            except Exception as e:
                err = f"Delete stale local points failed: {e}"
                print(f"  [SYNC] ⚠  {err}")
                errors.append(err)

        return vectors_added, vectors_deleted

    def _sync_pdfs(self, local, errors: list[str]) -> int:
        """
        Download PDFs that exist in the local vector store but not on disk.

        Uses the manifest URL for download links.
        Returns number of PDFs downloaded.
        """
        if not self.manifest_url:
            return 0

        from config import PDFS_DIR
        pdfs_dir = Path(PDFS_DIR)
        pdfs_dir.mkdir(parents=True, exist_ok=True)

        # Get all sources currently in the local store
        local_sources = set(local.list_sources())
        if not local_sources:
            return 0

        # Fetch the manifest for download URLs
        try:
            manifest = self._fetch_manifest()
        except Exception as e:
            errors.append(f"Manifest fetch for PDF sync failed: {e}")
            return 0

        url_map: dict[str, str] = {
            doc["filename"]: doc["url"]
            for doc in manifest.get("docs", [])
            if doc.get("filename") and doc.get("url")
        }

        # Download only sources that are missing from disk
        downloaded = 0
        for source in local_sources:
            pdf_path = pdfs_dir / source
            if pdf_path.exists():
                continue  # already on disk

            url = url_map.get(source)
            if not url:
                continue  # no URL in manifest (admin-uploaded directly?)

            try:
                print(f"  [SYNC] Downloading PDF: {source}")
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    pdf_path.write_bytes(resp.read())
                downloaded += 1
            except Exception as e:
                err = f"PDF download failed for '{source}': {e}"
                print(f"  [SYNC] ⚠  {err}")
                errors.append(err)

        if downloaded:
            print(f"  [SYNC] ✅ Downloaded {downloaded} PDF(s)")
        return downloaded

    def _fetch_manifest(self) -> dict:
        with urllib.request.urlopen(self.manifest_url, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _log_sync(self, result: dict) -> None:
        log: list[dict] = []
        if _SYNC_LOG_PATH.exists():
            try:
                log = json.loads(_SYNC_LOG_PATH.read_text())
            except Exception:
                pass

        log.append({
            "timestamp"      : datetime.now(timezone.utc).isoformat(),
            "vectors_added"  : result.get("vectors_added",   0),
            "vectors_deleted": result.get("vectors_deleted",  0),
            "pdfs_downloaded": result.get("pdfs_downloaded",  0),
            "duration_s"     : result.get("duration_s",       0),
            "errors"         : result.get("errors",           []),
        })

        # Keep last 50 entries
        log = log[-50:]
        _SYNC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SYNC_LOG_PATH.write_text(json.dumps(log, indent=2))


# ── BACKWARD COMPATIBILITY ALIAS ─────────────────────────────────────────────
# Old code that imports: from services.sync_service import SyncService
# will get the new VectorPullSyncService transparently.
SyncService = VectorPullSyncService

__all__ = ["VectorPullSyncService", "SyncService"]