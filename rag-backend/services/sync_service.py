# services/sync_service.py
#
# CHANGES vs previous version (VectorPullSyncService rewrite):
#
# FIX — Sync now works WITHOUT SYNC_MANIFEST_URL.
#
#   OLD behaviour:
#     The routers/sync.py trigger and network_monitor._on_reconnect both
#     checked `if not settings.sync_manifest_url: return "skipped"`.
#     This meant the entire sync was gated on a manifest URL that most
#     deployments don't have — so the sync never ran even when Qdrant Cloud
#     credentials were fully configured.
#
#   NEW behaviour:
#     SYNC_MANIFEST_URL is now OPTIONAL.  The sync is split into two
#     independent steps:
#
#       STEP 1 — Vector sync (ALWAYS runs if cloud store is configured)
#         Diffs cloud vs local Qdrant by point IDs.
#         Pulls missing points from cloud into local — no re-embedding needed.
#         Deletes local points that no longer exist in the cloud.
#         This step requires: QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY
#         (or LanceDB/Chroma cloud creds for other vendors).
#         Does NOT require SYNC_MANIFEST_URL.
#
#       STEP 2 — PDF sync (ONLY runs if SYNC_MANIFEST_URL is set)
#         Downloads missing PDF files from the manifest for the viewer.
#         Skipped silently when SYNC_MANIFEST_URL is empty.
#         Chat and retrieval work fine without PDFs — only the PDF viewer
#         modal (frontend) is affected.
#
#       STEP 3 — BM25 rebuild (async, after any vector changes)
#         Same as before.
#
#   SUMMARY:
#     - Set QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY → vector sync works.
#     - Set SYNC_MANIFEST_URL too → PDF files also download for the viewer.
#     - Neither set → sync is skipped (cloud store not configured).
#
# BACKWARD COMPATIBILITY:
#   SyncService alias preserved. All import sites unchanged.

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

    How it works:
      1. Compare point IDs in the cloud store vs local store.
      2. Pull any points that exist in cloud but not locally (no re-embedding).
      3. Delete any local points that were removed from the cloud.
      4. Optionally download missing PDFs (only if SYNC_MANIFEST_URL is set).
      5. Rebuild the BM25 index asynchronously.

    Requirements:
      - Cloud vector store credentials (QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY
        for Qdrant, or LANCEDB_CLOUD_URI / CHROMA_HOST for other vendors).
      - SYNC_MANIFEST_URL is optional — only needed for PDF file downloads.

    Thread-safe: only one sync runs at a time.
    """

    def __init__(self):
        from config import settings
        self.manifest_url = settings.sync_manifest_url   # may be empty — that's OK
        self.timeout      = 30   # seconds for HTTP requests

    # ── PUBLIC ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Main sync entry point. Thread-safe — only one sync runs at a time.

        Vector sync runs whenever cloud store creds are configured.
        PDF sync runs only when SYNC_MANIFEST_URL is also set.

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

        # Determine a human-readable status message
        import services.rag_service as _svc
        cloud_configured = _svc.get_cloud_store() is not None
        if not cloud_configured:
            message = "Cloud store not configured (set QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY)"
        elif not self.manifest_url:
            message = "Vector sync ready (SYNC_MANIFEST_URL not set — PDF download disabled)"
        else:
            message = "Sync service ready"

        return {
            "last_synced"  : last_synced,
            "is_syncing"   : is_syncing,
            "pending_count": pending_count,
            "message"      : message,
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
                "message": (
                    "Cloud store not configured — set QDRANT_CLOUD_URL + "
                    "QDRANT_CLOUD_API_KEY (or equivalent for your vendor)"
                ),
                "vectors_added": 0, "vectors_deleted": 0,
                "pdfs_downloaded": 0, "errors": [],
            }

        # ── STEP 1: Vector sync (always runs when cloud is configured) ─────
        vectors_added   = 0
        vectors_deleted = 0

        try:
            vectors_added, vectors_deleted = self._sync_vectors(cloud, local, errors)
        except Exception as e:
            err = f"Vector sync failed: {e}"
            print(f"  [SYNC] ❌ {err}")
            errors.append(err)

        # ── STEP 2: PDF sync (only when manifest URL is set) ───────────────
        pdfs_downloaded = 0
        if self.manifest_url:
            try:
                pdfs_downloaded = self._sync_pdfs(local, errors)
            except Exception as e:
                err = f"PDF sync failed: {e}"
                print(f"  [SYNC] ⚠  {err}")
                errors.append(err)
        else:
            print(
                "  [SYNC] PDF sync skipped — "
                "SYNC_MANIFEST_URL not set (vector sync completed normally)"
            )

        # ── STEP 3: BM25 rebuild (async, non-blocking) ─────────────────────
        if vectors_added > 0 or vectors_deleted > 0:
            try:
                rag_svc.rebuild_bm25_async()
            except Exception as e:
                errors.append(f"BM25 rebuild trigger failed: {e}")

        status = "error" if errors and vectors_added == 0 and vectors_deleted == 0 else "ok"
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

        This is the core sync mechanism — no SYNC_MANIFEST_URL needed.
        It works by:
          1. Fetching all point IDs from both the cloud and local stores.
          2. Computing the set difference.
          3. Pulling missing points from cloud → local (with their vectors + payloads).
          4. Deleting local points that no longer exist in the cloud.

        Returns (vectors_added, vectors_deleted).
        """
        print("  [SYNC] Diffing cloud vs local point IDs...")

        # Fetch IDs from both stores (no vectors — fast path)
        cloud_entries = cloud.get_all_ids(with_payload_fields=["source"])
        local_entries = local.get_all_ids(with_payload_fields=["source"])

        cloud_ids = {e["id"] for e in cloud_entries}
        local_ids = {e["id"] for e in local_entries}

        to_pull   = list(cloud_ids - local_ids)   # in cloud but not local → pull
        to_delete = list(local_ids - cloud_ids)   # in local but not cloud → delete

        print(
            f"  [SYNC] Cloud: {len(cloud_ids)} pts | Local: {len(local_ids)} pts | "
            f"To pull: {len(to_pull)} | To delete: {len(to_delete)}"
        )

        if len(to_pull) == 0 and len(to_delete) == 0:
            print("  [SYNC] ✅ Local store is already in sync with cloud — nothing to do")
            return 0, 0

        # ── Pull missing points from cloud ─────────────────────────────────
        vectors_added = 0
        if to_pull:
            print(f"  [SYNC] Pulling {len(to_pull)} missing points from cloud...")
            batch_size = 100
            for i in range(0, len(to_pull), batch_size):
                batch = to_pull[i : i + batch_size]
                try:
                    points = cloud.get_points_by_ids(batch)
                    if points:
                        local.upsert_from_points(points)
                        vectors_added += len(points)
                        print(
                            f"  [SYNC] Pulled batch {i // batch_size + 1}: "
                            f"{len(points)} points"
                        )
                except Exception as e:
                    err = f"Pull batch {i // batch_size + 1} failed: {e}"
                    print(f"  [SYNC] ⚠  {err}")
                    errors.append(err)

        # ── Delete stale local points ───────────────────────────────────────
        vectors_deleted = 0
        if to_delete:
            print(f"  [SYNC] Deleting {len(to_delete)} stale local points...")
            try:
                vectors_deleted = local.delete_by_ids(to_delete)
            except Exception as e:
                err = f"Delete stale local points failed: {e}"
                print(f"  [SYNC] ⚠  {err}")
                errors.append(err)

        if vectors_added > 0:
            print(f"  [SYNC] ✅ Pulled {vectors_added} new vectors from cloud")
        if vectors_deleted > 0:
            print(f"  [SYNC] 🗑  Deleted {vectors_deleted} stale local vectors")

        return vectors_added, vectors_deleted

    def _sync_pdfs(self, local, errors: list[str]) -> int:
        """
        Download PDFs that exist in the local vector store but not on disk.

        This step is OPTIONAL — only runs when SYNC_MANIFEST_URL is set.
        The manifest provides download URLs for each source PDF.

        Manifest format:
            { "docs": [{"filename": "x.pdf", "sha256": "abc", "url": "https://..."}] }

        Returns number of PDFs downloaded.
        """
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
# will get VectorPullSyncService transparently.
SyncService = VectorPullSyncService

__all__ = ["VectorPullSyncService", "SyncService"]