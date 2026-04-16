# services/sync_service.py
#
# NEW FILE.
# Handles document sync logic.
# On trigger:
#   1. Fetch manifest JSON from central server
#   2. Compare against local manifest.json
#   3. Download and ingest only new/changed PDFs
#   4. Update local manifest after successful sync
#   5. Log sync history

import json
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "sync_manifest.json"
_SYNC_LOG_PATH = Path(__file__).parent.parent / "data" / "sync_log.json"
_SYNC_LOCK     = threading.Lock()


class SyncService:
    """
    Compares a local manifest against a central server manifest
    and downloads + ingests any new or changed PDFs.

    Manifest format (both local and remote):
        {
          "version": 1,
          "docs": [
            { "filename": "engine_manual.pdf", "sha256": "abc123...", "url": "https://..." },
            ...
          ]
        }

    Local manifest is stored at data/sync_manifest.json.
    Sync log is stored at data/sync_log.json.
    """

    def __init__(self):
        from config import settings
        self.manifest_url  = settings.sync_manifest_url
        self.timeout       = 30   # seconds per download

    # ── PUBLIC ────────────────────────────────────────────

    def run(self) -> dict:
        """
        Main sync method. Thread-safe — only one sync runs at a time.
        Returns a status dict: {"synced": [...], "skipped": [...], "errors": [...]}
        """
        if not self.manifest_url:
            return {"status": "skipped", "message": "SYNC_MANIFEST_URL not set"}

        if not _SYNC_LOCK.acquire(blocking=False):
            return {"status": "skipped", "message": "Sync already in progress"}

        try:
            print("  [SYNC] Starting sync...")
            result = self._run_sync()
            self._log_sync(result)
            print(f"  [SYNC] Done. Synced: {len(result['synced'])} | "
                  f"Skipped: {len(result['skipped'])} | "
                  f"Errors: {len(result['errors'])}")
            return result
        finally:
            _SYNC_LOCK.release()

    # ── INTERNAL ──────────────────────────────────────────

    def _run_sync(self) -> dict:
        synced  : list[str] = []
        skipped : list[str] = []
        errors  : list[str] = []

        # 1. Fetch remote manifest
        try:
            remote = self._fetch_remote_manifest()
        except Exception as e:
            print(f"  [SYNC] Failed to fetch remote manifest: {e}")
            return {"status": "error", "synced": [], "skipped": [], "errors": [str(e)]}

        # 2. Load local manifest
        local = self._load_local_manifest()
        local_hashes = {doc["filename"]: doc.get("sha256", "") for doc in local.get("docs", [])}

        # 3. Find new / changed docs
        to_download = []
        for doc in remote.get("docs", []):
            fname  = doc.get("filename", "")
            sha256 = doc.get("sha256",   "")
            url    = doc.get("url",      "")

            if not fname or not url:
                continue

            if local_hashes.get(fname) == sha256:
                skipped.append(fname)
                continue

            to_download.append(doc)

        if not to_download:
            print("  [SYNC] No new documents to download.")
            return {"status": "ok", "synced": synced, "skipped": skipped, "errors": errors}

        # 4. Download and ingest each new doc
        import tempfile, shutil
        tmp_dir = Path(tempfile.mkdtemp(prefix="rag_sync_"))
        try:
            for doc in to_download:
                fname = doc["filename"]
                url   = doc["url"]
                try:
                    tmp_path = tmp_dir / fname
                    self._download_file(url, tmp_path)
                    self._ingest_file(str(tmp_path), fname)
                    synced.append(fname)
                except Exception as e:
                    print(f"  [SYNC] Error syncing '{fname}': {e}")
                    errors.append(f"{fname}: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # 5. Update local manifest
        self._save_local_manifest(remote)

        return {"status": "ok", "synced": synced, "skipped": skipped, "errors": errors}

    def _fetch_remote_manifest(self) -> dict:
        with urllib.request.urlopen(self.manifest_url, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _load_local_manifest(self) -> dict:
        if _MANIFEST_PATH.exists():
            try:
                return json.loads(_MANIFEST_PATH.read_text())
            except Exception:
                pass
        return {"docs": []}

    def _save_local_manifest(self, manifest: dict) -> None:
        _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    def _download_file(self, url: str, dest: Path) -> None:
        print(f"  [SYNC] Downloading: {dest.name}")
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            dest.write_bytes(resp.read())

    def _ingest_file(self, tmp_path: str, filename: str) -> None:
        """Ingest a single PDF using the same pipeline as the ingest router."""
        from routers.ingest import _ingest_files_sync
        _ingest_files_sync([(tmp_path, filename)])

    def _log_sync(self, result: dict) -> None:
        log: list[dict] = []
        if _SYNC_LOG_PATH.exists():
            try:
                log = json.loads(_SYNC_LOG_PATH.read_text())
            except Exception:
                pass

        log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "synced"   : result.get("synced",  []),
            "skipped"  : result.get("skipped", []),
            "errors"   : result.get("errors",  []),
        })

        # Keep only last 50 log entries
        log = log[-50:]
        _SYNC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SYNC_LOG_PATH.write_text(json.dumps(log, indent=2))

    # ── STATUS ────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Return current sync status for the /sync/status endpoint.
        """
        is_syncing    = not _SYNC_LOCK.acquire(blocking=False)
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

        # Check remote manifest for pending count
        pending_count = 0
        try:
            remote = self._fetch_remote_manifest()
            local  = self._load_local_manifest()
            local_hashes = {
                doc["filename"]: doc.get("sha256", "")
                for doc in local.get("docs", [])
            }
            for doc in remote.get("docs", []):
                if local_hashes.get(doc.get("filename", "")) != doc.get("sha256", ""):
                    pending_count += 1
        except Exception:
            pass   # offline or manifest not configured

        return {
            "last_synced"  : last_synced,
            "is_syncing"   : is_syncing,
            "pending_count": pending_count,
            "message"      : "Sync service ready" if self.manifest_url else "SYNC_MANIFEST_URL not configured",
        }


__all__ = ["SyncService"]