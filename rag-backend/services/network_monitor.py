# services/network_monitor.py
#
# CHANGES vs previous version:
#   - _on_reconnect() no longer requires SYNC_MANIFEST_URL to be set.
#     Sync now runs automatically on reconnect whenever cloud store creds
#     are configured (QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY).
#     Previously, an empty SYNC_MANIFEST_URL silently prevented auto-sync
#     on reconnect, so vectors would never sync unless triggered manually.

import threading
import time
import urllib.request
from datetime import datetime
import os

class NetworkMonitor:
    """
    Background thread that polls a URL every poll_interval seconds.
    Exposes is_online: bool — read by rag_service and the chat router.

    On offline → online transition, automatically calls SyncService.run()
    if cloud store credentials are configured (SYNC_MANIFEST_URL is optional).

    Usage:
        monitor = NetworkMonitor(check_url="https://8.8.8.8", poll_interval=30)
        monitor.start()
        ...
        if monitor.is_online:
            ...
    """

    def __init__(
        self,
        check_url    : str = "https://8.8.8.8",
        poll_interval: int = 30,
        timeout      : int = 5,
    ):
        self.check_url     = check_url
        self.poll_interval = poll_interval
        self.timeout       = timeout

        # Assume online at startup — first poll will correct if wrong
        self._is_online    = True
        self._last_checked : datetime | None = None
        self._lock         = threading.Lock()
        self._thread       : threading.Thread | None = None
        self._stop_event   = threading.Event()

    @property
    def is_online(self) -> bool:
        with self._lock:
            return self._is_online

    @property
    def last_checked(self) -> datetime | None:
        with self._lock:
            return self._last_checked

    def start(self) -> None:
        """Start the background polling thread."""
        self._is_online = self._check()
        self._last_checked = datetime.utcnow()
        self._thread = threading.Thread(
            target  = self._poll_loop,
            daemon  = True,   # dies when main process exits
            name    = "NetworkMonitor",
        )
        self._thread.start()
        print("  [NETWORK] Monitor started. Polling every "
              f"{self.poll_interval}s → {self.check_url}")

    def stop(self) -> None:
        self._stop_event.set()

    # ── INTERNAL ──────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            previous_state = self.is_online
            current_state  = self._check()

            with self._lock:
                self._is_online    = current_state
                self._last_checked = datetime.utcnow()

            if not previous_state and current_state:
                # Transitioned offline → online
                print("  [NETWORK] 🌐 Back online — triggering sync")
                self._on_reconnect()

            elif previous_state and not current_state:
                print("  [NETWORK] 📵 Offline — retrieval-only mode active")

            self._stop_event.wait(timeout=self.poll_interval)

    def _check(self) -> bool:
        """
        Try to reach check_url with a short timeout.
        Returns True if reachable, False otherwise.

        IS_ONLINE_OVERRIDE (env var):
          "true"  → always return True  (simulate online  — Mode 1)
          "false" → always return False (simulate offline — Mode 2)
          unset   → real network check (production behaviour)

        This lets dev-start.sh run two instances on different ports where
        one pretends to have internet and the other pretends it doesn't,
        without actually changing the network.
        """
        override = os.environ.get("IS_ONLINE_OVERRIDE", "").strip().lower()
        if override == "true":
            return True
        if override == "false":
            return False

        try:
            urllib.request.urlopen(self.check_url, timeout=self.timeout)
            return True
        except Exception:
            return False

    def _on_reconnect(self) -> None:
        """
        Called when connectivity is restored.
        Triggers the sync service if cloud store credentials are configured.
        SYNC_MANIFEST_URL is NOT required — vector sync works without it.
        Runs in the monitor thread — keep it fast and non-blocking.
        """
        try:
            import services.rag_service as rag_svc
            # Only trigger sync if a cloud store was successfully configured at startup
            if rag_svc.get_cloud_store() is None:
                print("  [NETWORK] Cloud store not configured — skipping auto-sync")
                return

            from services.sync_service import SyncService
            sync = SyncService()
            # Run sync in a separate thread so monitor loop isn't blocked
            threading.Thread(
                target = sync.run,
                daemon = True,
                name   = "SyncOnReconnect",
            ).start()
        except Exception as e:
            print(f"  [NETWORK] Sync trigger failed: {e}")


__all__ = ["NetworkMonitor"]