# routers/sync.py
#
# CHANGES vs previous version:
#   - POST /sync/trigger no longer requires SYNC_MANIFEST_URL to be set.
#     Vector sync (cloud → local) runs whenever cloud store creds exist.
#     The manifest URL is only needed for PDF file downloads (optional).
#     Previously, an empty SYNC_MANIFEST_URL silently skipped the entire sync.

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from schemas import SyncStatusResponse

router = APIRouter(prefix="/sync", tags=["sync"])


@router.get("/status", response_model=SyncStatusResponse)
async def sync_status():
    """
    Returns current sync state:
    - last_synced   : ISO timestamp of last successful sync (or null)
    - is_syncing    : true if a sync is currently running
    - pending_count : number of docs on the cloud not yet pulled locally
    - message       : human-readable status string
    """
    from services.sync_service import SyncService
    sync   = SyncService()
    status = await run_in_threadpool(sync.get_status)
    return SyncStatusResponse(**status)


@router.post("/trigger")
async def trigger_sync():
    """
    Manually trigger a document sync.
    Returns immediately — sync runs in background.

    Vector sync always runs when cloud store is configured.
    PDF sync only runs when SYNC_MANIFEST_URL is also set.
    """
    import services.rag_service as rag_svc
    from services.sync_service import SyncService

    # Check if cloud store is configured — if not, nothing to sync
    if rag_svc.get_cloud_store() is None:
        return {
            "status" : "skipped",
            "message": (
                "Cloud store not configured. "
                "Set QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY in .env to enable sync."
            ),
        }

    import asyncio
    sync = SyncService()
    asyncio.create_task(run_in_threadpool(sync.run))
    return {"status": "triggered", "message": "Sync started in background."}