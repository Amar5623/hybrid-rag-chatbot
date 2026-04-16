# routers/sync.py
#
# NEW FILE.
# Exposes sync-related API routes.
#   GET  /sync/status  — last sync time, is sync running, pending doc count
#   POST /sync/trigger — manually kick off a sync

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
    - pending_count : number of docs on the server not yet downloaded
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
    """
    from services.sync_service import SyncService
    from config import settings

    if not settings.sync_manifest_url:
        return {
            "status" : "skipped",
            "message": "SYNC_MANIFEST_URL not configured in .env",
        }

    import asyncio
    sync = SyncService()
    asyncio.create_task(run_in_threadpool(sync.run))
    return {"status": "triggered", "message": "Sync started in background."}