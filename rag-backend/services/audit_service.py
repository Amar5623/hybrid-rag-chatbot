# services/audit_service.py
#
# Phase 5-C — Append-Only Audit Log Service
#
# PURPOSE:
#   Provides a single async function `log_audit()` that writes one row to the
#   Supabase `audit_log` table.  Every super-admin write operation and every
#   super-admin portal access MUST call this function so there is a complete,
#   tamper-evident trail of who did what and when.
#
# TABLE SCHEMA (created in Phase 0):
#   audit_log (
#     id          uuid primary key default gen_random_uuid(),
#     actor_email text not null,
#     tenant_id   uuid references tenants(id),   -- nullable; NULL for global actions
#     action      text not null,
#     payload     jsonb not null default '{}',
#     created_at  timestamptz default now()
#   )
#
# DESIGN DECISIONS:
#   - The function is async to allow easy `await log_audit(...)` from async
#     FastAPI handlers, but the underlying Supabase Python SDK call is
#     synchronous. We wrap it with `run_in_threadpool` so we never block the
#     event loop on a network I/O call.
#
#   - Audit writes are fire-and-forget in the auth middleware (the middleware
#     catches exceptions so audit failure never blocks a request). However when
#     called explicitly from router handlers the caller may choose to surface
#     the error — this is intentional.
#
#   - Payloads are stored as JSONB. Keep them small and structured. Never store
#     secrets, passwords, or access tokens in a payload.
#
# USAGE:
#   from services.audit_service import log_audit
#
#   await log_audit(
#       actor_email = request.state.user_email,
#       action      = "plan_changed",
#       tenant_id   = tenant_id,
#       payload     = {"old_plan": old_plan_id, "new_plan": new_plan_id},
#   )
#
# STANDARD ACTION STRINGS:
#   "super_admin_access"       — every authenticated super-admin request
#   "tenant_plan_changed"      — PATCH /super-admin/tenants/{id} plan_id
#   "tenant_status_changed"    — suspend / reactivate
#   "tenant_config_changed"    — config_overrides updated
#   "tenant_trial_extended"    — trial_ends_at pushed forward
#   "tenant_impersonated"      — POST /super-admin/tenants/{id}/impersonate
#   "tenant_reconciled"        — POST /super-admin/tenants/{id}/reconcile
#   "document_deleted"         — DELETE /super-admin/tenants/{id}/documents/{doc_id}
#   "member_removed"           — DELETE /super-admin/tenants/{id}/members/{user_id}
#   "member_role_changed"      — PATCH  /super-admin/tenants/{id}/members/{user_id}/promote
#   "plan_created"             — POST /super-admin/plans
#   "plan_updated"             — PATCH /super-admin/plans/{plan_id}
#   "plan_retired"             — PATCH /super-admin/plans/{plan_id}/retire
#   "bulk_plan_change"         — POST /super-admin/bulk/plan-change
#   "bulk_trial_extend"        — POST /super-admin/bulk/trial-extend
#   "bulk_suspend"             — POST /super-admin/bulk/suspend
#   "bulk_config_push"         — POST /super-admin/bulk/config-push
#   "alert_read"               — PATCH /super-admin/alerts/{id}/read

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool

from utils.logger import get_logger

logger = get_logger(__name__)


async def log_audit(
    actor_email : str,
    action      : str,
    tenant_id   : str | None = None,
    payload     : dict       | None = None,
) -> None:
    """
    Write one append-only row to the `audit_log` table in Supabase.

    This is an async function so callers can `await` it without blocking, but
    the underlying Supabase SDK call is synchronous — we use run_in_threadpool
    to keep the event loop free.

    Args:
        actor_email : Email of the user who performed the action.
                      Use request.state.user_email (set by resolve_tenant).
        action      : Slug describing what happened. Use the standard action
                      strings listed at the top of this module.
        tenant_id   : UUID of the affected tenant (optional — pass None for
                      global / cross-tenant actions).
        payload     : Additional structured context (optional). Keep small.
                      Never include secrets or raw tokens.

    Raises:
        Exception — if the Supabase insert fails. Callers in the auth
        middleware catch this; callers in router handlers may propagate it.
    """
    row = {
        "actor_email": actor_email or "unknown",
        "action"     : action,
        "payload"    : payload or {},
    }
    if tenant_id:
        row["tenant_id"] = tenant_id

    def _write() -> None:
        from services.supabase_client import get_supabase_admin
        sb = get_supabase_admin()
        sb.table("audit_log").insert(row).execute()

    await run_in_threadpool(_write)

    logger.debug(
        "[AUDIT] Logged — actor=%s  action=%s  tenant=%s",
        actor_email, action, tenant_id or "—",
    )


__all__ = ["log_audit"]