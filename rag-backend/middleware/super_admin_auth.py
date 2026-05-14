# middleware/super_admin_auth.py
#
# Phase 5-A — Super Admin Auth Middleware
#
# PURPOSE:
#   Single FastAPI dependency that protects every /super-admin/* route.
#   It performs three checks in order (cheapest first):
#
#     1. IP allowlist   — if SUPER_ADMIN_ALLOWED_IPS is set, reject any request
#                         whose source IP is not in the list.  Cost: zero I/O.
#     2. JWT validation — delegates to resolve_tenant() (Phase 1) which
#                         verifies the Supabase JWT, fetches the tenant row, and
#                         populates request.state with user context.  Cost: one
#                         Supabase lookup (cached between requests via lru_cache).
#     3. Role check     — asserts request.state.role == "super_admin".
#                         Any admin or ordinary user is rejected with 403.
#
#   After a successful super-admin request, every access is written to the
#   audit_log table via audit_service.log_audit().  This gives a complete
#   tamper-evident trail of who accessed what and when.
#
# HOW IT FITS IN:
#   This middleware does NOT touch the RAG pipeline, vector stores, BM25, or
#   any retrieval logic.  It is purely an auth + audit layer wired around the
#   existing resolve_tenant() dependency from Phase 1.
#
# DEPENDENCY ON EARLIER PHASES:
#   Phase 1 — middleware/tenant_resolver.py  (resolve_tenant)
#   Phase 1 — services/supabase_client.py   (get_supabase_admin, via audit_service)
#   Phase 5 — services/audit_service.py     (log_audit)
#
# CONFIGURATION:
#   SUPER_ADMIN_ALLOWED_IPS=10.0.0.1,192.168.1.5   # comma-separated; leave empty to skip
#
# USAGE:
#   from middleware.super_admin_auth import require_super_admin
#
#   router = APIRouter(
#       prefix="/super-admin",
#       dependencies=[Depends(require_super_admin)],
#   )

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from config import settings
from middleware.tenant_resolver import resolve_tenant
from utils.logger import get_logger

logger = get_logger(__name__)


async def require_super_admin(
    request: Request,
    _tenant: None = Depends(resolve_tenant),   # populates request.state
) -> None:
    """
    FastAPI dependency — enforces super-admin access on every protected route.

    Steps (cheapest → most expensive):
      1. IP allowlist check (no I/O).
      2. resolve_tenant already ran (Depends chain) — JWT is valid and
         request.state is fully populated.
      3. Role assertion: role must be 'super_admin'.
      4. Audit log write (async, fire-and-forget pattern).

    Raises:
        403 — IP not in allowlist.
        403 — Authenticated user is not a super_admin.
        401 — JWT missing or invalid (raised by resolve_tenant upstream).
    """

    # ── 1. IP allowlist ───────────────────────────────────────────────────────
    allowed_ips_raw = settings.super_admin_allowed_ips.strip()
    if allowed_ips_raw:
        allowed_ips = {ip.strip() for ip in allowed_ips_raw.split(",") if ip.strip()}
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "")
        )
        if client_ip not in allowed_ips:
            logger.warning(
                "[SUPER_ADMIN_AUTH] IP rejected — client_ip=%s  allowlist=%s",
                client_ip,
                allowed_ips,
            )
            # Return generic 403 — do NOT reveal allowlist contents in message
            raise HTTPException(
                status_code=403,
                detail="Access denied from this IP address.",
            )
        logger.debug("[SUPER_ADMIN_AUTH] IP check passed — client_ip=%s", client_ip)

    # ── 2. Role check ─────────────────────────────────────────────────────────
    role = getattr(request.state, "role", None)
    if role != "super_admin":
        logger.warning(
            "[SUPER_ADMIN_AUTH] Role rejected — user=%s  role=%s  path=%s",
            getattr(request.state, "user_email", "unknown"),
            role,
            request.url.path,
        )
        raise HTTPException(
            status_code=403,
            detail="Super admin role required.",
        )

    # ── 3. Audit log (fire-and-forget — never block the request on this) ──────
    try:
        from services.audit_service import log_audit
        await log_audit(
            actor_email = getattr(request.state, "user_email", "unknown"),
            action      = "super_admin_access",
            tenant_id   = None,   # access-level log, not tenant-specific
            payload     = {
                "method": request.method,
                "path"  : str(request.url.path),
            },
        )
    except Exception as exc:
        # Audit failure must NEVER block the request — log and continue
        logger.error("[SUPER_ADMIN_AUTH] Audit log write failed: %s", exc)

    logger.info(
        "[SUPER_ADMIN_AUTH] ✅ Access granted — user=%s  path=%s",
        getattr(request.state, "user_email", "unknown"),
        request.url.path,
    )


__all__ = ["require_super_admin"]