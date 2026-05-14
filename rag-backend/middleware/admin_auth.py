# middleware/admin_auth.py
#
# ⚠  DEPRECATED — Phase 2
#
# This file is retained for backward compatibility in local single-tenant
# dev mode only. In production and in all multi-tenant deployments, this
# module is NOT used.
#
# What replaced it:
#   middleware/tenant_resolver.py  —  resolve_tenant() + require_admin_role()
#   These validate Supabase JWTs and extract tenant_id / role from app_metadata.
#
# Migration:
#   Any router that previously used:
#       dependencies=[Depends(require_admin)]
#   should now use:
#       dependencies=[Depends(resolve_tenant), Depends(require_admin_role)]
#
# This file is safe to delete once all routes have been migrated and
# single-tenant dev mode is no longer needed.

import os
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer      = HTTPBearer(auto_error=False)
_ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def require_admin(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
):
    """
    DEPRECATED — kept for local dev fallback only.

    FastAPI dependency — require ADMIN_TOKEN header on admin routes.
    Set ADMIN_TOKEN in .env. If empty, admin routes are open (dev mode).

    In production, replace with:
        Depends(resolve_tenant), Depends(require_admin_role)
    """
    if not _ADMIN_TOKEN:
        return  # dev mode — no token required

    if not credentials or credentials.credentials != _ADMIN_TOKEN:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or missing admin token.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )