# middleware/admin_auth.py
#
# Admin token authentication dependency.
# Used to guard all /admin/* routes.
#
# How it works:
#   - Reads ADMIN_TOKEN from environment (via config.py settings).
#   - If ADMIN_TOKEN is empty (dev mode), all admin routes are open — no token required.
#   - If ADMIN_TOKEN is set, every request to /admin/* must include:
#       Authorization: Bearer <your-token>
#   - A missing or wrong token returns HTTP 401 Unauthorized.
#
# Usage (in any router):
#   from middleware.admin_auth import require_admin
#   router = APIRouter(dependencies=[Depends(require_admin)])

import os
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer(auto_error=False)
_ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def require_admin(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
):
    """
    FastAPI dependency — require ADMIN_TOKEN header on admin routes.
    Set ADMIN_TOKEN in .env.  If empty, admin routes are open (dev mode).
    """
    if not _ADMIN_TOKEN:
        return  # dev mode — no token required

    if not credentials or credentials.credentials != _ADMIN_TOKEN:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or missing admin token.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )