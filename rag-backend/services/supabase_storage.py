# services/supabase_storage.py
#
# Supabase Storage helper — lightweight, uses only `requests` (no supabase SDK).
#
# FIX — "Invalid Compact JWS" / HTTP 400 error:
#   The Supabase Storage REST API requires BOTH headers:
#     Authorization: Bearer <service_role_key>
#     apikey: <service_role_key>
#   Sending only Authorization was accepted by older Supabase versions but
#   newer versions (2024+) enforce the apikey header as well.
#
#   Additionally, the key is now .strip()-ed to remove any trailing newlines
#   or spaces that can appear when copying from the Supabase dashboard or
#   pasting multi-line values into .env files.
#
# Public API:
#   upload_pdf_to_supabase(file_path: str) -> str | None
#   supabase_enabled() -> bool
#   download_pdf_from_url(url: str, dest_path: str) -> bool
#
# Backward compatibility:
#   If SUPABASE_URL, SUPABASE_SERVICE_KEY, or SUPABASE_BUCKET are not set,
#   every function is a no-op and returns None/False.

import os
from pathlib import Path


def supabase_enabled() -> bool:
    """Return True only when all three Supabase settings are non-empty."""
    from config import settings
    return bool(
        settings.supabase_url
        and settings.supabase_service_key
        and settings.supabase_bucket
    )


def upload_pdf_to_supabase(file_path: str) -> str | None:
    """
    Upload a PDF file to Supabase Storage and return its permanent public URL.

    FIX: Now sends both 'Authorization' and 'apikey' headers, and strips the
    service key of whitespace before use.  Supabase Storage REST API (v1)
    requires the apikey header in addition to Authorization — sending only
    Authorization causes HTTP 400 "Invalid Compact JWS" on newer projects.

    Args:
        file_path: Absolute or relative path to the PDF file on disk.

    Returns:
        Public URL string on success:
            https://<project>.supabase.co/storage/v1/object/public/<bucket>/<filename>
        None if Supabase is not configured, file not found, or upload fails.

    The function is idempotent: uploading the same filename twice will
    upsert (overwrite) the existing object in Supabase Storage.
    """
    if not supabase_enabled():
        return None

    from config import settings
    import requests

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        print(f"  [SUPABASE] ⚠  File not found, skipping upload: {file_path}")
        return None

    filename = file_path_obj.name

    # ── Strip whitespace from key ──────────────────────────────────────────
    # The most common cause of "Invalid Compact JWS" is a key that was
    # copied with a trailing newline, space, or carriage return from the
    # Supabase dashboard or a multi-line .env file.
    service_key = settings.supabase_service_key.strip()
    base_url    = settings.supabase_url.rstrip("/")
    bucket      = settings.supabase_bucket.strip()

    # ── Sanity-check the key format ────────────────────────────────────────
    # A valid Supabase JWT has exactly 3 dot-separated base64 segments.
    jwt_parts = service_key.split(".")
    if len(jwt_parts) != 3:
        print(
            f"  [SUPABASE] ⚠  SUPABASE_SERVICE_KEY looks malformed: "
            f"expected 3 JWT segments (header.payload.signature), got {len(jwt_parts)}. "
            f"Re-copy the service_role key from Supabase → Project Settings → API."
        )

    # ── Build the upload URL ───────────────────────────────────────────────
    # POST /storage/v1/object/<bucket>/<filename>?upsert=true
    upload_url = f"{base_url}/storage/v1/object/{bucket}/{filename}?upsert=true"

    # ── Build headers ─────────────────────────────────────────────────────
    # Supabase Storage REST API requires BOTH headers (as of 2024):
    #   Authorization : Bearer token — standard HTTP auth
    #   apikey        : same token   — Supabase-specific routing / RLS bypass
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey"       : service_key,
        "Content-Type" : "application/octet-stream",
        "x-upsert"     : "true",
    }

    try:
        print(f"  [SUPABASE] Uploading '{filename}' → {upload_url}")

        with open(file_path_obj, "rb") as fh:
            response = requests.post(
                upload_url,
                headers = headers,
                data    = fh,
                timeout = 120,   # large PDFs may take a while
            )

        if response.status_code in (200, 201):
            public_url = (
                f"{base_url}/storage/v1/object/public/{bucket}/{filename}"
            )
            print(f"  [SUPABASE] ✅ Uploaded '{filename}' → {public_url}")
            return public_url
        else:
            # Log the full response body for easier debugging
            print(
                f"  [SUPABASE] ❌ Upload failed for '{filename}': "
                f"HTTP {response.status_code} — {response.text[:500]}"
            )
            # Extra hint for the most common errors
            if response.status_code in (401, 403):
                print(
                    f"  [SUPABASE] 💡 Auth hint: check that SUPABASE_SERVICE_KEY "
                    f"is the 'service_role' key (not 'anon'), and that it has no "
                    f"leading/trailing whitespace in your .env file."
                )
            elif response.status_code == 404:
                print(
                    f"  [SUPABASE] 💡 Bucket hint: make sure the bucket '{bucket}' "
                    f"exists in Supabase Storage and is set to PUBLIC."
                )
            return None

    except Exception as exc:
        print(f"  [SUPABASE] ❌ Upload exception for '{filename}': {exc}")
        return None


def download_pdf_from_url(url: str, dest_path: str) -> bool:
    """
    Stream-download a PDF from `url` and save it to `dest_path`.

    Used by the sync engine to download PDFs referenced in chunk metadata
    (source_url) to the local data/pdfs/ directory for the offline viewer.

    The download is streamed in 8 KB chunks so large PDFs don't exhaust RAM.
    """
    import requests

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  [SUPABASE] Downloading PDF from {url}")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)

        print(f"  [SUPABASE] ✅ Saved to {dest}")
        return True

    except Exception as exc:
        print(f"  [SUPABASE] ❌ Download failed from {url}: {exc}")
        if dest.exists():
            try:
                dest.unlink()
            except Exception:
                pass
        return False


__all__ = [
    "supabase_enabled",
    "upload_pdf_to_supabase",
    "download_pdf_from_url",
]