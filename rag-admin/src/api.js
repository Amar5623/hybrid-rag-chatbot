// src/api.js
// All admin API calls go directly to /admin/* on the FastAPI backend.
// The Vite dev server proxies /admin → http://localhost:8000/admin.
//
// Auth: every request includes Authorization: Bearer <token>
// Token is read from localStorage key 'admin_token'.
// If ADMIN_TOKEN is empty on the server, the header is still sent
// but the server ignores it (dev mode — no token required).

const BASE = '/admin'

function getToken() {
  return localStorage.getItem('admin_token') || ''
}

function authHeaders(extra = {}) {
  const token = getToken()
  return {
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    ...extra,
  }
}

async function handleResponse(res) {
  if (res.status === 401) {
    throw new Error('Invalid or missing admin token. Check your token in Settings.')
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

// ── Files ─────────────────────────────────────────────────────
export async function adminListFiles() {
  const res = await fetch(`${BASE}/files`, {
    headers: authHeaders(),
  })
  return handleResponse(res)   // { files: string[] }
}

// ── Stats ──────────────────────────────────────────────────────
export async function adminStats() {
  const res = await fetch(`${BASE}/stats`, {
    headers: authHeaders(),
  })
  return handleResponse(res)
}

// ── Ingest ────────────────────────────────────────────────────
export async function adminIngest(files) {
  const fd = new FormData()
  for (const f of files) fd.append('files', f)

  const res = await fetch(`${BASE}/ingest`, {
    method : 'POST',
    headers: authHeaders(),   // no Content-Type — browser sets multipart boundary
    body   : fd,
  })
  return handleResponse(res)   // IngestResponse
}

// ── Delete file ───────────────────────────────────────────────
export async function adminDeleteFile(filename) {
  const res = await fetch(`${BASE}/file/${encodeURIComponent(filename)}`, {
    method : 'DELETE',
    headers: authHeaders(),
  })
  return handleResponse(res)   // DeleteFileResponse
}

// ── Wipe collection ───────────────────────────────────────────
export async function adminWipe() {
  const res = await fetch(`${BASE}/collection`, {
    method : 'DELETE',
    headers: authHeaders(),
  })
  return handleResponse(res)   // WipeResponse
}