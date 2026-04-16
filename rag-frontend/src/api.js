// src/api.js
// All calls go through the Vite proxy → FastAPI on :8000
// Auth removed — no JWT, no login, no token storage.

const BASE = '/api'

// ── Health / network status ───────────────────────────────────
export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()   // { status, version, groq_configured, is_online }
}

/**
 * Ping the backend /health endpoint to check if the backend itself is up
 * AND to get the current network status (is_online) from the NetworkMonitor.
 * Returns { is_online: bool } or throws if backend is down.
 */
export async function checkNetworkStatus() {
  const data = await fetchHealth()
  return { is_online: data.is_online ?? true }
}

// ── Stats (public — sidebar polls this) ───────────────────────
export async function fetchStats() {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}

// ── Documents ─────────────────────────────────────────────────
export async function fetchDocuments() {
  const res = await fetch(`${BASE}/documents`)
  if (!res.ok) throw new Error('Failed to fetch documents')
  return res.json()
}

// ── Ingest ────────────────────────────────────────────────────
export async function ingestFiles(files) {
  const form = new FormData()
  for (const f of files) form.append('files', f)

  const res = await fetch(`${BASE}/ingest`, {
    method: 'POST',
    body  : form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Ingest failed')
  }
  return res.json()
}

// ── Delete file ───────────────────────────────────────────────
export async function deleteFile(filename) {
  const res = await fetch(`${BASE}/ingest/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Delete failed')
  }
  return res.json()
}

// ── Wipe ──────────────────────────────────────────────────────
export async function wipeCollection() {
  const res = await fetch(`${BASE}/collection`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Wipe failed')
  return res.json()
}

// ── Clear session memory ──────────────────────────────────────
export async function clearSession() {
  await fetch(`${BASE}/session/clear`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ session_id: 'default' }),
  })
}

// ── Streaming chat (online — SSE) ─────────────────────────────
// Yields: { type: 'token', token }  |  { type: 'done', ... }  |  { type: 'error', message }
export async function* streamChat(question) {
  const res = await fetch(`${BASE}/chat/stream`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ question, session_id: 'default' }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    yield { type: 'error', message: err.detail || 'Chat failed' }
    return
  }

  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (!raw) continue
      try {
        const data = JSON.parse(raw)
        if (data.token !== undefined) {
          yield { type: 'token', token: data.token }
        } else if (data.done) {
          yield { type: 'done', ...data }
        } else if (data.error) {
          yield { type: 'error', message: data.error }
        }
      } catch { /* malformed line — skip */ }
    }
  }
}

// ── Offline chat (no SSE — returns chunk cards) ───────────────
/**
 * Called when is_online is false.
 * POSTs to /chat/stream which returns a plain JSON OfflineQueryResponse.
 * Returns { query, chunks: [{source, page, heading, section_path, content, score}], total, is_offline }
 */
export async function fetchOfflineResponse(question) {
  const res = await fetch(`${BASE}/chat/stream`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ question, session_id: 'default' }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Offline retrieval failed')
  }
  return res.json()   // OfflineQueryResponse
}

// ── Sync ──────────────────────────────────────────────────────
export async function fetchSyncStatus() {
  const res = await fetch(`${BASE}/sync/status`)
  if (!res.ok) throw new Error('Failed to fetch sync status')
  return res.json()   // SyncStatusResponse
}

export async function triggerSync() {
  const res = await fetch(`${BASE}/sync/trigger`, { method: 'POST' })
  if (!res.ok) throw new Error('Sync trigger failed')
  return res.json()
}

// ── Pin / Unpin source ────────────────────────────────────────
export async function pinFile(filename) {
  const res = await fetch(`${BASE}/session/pin`, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ filename }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Pin failed')
  }
  return res.json()
}

export async function unpinFile() {
  const res = await fetch(`${BASE}/session/pin`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Unpin failed')
  return res.json()
}