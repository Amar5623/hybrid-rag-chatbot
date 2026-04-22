// src/components/FileManager.jsx
//
// Handles:
//   - PDF drag-and-drop / browse upload → POST /admin/ingest
//   - Indexed file list with per-file delete → DELETE /admin/file/{filename}
//   - Wipe entire KB → DELETE /admin/collection
//
// Styling matches rag-frontend Sidebar exactly (same CSS variables, same
// component patterns) so the two panels feel like one product.

import { useState } from 'react'
import { adminIngest, adminDeleteFile, adminWipe } from '../api'

const INPUT_ID = 'admin-file-input'

export default function FileManager({ files, onRefresh, disabled }) {
  const [pending,  setPending]  = useState([])
  const [busy,     setBusy]     = useState(false)
  const [busyMsg,  setBusyMsg]  = useState('')
  const [toast,    setToast]    = useState(null)   // { type: 'ok'|'err', text }
  const [drag,     setDrag]     = useState(false)

  // ── Toast helper ──────────────────────────────────────────────
  const showToast = (type, text) => {
    setToast({ type, text })
    setTimeout(() => setToast(null), 4000)
  }

  // ── File staging ──────────────────────────────────────────────
  const addFiles = (fileList) => {
    const valid = [...fileList].filter(f => f.name.toLowerCase().endsWith('.pdf'))
    if (valid.length) setPending(p => [...p, ...valid])
  }

  const handleDrop   = e => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files) }
  const handleChange = e => { addFiles(e.target.files); e.target.value = '' }
  const removeFile   = i => setPending(p => p.filter((_, j) => j !== i))

  // ── Ingest ────────────────────────────────────────────────────
  const doIngest = async () => {
    if (!pending.length) return
    setBusy(true); setBusyMsg(`Indexing ${pending.length} file(s)…`)
    try {
      const result = await adminIngest(pending)
      setPending([])
      await onRefresh()
      showToast('ok',
        `Indexed ${result.files_indexed?.length ?? 0} file(s). ` +
        `${result.total_chunks ?? 0} chunks created.`
      )
    } catch (e) {
      showToast('err', e.message)
    } finally {
      setBusy(false); setBusyMsg('')
    }
  }

  // ── Delete single file ────────────────────────────────────────
  const doDelete = async (filename) => {
    if (!confirm(`Delete "${filename}" from the knowledge base?\nThis cannot be undone.`)) return
    setBusy(true); setBusyMsg(`Deleting ${filename}…`)
    try {
      await adminDeleteFile(filename)
      await onRefresh()
      showToast('ok', `"${filename}" deleted.`)
    } catch (e) {
      showToast('err', e.message)
    } finally {
      setBusy(false); setBusyMsg('')
    }
  }

  // ── Wipe ──────────────────────────────────────────────────────
  const doWipe = async () => {
    if (!confirm('Wipe the ENTIRE knowledge base?\nAll vectors, BM25 index, and hash registry will be erased.\nThis cannot be undone.')) return
    setBusy(true); setBusyMsg('Wiping…')
    try {
      await adminWipe()
      await onRefresh()
      showToast('ok', 'Knowledge base wiped.')
    } catch (e) {
      showToast('err', e.message)
    } finally {
      setBusy(false); setBusyMsg('')
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* ── Toast ── */}
      {toast && (
        <div style={{
          padding: '10px 16px', borderRadius: 'var(--r-md)',
          background: toast.type === 'ok' ? 'var(--success-bg)' : 'var(--danger-bg)',
          border: `1px solid ${toast.type === 'ok' ? 'var(--success-border)' : 'var(--danger-border)'}`,
          color: toast.type === 'ok' ? 'var(--success)' : 'var(--danger)',
          fontSize: '.8rem', fontFamily: 'var(--font-mono)',
          animation: 'fadeUp .2s var(--ease)',
        }}>
          {toast.type === 'ok' ? '✓ ' : '⚠ '}{toast.text}
        </div>
      )}

      {/* ── Upload zone ── */}
      <div>
        <SLabel>Upload PDFs</SLabel>

        <label
          htmlFor={INPUT_ID}
          onDragOver={e => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={handleDrop}
          style={{
            display: 'block',
            border: `1.5px dashed ${drag ? 'var(--accent)' : 'var(--border-md)'}`,
            borderRadius: 'var(--r-lg)', padding: '28px 16px',
            textAlign: 'center', cursor: 'pointer',
            background: drag ? 'var(--accent-glow)' : 'var(--bg-2)',
            transition: 'all .2s', marginBottom: 10,
          }}
        >
          <div style={{ fontSize: '1.5rem', marginBottom: 6, opacity: .7 }}>⊕</div>
          <div style={{ fontSize: '.82rem', color: 'var(--text-2)', lineHeight: 1.5 }}>
            Drop PDFs here or{' '}
            <span style={{ color: 'var(--accent-text)' }}>browse</span>
          </div>
          <div style={{ fontSize: '.68rem', color: 'var(--text-3)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
            .pdf only
          </div>
        </label>

        <input id={INPUT_ID} type="file" multiple accept=".pdf"
          onChange={handleChange} style={{ display: 'none' }} />

        {/* Staged files */}
        {pending.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 10 }}>
            {pending.map((f, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                background: 'var(--bg-3)', border: '1px solid var(--border)',
                borderRadius: 'var(--r-sm)', padding: '6px 10px',
                fontSize: '.76rem', color: 'var(--text-1)',
              }}>
                <span style={{ fontSize: 12, flexShrink: 0 }}>📄</span>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                <span
                  onClick={e => { e.preventDefault(); removeFile(i) }}
                  style={{ cursor: 'pointer', color: 'var(--text-3)', lineHeight: 1, flexShrink: 0 }}
                >✕</span>
              </div>
            ))}
          </div>
        )}

        <ABtn
          primary
          onClick={doIngest}
          disabled={!pending.length || busy || disabled}
        >
          {busy && busyMsg.startsWith('Indexing')
            ? busyMsg
            : pending.length
              ? `Index ${pending.length} file${pending.length > 1 ? 's' : ''}`
              : 'Select files above'}
        </ABtn>
      </div>

      {/* ── Indexed files list ── */}
      <div>
        <SLabel>Indexed files {files.length > 0 && `(${files.length})`}</SLabel>

        {files.length === 0 ? (
          <div style={{
            padding: '20px 16px', textAlign: 'center',
            background: 'var(--bg-2)', border: '1px solid var(--border)',
            borderRadius: 'var(--r-md)',
            color: 'var(--text-3)', fontSize: '.78rem', fontFamily: 'var(--font-mono)',
          }}>
            No documents indexed yet
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {files.map(f => (
              <div key={f} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 12px',
                background: 'var(--bg-2)', border: '1px solid var(--border)',
                borderRadius: 'var(--r-md)',
                fontSize: '.78rem', color: 'var(--text-1)',
                transition: 'border-color .15s',
              }}>
                <span style={{ fontSize: 12, flexShrink: 0 }}>📄</span>
                <span style={{
                  flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  fontFamily: 'var(--font-mono)', fontSize: '.74rem',
                }}>{f}</span>

                {/* PDF preview link — opens the file served from /pdfs/{filename} */}
                <a
                  href={`/pdfs/${encodeURIComponent(f)}`}
                  target="_blank"
                  rel="noreferrer"
                  title={`Preview ${f}`}
                  style={{
                    flexShrink: 0, width: 26, height: 26,
                    border: '1px solid var(--border-md)',
                    borderRadius: 6, background: 'transparent',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '.72rem', color: 'var(--teal)', textDecoration: 'none',
                    transition: 'all .15s',
                  }}
                >↗</a>

                <button
                  onClick={() => doDelete(f)}
                  disabled={busy || disabled}
                  title={`Delete ${f}`}
                  style={{
                    flexShrink: 0, width: 26, height: 26,
                    border: 'var(--danger-border) 1px solid',
                    borderRadius: 6, background: 'transparent',
                    cursor: busy ? 'not-allowed' : 'pointer',
                    color: 'var(--danger)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '.65rem', opacity: busy ? .4 : 1,
                    transition: 'all .15s',
                  }}
                >🗑</button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Danger zone ── */}
      {files.length > 0 && (
        <div>
          <SLabel>Danger zone</SLabel>
          <div style={{
            background: 'var(--danger-bg)', border: '1px solid var(--danger-border)',
            borderRadius: 'var(--r-lg)', padding: '14px 16px',
          }}>
            <div style={{ fontSize: '.78rem', color: 'var(--text-1)', marginBottom: 12, lineHeight: 1.5 }}>
              Wipe erases all vectors, the BM25 index, and the duplicate-detection
              registry. All {files.length} indexed file{files.length > 1 ? 's' : ''} will
              be removed. This cannot be undone.
            </div>
            <ABtn
              danger
              onClick={doWipe}
              disabled={busy || disabled}
            >
              {busy && busyMsg === 'Wiping…' ? 'Wiping…' : 'Wipe entire knowledge base'}
            </ABtn>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Shared sub-components ──────────────────────────────────────

function SLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-mono)', fontSize: '.62rem', fontWeight: 500,
      letterSpacing: '.12em', textTransform: 'uppercase', color: 'var(--text-3)',
      marginBottom: 10, paddingBottom: 6, borderBottom: '1px solid var(--border)',
    }}>{children}</div>
  )
}

function ABtn({ children, onClick, disabled, primary, danger, style }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        width: '100%', padding: '10px 16px', borderRadius: 'var(--r-md)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontFamily: 'var(--font-display)', fontWeight: 700,
        fontSize: '.78rem', letterSpacing: '.03em',
        transition: 'all .15s', opacity: disabled ? .5 : 1,
        border  : primary ? 'none' : danger ? '1px solid var(--danger-border)' : '1px solid var(--border-md)',
        background: primary ? 'linear-gradient(135deg, var(--accent), var(--accent-dim))' : 'transparent',
        color   : primary ? '#fff' : danger ? 'var(--danger)' : 'var(--text-1)',
        ...style,
      }}
    >{children}</button>
  )
}