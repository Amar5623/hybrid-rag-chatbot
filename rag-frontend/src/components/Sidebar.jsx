// src/components/Sidebar.jsx
//
// CHANGES vs original:
//   - currentUser / onLogout props REMOVED — no auth.
//   - Sync status section added: "Last synced: X" and a "Sync now" button.
//   - isOnline prop used to disable sync button when offline.
//   - Everything else unchanged.

import { useState, useEffect } from 'react'
import {
  fetchStats, ingestFiles, wipeCollection,
  deleteFile, pinFile, unpinFile,
  fetchSyncStatus, triggerSync,
} from '../api'

export default function Sidebar({
  onClearChat, kbReady, setKbReady, refreshKey,
  pinnedFile, onPin, onUnpin, isOnline,
}) {
  const [collapsed,   setCollapsed]  = useState(false)
  const [stats,       setStats]      = useState(null)
  const [syncStatus,  setSyncStatus] = useState(null)
  const [drag,        setDrag]       = useState(false)
  const [pending,     setPending]    = useState([])
  const [busy,        setBusy]       = useState(false)
  const [busyMsg,     setBusyMsg]    = useState('')

  const refresh = async () => {
    try {
      const s = await fetchStats()
      setStats(s)
      setKbReady(s.total_vectors > 0)
    } catch { /* backend not ready yet */ }
  }

  const refreshSync = async () => {
    try {
      const s = await fetchSyncStatus()
      setSyncStatus(s)
    } catch { /* sync not configured */ }
  }

  useEffect(() => {
    refresh()
    refreshSync()
  }, [refreshKey])

  const INPUT_ID = 'sidebar-file-input'

  const addFiles = files => {
    const valid = [...files].filter(f =>
      f.name.toLowerCase().endsWith('.pdf')
    )
    if (valid.length) setPending(p => [...p, ...valid])
  }

  const handleDrop   = e => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files) }
  const handleChange = e => { addFiles(e.target.files); e.target.value = '' }
  const removeFile   = i => setPending(p => p.filter((_, j) => j !== i))

  const doIngest = async () => {
    if (!pending.length) return
    setBusy(true); setBusyMsg(`Indexing ${pending.length} file(s)…`)
    try {
      await ingestFiles(pending); setPending([]); await refresh()
    } catch (e) { alert(e.message) }
    finally { setBusy(false); setBusyMsg('') }
  }

  const doWipe = async () => {
    if (!confirm('Wipe the entire knowledge base? This cannot be undone.')) return
    setBusy(true); setBusyMsg('Wiping…')
    try {
      await wipeCollection()
      onUnpin?.()
      await refresh()
      onClearChat()
    } catch (e) { alert(e.message) }
    finally { setBusy(false); setBusyMsg('') }
  }

  const doDeleteFile = async (filename) => {
    if (!confirm(`Delete "${filename}" from the knowledge base?\nThis cannot be undone.`)) return
    setBusy(true); setBusyMsg(`Deleting ${filename}…`)
    try {
      if (pinnedFile === filename) onUnpin?.()
      await deleteFile(filename)
      await refresh()
    } catch (e) { alert(e.message) }
    finally { setBusy(false); setBusyMsg('') }
  }

  const doTogglePin = async (filename) => {
    try {
      if (pinnedFile === filename) {
        await unpinFile(); onUnpin?.()
      } else {
        await pinFile(filename); onPin?.(filename)
      }
    } catch (e) { alert(e.message) }
  }

  const doSync = async () => {
    if (!isOnline) { alert('Cannot sync while offline.'); return }
    setBusy(true); setBusyMsg('Syncing…')
    try {
      await triggerSync()
      // Wait a moment then refresh stats
      setTimeout(async () => { await refresh(); await refreshSync() }, 3000)
    } catch (e) { alert(e.message) }
    finally { setBusy(false); setBusyMsg('') }
  }

  const formatSyncTime = (iso) => {
    if (!iso) return 'Never'
    try {
      return new Date(iso).toLocaleString()
    } catch {
      return iso
    }
  }

  const W = collapsed ? 52 : 268

  return (
    <aside style={{
      width: W, minWidth: W,
      background: 'var(--bg-1)',
      borderRight: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column',
      height: '100%', overflow: 'hidden',
      transition: 'width .25s cubic-bezier(.16,1,.3,1)',
      flexShrink: 0,
    }}>

      {/* Header */}
      <div style={{
        height: 56, padding: '0 14px',
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center',
        justifyContent: collapsed ? 'center' : 'space-between',
        flexShrink: 0,
      }}>
        {!collapsed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, overflow: 'hidden' }}>
            <Logo />
            <div style={{ overflow: 'hidden' }}>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '.95rem', color: 'var(--text-0)', whiteSpace: 'nowrap' }}>DocMind</div>
              <div style={{ fontSize: '.58rem', color: 'var(--text-3)', letterSpacing: '.1em', textTransform: 'uppercase' }}>RAG Intelligence</div>
            </div>
          </div>
        )}
        {collapsed && <Logo />}
        <button onClick={() => setCollapsed(c => !c)} style={{
          width: 26, height: 26, borderRadius: 6, border: '1px solid var(--border)',
          background: 'transparent', cursor: 'pointer', color: 'var(--text-2)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '.75rem', flexShrink: 0, transition: 'all .15s',
        }} title={collapsed ? 'Expand' : 'Collapse'}>
          {collapsed ? '›' : '‹'}
        </button>
      </div>

      {/* Body */}
      {!collapsed && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 14px', display: 'flex', flexDirection: 'column' }}>

          {/* KB status pill */}
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6, marginBottom: 16,
            background: kbReady ? 'rgba(45,212,191,.08)' : 'rgba(255,255,255,.03)',
            border: `1px solid ${kbReady ? 'rgba(45,212,191,.2)' : 'var(--border)'}`,
            borderRadius: 20, padding: '4px 11px',
            fontSize: '.68rem', color: kbReady ? 'var(--teal)' : 'var(--text-3)',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
              background: kbReady ? 'var(--teal)' : 'var(--text-3)',
              animation: kbReady ? 'pulse 2.5s ease infinite' : 'none',
            }} />
            {kbReady ? 'KB ready' : 'No documents'}
          </div>

          <SLabel>Upload documents</SLabel>

          <label
            htmlFor={INPUT_ID}
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={handleDrop}
            style={{
              display: 'block',
              border: `1.5px dashed ${drag ? 'var(--accent)' : 'var(--border-md)'}`,
              borderRadius: 'var(--r-lg)', padding: '18px 12px',
              textAlign: 'center', cursor: 'pointer',
              background: drag ? 'var(--accent-glow)' : 'var(--bg-0)',
              transition: 'all .2s', marginBottom: 8,
            }}
          >
            <div style={{ fontSize: '1.2rem', marginBottom: 5, opacity: .7 }}>⊕</div>
            <div style={{ fontSize: '.75rem', color: 'var(--text-2)', lineHeight: 1.5 }}>
              Drop files or <span style={{ color: 'var(--accent-text)' }}>browse</span>
            </div>
            <div style={{ fontSize: '.64rem', color: 'var(--text-3)', marginTop: 3, fontFamily: 'var(--font-mono)' }}>
              pdf only
            </div>
          </label>

          <input id={INPUT_ID} type="file" multiple accept=".pdf"
            onChange={handleChange} style={{ display: 'none' }} />

          {pending.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginBottom: 8 }}>
              {pending.map((f, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 7,
                  background: 'var(--bg-2)', border: '1px solid var(--border)',
                  borderRadius: 'var(--r-sm)', padding: '5px 9px',
                  fontSize: '.71rem', color: 'var(--text-1)',
                }}>
                  <span style={{ fontSize: 11, flexShrink: 0 }}>📄</span>
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                  <span onClick={e => { e.preventDefault(); removeFile(i) }}
                    style={{ cursor: 'pointer', color: 'var(--text-3)', lineHeight: 1, flexShrink: 0 }}>✕</span>
                </div>
              ))}
            </div>
          )}

          <SBtn primary onClick={doIngest} disabled={!pending.length || busy}>
            {busy ? busyMsg : pending.length ? `Index ${pending.length} file${pending.length > 1 ? 's' : ''}` : 'Select files above'}
          </SBtn>

          {kbReady && (<>
            <SLabel>Chat</SLabel>
            <SBtn onClick={onClearChat} style={{ marginBottom: 6 }}>Clear conversation</SBtn>
            <SBtn danger onClick={doWipe}>Wipe knowledge base</SBtn>
          </>)}

          {/* ── Sync status ── */}
          <SLabel>Sync</SLabel>
          <div style={{
            background: 'var(--bg-2)', border: '1px solid var(--border)',
            borderRadius: 'var(--r-md)', padding: '10px 12px', marginBottom: 8,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: '.68rem' }}>
              <span style={{ color: 'var(--text-3)' }}>Last synced</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-text)', fontSize: '.65rem' }}>
                {syncStatus ? formatSyncTime(syncStatus.last_synced) : '—'}
              </span>
            </div>
            {syncStatus?.pending_count > 0 && (
              <div style={{ fontSize: '.65rem', color: '#f59e0b', marginBottom: 4 }}>
                {syncStatus.pending_count} doc{syncStatus.pending_count > 1 ? 's' : ''} pending
              </div>
            )}
            {syncStatus?.is_syncing && (
              <div style={{ fontSize: '.65rem', color: 'var(--teal)', marginBottom: 4 }}>
                Syncing…
              </div>
            )}
          </div>
          <SBtn
            onClick={doSync}
            disabled={busy || !isOnline || syncStatus?.is_syncing}
            title={!isOnline ? 'Cannot sync while offline' : 'Check for new documents on the server'}
          >
            {busy && busyMsg === 'Syncing…' ? 'Syncing…' : isOnline ? 'Sync now' : 'Offline — sync unavailable'}
          </SBtn>

          {stats && (<>
            <SLabel>Stats</SLabel>
            {[
              ['Vectors',  stats.total_vectors],
              ['BM25',     stats.bm25_docs],
              ['Model',    stats.llm_model?.split('-').slice(0, 3).join('-') + '…'],
              ['Embedder', stats.embedding_model?.split('/').pop()],
            ].map(([l, v]) => (
              <div key={l} style={{
                display: 'flex', justifyContent: 'space-between',
                padding: '5px 0', borderBottom: '1px solid var(--border)', fontSize: '.72rem',
              }}>
                <span style={{ color: 'var(--text-2)' }}>{l}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '.69rem', color: 'var(--accent-text)' }}>{v}</span>
              </div>
            ))}

            {stats.indexed_files?.length > 0 && (<>
              <SLabel>Indexed files</SLabel>
              {stats.indexed_files.map(f => {
                const isPinned = pinnedFile === f
                return (
                  <div key={f} style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    padding: '5px 6px', marginBottom: 3,
                    borderRadius: 'var(--r-sm)',
                    background: isPinned ? 'rgba(124,106,247,.1)' : 'transparent',
                    border: `1px solid ${isPinned ? 'rgba(124,106,247,.35)' : 'var(--border)'}`,
                    fontSize: '.71rem', color: isPinned ? 'var(--accent-text)' : 'var(--text-2)',
                    transition: 'all .15s',
                  }}>
                    <span style={{ fontSize: 11, flexShrink: 0 }}>📄</span>
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f}</span>

                    <button onClick={() => doTogglePin(f)} disabled={busy}
                      title={isPinned ? 'Unpin' : 'Pin — search this file only'}
                      style={{
                        flexShrink: 0, width: 20, height: 20,
                        border: `1px solid ${isPinned ? 'rgba(124,106,247,.4)' : 'var(--border-md)'}`,
                        borderRadius: 4, background: isPinned ? 'rgba(124,106,247,.15)' : 'transparent',
                        cursor: busy ? 'not-allowed' : 'pointer',
                        color: isPinned ? 'var(--accent-text)' : 'var(--text-3)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '.7rem', transition: 'all .15s',
                      }}>📌</button>

                    <button onClick={() => doDeleteFile(f)} disabled={busy}
                      title={`Delete ${f}`}
                      style={{
                        flexShrink: 0, width: 20, height: 20,
                        border: '1px solid rgba(239,68,68,.2)',
                        borderRadius: 4, background: 'transparent',
                        cursor: busy ? 'not-allowed' : 'pointer',
                        color: 'rgba(239,100,100,.7)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '.65rem', transition: 'all .15s',
                      }}>🗑</button>
                  </div>
                )
              })}
            </>)}
          </>)}

          <div style={{ flex: 1, minHeight: 16 }} />
        </div>
      )}

      {collapsed && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 14, gap: 12 }}>
          <label htmlFor={INPUT_ID} title="Upload files" style={{
            width: 32, height: 32, borderRadius: 8, cursor: 'pointer',
            background: 'var(--bg-3)', border: '1px solid var(--border-md)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', color: 'var(--text-1)',
          }}>⊕</label>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: kbReady ? 'var(--teal)' : 'var(--text-3)',
            animation: kbReady ? 'pulse 2.5s ease infinite' : 'none',
          }} title={kbReady ? 'KB ready' : 'No documents'} />
          {pinnedFile && (
            <div title={`Pinned: ${pinnedFile}`} style={{
              width: 8, height: 8, borderRadius: '50%',
              background: 'var(--accent)',
              animation: 'glow-pulse 2s ease infinite',
            }} />
          )}
        </div>
      )}
    </aside>
  )
}

function Logo() {
  return (
    <div style={{
      width: 28, height: 28, borderRadius: 7, flexShrink: 0,
      background: 'linear-gradient(135deg, var(--accent) 0%, #5b4dd4 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: '.85rem', color: '#fff',
    }}>✦</div>
  )
}

function SLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-mono)', fontSize: '.59rem', fontWeight: 500,
      letterSpacing: '.12em', textTransform: 'uppercase', color: 'var(--text-3)',
      margin: '18px 0 8px', paddingBottom: 5, borderBottom: '1px solid var(--border)',
    }}>{children}</div>
  )
}

function SBtn({ children, onClick, disabled, primary, danger, style, title }) {
  return (
    <button onClick={onClick} disabled={disabled} title={title} style={{
      width: '100%', padding: '8px 12px', borderRadius: 'var(--r-md)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      fontFamily: 'var(--font-display)', fontWeight: 700,
      fontSize: '.72rem', letterSpacing: '.03em',
      transition: 'all .15s', marginBottom: 6, opacity: disabled ? .5 : 1,
      border  : primary ? 'none' : danger ? '1px solid rgba(239,68,68,.25)' : '1px solid var(--border-md)',
      background: primary ? 'linear-gradient(135deg, var(--accent), var(--accent-dim))' : 'transparent',
      color   : primary ? '#fff' : danger ? 'rgba(239,100,100,.85)' : 'var(--text-1)',
      ...style,
    }}>{children}</button>
  )
}