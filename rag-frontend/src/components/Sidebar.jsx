// src/components/Sidebar.jsx
//
// CHANGES vs previous version:
//   - All ingest UI REMOVED (upload zone, pending list, Index button, Wipe button,
//     Delete file buttons). Document management has moved to the rag-admin panel.
//   - Removed imports: ingestFiles, wipeCollection, deleteFile
//   - Removed state: drag, pending, busy, busyMsg
//   - Removed functions: addFiles, handleDrop, handleChange, removeFile,
//     doIngest, doWipe, doDeleteFile
//   - Removed INPUT_ID constant and the hidden <input type="file"> element
//   - Removed the collapsed ⊕ upload shortcut icon
//   - File list is now read-only: pin/unpin buttons remain, delete button removed
//   - Everything else (header, KB pill, Chat section, Sync section, Stats,
//     Indexed files with pin, collapsed dot indicators) is UNCHANGED.

import { useState, useEffect } from 'react'
import {
  fetchStats, pinFile, unpinFile,
  fetchSyncStatus, triggerSync,
} from '../api'

export default function Sidebar({
  onClearChat, kbReady, setKbReady, refreshKey,
  pinnedFile, onPin, onUnpin, isOnline,
}) {
  const [collapsed,  setCollapsed]  = useState(false)
  const [stats,      setStats]      = useState(null)
  const [syncStatus, setSyncStatus] = useState(null)
  const [syncing,    setSyncing]    = useState(false)

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
    setSyncing(true)
    try {
      await triggerSync()
      // Wait a moment then refresh stats
      setTimeout(async () => { await refresh(); await refreshSync() }, 3000)
    } catch (e) { alert(e.message) }
    finally { setSyncing(false) }
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

          {/* ── Chat ── */}
          <SLabel>Chat</SLabel>
          <SBtn onClick={onClearChat}>Clear conversation</SBtn>

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
            disabled={syncing || !isOnline || syncStatus?.is_syncing}
            title={!isOnline ? 'Cannot sync while offline' : 'Check for new documents on the server'}
          >
            {syncing ? 'Syncing…' : isOnline ? 'Sync now' : 'Offline — sync unavailable'}
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

                    <button onClick={() => doTogglePin(f)}
                      title={isPinned ? 'Unpin' : 'Pin — search this file only'}
                      style={{
                        flexShrink: 0, width: 20, height: 20,
                        border: `1px solid ${isPinned ? 'rgba(124,106,247,.4)' : 'var(--border-md)'}`,
                        borderRadius: 4, background: isPinned ? 'rgba(124,106,247,.15)' : 'transparent',
                        cursor: 'pointer',
                        color: isPinned ? 'var(--accent-text)' : 'var(--text-3)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '.7rem', transition: 'all .15s',
                      }}>📌</button>
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

function SBtn({ children, onClick, disabled, style, title }) {
  return (
    <button onClick={onClick} disabled={disabled} title={title} style={{
      width: '100%', padding: '8px 12px', borderRadius: 'var(--r-md)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      fontFamily: 'var(--font-display)', fontWeight: 700,
      fontSize: '.72rem', letterSpacing: '.03em',
      transition: 'all .15s', marginBottom: 6, opacity: disabled ? .5 : 1,
      border: '1px solid var(--border-md)',
      background: 'transparent',
      color: 'var(--text-1)',
      ...style,
    }}>{children}</button>
  )
}