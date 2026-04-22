// src/App.jsx
//
// Admin panel root.
//
// Flow:
//   1. On load, reads admin_token from localStorage and tries GET /admin/stats.
//      - If it returns 200  → show the main panel.
//      - If it returns 401  → show the login / token-entry screen.
//      - If ADMIN_TOKEN is empty on the server → 200 (dev mode, no token needed).
//   2. Main panel has two columns:
//      - Left : StatsPanel (system stats)
//      - Right: FileManager (upload, file list, delete, wipe)
//   3. Refresh button at top re-fetches stats + file list.

import { useState, useEffect, useCallback } from 'react'
import { adminStats, adminListFiles } from './api'
import FileManager from './components/FileManager'
import StatsPanel  from './components/StatsPanel'

export default function App() {
  const [authed,      setAuthed]      = useState(false)
  const [authChecked, setAuthChecked] = useState(false)
  const [tokenInput,  setTokenInput]  = useState('')
  const [loginError,  setLoginError]  = useState('')
  const [stats,       setStats]       = useState(null)
  const [files,       setFiles]       = useState([])
  const [loading,     setLoading]     = useState(false)

  // ── Fetch stats + files ────────────────────────────────────────
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [s, f] = await Promise.all([adminStats(), adminListFiles()])
      setStats(s)
      setFiles(f.files || [])
    } catch (e) {
      if (e.message.includes('401') || e.message.toLowerCase().includes('token')) {
        setAuthed(false)
      }
    } finally {
      setLoading(false)
    }
  }, [])

  // ── Check auth on mount ────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        await adminStats()
        setAuthed(true)
        setAuthChecked(true)
      } catch (e) {
        // 401 → show login. Any other error (backend down) → also show login.
        setAuthed(false)
        setAuthChecked(true)
      }
    }
    check()
  }, [])

  // ── Load data once authed ──────────────────────────────────────
  useEffect(() => {
    if (authed) fetchData()
  }, [authed, fetchData])

  // ── Token submit ───────────────────────────────────────────────
  const handleLogin = async (e) => {
    e.preventDefault()
    setLoginError('')
    localStorage.setItem('admin_token', tokenInput.trim())
    try {
      await adminStats()
      setAuthed(true)
    } catch {
      localStorage.removeItem('admin_token')
      setLoginError('Token rejected — check your ADMIN_TOKEN value.')
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('admin_token')
    setAuthed(false)
    setStats(null)
    setFiles([])
  }

  // ── Loading spinner (initial auth check) ──────────────────────
  if (!authChecked) {
    return (
      <div style={centerStyle}>
        <Spinner />
        <span style={{ marginTop: 16, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', fontSize: '.78rem' }}>
          Connecting…
        </span>
      </div>
    )
  }

  // ── Login screen ───────────────────────────────────────────────
  if (!authed) {
    return (
      <div style={centerStyle}>
        <div style={{
          width: '100%', maxWidth: 380,
          background: 'var(--bg-1)', border: '1px solid var(--border-md)',
          borderRadius: 'var(--r-xl)', padding: '36px 32px',
          animation: 'fadeUp .25s var(--ease)',
        }}>
          {/* Logo + title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 28 }}>
            <Logo />
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '1.1rem', color: 'var(--text-0)' }}>
                DocMind Admin
              </div>
              <div style={{ fontSize: '.65rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)', letterSpacing: '.1em', textTransform: 'uppercase', marginTop: 2 }}>
                RAG Management Panel
              </div>
            </div>
          </div>

          <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <label style={{ fontSize: '.76rem', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
              Admin Token
            </label>
            <input
              type="password"
              value={tokenInput}
              onChange={e => setTokenInput(e.target.value)}
              placeholder="Paste your ADMIN_TOKEN here"
              autoFocus
              style={{
                background: 'var(--bg-3)', border: '1px solid var(--border-md)',
                borderRadius: 'var(--r-md)', padding: '10px 14px',
                color: 'var(--text-0)', fontFamily: 'var(--font-mono)',
                fontSize: '.82rem', outline: 'none', width: '100%',
                transition: 'border-color .15s',
              }}
            />
            {loginError && (
              <div style={{
                fontSize: '.74rem', color: 'var(--danger)',
                fontFamily: 'var(--font-mono)',
                background: 'var(--danger-bg)', border: '1px solid var(--danger-border)',
                borderRadius: 'var(--r-sm)', padding: '6px 10px',
              }}>
                {loginError}
              </div>
            )}
            <button
              type="submit"
              disabled={!tokenInput.trim()}
              style={{
                marginTop: 4, padding: '11px 0',
                background: 'linear-gradient(135deg, var(--accent), var(--accent-dim))',
                border: 'none', borderRadius: 'var(--r-md)',
                color: '#fff', cursor: tokenInput.trim() ? 'pointer' : 'not-allowed',
                fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '.82rem',
                opacity: tokenInput.trim() ? 1 : .5, transition: 'opacity .15s',
              }}
            >
              Enter Admin Panel
            </button>
          </form>

          <div style={{
            marginTop: 20, paddingTop: 16, borderTop: '1px solid var(--border)',
            fontSize: '.68rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)',
            lineHeight: 1.6,
          }}>
            If <span style={{ color: 'var(--accent-text)' }}>ADMIN_TOKEN</span> is empty
            in <span style={{ color: 'var(--accent-text)' }}>.env</span>, leave this blank
            and submit to enter dev mode.
          </div>
        </div>
      </div>
    )
  }

  // ── Main panel ─────────────────────────────────────────────────
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* ── Topbar ── */}
      <header style={{
        height: 56, padding: '0 28px',
        background: 'var(--bg-1)', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexShrink: 0, position: 'sticky', top: 0, zIndex: 10,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Logo />
          <div>
            <span style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '.95rem', color: 'var(--text-0)' }}>
              DocMind Admin
            </span>
            <span style={{
              marginLeft: 10, fontSize: '.62rem', fontFamily: 'var(--font-mono)',
              letterSpacing: '.1em', color: 'var(--text-3)', textTransform: 'uppercase',
            }}>
              RAG Management
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* KB ready indicator */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: files.length > 0 ? 'var(--teal-dim)' : 'rgba(255,255,255,.03)',
            border: `1px solid ${files.length > 0 ? 'rgba(45,212,191,.2)' : 'var(--border)'}`,
            borderRadius: 20, padding: '4px 12px',
            fontSize: '.68rem', color: files.length > 0 ? 'var(--teal)' : 'var(--text-3)',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
              background: files.length > 0 ? 'var(--teal)' : 'var(--text-3)',
              animation: files.length > 0 ? 'pulse 2.5s ease infinite' : 'none',
            }} />
            {files.length > 0 ? `${files.length} file${files.length > 1 ? 's' : ''} indexed` : 'No documents'}
          </div>

          {/* Refresh */}
          <button
            onClick={fetchData}
            disabled={loading}
            title="Refresh"
            style={{
              width: 32, height: 32, borderRadius: 'var(--r-md)',
              border: '1px solid var(--border-md)', background: 'transparent',
              cursor: loading ? 'not-allowed' : 'pointer',
              color: 'var(--text-2)', fontSize: '.9rem',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              opacity: loading ? .5 : 1, transition: 'all .15s',
            }}
          >
            <span style={{ display: 'inline-block', animation: loading ? 'spin 1s linear infinite' : 'none' }}>↻</span>
          </button>

          {/* Logout */}
          <button
            onClick={handleLogout}
            title="Sign out"
            style={{
              height: 32, padding: '0 12px', borderRadius: 'var(--r-md)',
              border: '1px solid var(--border-md)', background: 'transparent',
              cursor: 'pointer', color: 'var(--text-2)',
              fontFamily: 'var(--font-mono)', fontSize: '.7rem',
              transition: 'all .15s',
            }}
          >
            Sign out
          </button>
        </div>
      </header>

      {/* ── Content ── */}
      <main style={{
        flex: 1, padding: '28px',
        display: 'grid',
        gridTemplateColumns: '280px 1fr',
        gap: 24,
        alignItems: 'start',
        maxWidth: 1100, width: '100%', margin: '0 auto',
      }}>

        {/* Left column — Stats */}
        <aside style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <StatsPanel stats={stats} />

          {/* Connection info */}
          <div style={{
            background: 'var(--bg-2)', border: '1px solid var(--border)',
            borderRadius: 'var(--r-lg)', padding: '14px 16px',
          }}>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: '.62rem', fontWeight: 500,
              letterSpacing: '.12em', textTransform: 'uppercase', color: 'var(--text-3)',
              marginBottom: 10, paddingBottom: 6, borderBottom: '1px solid var(--border)',
            }}>Backend</div>
            <div style={{ fontSize: '.74rem', color: 'var(--text-2)', lineHeight: 1.7 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>URL</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '.7rem', color: 'var(--accent-text)' }}>
                  localhost:8000
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <span>Auth</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '.7rem', color: localStorage.getItem('admin_token') ? 'var(--teal)' : 'var(--warn)' }}>
                  {localStorage.getItem('admin_token') ? 'Token set' : 'Dev mode'}
                </span>
              </div>
            </div>
          </div>
        </aside>

        {/* Right column — File manager */}
        <section style={{
          background: 'var(--bg-1)', border: '1px solid var(--border)',
          borderRadius: 'var(--r-xl)', padding: '24px',
          animation: 'fadeUp .2s var(--ease)',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            marginBottom: 24,
          }}>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: '1.05rem', color: 'var(--text-0)' }}>
                Knowledge Base
              </div>
              <div style={{ fontSize: '.74rem', color: 'var(--text-2)', marginTop: 2 }}>
                Upload, manage, and delete indexed documents
              </div>
            </div>
          </div>

          <FileManager
            files={files}
            onRefresh={fetchData}
            disabled={loading}
          />
        </section>
      </main>
    </div>
  )
}

// ── Logo ──────────────────────────────────────────────────────
function Logo() {
  return (
    <div style={{
      width: 30, height: 30, borderRadius: 8, flexShrink: 0,
      background: 'linear-gradient(135deg, var(--accent) 0%, #5b4dd4 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: '.9rem', color: '#fff',
    }}>✦</div>
  )
}

// ── Spinner ───────────────────────────────────────────────────
function Spinner() {
  return (
    <div style={{
      width: 28, height: 28, borderRadius: '50%',
      border: '2px solid var(--bg-3)',
      borderTopColor: 'var(--accent)',
      animation: 'spin .8s linear infinite',
    }} />
  )
}

// ── Center layout helper ──────────────────────────────────────
const centerStyle = {
  minHeight: '100vh',
  display: 'flex', flexDirection: 'column',
  alignItems: 'center', justifyContent: 'center',
  padding: 24,
}