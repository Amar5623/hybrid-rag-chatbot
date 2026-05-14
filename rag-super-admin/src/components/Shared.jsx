// components/Shared.jsx
// Reusable UI primitives used across all pages.

import React, { useState } from 'react'

// ── Status badge ─────────────────────────────────────────────────────────────
const STATUS_DOT = {
  active:     '#34d399',
  trial:      '#60a5fa',
  suspended:  '#f87171',
  over_quota: '#fbbf24',
}

export function StatusBadge({ status }) {
  const s = (status || 'unknown').toLowerCase()
  const label = s.replace('_', ' ')
  return (
    <span className={`badge ${s}`}>
      <span className="badge-dot" style={{ background: STATUS_DOT[s] || '#4a5580' }} />
      {label}
    </span>
  )
}

// ── Plan badge ───────────────────────────────────────────────────────────────
export function PlanBadge({ name }) {
  const cls = (name || '').toLowerCase().split(' ')[0] // starter | growth | enterprise
  return <span className={`badge ${cls}`}>{name || '—'}</span>
}

// ── Usage meter ──────────────────────────────────────────────────────────────
export function UsageMeter({ label, used, limit, unit = '' }) {
  const pct = limit > 0 ? Math.min((used / limit) * 100, 100) : 0
  const cls  = pct >= 100 ? 'danger' : pct >= 80 ? 'warning' : 'normal'
  const fmt  = (n) => n >= 1_000_000 ? `${(n/1_000_000).toFixed(1)}M` : n >= 1000 ? `${(n/1000).toFixed(1)}k` : String(n)

  return (
    <div className="usage-meter">
      <div className="usage-meter-header">
        <span className="usage-meter-label">{label}</span>
        <span className="usage-meter-value">{fmt(used)}{unit} / {fmt(limit)}{unit}</span>
      </div>
      <div className="usage-bar-track">
        <div className={`usage-bar-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

// ── Spinner ──────────────────────────────────────────────────────────────────
export function Spinner({ text = 'Loading…' }) {
  return (
    <div className="loading-overlay">
      <div className="spinner" />
      {text && <span>{text}</span>}
    </div>
  )
}

// ── Empty state ──────────────────────────────────────────────────────────────
export function EmptyState({ icon = '📭', title, body }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">{icon}</div>
      {title && <div className="empty-state-title">{title}</div>}
      {body  && <div className="empty-state-body">{body}</div>}
    </div>
  )
}

// ── Confirm modal ─────────────────────────────────────────────────────────────
export function ConfirmModal({ title, message, confirmLabel = 'Confirm', danger = false, onConfirm, onCancel }) {
  const [loading, setLoading] = useState(false)

  async function handleConfirm() {
    setLoading(true)
    try { await onConfirm() } finally { setLoading(false) }
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">{title}</span>
          <button className="btn-icon" onClick={onCancel}>✕</button>
        </div>
        <div className="modal-body">
          <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.6 }}>{message}</p>
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onCancel} disabled={loading}>Cancel</button>
          <button
            className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`}
            onClick={handleConfirm}
            disabled={loading}
          >
            {loading ? <><span className="spinner" style={{width:14,height:14}} /> Working…</> : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Pagination controls ───────────────────────────────────────────────────────
export function Pagination({ page, pageSize, total, onChange }) {
  const totalPages = Math.ceil(total / pageSize)
  const from = (page - 1) * pageSize + 1
  const to   = Math.min(page * pageSize, total)

  return (
    <div className="pagination">
      <span className="pagination-info">
        {total > 0 ? `${from}–${to} of ${total}` : '0 results'}
      </span>
      <div className="pagination-controls">
        <button
          className="btn btn-secondary btn-sm"
          disabled={page <= 1}
          onClick={() => onChange(page - 1)}
        >← Prev</button>
        <span className="mono-sm" style={{ padding: '0 8px', color: 'var(--text-muted)' }}>
          {page} / {totalPages || 1}
        </span>
        <button
          className="btn btn-secondary btn-sm"
          disabled={page >= totalPages}
          onClick={() => onChange(page + 1)}
        >Next →</button>
      </div>
    </div>
  )
}

// ── JSON editor field ─────────────────────────────────────────────────────────
export function JsonEditor({ value, onChange, label, rows = 8 }) {
  const [error, setError] = useState('')

  function handleChange(e) {
    const raw = e.target.value
    onChange(raw) // pass raw string up
    try {
      JSON.parse(raw)
      setError('')
    } catch {
      setError('Invalid JSON')
    }
  }

  return (
    <div className="form-field">
      {label && <label className="label">{label}</label>}
      <textarea
        className={`json-editor ${error ? 'error' : ''}`}
        rows={rows}
        value={value}
        onChange={handleChange}
        spellCheck={false}
      />
      {error && <span style={{ color: 'var(--red)', fontSize: 11, fontFamily: 'var(--font-mono)' }}>{error}</span>}
    </div>
  )
}

// ── Section header ────────────────────────────────────────────────────────────
export function SectionHeader({ title, sub, children }) {
  return (
    <div className="section-header">
      <div>
        <h1 className="section-title">{title}</h1>
        {sub && <p className="section-sub">{sub}</p>}
      </div>
      {children && <div style={{ display: 'flex', gap: 8 }}>{children}</div>}
    </div>
  )
}

// ── Relative time formatter ───────────────────────────────────────────────────
export function RelTime({ iso }) {
  if (!iso) return <span className="text-muted">—</span>
  const d     = new Date(iso)
  const now   = Date.now()
  const diff  = now - d.getTime()
  const mins  = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days  = Math.floor(diff / 86400000)

  let label
  if (mins  < 1)    label = 'just now'
  else if (mins  < 60)  label = `${mins}m ago`
  else if (hours < 24)  label = `${hours}h ago`
  else if (days  < 7)   label = `${days}d ago`
  else               label = d.toLocaleDateString()

  return (
    <span className="mono-xs text-muted" title={d.toLocaleString()}>
      {label}
    </span>
  )
}

// ── Mono value display ────────────────────────────────────────────────────────
export function MonoValue({ children, dim = false }) {
  return (
    <span className="mono-sm" style={{ color: dim ? 'var(--text-muted)' : 'var(--text-secondary)' }}>
      {children}
    </span>
  )
}