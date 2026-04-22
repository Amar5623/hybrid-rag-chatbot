// src/components/StatsPanel.jsx
// Displays key/value stats fetched from GET /admin/stats.

export default function StatsPanel({ stats }) {
  if (!stats) return null

  const rows = [
    ['Total vectors',   stats.total_vectors ?? '—'],
    ['BM25 docs',       stats.bm25_docs     ?? '—'],
    ['LLM model',       stats.llm_model     ?? '—'],
    ['Embedding model', stats.embedding_model?.split('/').pop() ?? '—'],
    ['Collection',      stats.collection    ?? '—'],
  ]

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--r-lg)',
      overflow: 'hidden',
    }}>
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <span style={{ fontSize: '1rem' }}>📊</span>
        <span style={{
          fontFamily: 'var(--font-display)', fontWeight: 700,
          fontSize: '.85rem', color: 'var(--text-0)',
        }}>System Stats</span>
      </div>

      <div style={{ padding: '4px 0' }}>
        {rows.map(([label, value]) => (
          <div key={label} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 16px',
            borderBottom: '1px solid var(--border)',
          }}>
            <span style={{ fontSize: '.78rem', color: 'var(--text-2)' }}>{label}</span>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: '.75rem',
              color: 'var(--accent-text)',
            }}>{String(value)}</span>
          </div>
        ))}
      </div>

      {/* Indexed files count badge */}
      <div style={{
        padding: '10px 16px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span style={{ fontSize: '.78rem', color: 'var(--text-2)' }}>Indexed files</span>
        <span style={{
          background: 'var(--accent-glow)', border: '1px solid rgba(124,106,247,.3)',
          borderRadius: 20, padding: '2px 10px',
          fontFamily: 'var(--font-mono)', fontSize: '.72rem', color: 'var(--accent-text)',
        }}>
          {stats.indexed_files?.length ?? 0}
        </span>
      </div>
    </div>
  )
}