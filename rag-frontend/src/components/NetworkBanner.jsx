// src/components/NetworkBanner.jsx
// NEW FILE. Top-of-screen banner showing Online/Offline state.

export default function NetworkBanner({ isOnline }) {
  if (isOnline) return null   // nothing when online — no clutter

  return (
    <div style={{
      width: '100%', padding: '7px 20px',
      background: 'rgba(245,158,11,.12)',
      borderBottom: '1px solid rgba(245,158,11,.25)',
      display: 'flex', alignItems: 'center', gap: 10,
      flexShrink: 0,
    }}>
      <span style={{ fontSize: '.8rem' }}>📵</span>
      <span style={{
        fontSize: '.72rem', fontFamily: 'var(--font-mono)',
        color: '#f59e0b', letterSpacing: '.04em',
      }}>
        Offline Mode — Retrieval Only. AI generation unavailable until connection restored.
      </span>
    </div>
  )
}