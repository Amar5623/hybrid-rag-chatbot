// src/pages/VerifyEmailPage.jsx
// Shown after signup when Supabase email confirmation is required.
// The user clicks the link in the email → Supabase redirects back with
// a session → onAuthStateChange fires → App routes them to /plans.

import { Link } from 'react-router-dom'
import { cardStyle } from './LoginPage'

function MailIcon() {
  return (
    <div style={{
      width: 56, height: 56, borderRadius: 16,
      background: 'var(--accent-glow)', border: '1px solid rgba(124,106,247,.3)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: '1.6rem', marginBottom: 20,
    }}>
      📧
    </div>
  )
}

export default function VerifyEmailPage() {
  return (
    <div style={{
      minHeight: '100vh', display: 'flex',
      alignItems: 'center', justifyContent: 'center', padding: 24,
    }}>
      <div style={{ ...cardStyle, textAlign: 'center' }}>
        <MailIcon />

        <h2 style={{
          fontFamily: 'var(--font-display)', fontWeight: 800,
          fontSize: '1.2rem', color: 'var(--text-0)', marginBottom: 12,
        }}>
          Check your inbox
        </h2>

        <p style={{
          fontSize: '.85rem', color: 'var(--text-2)',
          lineHeight: 1.7, marginBottom: 24,
        }}>
          We've sent a confirmation link to your email address.
          Click it to verify your account and continue to plan selection.
        </p>

        <div style={{
          background: 'var(--bg-2)', border: '1px solid var(--border)',
          borderRadius: 'var(--r-md)', padding: '14px 16px',
          fontSize: '.78rem', color: 'var(--text-3)',
          fontFamily: 'var(--font-mono)', lineHeight: 1.6,
          textAlign: 'left', marginBottom: 24,
        }}>
          <strong style={{ color: 'var(--text-2)' }}>Tip:</strong> Check your spam folder if you
          don't see it within 2 minutes.
        </div>

        <div style={{ fontSize: '.75rem', color: 'var(--text-3)' }}>
          Wrong email?{' '}
          <Link to="/signup" style={{ color: 'var(--accent-text)', textDecoration: 'none', fontWeight: 600 }}>
            Start over
          </Link>
        </div>
      </div>
    </div>
  )
}