// src/pages/PaymentPage.jsx
//
// Payment placeholder — shown after plan selection.
// Payment integration is not yet implemented.
// The user can bypass and continue to login, or come back later.
//
// Future: wire up Stripe / Razorpay here.
// The selected plan is passed via router state: location.state.planId
//
// Flow:
//   /plans  →  /payment  →  /login  →  /onboarding  →  /

import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { cardStyle } from './LoginPage'

const PLAN_LABELS = {
  starter:    { name: 'Starter',    price: 'Free',      color: 'var(--teal)' },
  growth:     { name: 'Growth',     price: '$99 / mo',  color: 'var(--accent)' },
  enterprise: { name: 'Enterprise', price: '$499 / mo', color: '#f59e0b' },
}

function Logo() {
  return (
    <div style={{
      width: 36, height: 36, borderRadius: 10, flexShrink: 0,
      background: 'linear-gradient(135deg, var(--accent) 0%, #5b4dd4 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: '1.1rem', color: '#fff',
    }}>✦</div>
  )
}

// ── Card icon ─────────────────────────────────────────────────────────────────
function PaymentIcon() {
  return (
    <div style={{
      width: 56, height: 56, borderRadius: 16,
      background: 'var(--accent-glow)',
      border: '1px solid rgba(124,106,247,.3)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: '1.6rem', marginBottom: 20,
    }}>
      💳
    </div>
  )
}

// ── Plan summary pill ─────────────────────────────────────────────────────────
function PlanSummary({ planId }) {
  const plan = PLAN_LABELS[planId] ?? PLAN_LABELS.starter
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      background: 'var(--bg-2)',
      border: '1px solid var(--border-md)',
      borderRadius: 'var(--r-md)',
      padding: '12px 16px',
      marginBottom: 24,
    }}>
      <div>
        <div style={{
          fontSize: '.62rem', fontFamily: 'var(--font-mono)',
          letterSpacing: '.12em', textTransform: 'uppercase',
          color: 'var(--text-3)', marginBottom: 4,
        }}>
          Selected plan
        </div>
        <div style={{
          fontFamily: 'var(--font-display)', fontWeight: 800,
          fontSize: '1rem', color: plan.color,
        }}>
          {plan.name}
        </div>
      </div>
      <div style={{
        fontFamily: 'var(--font-display)', fontWeight: 700,
        fontSize: '1.1rem', color: 'var(--text-0)',
      }}>
        {plan.price}
      </div>
    </div>
  )
}

// ── Payment placeholder slots ─────────────────────────────────────────────────
function PlaceholderField({ label }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span style={labelStyle}>{label}</span>
      <div style={{
        ...placeholderStyle,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{
          fontSize: '.7rem', fontFamily: 'var(--font-mono)',
          color: 'var(--text-3)', letterSpacing: '.08em',
        }}>
          Coming soon
        </span>
      </div>
    </label>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function PaymentPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const planId = location.state?.planId ?? 'starter'
  const [loading, setLoading] = useState(false)

  const handleContinue = () => {
    setLoading(true)
    // Simulate a brief "processing" moment, then go to login
    setTimeout(() => navigate('/login', { replace: true }), 600)
  }

  return (
    <div style={{
      minHeight: '100vh', display: 'flex',
      alignItems: 'center', justifyContent: 'center', padding: 24,
    }}>
      <div style={{ ...cardStyle, maxWidth: 440 }}>

        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
          <Logo />
          <div>
            <div style={{
              fontFamily: 'var(--font-display)', fontWeight: 800,
              fontSize: '1.1rem', color: 'var(--text-0)',
            }}>
              DocMind Admin
            </div>
            <div style={{
              fontSize: '.65rem', color: 'var(--text-3)',
              fontFamily: 'var(--font-mono)', letterSpacing: '.1em',
              textTransform: 'uppercase', marginTop: 2,
            }}>
              Complete your order
            </div>
          </div>
        </div>

        {/* Plan summary */}
        <PlanSummary planId={planId} />

        {/* Coming soon notice */}
        <div style={{
          background: 'var(--accent-glow)',
          border: '1px solid rgba(124,106,247,.25)',
          borderRadius: 'var(--r-md)',
          padding: '14px 16px',
          marginBottom: 24,
          display: 'flex', gap: 12, alignItems: 'flex-start',
        }}>
          <span style={{ fontSize: '1rem', flexShrink: 0, marginTop: 1 }}>🚧</span>
          <div>
            <div style={{
              fontSize: '.8rem', fontWeight: 700, color: 'var(--accent-text)',
              fontFamily: 'var(--font-display)', marginBottom: 4,
            }}>
              Payment integration coming soon
            </div>
            <div style={{
              fontSize: '.75rem', color: 'var(--text-2)', lineHeight: 1.6,
            }}>
              We're setting up secure payment processing. For now, continue to
              activate your account — billing will be added before launch.
            </div>
          </div>
        </div>

        {/* Payment form placeholders */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 24 }}>
          <PlaceholderField label="Card number" />
          <div style={{ display: 'flex', gap: 12 }}>
            <div style={{ flex: 1 }}><PlaceholderField label="Expiry" /></div>
            <div style={{ flex: 1 }}><PlaceholderField label="CVC" /></div>
          </div>
          <PlaceholderField label="Name on card" />
        </div>

        {/* Continue button */}
        <button
          onClick={handleContinue}
          disabled={loading}
          style={{
            width: '100%', padding: '12px 0',
            background: loading
              ? 'var(--bg-3)'
              : 'linear-gradient(135deg, var(--accent), var(--accent-dim))',
            border: 'none', borderRadius: 'var(--r-md)',
            color: loading ? 'var(--text-3)' : '#fff',
            fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '.9rem',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all .15s',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
          }}
        >
          {loading && (
            <span style={{
              width: 14, height: 14, borderRadius: '50%',
              border: '2px solid rgba(255,255,255,.3)',
              borderTopColor: '#fff',
              animation: 'spin .7s linear infinite', display: 'inline-block',
            }} />
          )}
          {loading ? 'Activating account…' : 'Continue to login →'}
        </button>

        {/* Skip note */}
        <div style={{
          marginTop: 16, textAlign: 'center',
          fontSize: '.72rem', color: 'var(--text-3)',
          fontFamily: 'var(--font-mono)',
        }}>
          No payment required during early access
        </div>
      </div>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────
const labelStyle = {
  fontSize: '.72rem', color: 'var(--text-2)',
  fontFamily: 'var(--font-mono)', letterSpacing: '.08em', textTransform: 'uppercase',
}

const placeholderStyle = {
  background: 'var(--bg-2)',
  border: '1px dashed var(--border-md)',
  borderRadius: 'var(--r-md)',
  padding: '10px 14px',
  height: 42,
  opacity: 0.6,
}