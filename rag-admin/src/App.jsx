// src/App.jsx
// Router shell only — no UI here.
// All page content lives in src/pages/*.
//
// Route guards:
//   ProtectedRoute — redirects to /login if no session.
//   After first successful login checks isOnboardingComplete():
//     false → redirects to /onboarding (shown once per new tenant admin)
//     true  → renders the requested page normally

import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useAuth } from './context/AuthContext'

import LoginPage         from './pages/LoginPage'
import SignupPage        from './pages/SignupPage'
import VerifyEmailPage   from './pages/VerifyEmailPage'
import PlanSelectionPage from './pages/PlanSelectionPage'
import OnboardingPage    from './pages/OnboardingPage'
import DashboardPage     from './pages/DashboardPage'

// ── Spinner shown during initial session restore ───────────────────────────────
function SplashScreen() {
  return (
    <div style={{
      minHeight: '100vh', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 16,
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: '50%',
        border: '2px solid var(--bg-3)',
        borderTopColor: 'var(--accent)',
        animation: 'spin .8s linear infinite',
      }} />
      <span style={{
        color: 'var(--text-3)', fontFamily: 'var(--font-mono)',
        fontSize: '.72rem', letterSpacing: '.12em',
      }}>INITIALISING…</span>
    </div>
  )
}

// ── Protected route wrapper ────────────────────────────────────────────────────
function ProtectedRoute({ children }) {
  const { isAuthenticated, isOnboardingComplete, loading } = useAuth()
  const location = useLocation()

  if (loading) return <SplashScreen />
  if (!isAuthenticated) return <Navigate to="/login" state={{ from: location }} replace />

  // Skip onboarding redirect if we're already going there
  if (!isOnboardingComplete() && location.pathname !== '/onboarding') {
    return <Navigate to="/onboarding" replace />
  }

  return children
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const { loading } = useAuth()

  if (loading) return <SplashScreen />

  return (
    <Routes>
      {/* Public */}
      <Route path="/login"  element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/verify" element={<VerifyEmailPage />} />
      <Route path="/plans"  element={<PlanSelectionPage />} />

      {/* Protected */}
      <Route
        path="/onboarding"
        element={
          <ProtectedRoute>
            <OnboardingPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        }
      />
    </Routes>
  )
}