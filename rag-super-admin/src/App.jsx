// rag-super-admin/src/App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ToastProvider } from './context/ToastContext';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import TenantListPage from './pages/TenantListPage';
import TenantDetailPage from './pages/TenantDetailPage';
import PlanManagementPage from './pages/PlanManagementPage';
import ActivityFeedPage from './pages/ActivityFeedPage';
import AlertsPage from './pages/AlertsPage';

// Protected route wrapper
function ProtectedRoute({ children }) {
  const { session, loading } = useAuth();
  
  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        height: '100vh',
        background: 'var(--bg-base)'
      }}>
        <div className="spinner" />
      </div>
    );
  }
  
  if (!session) {
    return <Navigate to="/login" replace />;
  }
  
  return children;
}

// Layout with sidebar for authenticated pages
import Sidebar from './components/Sidebar';

function AuthenticatedLayout({ children }) {
  const { userEmail } = useAuth();
  // We'll need alert count from somewhere – could be fetched in Sidebar itself
  return (
    <div className="app-shell">
      <Sidebar alertCount={0} />
      <div className="main-area">
        <div className="topbar">
          <div className="topbar-title">Control Tower</div>
          <div className="topbar-right">
            <span className="mono-xs text-muted">{userEmail}</span>
          </div>
        </div>
        <div className="page-content">
          {children}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <ToastProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <AuthenticatedLayout>
                    <DashboardPage />
                  </AuthenticatedLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/tenants"
              element={
                <ProtectedRoute>
                  <AuthenticatedLayout>
                    <TenantListPage />
                  </AuthenticatedLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/tenants/:id"
              element={
                <ProtectedRoute>
                  <AuthenticatedLayout>
                    <TenantDetailPage />
                  </AuthenticatedLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/plans"
              element={
                <ProtectedRoute>
                  <AuthenticatedLayout>
                    <PlanManagementPage />
                  </AuthenticatedLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/activity"
              element={
                <ProtectedRoute>
                  <AuthenticatedLayout>
                    <ActivityFeedPage />
                  </AuthenticatedLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/alerts"
              element={
                <ProtectedRoute>
                  <AuthenticatedLayout>
                    <AlertsPage />
                  </AuthenticatedLayout>
                </ProtectedRoute>
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ToastProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}