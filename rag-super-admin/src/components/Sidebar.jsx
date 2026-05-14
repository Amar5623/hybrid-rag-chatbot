// components/Sidebar.jsx
import React from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const NAV_SECTIONS = [
  {
    label: 'Overview',
    items: [
      { path: '/',         icon: '⬡', label: 'Dashboard'     },
      { path: '/alerts',   icon: '◈', label: 'Alerts',       alertKey: true },
    ],
  },
  {
    label: 'Tenants',
    items: [
      { path: '/tenants',  icon: '◧', label: 'All Tenants'   },
    ],
  },
  {
    label: 'Config',
    items: [
      { path: '/plans',    icon: '◫', label: 'Plans'         },
    ],
  },
  {
    label: 'Audit',
    items: [
      { path: '/activity', icon: '◲', label: 'Activity Feed' },
    ],
  },
]

export default function Sidebar({ alertCount = 0 }) {
  const { userEmail, logout } = useAuth()
  const navigate  = useNavigate()
  const location  = useLocation()

  const initials = userEmail
    ? userEmail.slice(0, 2).toUpperCase()
    : 'SA'

  function isActive(path) {
    if (path === '/') return location.pathname === '/'
    return location.pathname.startsWith(path)
  }

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="logo-mark">
          <div className="logo-icon">⬡</div>
          <div className="logo-text">
            <div className="logo-name">Control Tower</div>
            <div className="logo-sub">Super Admin</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      {NAV_SECTIONS.map(section => (
        <div className="sidebar-section" key={section.label}>
          <div className="sidebar-section-label">{section.label}</div>
          {section.items.map(item => (
            <div
              key={item.path}
              className={`sidebar-item ${isActive(item.path) ? 'active' : ''}`}
              onClick={() => navigate(item.path)}
            >
              <span className="item-icon">{item.icon}</span>
              {item.label}
              {item.alertKey && alertCount > 0 && (
                <span className="sidebar-badge">{alertCount > 99 ? '99+' : alertCount}</span>
              )}
            </div>
          ))}
        </div>
      ))}

      {/* User footer */}
      <div className="sidebar-footer">
        <div className="sidebar-user">
          <div className="sidebar-avatar">{initials}</div>
          <div className="sidebar-user-info">
            <div className="sidebar-user-name" title={userEmail}>
              {userEmail || 'Super Admin'}
            </div>
            <div className="sidebar-user-role">super_admin</div>
          </div>
          <button
            className="btn-logout"
            onClick={logout}
            title="Sign out"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
          </button>
        </div>
      </div>
    </aside>
  )
}