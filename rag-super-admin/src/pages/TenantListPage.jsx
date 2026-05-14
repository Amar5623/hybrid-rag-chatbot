// pages/TenantListPage.jsx
import React, { useEffect, useState, useCallback, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  listTenants, listPlans, bulkPlanChange, bulkTrialExtend, bulkSuspend,
} from '../api/superAdmin'
import {
  StatusBadge, PlanBadge, Spinner, Pagination, SectionHeader,
  RelTime, EmptyState, ConfirmModal,
} from '../components/Shared'
import { useToast } from '../context/ToastContext'

// Debounce hook
function useDebounce(value, delay) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return debounced
}

export default function TenantListPage() {
  const navigate          = useNavigate()
  const [searchParams]    = useSearchParams()
  const { addToast }      = useToast()

  const [tenants,   setTenants]   = useState([])
  const [plans,     setPlans]     = useState([])
  const [total,     setTotal]     = useState(0)
  const [page,      setPage]      = useState(1)
  const [loading,   setLoading]   = useState(true)

  const [search,    setSearch]    = useState('')
  const [planFilter,setPlanFilter]= useState('')
  const [statusFilter, setStatus] = useState(searchParams.get('status') || '')

  const [selected,  setSelected]  = useState(new Set())
  const [bulkModal, setBulkModal] = useState(null) // { type, planId? }
  const [bulkPlanId,setBulkPlanId]= useState('')

  const PAGE_SIZE = 25
  const debouncedSearch = useDebounce(search, 350)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await listTenants({
        page, pageSize: PAGE_SIZE,
        search: debouncedSearch,
        planId: planFilter,
        status: statusFilter,
      })
      setTenants(data.items || [])
      setTotal(data.total || 0)
    } catch (e) {
      addToast(e.message, 'error')
    }
    setLoading(false)
  }, [page, debouncedSearch, planFilter, statusFilter])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    listPlans().then(p => setPlans(p || [])).catch(() => {})
  }, [])

  // Reset page on filter change
  useEffect(() => { setPage(1) }, [debouncedSearch, planFilter, statusFilter])

  // Selection
  function toggleAll() {
    if (selected.size === tenants.length) setSelected(new Set())
    else setSelected(new Set(tenants.map(t => t.id)))
  }

  function toggleOne(id) {
    const s = new Set(selected)
    s.has(id) ? s.delete(id) : s.add(id)
    setSelected(s)
  }

  // Bulk actions
  async function executeBulk() {
    const ids = [...selected]
    try {
      if (bulkModal === 'suspend') {
        await bulkSuspend(ids)
        addToast(`Suspended ${ids.length} tenant(s).`, 'success')
      } else if (bulkModal === 'plan') {
        await bulkPlanChange(ids, bulkPlanId)
        addToast(`Plan updated for ${ids.length} tenant(s).`, 'success')
      } else if (bulkModal === 'trial') {
        await bulkTrialExtend(ids, 7)
        addToast(`Trial extended 7 days for ${ids.length} tenant(s).`, 'success')
      }
    } catch (e) {
      addToast(e.message, 'error')
    }
    setBulkModal(null)
    setSelected(new Set())
    load()
  }

  const fmt = n => n >= 1e6 ? `${(n/1e6).toFixed(1)}M` : n >= 1000 ? `${(n/1000).toFixed(0)}k` : String(n || 0)

  return (
    <div className="page-enter">
      <SectionHeader
        title="All Tenants"
        sub={`${total} tenant${total !== 1 ? 's' : ''} total`}
      />

      {/* Filters */}
      <div className="filters-row">
        {/* Search */}
        <div className="search-wrap" style={{ flex: '1 1 240px', maxWidth: 320 }}>
          <span className="search-icon" style={{fontSize:13}}>⌕</span>
          <input
            className="input"
            placeholder="Search name or slug…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <select
          className="input"
          style={{ width: 'auto', minWidth: 140 }}
          value={planFilter}
          onChange={e => setPlanFilter(e.target.value)}
        >
          <option value="">All Plans</option>
          {plans.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>

        <select
          className="input"
          style={{ width: 'auto', minWidth: 150 }}
          value={statusFilter}
          onChange={e => setStatus(e.target.value)}
        >
          <option value="">All Statuses</option>
          <option value="active">Active</option>
          <option value="trial">Trial</option>
          <option value="over_quota">Over Quota</option>
          <option value="suspended">Suspended</option>
        </select>

        {(search || planFilter || statusFilter) && (
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => { setSearch(''); setPlanFilter(''); setStatus('') }}
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Table */}
      <div className="card">
        {loading ? (
          <Spinner />
        ) : tenants.length === 0 ? (
          <EmptyState icon="◧" title="No tenants found" body="Try adjusting your search filters." />
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th style={{ width: 36 }}>
                      <input
                        type="checkbox"
                        checked={selected.size === tenants.length && tenants.length > 0}
                        onChange={toggleAll}
                        style={{ accentColor: 'var(--accent)', cursor:'pointer' }}
                      />
                    </th>
                    <th>Tenant</th>
                    <th>Plan</th>
                    <th>Status</th>
                    <th>Vectors</th>
                    <th>Users</th>
                    <th>Trial Ends</th>
                    <th>Joined</th>
                  </tr>
                </thead>
                <tbody>
                  {tenants.map(t => {
                    const plan  = t.plans         || {}
                    const usage = t.tenant_usage   || {}
                    const pct   = plan.max_vectors > 0
                      ? (usage.vector_count / plan.max_vectors) * 100
                      : 0

                    return (
                      <tr
                        key={t.id}
                        onClick={() => navigate(`/tenants/${t.id}`)}
                        style={{ background: selected.has(t.id) ? 'var(--accent-subtle)' : undefined }}
                      >
                        <td onClick={e => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={selected.has(t.id)}
                            onChange={() => toggleOne(t.id)}
                            style={{ accentColor: 'var(--accent)', cursor:'pointer' }}
                          />
                        </td>
                        <td className="td-primary" style={{ minWidth: 180 }}>
                          {t.display_name}
                          <div className="mono-xs text-muted">{t.slug}</div>
                        </td>
                        <td><PlanBadge name={plan.name} /></td>
                        <td><StatusBadge status={t.status} /></td>
                        <td>
                          <span className="mono-sm" style={{ color: pct > 100 ? 'var(--red)' : pct > 80 ? 'var(--amber)' : 'var(--text-secondary)' }}>
                            {fmt(usage.vector_count)}
                          </span>
                          <span className="mono-xs text-muted"> / {fmt(plan.max_vectors)}</span>
                        </td>
                        <td className="mono-sm">
                          {usage.user_count || 0}
                          <span className="text-muted"> / {plan.max_users || '—'}</span>
                        </td>
                        <td>
                          {t.trial_ends_at
                            ? <span className="mono-xs" style={{color: new Date(t.trial_ends_at) < new Date() ? 'var(--red)' : 'var(--text-muted)'}}>
                                {new Date(t.trial_ends_at).toLocaleDateString()}
                              </span>
                            : <span className="text-muted">—</span>
                          }
                        </td>
                        <td><RelTime iso={t.created_at} /></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <Pagination
              page={page}
              pageSize={PAGE_SIZE}
              total={total}
              onChange={setPage}
            />
          </>
        )}
      </div>

      {/* Bulk actions bar */}
      {selected.size > 0 && (
        <div className="bulk-bar">
          <span className="bulk-bar-count">{selected.size} selected</span>
          <button className="btn btn-secondary btn-sm" onClick={() => setBulkModal('trial')}>
            +7d Trial
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => setBulkModal('plan')}>
            Change Plan
          </button>
          <button className="btn btn-danger btn-sm" onClick={() => setBulkModal('suspend')}>
            Suspend
          </button>
          <button
            className="btn btn-ghost btn-sm"
            style={{ marginLeft: 'auto' }}
            onClick={() => setSelected(new Set())}
          >
            Clear selection
          </button>
        </div>
      )}

      {/* Bulk modals */}
      {bulkModal === 'suspend' && (
        <ConfirmModal
          title="Suspend Selected Tenants"
          message={`This will suspend ${selected.size} tenant(s). Their users will immediately lose access. You can reactivate them individually from the tenant detail page.`}
          confirmLabel="Suspend All"
          danger
          onConfirm={executeBulk}
          onCancel={() => setBulkModal(null)}
        />
      )}

      {bulkModal === 'trial' && (
        <ConfirmModal
          title="Extend Trial by 7 Days"
          message={`This will extend the trial period of ${selected.size} tenant(s) by 7 days each.`}
          confirmLabel="Extend Trials"
          onConfirm={executeBulk}
          onCancel={() => setBulkModal(null)}
        />
      )}

      {bulkModal === 'plan' && (
        <div className="modal-overlay" onClick={() => setBulkModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">Change Plan for {selected.size} Tenant(s)</span>
              <button className="btn-icon" onClick={() => setBulkModal(null)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="form-field">
                <label className="label">Select New Plan</label>
                <select
                  className="input"
                  value={bulkPlanId}
                  onChange={e => setBulkPlanId(e.target.value)}
                >
                  <option value="">— choose a plan —</option>
                  {plans.filter(p => p.is_active).map(p => (
                    <option key={p.id} value={p.id}>{p.name} — ${p.price_monthly}/mo</option>
                  ))}
                </select>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setBulkModal(null)}>Cancel</button>
              <button
                className="btn btn-primary"
                disabled={!bulkPlanId}
                onClick={executeBulk}
              >Apply Plan</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}