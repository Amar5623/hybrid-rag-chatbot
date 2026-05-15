// rag-super-admin/src/pages/ActivityFeedPage.jsx
import React, { useEffect, useState } from 'react';
import { getActivity } from '../api/superAdmin';
import { Pagination, Spinner, EmptyState, RelTime, SectionHeader } from '../components/Shared';
import { useToast } from '../context/ToastContext';

const ACTION_COLOR = {
  tenant_updated:      'info',
  tenant_reconciled:   'success',
  tenant_impersonated: 'warning',
  document_deleted:    'danger',
  member_removed:      'danger',
  member_role_changed: 'info',
  plan_created:        'success',
  plan_updated:        'warning',
  plan_retired:        'neutral',
  bulk_plan_change:    'info',
  bulk_trial_extend:   'success',
  bulk_suspend:        'danger',
  bulk_config_push:    'info',
  alert_read:          'neutral',
  super_admin_access:  'info',
};

export default function ActivityFeedPage() {
  const { addToast } = useToast();
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [tenantFilter, setTenantFilter] = useState('');
  const [actionFilter, setActionFilter] = useState('');
  const PAGE_SIZE = 25;

  useEffect(() => {
    load();
  }, [page, tenantFilter, actionFilter]);

  async function load() {
    setLoading(true);
    try {
      const data = await getActivity({
        page,
        pageSize: PAGE_SIZE,
        tenantId: tenantFilter,
        action: actionFilter,
      });
      setEntries(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      addToast(err.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  // Extract unique action types from current entries for filter dropdown (could also be static)
  const actionTypes = [...new Set(entries.map(e => e.action))].sort();

  return (
    <div className="page-enter">
      <SectionHeader
        title="Activity Feed"
        sub="Audit trail across all tenants"
      />

      {/* Filters */}
      <div className="filters-row">
        <input
          className="input"
          placeholder="Tenant ID (optional)"
          value={tenantFilter}
          onChange={e => {
            setTenantFilter(e.target.value);
            setPage(1);
          }}
          style={{ width: 280 }}
        />
        <select
          className="input"
          value={actionFilter}
          onChange={e => {
            setActionFilter(e.target.value);
            setPage(1);
          }}
          style={{ width: 200 }}
        >
          <option value="">All actions</option>
          {actionTypes.map(a => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>
        {(tenantFilter || actionFilter) && (
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              setTenantFilter('');
              setActionFilter('');
              setPage(1);
            }}
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Activity list */}
      <div className="card">
        {loading ? (
          <Spinner />
        ) : entries.length === 0 ? (
          <EmptyState icon="📋" title="No activity found" body="Try adjusting filters." />
        ) : (
          <>
            <div className="card-body-flush">
              {entries.map(entry => (
                <div key={entry.id} className="activity-item">
                  <div className={`activity-dot ${ACTION_COLOR[entry.action] || ''}`} />
                  <div className="activity-content">
                    <div className="activity-action">
                      <span className="mono-xs" style={{ color: 'var(--accent)', marginRight: 8 }}>
                        {entry.action}
                      </span>
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        {entry.actor_email}
                      </span>
                    </div>
                    {entry.tenant_id && (
                      <div className="activity-meta" style={{ marginTop: 2 }}>
                        <span>Tenant: {entry.tenant_id}</span>
                      </div>
                    )}
                    {entry.payload && Object.keys(entry.payload).length > 0 && (
                      <pre
                        className="mono-xs text-muted"
                        style={{
                          marginTop: 8,
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          maxWidth: '100%',
                          background: 'var(--bg-base)',
                          padding: 8,
                          borderRadius: 'var(--r-sm)',
                          fontSize: 11,
                        }}
                      >
                        {JSON.stringify(entry.payload, null, 2).slice(0, 300)}
                        {JSON.stringify(entry.payload).length > 300 ? '…' : ''}
                      </pre>
                    )}
                    <div className="activity-meta" style={{ marginTop: 6 }}>
                      <RelTime iso={entry.created_at} />
                    </div>
                  </div>
                </div>
              ))}
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
    </div>
  );
}