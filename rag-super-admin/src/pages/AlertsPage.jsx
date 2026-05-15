// rag-super-admin/src/pages/AlertsPage.jsx
import React, { useEffect, useState } from 'react';
import { getAlerts, markAlertRead } from '../api/superAdmin';
import { Pagination, Spinner, EmptyState, RelTime, SectionHeader, ConfirmModal } from '../components/Shared';
import { useToast } from '../context/ToastContext';

const ALERT_TYPE_COLOR = {
  reconciliation_drift: 'warning',
  trial_expiring:       'info',
  over_quota_3d:        'danger',
  manual_reconciliation:'success',
  default:              'neutral',
};

export default function AlertsPage() {
  const { addToast } = useToast();
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [unreadOnly, setUnreadOnly] = useState(true);
  const [markingId, setMarkingId] = useState(null);
  const PAGE_SIZE = 25;

  useEffect(() => {
    load();
  }, [page, unreadOnly]);

  async function load() {
    setLoading(true);
    try {
      const data = await getAlerts({ page, pageSize: PAGE_SIZE, unreadOnly });
      setAlerts(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      addToast(err.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  async function handleMarkRead(alertId) {
    setMarkingId(alertId);
    try {
      await markAlertRead(alertId);
      addToast('Alert marked as read.', 'success');
      // Refresh current page
      await load();
    } catch (err) {
      addToast(err.message, 'error');
    } finally {
      setMarkingId(null);
    }
  }

  function getTypeStyle(type) {
    return ALERT_TYPE_COLOR[type] || ALERT_TYPE_COLOR.default;
  }

  return (
    <div className="page-enter">
      <SectionHeader
        title="Alerts"
        sub="System alerts across all tenants"
      >
        <button
          className={`btn btn-sm ${unreadOnly ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => {
            setUnreadOnly(!unreadOnly);
            setPage(1);
          }}
        >
          {unreadOnly ? 'Showing unread only' : 'Show all'}
        </button>
      </SectionHeader>

      <div className="card">
        {loading ? (
          <Spinner />
        ) : alerts.length === 0 ? (
          <EmptyState icon="🔔" title="No alerts" body={unreadOnly ? 'All clear – no unread alerts.' : 'No alerts found.'} />
        ) : (
          <>
            <div className="card-body-flush">
              {alerts.map(alert => (
                <div
                  key={alert.id}
                  className={`alert-item ${!alert.is_read ? 'unread' : ''}`}
                >
                  <div
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      background: `var(--${getTypeStyle(alert.type)})`,
                      marginTop: 6,
                      flexShrink: 0,
                    }}
                  />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 4 }}>
                      <span className="badge" style={{ background: `var(--${getTypeStyle(alert.type)}-bg)`, color: `var(--${getTypeStyle(alert.type)})` }}>
                        {alert.type}
                      </span>
                      {!alert.is_read && (
                        <span className="badge warning" style={{ fontSize: 9 }}>UNREAD</span>
                      )}
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 4 }}>
                      {alert.message}
                    </div>
                    {alert.tenant_id && (
                      <div className="mono-xs text-muted" style={{ marginBottom: 4 }}>
                        Tenant: {alert.tenant_id}
                      </div>
                    )}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 6 }}>
                      <RelTime iso={alert.created_at} />
                      {!alert.is_read && (
                        <button
                          className="btn btn-ghost btn-xs"
                          onClick={() => handleMarkRead(alert.id)}
                          disabled={markingId === alert.id}
                        >
                          {markingId === alert.id ? <span className="spinner" style={{ width: 12, height: 12 }} /> : 'Mark read'}
                        </button>
                      )}
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