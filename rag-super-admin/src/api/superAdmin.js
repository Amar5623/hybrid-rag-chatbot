// api/superAdmin.js
// All calls to /super-admin/* backend endpoints.
// Automatically injects the Supabase JWT from session storage.

import { supabase } from '../supabase'

const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function authHeaders() {
  const { data: { session } } = await supabase.auth.getSession()
  const token = session?.access_token
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function apiFetch(path, opts = {}) {
  const headers = await authHeaders()
  const res = await fetch(`${BASE}${path}`, { ...opts, headers: { ...headers, ...opts.headers } })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const json = await res.json()
      detail = json.detail?.message || json.detail || detail
    } catch {}
    throw new Error(detail)
  }

  const ct = res.headers.get('content-type') || ''
  if (ct.includes('application/json')) return res.json()
  return res.text()
}

// ── Tenants ─────────────────────────────────────────────────────────────────

export function listTenants({ page = 1, pageSize = 25, search = '', planId = '', status = '' } = {}) {
  const params = new URLSearchParams({
    page, page_size: pageSize,
    ...(search  ? { search }  : {}),
    ...(planId  ? { plan_id: planId } : {}),
    ...(status  ? { status }  : {}),
  })
  return apiFetch(`/super-admin/tenants?${params}`)
}

export function getTenant(tenantId) {
  return apiFetch(`/super-admin/tenants/${tenantId}`)
}

export function patchTenant(tenantId, body) {
  return apiFetch(`/super-admin/tenants/${tenantId}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  })
}

export function reconcileTenant(tenantId) {
  return apiFetch(`/super-admin/tenants/${tenantId}/reconcile`, { method: 'POST' })
}

export function impersonateTenant(tenantId) {
  return apiFetch(`/super-admin/tenants/${tenantId}/impersonate`, { method: 'POST' })
}

export function deleteTenantDocument(tenantId, docId) {
  return apiFetch(`/super-admin/tenants/${tenantId}/documents/${docId}`, { method: 'DELETE' })
}

// ── Plans ────────────────────────────────────────────────────────────────────

export function listPlans() {
  return apiFetch('/super-admin/plans')
}

export function createPlan(body) {
  return apiFetch('/super-admin/plans', { method: 'POST', body: JSON.stringify(body) })
}

export function patchPlan(planId, body) {
  return apiFetch(`/super-admin/plans/${planId}`, { method: 'PATCH', body: JSON.stringify(body) })
}

export function retirePlan(planId) {
  return apiFetch(`/super-admin/plans/${planId}/retire`, { method: 'PATCH' })
}

// ── Members ──────────────────────────────────────────────────────────────────

export function listMembers(tenantId) {
  return apiFetch(`/super-admin/tenants/${tenantId}/members`)
}

export function removeMember(tenantId, userId) {
  return apiFetch(`/super-admin/tenants/${tenantId}/members/${userId}`, { method: 'DELETE' })
}

export function promoteMember(tenantId, userId, role) {
  return apiFetch(`/super-admin/tenants/${tenantId}/members/${userId}/promote`, {
    method: 'PATCH',
    body: JSON.stringify({ role }),
  })
}

// ── Bulk operations ───────────────────────────────────────────────────────────

export function bulkPlanChange(tenantIds, planId) {
  return apiFetch('/super-admin/bulk/plan-change', {
    method: 'POST',
    body: JSON.stringify({ tenant_ids: tenantIds, plan_id: planId }),
  })
}

export function bulkTrialExtend(tenantIds, days) {
  return apiFetch('/super-admin/bulk/trial-extend', {
    method: 'POST',
    body: JSON.stringify({ tenant_ids: tenantIds, days }),
  })
}

export function bulkSuspend(tenantIds) {
  return apiFetch('/super-admin/bulk/suspend', {
    method: 'POST',
    body: JSON.stringify({ tenant_ids: tenantIds }),
  })
}

export function bulkConfigPush(planId, configPatch) {
  return apiFetch('/super-admin/bulk/config-push', {
    method: 'POST',
    body: JSON.stringify({ plan_id: planId, config_patch: configPatch }),
  })
}

// ── Activity & alerts ─────────────────────────────────────────────────────────

export function getActivity({ page = 1, pageSize = 25, tenantId = '', action = '' } = {}) {
  const params = new URLSearchParams({
    page, page_size: pageSize,
    ...(tenantId ? { tenant_id: tenantId } : {}),
    ...(action   ? { action }              : {}),
  })
  return apiFetch(`/super-admin/activity?${params}`)
}

export function getAlerts({ page = 1, pageSize = 50, unreadOnly = true } = {}) {
  const params = new URLSearchParams({ page, page_size: pageSize, unread_only: unreadOnly })
  return apiFetch(`/super-admin/alerts?${params}`)
}

export function markAlertRead(alertId) {
  return apiFetch(`/super-admin/alerts/${alertId}/read`, { method: 'PATCH' })
}