// pages/PlanManagementPage.jsx
import React, { useEffect, useState } from 'react'
import { listPlans, createPlan, patchPlan, retirePlan, bulkConfigPush } from '../api/superAdmin'
import { SectionHeader, ConfirmModal, JsonEditor, EmptyState, Spinner } from '../components/Shared'
import { useToast } from '../context/ToastContext'

const BLANK_PLAN = {
  name: '', max_users: 5, max_vectors: 10000, max_batch_pdfs: 3,
  allowed_modes: ['online'], price_monthly: 0,
}

function PlanCard({ plan, onEdit, onRetire }) {
  const retired = !plan.is_active

  return (
    <div className={`plan-card ${retired ? 'retired' : ''}`}>
      {retired && (
        <div style={{ position:'absolute', top:12, right:12 }}>
          <span className="badge neutral">Retired</span>
        </div>
      )}
      <div className="plan-name">{plan.name}</div>
      <div className="plan-price">
        ${plan.price_monthly}<span>/mo</span>
      </div>
      <div className="plan-limits">
        {[
          ['Max Vectors',    (plan.max_vectors || 0).toLocaleString()],
          ['Max Users',      plan.max_users],
          ['Max Batch PDFs', plan.max_batch_pdfs],
          ['Allowed Modes',  (plan.allowed_modes || []).join(', ')],
        ].map(([k,v]) => (
          <div key={k} className="plan-limit-row">
            <span>{k}</span>
            <span className="val">{v}</span>
          </div>
        ))}
      </div>
      <div style={{ display:'flex', gap:8, marginTop:20 }}>
        <button className="btn btn-secondary btn-sm" onClick={() => onEdit(plan)} style={{ flex:1 }}>
          Edit
        </button>
        {!retired && (
          <button className="btn btn-danger btn-sm" onClick={() => onRetire(plan)}>
            Retire
          </button>
        )}
      </div>
    </div>
  )
}

export default function PlanManagementPage() {
  const { addToast }  = useToast()
  const [plans,       setPlans]     = useState([])
  const [loading,     setLoading]   = useState(true)
  const [showCreate,  setShowCreate] = useState(false)
  const [editPlan,    setEditPlan]  = useState(null)
  const [retireTarget,setRetireTarget] = useState(null)
  const [form,        setForm]      = useState(BLANK_PLAN)
  const [saving,      setSaving]    = useState(false)

  // Bulk config push state
  const [showConfigPush, setShowConfigPush] = useState(false)
  const [pushPlanId,     setPushPlanId]     = useState('')
  const [pushJson,       setPushJson]       = useState('{}')

  useEffect(() => { load() }, [])

  async function load() {
    setLoading(true)
    try { setPlans(await listPlans()) }
    catch (e) { addToast(e.message, 'error') }
    setLoading(false)
  }

  function openCreate() {
    setForm(BLANK_PLAN)
    setEditPlan(null)
    setShowCreate(true)
  }

  function openEdit(plan) {
    setForm({
      name:           plan.name,
      max_users:      plan.max_users,
      max_vectors:    plan.max_vectors,
      max_batch_pdfs: plan.max_batch_pdfs,
      allowed_modes:  plan.allowed_modes || [],
      price_monthly:  plan.price_monthly,
    })
    setEditPlan(plan)
    setShowCreate(true)
  }

  async function savePlan() {
    setSaving(true)
    const body = {
      ...form,
      max_users:      Number(form.max_users),
      max_vectors:    Number(form.max_vectors),
      max_batch_pdfs: Number(form.max_batch_pdfs),
      price_monthly:  Number(form.price_monthly),
      allowed_modes:  typeof form.allowed_modes === 'string'
        ? form.allowed_modes.split(',').map(s => s.trim())
        : form.allowed_modes,
    }
    try {
      if (editPlan) {
        await patchPlan(editPlan.id, body)
        addToast('Plan updated.', 'success')
      } else {
        await createPlan(body)
        addToast('Plan created.', 'success')
      }
      setShowCreate(false)
      load()
    } catch (e) {
      addToast(e.message, 'error')
    }
    setSaving(false)
  }

  async function doRetire() {
    try {
      await retirePlan(retireTarget.id)
      addToast(`"${retireTarget.name}" retired.`, 'success')
      setRetireTarget(null)
      load()
    } catch (e) {
      addToast(e.message, 'error')
    }
  }

  async function doBulkConfigPush() {
    let parsed
    try { parsed = JSON.parse(pushJson) }
    catch { addToast('Invalid JSON.', 'error'); return }
    try {
      const r = await bulkConfigPush(pushPlanId, parsed)
      addToast(`Config pushed to ${Object.keys(r.results).length} tenants.`, 'success')
      setShowConfigPush(false)
    } catch (e) {
      addToast(e.message, 'error')
    }
  }

  const f = (k, v) => setForm(prev => ({ ...prev, [k]: v }))

  if (loading) return <Spinner text="Loading plans…" />

  const active  = plans.filter(p => p.is_active)
  const retired = plans.filter(p => !p.is_active)

  return (
    <div className="page-enter">
      <SectionHeader
        title="Plan Management"
        sub={`${active.length} active plan${active.length !== 1 ? 's' : ''}`}
      >
        <button className="btn btn-secondary btn-sm" onClick={() => setShowConfigPush(true)}>
          Push Config
        </button>
        <button className="btn btn-primary btn-sm" onClick={openCreate}>
          + New Plan
        </button>
      </SectionHeader>

      {/* Active plans */}
      <div className="three-col mb-24">
        {active.map(p => (
          <PlanCard key={p.id} plan={p} onEdit={openEdit} onRetire={setRetireTarget} />
        ))}
        {active.length === 0 && <EmptyState icon="◫" title="No active plans" />}
      </div>

      {/* Retired plans */}
      {retired.length > 0 && (
        <>
          <div style={{ fontSize:12, color:'var(--text-muted)', marginBottom:12, fontFamily:'var(--font-mono)', letterSpacing:'0.1em', textTransform:'uppercase' }}>
            Retired Plans
          </div>
          <div className="three-col">
            {retired.map(p => (
              <PlanCard key={p.id} plan={p} onEdit={openEdit} onRetire={() => {}} />
            ))}
          </div>
        </>
      )}

      {/* Create / Edit modal */}
      {showCreate && (
        <div className="modal-overlay" onClick={() => setShowCreate(false)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth:480 }}>
            <div className="modal-header">
              <span className="modal-title">{editPlan ? `Edit Plan — ${editPlan.name}` : 'Create New Plan'}</span>
              <button className="btn-icon" onClick={() => setShowCreate(false)}>✕</button>
            </div>
            <div className="modal-body">
              {editPlan && (
                <div style={{ background:'var(--amber-bg)', border:'1px solid rgba(251,191,36,0.3)', borderRadius:'var(--r-md)', padding:'8px 12px', marginBottom:16, fontSize:12, color:'var(--amber)' }}>
                  ⚠ Changes apply immediately to all tenants on this plan.
                </div>
              )}
              <div className="two-col" style={{ gap:12 }}>
                <div className="form-field" style={{ gridColumn:'1/-1' }}>
                  <label className="label">Plan Name</label>
                  <input className="input" value={form.name} onChange={e => f('name', e.target.value)} placeholder="e.g. Growth" />
                </div>
                <div className="form-field">
                  <label className="label">Max Vectors</label>
                  <input type="number" className="input input-mono" value={form.max_vectors} onChange={e => f('max_vectors', e.target.value)} />
                </div>
                <div className="form-field">
                  <label className="label">Max Users</label>
                  <input type="number" className="input input-mono" value={form.max_users} onChange={e => f('max_users', e.target.value)} />
                </div>
                <div className="form-field">
                  <label className="label">Max Batch PDFs</label>
                  <input type="number" className="input input-mono" value={form.max_batch_pdfs} onChange={e => f('max_batch_pdfs', e.target.value)} />
                </div>
                <div className="form-field">
                  <label className="label">Price / Month ($)</label>
                  <input type="number" className="input input-mono" value={form.price_monthly} onChange={e => f('price_monthly', e.target.value)} step="0.01" />
                </div>
                <div className="form-field" style={{ gridColumn:'1/-1' }}>
                  <label className="label">Allowed Modes (comma-separated)</label>
                  <input
                    className="input input-mono"
                    value={Array.isArray(form.allowed_modes) ? form.allowed_modes.join(', ') : form.allowed_modes}
                    onChange={e => f('allowed_modes', e.target.value)}
                    placeholder="online, offline, hybrid"
                  />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowCreate(false)}>Cancel</button>
              <button
                className="btn btn-primary"
                onClick={savePlan}
                disabled={saving || !form.name}
              >
                {saving ? 'Saving…' : editPlan ? 'Save Changes' : 'Create Plan'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Retire confirm */}
      {retireTarget && (
        <ConfirmModal
          title={`Retire "${retireTarget.name}"`}
          message="Existing tenants on this plan are unaffected. Only new signups will be blocked from choosing it."
          confirmLabel="Retire Plan"
          danger
          onConfirm={doRetire}
          onCancel={() => setRetireTarget(null)}
        />
      )}

      {/* Bulk config push modal */}
      {showConfigPush && (
        <div className="modal-overlay" onClick={() => setShowConfigPush(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">Push Config to Plan</span>
              <button className="btn-icon" onClick={() => setShowConfigPush(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p className="text-secondary text-sm mb-16">
                Pushes config keys to all tenants on the selected plan — only for keys they have not already customized.
              </p>
              <div className="form-field">
                <label className="label">Target Plan</label>
                <select className="input" value={pushPlanId} onChange={e => setPushPlanId(e.target.value)}>
                  <option value="">— select a plan —</option>
                  {plans.filter(p => p.is_active).map(p => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </div>
              <JsonEditor
                label="Config Patch (JSON keys to push)"
                value={pushJson}
                onChange={setPushJson}
                rows={6}
              />
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowConfigPush(false)}>Cancel</button>
              <button
                className="btn btn-primary"
                disabled={!pushPlanId}
                onClick={doBulkConfigPush}
              >Push Config</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}