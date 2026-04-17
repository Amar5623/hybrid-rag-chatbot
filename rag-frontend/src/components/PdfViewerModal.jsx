// src/components/PdfViewerModal.jsx
//
// FIXES vs previous version:
//   - renderPage now receives `scale` as a parameter instead of closing over
//     stale state — eliminates the stale-closure bug where zoom changes
//     wouldn't re-render at the correct scale.
//   - Overlay canvas is now inside a `position:relative` wrapper div that
//     matches the PDF canvas size exactly, so the highlight bbox aligns
//     correctly over the rendered text regardless of the flex container.
//   - useCallback removed (simpler: just pass scale explicitly each call).
//   - Everything else (props API, styling, zoom/nav controls) unchanged.

import { useEffect, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'

// Use the legacy build (more reliable with Vite) + correct worker
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker

export default function PdfViewerModal({
  filename,
  page: initialPage = 1,
  bbox = null,           // [x0, y0, x1, y1] in PDF point coordinates
  sectionPath = '',
  onClose,
}) {
  const canvasRef  = useRef(null)
  const overlayRef = useRef(null)
  const [pdfDoc,      setPdfDoc]      = useState(null)
  const [currentPage, setCurrentPage] = useState(initialPage)
  const [scale,       setScale]       = useState(1.5)
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState(null)

  // ── Render a single page at the given scale ──────────────────
  // Scale is passed explicitly to avoid stale-closure issues.
  const renderPage = async (pdf, pageNum, renderScale) => {
    if (!pdf || !canvasRef.current) return

    const page     = await pdf.getPage(pageNum)
    const viewport = page.getViewport({ scale: renderScale })

    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')
    canvas.width  = viewport.width
    canvas.height = viewport.height

    await page.render({ canvasContext: ctx, viewport }).promise

    // ── Draw highlight bbox over matched text ─────────────────
    // PDF coordinate origin is bottom-left; canvas origin is top-left.
    // Convert:  canvasY = pageHeight - pdfY  (in PDF points, before scale)
    // Then multiply by scale for final pixel position.
    if (bbox && overlayRef.current) {
      const overlay = overlayRef.current
      overlay.width  = viewport.width
      overlay.height = viewport.height

      const oCtx = overlay.getContext('2d')
      oCtx.clearRect(0, 0, overlay.width, overlay.height)

      const [x0, y0, x1, y1] = bbox
      const pageHeightPt = viewport.height / renderScale  // original PDF height in points

      // Convert PDF point coords → canvas pixel coords
      const cx = x0 * renderScale
      const cy = (pageHeightPt - y1) * renderScale   // flip Y axis
      const cw = (x1 - x0) * renderScale
      const ch = (y1 - y0) * renderScale

      oCtx.fillStyle   = 'rgba(251, 191, 36, 0.35)'
      oCtx.strokeStyle = 'rgba(251, 191, 36, 0.8)'
      oCtx.lineWidth   = 1.5
      oCtx.fillRect(cx, cy, cw, ch)
      oCtx.strokeRect(cx, cy, cw, ch)
    }
  }

  // ── Load PDF on filename change ───────────────────────────────
  useEffect(() => {
    let cancelled = false

    const loadPdf = async () => {
      try {
        setLoading(true)
        setError(null)
        const pdf = await pdfjsLib.getDocument(`/pdfs/${filename}`).promise
        if (cancelled) return
        setPdfDoc(pdf)
        setLoading(false)
        // Render immediately with current values — no stale closure
        await renderPage(pdf, currentPage, scale)
      } catch (err) {
        if (cancelled) return
        console.error(err)
        setError(`Could not load PDF: ${filename}. Make sure the backend serves PDFs at /pdfs/`)
        setLoading(false)
      }
    }

    loadPdf()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filename])

  // ── Re-render on page or scale change ────────────────────────
  useEffect(() => {
    if (pdfDoc) renderPage(pdfDoc, currentPage, scale)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPage, scale, pdfDoc])

  const nextPage = () => setCurrentPage(p => Math.min(p + 1, pdfDoc?.numPages || 1))
  const prevPage = () => setCurrentPage(p => Math.max(p - 1, 1))

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.9)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 9999,
      }}
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <div style={{
        background: 'var(--bg-1)', borderRadius: 'var(--r-lg)',
        width: '95%', maxWidth: 1100, maxHeight: '95vh',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>

        {/* ── Header ── */}
        <div style={{
          padding: '12px 20px', borderBottom: '1px solid var(--border)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          flexShrink: 0,
        }}>
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '.9rem', color: 'var(--teal)' }}>
              📖 {filename}
            </div>
            {sectionPath && (
              <div style={{ fontSize: '.75rem', color: 'var(--text-2)', marginTop: 2 }}>
                {sectionPath}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <button onClick={prevPage} disabled={currentPage <= 1} style={navBtn}>‹ Prev</button>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-1)', fontSize: '.82rem' }}>
              Page {currentPage} / {pdfDoc ? pdfDoc.numPages : '?'}
            </span>
            <button onClick={nextPage} disabled={pdfDoc && currentPage >= pdfDoc.numPages} style={navBtn}>Next ›</button>
            <button onClick={onClose} style={navBtn}>✕ Close</button>
          </div>
        </div>

        {/* ── PDF area ── */}
        <div style={{
          flex: 1, overflow: 'auto', background: '#1a1a1a',
          display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
          padding: '20px 0',
        }}>
          {loading && (
            <div style={{ color: 'var(--text-2)', alignSelf: 'center' }}>Loading PDF…</div>
          )}
          {error && (
            <div style={{ color: '#f59e0b', padding: 20, maxWidth: 500, alignSelf: 'center', textAlign: 'center' }}>
              {error}
            </div>
          )}

          {/* Wrapper keeps overlay perfectly on top of the PDF canvas */}
          {!loading && !error && (
            <div style={{ position: 'relative', flexShrink: 0 }}>
              <canvas
                ref={canvasRef}
                style={{ display: 'block', boxShadow: '0 10px 30px rgba(0,0,0,0.6)' }}
              />
              {/* Overlay canvas for bbox highlight — sits exactly over the PDF canvas */}
              <canvas
                ref={overlayRef}
                style={{
                  position: 'absolute', top: 0, left: 0,
                  pointerEvents: 'none',
                }}
              />
            </div>
          )}
        </div>

        {/* ── Zoom controls ── */}
        <div style={{
          padding: '10px 20px', borderTop: '1px solid var(--border)',
          display: 'flex', gap: 12, justifyContent: 'center', flexShrink: 0,
        }}>
          <button onClick={() => setScale(s => Math.max(0.5, parseFloat((s - 0.25).toFixed(2))))} style={zoomBtn}>− Zoom</button>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '.75rem', color: 'var(--text-2)', alignSelf: 'center' }}>
            {Math.round(scale * 100)}%
          </span>
          <button onClick={() => setScale(1.5)} style={zoomBtn}>Reset</button>
          <button onClick={() => setScale(s => parseFloat((s + 0.25).toFixed(2)))} style={zoomBtn}>+ Zoom</button>
        </div>
      </div>
    </div>
  )
}

const navBtn = {
  background: 'var(--bg-3)', border: '1px solid var(--border-md)',
  color: 'var(--text-1)', padding: '6px 16px',
  borderRadius: 8, cursor: 'pointer', fontSize: '.8rem',
}
const zoomBtn = { ...navBtn }