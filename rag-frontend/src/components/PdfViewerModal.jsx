// src/components/PdfViewerModal.jsx
import { useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';

// Use the legacy build (more reliable with Vite) + correct worker
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;

export default function PdfViewerModal({
  filename,
  page: initialPage = 1,
  bbox = null,           // [x0, y0, x1, y1]
  sectionPath = '',
  onClose,
}) {
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const [pdfDoc, setPdfDoc] = useState(null);
  const [currentPage, setCurrentPage] = useState(initialPage);
  const [scale, setScale] = useState(1.5);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadPdf = async () => {
      try {
        setLoading(true);
        const loadingTask = pdfjsLib.getDocument(`/pdfs/${filename}`);
        const pdf = await loadingTask.promise;
        setPdfDoc(pdf);
        setLoading(false);
        renderPage(pdf, currentPage);
      } catch (err) {
        console.error(err);
        setError(`Could not load PDF: ${filename}. Make sure the backend serves PDFs at /pdfs/`);
        setLoading(false);
      }
    };
    loadPdf();
  }, [filename]);

  const renderPage = async (pdf, pageNum) => {
    const page = await pdf.getPage(pageNum);
    const viewport = page.getViewport({ scale });

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.height = viewport.height;
    canvas.width = viewport.width;

    await page.render({ canvasContext: ctx, viewport }).promise;

    // Draw highlight rectangle if bbox exists
    if (bbox && overlayRef.current) {
      const overlay = overlayRef.current;
      overlay.width = viewport.width;
      overlay.height = viewport.height;
      const oCtx = overlay.getContext('2d');
      oCtx.clearRect(0, 0, overlay.width, overlay.height);

      const [x0, y0, x1, y1] = bbox;
      const width = x1 - x0;
      const height = y1 - y0;

      oCtx.fillStyle = 'rgba(251, 191, 36, 0.35)';
      oCtx.fillRect(
        x0 * scale,
        (viewport.height - y1 * scale),
        width * scale,
        height * scale
      );
    }
  };

  useEffect(() => {
    if (pdfDoc) renderPage(pdfDoc, currentPage);
  }, [currentPage, scale, pdfDoc]);

  const nextPage = () => setCurrentPage(p => Math.min(p + 1, pdfDoc?.numPages || 1));
  const prevPage = () => setCurrentPage(p => Math.max(p - 1, 1));

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.9)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 9999,
    }} onClick={e => e.target === e.currentTarget && onClose()}>
      <div style={{
        background: 'var(--bg-1)', borderRadius: 'var(--r-lg)',
        width: '95%', maxWidth: 1100, maxHeight: '95vh',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{ padding: '12px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '.9rem', color: 'var(--teal)' }}>📖 {filename}</div>
            {sectionPath && <div style={{ fontSize: '.75rem', color: 'var(--text-2)', marginTop: 2 }}>{sectionPath}</div>}
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <button onClick={prevPage} style={navBtn}>‹ Prev</button>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-1)' }}>
              Page {currentPage} / {pdfDoc ? pdfDoc.numPages : '?'}
            </span>
            <button onClick={nextPage} style={navBtn}>Next ›</button>
            <button onClick={onClose} style={{ background: 'var(--bg-3)', border: '1px solid var(--border-md)', color: 'var(--text-1)', padding: '6px 14px', borderRadius: 8, fontSize: '.8rem' }}>
              Close
            </button>
          </div>
        </div>

        {/* PDF Area */}
        <div style={{ position: 'relative', flex: 1, overflow: 'auto', background: '#1a1a1a', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {loading && <div style={{ color: 'var(--text-2)' }}>Loading PDF…</div>}
          {error && <div style={{ color: '#f59e0b' }}>{error}</div>}

          {!loading && !error && (
            <>
              <canvas ref={canvasRef} style={{ maxHeight: '100%', boxShadow: '0 10px 30px rgba(0,0,0,0.6)' }} />
              <canvas ref={overlayRef} style={{ position: 'absolute', pointerEvents: 'none', maxHeight: '100%' }} />
            </>
          )}
        </div>

        {/* Zoom controls */}
        <div style={{ padding: '10px 20px', borderTop: '1px solid var(--border)', display: 'flex', gap: 12, justifyContent: 'center' }}>
          <button onClick={() => setScale(s => Math.max(0.8, s - 0.3))} style={zoomBtn}>− Zoom</button>
          <button onClick={() => setScale(1.5)} style={zoomBtn}>Reset Zoom</button>
          <button onClick={() => setScale(s => s + 0.3)} style={zoomBtn}>+ Zoom</button>
        </div>
      </div>
    </div>
  );
}

const navBtn = { background: 'var(--bg-3)', border: '1px solid var(--border-md)', color: 'var(--text-1)', padding: '6px 16px', borderRadius: 8, cursor: 'pointer', fontSize: '.8rem' };
const zoomBtn = { ...navBtn };