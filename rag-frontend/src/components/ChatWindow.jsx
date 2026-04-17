// src/components/ChatWindow.jsx
//
// CHANGES vs previous version:
//   - MessageBubble now receives `query` prop (the last user question)
//     so B4 keyword highlighting works correctly.
//   - Everything else unchanged.

import { useEffect, useRef, useState } from 'react'
import MessageBubble from './MessageBubble'
import { ingestFiles, unpinFile } from '../api'

const SUGGESTIONS = [
  'Summarise the key points',
  'What are the main findings?',
  'List all dates and deadlines',
  'Compare the data in the tables',
  'What methodology was used?',
  'Give me an executive summary',
]

export default function ChatWindow({
  messages, streaming, statusText, onSend, kbReady, onFilesIndexed,
  pinnedFile, onUnpin, isOnline,
}) {
  const [input,        setInput]        = useState('')
  const [uploading,    setUploading]    = useState(false)
  const [uploadMsg,    setUploadMsg]    = useState('')
  const [pendingFiles, setPendingFiles] = useState([])
  const [focused,      setFocused]      = useState(false)
  const bottomRef  = useRef()
  const textRef    = useRef()
  const uploadRef  = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, statusText])

  const submit = () => {
    const q = input.trim()
    if (!q || streaming) return
    setInput('')
    if (textRef.current) textRef.current.style.height = 'auto'
    onSend(q)
  }

  const handleKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  const handleFileSelect = e => {
    const files = [...e.target.files]
    if (!files.length) return
    e.target.value = ''
    setPendingFiles(p => [...p, ...files])
  }

  const removePending = i => setPendingFiles(p => p.filter((_, j) => j !== i))

  const doIndex = async () => {
    if (!pendingFiles.length) return
    setUploading(true)
    setUploadMsg(`Indexing ${pendingFiles.length} file${pendingFiles.length > 1 ? 's' : ''}…`)
    try {
      await ingestFiles(pendingFiles)
      setPendingFiles([])
      onFilesIndexed?.()
    } catch (err) {
      alert(`Upload failed: ${err.message}`)
    } finally {
      setUploading(false)
      setUploadMsg('')
    }
  }

  const handleUnpin = async () => {
    try {
      await unpinFile()
      onUnpin?.()
    } catch (e) {
      alert(e.message)
    }
  }

  const extIcon = n => {
    const l = n.toLowerCase()
    if (l.endsWith('.pdf'))  return '📄'
    if (l.endsWith('.csv'))  return '📊'
    if (l.endsWith('.xlsx')) return '📗'
    return '📝'
  }

  const isEmpty = messages.length === 0

  const inputDisabled = streaming || uploading || !kbReady

  // Derive the last user query so MessageBubble can highlight keywords (B4)
  const lastUserQuery = [...messages].reverse().find(m => m.role === 'user')?.content || ''

  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      height: '100%', overflow: 'hidden', background: 'var(--bg-0)',
    }}>

      {/* ── Top bar ── */}
      <div style={{
        padding: '0 28px', height: 56,
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'var(--bg-1)', flexShrink: 0,
      }}>
        {/* Left */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            fontFamily: 'var(--font-display)', fontWeight: 700,
            fontSize: '.95rem', color: 'var(--text-0)', letterSpacing: '-.01em',
          }}>Document Intelligence</div>
          <div style={{ display: 'flex', gap: 5 }}>
            {['Hybrid retrieval', 'Cross-encoder', isOnline ? 'Online' : 'Offline'].map(t => (
              <span key={t} style={{
                fontSize: '.62rem', fontFamily: 'var(--font-mono)',
                color: t === 'Online' ? 'var(--teal)' : t === 'Offline' ? '#f59e0b' : 'var(--text-3)',
                background: 'var(--bg-3)',
                border: '1px solid var(--border)', padding: '3px 8px', borderRadius: 12,
              }}>{t}</span>
            ))}
          </div>
        </div>

        {/* Right */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>

          {/* Focus badge */}
          {pinnedFile && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              background: 'rgba(124,106,247,.12)',
              border: '1px solid rgba(124,106,247,.35)',
              borderRadius: 20, padding: '4px 10px',
              fontSize: '.7rem', color: 'var(--accent-text)',
              fontFamily: 'var(--font-mono)', maxWidth: 220,
            }}>
              <span style={{ fontSize: 11 }}>📌</span>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                {pinnedFile}
              </span>
              <button onClick={handleUnpin} title="Unpin — search all files" style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: 'var(--accent-text)', fontSize: '.75rem',
                lineHeight: 1, padding: '0 2px', opacity: .7,
              }}>✕</button>
            </div>
          )}

          {(streaming || uploading) && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 7,
              fontSize: '.72rem', color: 'var(--accent-text)',
              fontFamily: 'var(--font-mono)',
            }}>
              <div style={{
                width: 7, height: 7, borderRadius: '50%',
                background: 'var(--accent)', animation: 'pulse 1s ease infinite',
              }} />
              {uploading ? uploadMsg : (statusText || (isOnline ? 'Generating…' : 'Searching sections…'))}
            </div>
          )}
        </div>
      </div>

      {/* ── Messages area ── */}
      <div style={{
        flex: 1, overflowY: 'auto', padding: '28px 32px',
        display: 'flex', flexDirection: 'column',
      }}>
        {isEmpty && (
          <div style={{
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            height: '100%', gap: 36, animation: 'fadeUp .4s var(--ease)',
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                fontFamily: 'var(--font-display)', fontWeight: 800,
                fontSize: '2.2rem', letterSpacing: '-.04em',
                background: 'linear-gradient(135deg, var(--text-0) 0%, var(--text-1) 60%, var(--accent-text) 100%)',
                backgroundSize: '200%',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                animation: 'shimmer 6s linear infinite', lineHeight: 1.2, marginBottom: 10,
              }}>
                {pinnedFile ? `Focused on\n${pinnedFile}` : 'Ask your manuals\nanything.'}
              </div>
              <div style={{ fontSize: '.85rem', color: 'var(--text-2)', maxWidth: 420, lineHeight: 1.7 }}>
                {!isOnline
                  ? 'Offline mode — manual sections will be shown directly without AI generation.'
                  : pinnedFile
                  ? `All answers will come from "${pinnedFile}" only.`
                  : kbReady
                  ? 'Your knowledge base is ready. Try a suggestion below or ask your own question.'
                  : 'Upload PDF manuals via the sidebar or the + button below to get started.'}
              </div>
            </div>

            {kbReady && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7, justifyContent: 'center', maxWidth: 540 }}>
                {SUGGESTIONS.map(s => (
                  <button key={s} onClick={() => onSend(s)} style={{
                    background: 'var(--bg-2)', border: '1px solid var(--border-md)',
                    borderRadius: 20, padding: '7px 15px', fontSize: '.78rem',
                    color: 'var(--text-1)', cursor: 'pointer', transition: 'all .15s',
                    fontFamily: 'var(--font-body)',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent-text)' }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-md)'; e.currentTarget.style.color = 'var(--text-1)' }}
                  >{s}</button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Pass lastUserQuery so each assistant bubble can highlight keywords (B4) */}
        {messages.map((msg, idx) => {
          // For each assistant message, find the most recent preceding user message
          let q = ''
          if (msg.role === 'assistant') {
            for (let i = idx - 1; i >= 0; i--) {
              if (messages[i].role === 'user') { q = messages[i].content; break }
            }
          }
          return <MessageBubble key={msg.id} message={msg} query={q} />
        })}
        <div ref={bottomRef} />
      </div>

      {/* ── Pending files tray ── */}
      {pendingFiles.length > 0 && (
        <div style={{ padding: '10px 28px 0', background: 'var(--bg-1)', borderTop: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
            {pendingFiles.map((f, i) => (
              <div key={i} style={{
                display: 'inline-flex', alignItems: 'center', gap: 5,
                background: 'var(--bg-2)', border: '1px solid var(--border-md)',
                borderRadius: 20, padding: '3px 10px 3px 8px',
                fontSize: '.71rem', color: 'var(--text-1)',
              }}>
                <span style={{ fontSize: 11 }}>{extIcon(f.name)}</span>
                <span style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                <span onClick={() => removePending(i)} style={{ cursor: 'pointer', color: 'var(--text-3)', fontSize: '.8rem', lineHeight: 1, marginLeft: 2 }}>✕</span>
              </div>
            ))}
            <button onClick={doIndex} disabled={uploading} style={{
              background: 'linear-gradient(135deg, var(--accent), var(--accent-dim))',
              border: 'none', borderRadius: 20, padding: '4px 14px',
              fontSize: '.71rem', color: '#fff',
              cursor: uploading ? 'not-allowed' : 'pointer',
              fontFamily: 'var(--font-display)', fontWeight: 700, opacity: uploading ? .6 : 1,
            }}>
              {uploading ? uploadMsg : `Index ${pendingFiles.length} file${pendingFiles.length > 1 ? 's' : ''}`}
            </button>
          </div>
        </div>
      )}

      {/* ── Input bar ── */}
      <div style={{
        padding: '14px 28px 20px',
        borderTop: pendingFiles.length ? 'none' : '1px solid var(--border)',
        background: 'var(--bg-1)', flexShrink: 0,
      }}>
        <div style={{
          display: 'flex', alignItems: 'flex-end', gap: 8,
          background: 'var(--bg-0)',
          border: `1.5px solid ${focused ? 'var(--accent)' : !isOnline ? 'rgba(245,158,11,.3)' : pinnedFile ? 'rgba(124,106,247,.4)' : 'var(--border-md)'}`,
          borderRadius: 'var(--r-xl)', padding: '10px 12px',
          transition: 'border-color .2s',
          boxShadow: focused ? '0 0 0 3px var(--accent-glow)' : 'none',
        }}>
          <button onClick={() => uploadRef.current?.click()} disabled={uploading} title="Add files" style={{
            width: 32, height: 32, borderRadius: 'var(--r-md)',
            border: `1px solid ${pendingFiles.length ? 'var(--accent)' : 'var(--border-md)'}`,
            background: pendingFiles.length ? 'var(--accent-glow)' : 'var(--bg-2)',
            color: pendingFiles.length ? 'var(--accent-text)' : 'var(--text-1)',
            cursor: uploading ? 'not-allowed' : 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1.1rem', flexShrink: 0, transition: 'all .15s', position: 'relative',
          }}>
            {uploading
              ? <div style={{ width: 13, height: 13, border: '2px solid var(--border-md)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin .6s linear infinite' }} />
              : '+'}
            {pendingFiles.length > 0 && !uploading && (
              <span style={{
                position: 'absolute', top: -5, right: -5,
                width: 16, height: 16, borderRadius: '50%',
                background: 'var(--accent)', color: '#fff',
                fontSize: '.6rem', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontWeight: 700, fontFamily: 'var(--font-mono)',
              }}>{pendingFiles.length}</span>
            )}
          </button>
          <input ref={uploadRef} type="file" multiple accept=".pdf"
            style={{ display: 'none' }} onChange={handleFileSelect} />

          <textarea
            ref={textRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder={
              uploading    ? uploadMsg :
              !isOnline    ? 'Offline — will return relevant manual sections…' :
              pinnedFile   ? `Ask about ${pinnedFile}…` :
              kbReady      ? 'Ask anything about your manuals…' :
              'Upload PDF manuals to get started…'
            }
            disabled={inputDisabled}
            rows={1}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              resize: 'none', fontFamily: 'var(--font-body)', fontSize: '.9rem',
              color: 'var(--text-0)', lineHeight: 1.55,
              maxHeight: 130, overflowY: 'auto', padding: 0,
              cursor: inputDisabled ? 'not-allowed' : 'text',
            }}
            onInput={e => {
              e.target.style.height = 'auto'
              e.target.style.height = Math.min(e.target.scrollHeight, 130) + 'px'
            }}
          />

          <button onClick={submit} disabled={!input.trim() || streaming || !kbReady} style={{
            width: 34, height: 34, borderRadius: '50%', border: 'none',
            cursor: (!input.trim() || streaming || !kbReady) ? 'not-allowed' : 'pointer',
            background: (!input.trim() || streaming || !kbReady)
              ? 'var(--bg-4)'
              : isOnline
              ? 'linear-gradient(135deg, var(--accent), var(--accent-dim))'
              : 'linear-gradient(135deg, #92400e, #78350f)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', color: '#fff', flexShrink: 0, transition: 'all .2s',
            boxShadow: (!input.trim() || streaming || !kbReady) ? 'none' : '0 2px 12px var(--accent-glow)',
          }}>
            {streaming
              ? <div style={{ width: 12, height: 12, border: '2px solid rgba(255,255,255,.3)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin .6s linear infinite' }} />
              : '↑'}
          </button>
        </div>
        <div style={{
          textAlign: 'center', fontSize: '.62rem', color: 'var(--text-3)',
          marginTop: 8, fontFamily: 'var(--font-mono)',
        }}>
          {!isOnline
            ? '📵 Offline — retrieval only, no AI generation'
            : pinnedFile
            ? `📌 Focused on ${pinnedFile} — click the badge above to search all files`
            : 'Enter to send · Shift+Enter for newline · + to queue files'}
        </div>
      </div>
    </div>
  )
}