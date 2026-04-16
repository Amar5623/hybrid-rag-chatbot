// src/hooks/useChat.js
//
// CHANGES vs original:
//   - send() now accepts isOnline: bool as second argument.
//   - Online  → streamChat() SSE generator, same as before.
//   - Offline → fetchOfflineResponse() JSON call, renders chunk cards.
//   - Message format extended with is_offline + offline_chunks for offline responses.

import { useState, useCallback } from 'react'
import { streamChat, fetchOfflineResponse, clearSession } from '../api'

export function useChat() {
  const [messages,   setMessages]   = useState([])
  const [streaming,  setStreaming]  = useState(false)
  const [statusText, setStatusText] = useState('')

  const send = useCallback(async (question, isOnline = true) => {
    if (streaming) return

    const userMsg = { id: Date.now(), role: 'user', content: question }
    setMessages(prev => [...prev, userMsg])

    const assistantId = Date.now() + 1

    // ── OFFLINE ─────────────────────────────────────────
    if (!isOnline) {
      setMessages(prev => [...prev, {
        id             : assistantId,
        role           : 'assistant',
        content        : '',
        streaming      : false,
        is_offline     : true,
        offline_chunks : [],
        citations      : [],
        image_urls     : [],
        query_type     : 'offline',
        usage          : {},
      }])

      setStreaming(true)
      setStatusText('Searching manual sections…')

      try {
        const result = await fetchOfflineResponse(question)
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? {
                ...m,
                is_offline    : true,
                offline_chunks: result.chunks || [],
                content       : '',   // no LLM text in offline mode
              }
            : m
        ))
      } catch (err) {
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? { ...m, content: `⚠️ ${err.message}`, isError: true }
            : m
        ))
      } finally {
        setStreaming(false)
        setStatusText('')
      }
      return
    }

    // ── ONLINE (SSE stream) ──────────────────────────────
    setMessages(prev => [...prev, {
      id         : assistantId,
      role       : 'assistant',
      content    : '',
      streaming  : true,
      citations  : [],
      image_urls : [],
      query_type : 'document',
      usage      : {},
    }])

    setStreaming(true)
    setStatusText('Searching documents…')

    try {
      let firstToken = false
      for await (const event of streamChat(question)) {
        if (event.type === 'token') {
          if (!firstToken) { firstToken = true; setStatusText('') }
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: m.content + event.token }
              : m
          ))
        } else if (event.type === 'done') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? {
                  ...m,
                  streaming  : false,
                  citations  : event.citations  || [],
                  image_urls : event.image_urls || [],
                  query_type : event.query_type || 'document',
                  usage      : event.usage      || {},
                }
              : m
          ))
        } else if (event.type === 'error') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: `⚠️ ${event.message}`, streaming: false, isError: true }
              : m
          ))
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, content: `⚠️ ${err.message}`, streaming: false, isError: true }
          : m
      ))
    } finally {
      setStreaming(false)
      setStatusText('')
    }
  }, [streaming])

  const clear = useCallback(async () => {
    await clearSession()
    setMessages([])
  }, [])

  return { messages, streaming, statusText, send, clear }
}