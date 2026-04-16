// src/App.jsx
// Auth gate removed — app loads directly into chat.
// Network status is polled every 30s from /health and shared via prop drilling.

import { useState, useEffect } from 'react'
import Sidebar       from './components/Sidebar'
import ChatWindow    from './components/ChatWindow'
import NetworkBanner from './components/NetworkBanner'
import { useChat }   from './hooks/useChat'
import { checkNetworkStatus } from './api'

export default function App() {
  const [kbReady,      setKbReady]      = useState(false)
  const [refreshCount, setRefreshCount] = useState(0)
  const [pinnedFile,   setPinnedFile]   = useState(null)
  const [isOnline,     setIsOnline]     = useState(true)

  const { messages, streaming, statusText, send, clear } = useChat()

  // Poll network status every 30 seconds
  useEffect(() => {
    const poll = async () => {
      try {
        const { is_online } = await checkNetworkStatus()
        setIsOnline(is_online)
      } catch {
        // If backend is unreachable, treat as offline
        setIsOnline(false)
      }
    }

    poll()
    const interval = setInterval(poll, 30_000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <NetworkBanner isOnline={isOnline} />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar
          kbReady={kbReady}
          setKbReady={setKbReady}
          onClearChat={clear}
          refreshKey={refreshCount}
          pinnedFile={pinnedFile}
          onPin={setPinnedFile}
          onUnpin={() => setPinnedFile(null)}
          isOnline={isOnline}
        />
        <ChatWindow
          messages={messages}
          streaming={streaming}
          statusText={statusText}
          onSend={(q) => send(q, isOnline)}
          kbReady={kbReady}
          onFilesIndexed={() => setRefreshCount(c => c + 1)}
          pinnedFile={pinnedFile}
          onUnpin={() => setPinnedFile(null)}
          isOnline={isOnline}
        />
      </div>
    </div>
  )
}