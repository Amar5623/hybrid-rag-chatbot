// src/hooks/useAdminApi.js
//
// Thin wrapper that adds per-call loading + error state management around
// any api.js function.  Keeps pages clean — no repetitive try/catch blocks.
//
// Usage:
//   const { run: deleteDoc, loading, error } = useAdminApi()
//   await deleteDoc(() => adminDeleteDocument(id))

import { useState, useCallback } from 'react'

export function useAdminApi() {
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  const run = useCallback(async (apiFn) => {
    setLoading(true)
    setError(null)
    try {
      const result = await apiFn()
      return result
    } catch (err) {
      setError(err.message ?? 'Something went wrong')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const clearError = useCallback(() => setError(null), [])

  return { run, loading, error, clearError }
}