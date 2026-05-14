import React, { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '../supabase'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [session,  setSession]  = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [userEmail, setUserEmail] = useState('')

  useEffect(() => {
    // Restore session on mount
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      setUserEmail(session?.user?.email || '')
      setLoading(false)
    })

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
      setUserEmail(session?.user?.email || '')
    })

    return () => subscription.unsubscribe()
  }, [])

  async function login(email, password) {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) throw error

    // Verify role after login
    const role = data.session?.user?.app_metadata?.role
    if (role !== 'super_admin') {
      await supabase.auth.signOut()
      throw new Error('Access denied. Super admin role required.')
    }
    return data
  }

  async function logout() {
    await supabase.auth.signOut()
    setSession(null)
  }

  async function getToken() {
    const { data: { session } } = await supabase.auth.getSession()
    return session?.access_token || null
  }

  const value = {
    session,
    loading,
    userEmail,
    isAuthenticated: !!session,
    login,
    logout,
    getToken,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}