import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'codebase-qa-repos'

export const DEMO_REPOS = [
  {
    id: 'demo-fastapi',
    isDemo: true,
    name: 'tiangolo/fastapi',
    repoUrl: 'https://github.com/tiangolo/fastapi',
    languages: ['Python'],
    parents: 312,
    childrens: 1840,
    snippet: 'async def read_root():\n    return {"Hello": "World"}',
    indexedAt: '2025-06-20T10:00:00.000Z',
  },
  {
    id: 'demo-react',
    isDemo: true,
    name: 'facebook/react',
    repoUrl: 'https://github.com/facebook/react',
    languages: ['JavaScript', 'TypeScript'],
    parents: 890,
    childrens: 5200,
    snippet: 'function App() {\n  return <h1>Hello</h1>;\n}',
    indexedAt: '2025-06-18T14:30:00.000Z',
  },
  {
    id: 'demo-langchain',
    isDemo: true,
    name: 'langchain-ai/langchain',
    repoUrl: 'https://github.com/langchain-ai/langchain',
    languages: ['Python'],
    parents: 1204,
    childrens: 9100,
    snippet: 'chain = prompt | llm | StrOutputParser()',
    indexedAt: '2025-06-15T09:15:00.000Z',
  },
]

const ReposContext = createContext(null)

function parseRepoName(url) {
  try {
    const parts = new URL(url).pathname.split('/').filter(Boolean)
    if (parts.length >= 2) return `${parts[0]}/${parts[1]}`
  } catch {
    /* ignore */
  }
  return url
}

function loadStoredRepos() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function ReposProvider({ children }) {
  const [repos, setRepos] = useState(loadStoredRepos)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(repos))
  }, [repos])

  const addRepo = useCallback((ingestResult, repoUrl) => {
    const entry = {
      id: ingestResult.repoId,
      isDemo: false,
      name: parseRepoName(repoUrl),
      repoUrl,
      repoId: ingestResult.repoId,
      languages: ['Mixed'],
      parents: ingestResult.parents ?? 0,
      childrens: ingestResult.childrens ?? 0,
      snippet: `# Indexed ${ingestResult.parents} parent chunks\n# ${ingestResult.childrens} child chunks ready`,
      indexedAt: new Date().toISOString(),
    }
    setRepos((prev) => [entry, ...prev.filter((r) => r.id !== entry.id)])
    return entry
  }, [])

  const removeRepo = useCallback((id) => {
    setRepos((prev) => prev.filter((r) => r.id !== id))
  }, [])

  const getRepo = useCallback((id) => repos.find((r) => r.id === id || r.repoId === id), [repos])

  const value = useMemo(
    () => ({ repos, demoRepos: DEMO_REPOS, addRepo, removeRepo, getRepo }),
    [repos, addRepo, removeRepo, getRepo],
  )

  return <ReposContext.Provider value={value}>{children}</ReposContext.Provider>
}

export function useRepos() {
  const ctx = useContext(ReposContext)
  if (!ctx) throw new Error('useRepos must be used within ReposProvider')
  return ctx
}

export function formatRelativeTime(iso) {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}
