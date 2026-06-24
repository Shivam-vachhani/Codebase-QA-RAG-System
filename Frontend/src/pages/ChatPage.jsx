import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ChevronDown, PlusCircle } from 'lucide-react'
import ChatPanel from '../components/ChatPanel'
import { queryRepo } from '../api/client'
import { useRepos } from '../hooks/useRepos.jsx'

const MODELS = [
  { id: 'gpt-4o', label: 'GPT-4o' },
  { id: 'qwen-2.5', label: 'Qwen 2.5' },
]

export default function ChatPage() {
  const { repoId: paramRepoId } = useParams()
  const navigate = useNavigate()
  const { repos } = useRepos()
  const [selectedRepoId, setSelectedRepoId] = useState(paramRepoId ?? '')
  const [model, setModel] = useState('gpt-4o')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const msgId = useRef(0)

  const indexedRepos = useMemo(() => repos.filter((r) => !r.isDemo), [repos])

  useEffect(() => {
    if (paramRepoId) setSelectedRepoId(paramRepoId)
    else if (!selectedRepoId && indexedRepos.length === 1) {
      setSelectedRepoId(indexedRepos[0].repoId ?? indexedRepos[0].id)
    }
  }, [paramRepoId, indexedRepos, selectedRepoId])

  useEffect(() => {
    if (selectedRepoId && selectedRepoId !== paramRepoId) {
      navigate(`/chat/${selectedRepoId}`, { replace: true })
    }
  }, [selectedRepoId, paramRepoId, navigate])

  const activeRepo = indexedRepos.find(
    (r) => (r.repoId ?? r.id) === selectedRepoId,
  )

  const handleSend = useCallback(
    async (question) => {
      if (!selectedRepoId) return

      const userId = ++msgId.current
      const assistantId = ++msgId.current

      setError('')
      setMessages((prev) => [
        ...prev,
        { id: userId, role: 'user', content: question },
        { id: assistantId, role: 'assistant', content: '', loading: true },
      ])
      setLoading(true)

      try {
        const data = await queryRepo({
          question,
          repoId: selectedRepoId,
          model,
        })

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  loading: false,
                  content: data.response?.answer ?? 'No answer returned.',
                  sources: data.response?.sources ?? [],
                }
              : m,
          ),
        )
      } catch (err) {
        setMessages((prev) => prev.filter((m) => m.id !== assistantId))
        setError(err.message || 'Query failed.')
      } finally {
        setLoading(false)
      }
    },
    [selectedRepoId, model],
  )

  return (
    <div className="flex flex-1 flex-col">
      <header className="sticky top-0 z-10 border-b border-stone-200 bg-stone-50/95 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/95">
        <div className="mx-auto flex max-w-chat flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <div className="relative min-w-0 flex-1">
              <select
                value={selectedRepoId}
                onChange={(e) => {
                  setSelectedRepoId(e.target.value)
                  setMessages([])
                  setError('')
                }}
                className="w-full appearance-none truncate rounded-xl border border-stone-300 bg-white py-2 pl-3 pr-9 text-sm font-medium text-stone-800 outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
              >
                <option value="">Select repository…</option>
                {indexedRepos.map((repo) => (
                  <option key={repo.id} value={repo.repoId ?? repo.id}>
                    {repo.name}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-stone-400" />
            </div>
            {activeRepo && (
              <span className="hidden shrink-0 rounded-lg bg-stone-200/70 px-2 py-1 font-mono text-[10px] text-stone-500 dark:bg-zinc-800 dark:text-zinc-400 sm:inline">
                {activeRepo.repoId ?? activeRepo.id}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <div className="flex rounded-xl border border-stone-200 bg-stone-100 p-1 dark:border-zinc-700 dark:bg-zinc-900">
              {MODELS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setModel(m.id)}
                  className={[
                    'rounded-lg px-3 py-1.5 text-xs font-medium transition',
                    model === m.id
                      ? 'bg-white text-brand-700 shadow-sm dark:bg-zinc-800 dark:text-brand-300'
                      : 'text-stone-500 hover:text-stone-700 dark:text-zinc-500 dark:hover:text-zinc-300',
                  ].join(' ')}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <Link
              to="/ingest"
              className="hidden items-center gap-1 rounded-xl border border-stone-200 px-3 py-2 text-xs font-medium text-stone-600 hover:bg-stone-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800 sm:inline-flex"
            >
              <PlusCircle className="h-3.5 w-3.5" />
              New
            </Link>
          </div>
        </div>

        {error && (
          <p className="mx-auto mb-3 max-w-chat rounded-lg bg-red-50 px-4 py-2 text-xs text-red-600 dark:bg-red-950/50 dark:text-red-400 sm:px-6">
            {error}
          </p>
        )}
      </header>

      {!selectedRepoId ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-4 px-4 py-12 text-center">
          <p className="text-sm text-stone-500 dark:text-zinc-400">No indexed repository selected.</p>
          <Link
            to="/ingest"
            className="rounded-xl bg-brand-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-brand-600"
          >
            Index a repository
          </Link>
        </div>
      ) : (
        <ChatPanel
          messages={messages}
          onSend={handleSend}
          loading={loading}
          disabled={!selectedRepoId}
        />
      )}
    </div>
  )
}
