import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Check, ChevronRight, Loader2 } from 'lucide-react'
import { ingestRepo } from '../api/client'

const STEPS = [
  { id: 'clone', label: 'Clone Repository' },
  { id: 'parse', label: 'Parse Files' },
  { id: 'chunk', label: 'Chunk Code' },
  { id: 'embed', label: 'Embed & Store' },
  { id: 'ready', label: 'Ready' },
]

function StepIcon({ status }) {
  if (status === 'done') {
    return (
      <span className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald-500 text-white">
        <Check className="h-4 w-4" />
      </span>
    )
  }
  if (status === 'active') {
    return (
      <span className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-500 text-white">
        <Loader2 className="h-4 w-4 animate-spin" />
      </span>
    )
  }
  return (
    <span className="flex h-7 w-7 items-center justify-center rounded-full bg-stone-200 text-xs font-medium text-stone-500 dark:bg-zinc-800 dark:text-zinc-500">
      ·
    </span>
  )
}

export default function IngestWizard({ onComplete }) {
  const navigate = useNavigate()
  const [repoUrl, setRepoUrl] = useState('')
  const [phase, setPhase] = useState('idle')
  const [activeStep, setActiveStep] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const runStepAnimation = useCallback(async () => {
    for (let i = 0; i < STEPS.length - 1; i++) {
      setActiveStep(i)
      await new Promise((r) => setTimeout(r, 700 + i * 200))
    }
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)

    const url = repoUrl.trim()
    if (!url) {
      setError('Please enter a GitHub repository URL.')
      return
    }

    try {
      new URL(url)
    } catch {
      setError('Enter a valid URL (e.g. https://github.com/owner/repo).')
      return
    }

    setPhase('running')
    setActiveStep(0)

    const animation = runStepAnimation()

    try {
      const data = await ingestRepo(url)
      await animation
      setActiveStep(STEPS.length - 1)
      setResult(data)
      setPhase('done')
      onComplete?.(data, url)
    } catch (err) {
      setPhase('idle')
      setActiveStep(0)
      setError(err.message || 'Ingestion failed. Check the repo URL and backend logs.')
    }
  }

  useEffect(() => {
    if (phase !== 'running') return
    const onBeforeUnload = (e) => {
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [phase])

  return (
    <div className="mx-auto w-full max-w-2xl">
      <div className="mb-8 rounded-2xl bg-gradient-to-r from-brand-500 to-brand-700 px-6 py-8 text-white shadow-lg">
        <h1 className="text-xl font-bold sm:text-2xl">Index your repository</h1>
        <p className="mt-2 text-sm text-brand-100">
          Clone, parse with Tree-sitter, chunk, embed, and store in ChromaDB.
        </p>
      </div>

      <div className="mb-8 overflow-x-auto">
        <div className="flex min-w-[520px] items-center justify-between px-2">
          {STEPS.map((step, i) => (
            <div key={step.id} className="flex flex-1 items-center">
              <div className="flex flex-col items-center gap-1.5">
                <StepIcon
                  status={
                    phase === 'done'
                      ? 'done'
                      : phase === 'running' && i < activeStep
                        ? 'done'
                        : phase === 'running' && i === activeStep
                          ? 'active'
                          : 'pending'
                  }
                />
                <span className="max-w-[72px] text-center text-[10px] font-medium text-stone-600 dark:text-zinc-400 sm:text-xs">
                  {step.label}
                </span>
              </div>
              {i < STEPS.length - 1 && (
                <div
                  className={[
                    'mx-1 mb-5 h-0.5 flex-1 rounded',
                    phase === 'done' || (phase === 'running' && i < activeStep)
                      ? 'bg-emerald-400'
                      : 'bg-stone-200 dark:bg-zinc-800',
                  ].join(' ')}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-2xl border border-stone-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        {phase === 'idle' && (
          <form onSubmit={handleSubmit} className="space-y-4">
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium text-stone-700 dark:text-zinc-300">GitHub Repository URL</span>
              <input
                type="url"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/owner/repository"
                className="w-full rounded-xl border border-stone-300 bg-white px-4 py-3 text-sm text-stone-900 outline-none transition focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
              />
            </label>
            {error && (
              <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-950/50 dark:text-red-400">{error}</p>
            )}
            <button
              type="submit"
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-brand-500 px-4 py-3 text-sm font-semibold text-white hover:bg-brand-600 sm:w-auto"
            >
              Start Indexing
              <ChevronRight className="h-4 w-4" />
            </button>
          </form>
        )}

        {phase === 'running' && (
          <div className="space-y-3 py-4 text-center">
            <Loader2 className="mx-auto h-8 w-8 animate-spin text-brand-500" />
            <p className="text-sm font-medium text-stone-800 dark:text-zinc-200">
              {STEPS[activeStep]?.label ?? 'Processing…'}
            </p>
            <p className="text-xs text-stone-500 dark:text-zinc-500">
              This may take a minute for larger repositories. Keep this tab open.
            </p>
          </div>
        )}

        {phase === 'done' && result && (
          <div className="space-y-4">
            <div className="rounded-xl bg-emerald-50 px-4 py-3 text-sm text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-400">
              Repository indexed successfully.
            </div>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-xl bg-stone-50 p-3 dark:bg-zinc-800">
                <dt className="text-xs text-stone-500 dark:text-zinc-500">Repo ID</dt>
                <dd className="font-mono font-semibold text-stone-900 dark:text-zinc-100">{result.repoId}</dd>
              </div>
              <div className="rounded-xl bg-stone-50 p-3 dark:bg-zinc-800">
                <dt className="text-xs text-stone-500 dark:text-zinc-500">Parent chunks</dt>
                <dd className="font-semibold text-stone-900 dark:text-zinc-100">{result.parents}</dd>
              </div>
              <div className="rounded-xl bg-stone-50 p-3 dark:bg-zinc-800">
                <dt className="text-xs text-stone-500 dark:text-zinc-500">Child chunks</dt>
                <dd className="font-semibold text-stone-900 dark:text-zinc-100">{result.childrens}</dd>
              </div>
              <div className="rounded-xl bg-stone-50 p-3 dark:bg-zinc-800">
                <dt className="text-xs text-stone-500 dark:text-zinc-500">Status</dt>
                <dd className="font-semibold text-emerald-600 dark:text-emerald-400">{result.status}</dd>
              </div>
            </dl>
            <button
              type="button"
              onClick={() => navigate(`/chat/${result.repoId}`)}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-brand-500 px-4 py-3 text-sm font-semibold text-white hover:bg-brand-600"
            >
              Start Chatting
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
