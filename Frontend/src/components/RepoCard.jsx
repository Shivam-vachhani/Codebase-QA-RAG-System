import { Link } from 'react-router-dom'
import { ArrowRight, Clock, GitBranch, Sparkles } from 'lucide-react'
import { formatRelativeTime } from '../hooks/useRepos.jsx'

function SnippetPreview({ code }) {
  const lines = code.split('\n').slice(0, 3)
  return (
    <pre className="overflow-hidden rounded-lg bg-zinc-900 px-3 py-2 text-[11px] leading-relaxed text-zinc-300">
      {lines.map((line, i) => (
        <span key={i} className="block truncate">
          {line}
        </span>
      ))}
    </pre>
  )
}

export default function RepoCard({ repo, onRemove }) {
  const isDemo = repo.isDemo
  const chatLink = isDemo ? '/ingest' : `/chat/${repo.repoId ?? repo.id}`

  return (
    <article
      className={[
        'group flex flex-col rounded-2xl border bg-white p-5 shadow-sm transition-shadow hover:shadow-md dark:bg-zinc-900',
        isDemo ? 'border-dashed border-stone-300 dark:border-zinc-700' : 'border-stone-200 dark:border-zinc-800',
      ].join(' ')}
    >
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="mb-1 flex items-center gap-2">
            <GitBranch className="h-4 w-4 shrink-0 text-brand-500" />
            <h3 className="truncate text-sm font-semibold text-stone-900 dark:text-zinc-100">{repo.name}</h3>
          </div>
          {isDemo ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-950/50 dark:text-amber-400">
              <Sparkles className="h-3 w-3" />
              Demo preview
            </span>
          ) : (
            <span className="inline-flex rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400">
              Indexed · {repo.repoId ?? repo.id}
            </span>
          )}
        </div>
        {!isDemo && onRemove && (
          <button
            type="button"
            onClick={() => onRemove(repo.id)}
            className="shrink-0 text-xs text-stone-400 hover:text-red-500 dark:text-zinc-500"
          >
            Remove
          </button>
        )}
      </div>

      <div className="mb-3 flex flex-wrap gap-1.5">
        {repo.languages.map((lang) => (
          <span
            key={lang}
            className="rounded-md bg-stone-100 px-2 py-0.5 text-[10px] font-medium text-stone-600 dark:bg-zinc-800 dark:text-zinc-400"
          >
            {lang}
          </span>
        ))}
      </div>

      <SnippetPreview code={repo.snippet} />

      <div className="mt-3 flex items-center justify-between text-xs text-stone-500 dark:text-zinc-500">
        <span>{repo.parents + repo.childrens} chunks</span>
        <span className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {formatRelativeTime(repo.indexedAt)}
        </span>
      </div>

      <Link
        to={chatLink}
        className="mt-4 inline-flex items-center justify-center gap-2 rounded-xl bg-brand-500 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-brand-600"
      >
        {isDemo ? 'Index a real repo' : 'Open Chat'}
        <ArrowRight className="h-4 w-4" />
      </Link>
    </article>
  )
}

export function AddRepoCard() {
  return (
    <Link
      to="/ingest"
      className="flex min-h-[280px] flex-col items-center justify-center rounded-2xl border-2 border-dashed border-brand-300 bg-brand-50/40 p-6 text-center transition-colors hover:border-brand-500 hover:bg-brand-50 dark:border-brand-700 dark:bg-brand-950/20 dark:hover:border-brand-500 dark:hover:bg-brand-950/40"
    >
      <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-brand-100 text-brand-600 dark:bg-brand-900/50 dark:text-brand-400">
        <span className="text-2xl leading-none">+</span>
      </div>
      <h3 className="text-sm font-semibold text-stone-900 dark:text-zinc-100">Add Repository</h3>
      <p className="mt-1 max-w-[200px] text-xs text-stone-500 dark:text-zinc-500">
        Paste a GitHub URL to clone, chunk, and index your codebase
      </p>
    </Link>
  )
}
