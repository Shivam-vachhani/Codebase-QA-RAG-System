import { ChevronDown, FileCode2, ArrowUp } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import Logo from './Logo'
import MarkdownContent from './MarkdownContent'

function SourceList({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources?.length) return null

  return (
    <div className="mt-6 border-t border-stone-200 pt-4 dark:border-zinc-800">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-xs font-medium text-stone-500 transition hover:text-brand-600 dark:text-zinc-400 dark:hover:text-brand-400"
      >
        <FileCode2 className="h-3.5 w-3.5" />
        {sources.length} source{sources.length !== 1 ? 's' : ''} referenced
        <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <ul className="mt-3 space-y-2">
          {sources.map((src, i) => (
            <li
              key={`${src.file}-${src.line}-${i}`}
              className="rounded-lg bg-stone-100 px-3 py-2 font-mono text-[11px] text-stone-600 dark:bg-zinc-900 dark:text-zinc-400"
            >
              <span className="text-brand-600 dark:text-brand-400">{src.file}</span>
              <span className="text-stone-400 dark:text-zinc-600"> · line {src.line}</span>
              {src.language && (
                <span className="ml-2 rounded bg-stone-200 px-1.5 py-0.5 text-[10px] dark:bg-zinc-800">
                  {src.language}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function UserQuery({ content }) {
  return (
    <div className="py-8 text-center">
      <p className="mx-auto max-w-chat text-lg font-medium leading-relaxed text-stone-900 dark:text-zinc-100 sm:text-xl">
        {content}
      </p>
    </div>
  )
}

function AssistantBlock({ content, sources, loading }) {
  return (
    <div className="flex gap-4 pb-10">
      <div className="sticky top-4 shrink-0 self-start">
        <Logo size={28} animated={loading} />
      </div>
      <div className="min-w-0 flex-1">
        {loading ? (
          <div className="space-y-3 pt-1">
            <p className="text-shimmer text-sm font-medium">Searching codebase…</p>
            <div className="space-y-2">
              <div className="h-3 w-4/5 animate-pulse rounded bg-stone-200 dark:bg-zinc-800" />
              <div className="h-3 w-3/5 animate-pulse rounded bg-stone-200 dark:bg-zinc-800" />
              <div className="h-3 w-2/5 animate-pulse rounded bg-stone-200 dark:bg-zinc-800" />
            </div>
          </div>
        ) : (
          <>
            <MarkdownContent content={content} />
            <SourceList sources={sources} />
          </>
        )}
      </div>
    </div>
  )
}

function EmptyState({ onSend, disabled, loading }) {
  const suggestions = [
    'How does the RAG pipeline work?',
    'Explain the hybrid retriever',
    'Where is the LLM prompt defined?',
  ]

  return (
    <div className="flex flex-col items-center px-4 py-16 text-center">
      <Logo size={56} className="mb-6" />
      <h2 className="text-xl font-semibold text-stone-900 dark:text-zinc-50">
        Ask anything about your codebase
      </h2>
      <p className="mt-2 max-w-md text-sm text-stone-500 dark:text-zinc-400">
        Hybrid BM25 + vector search with source citations. Questions appear centered, answers flow below.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {suggestions.map((s) => (
          <button
            key={s}
            type="button"
            disabled={disabled || loading}
            onClick={() => onSend(s)}
            className="rounded-full border border-stone-200 bg-white px-4 py-2 text-xs text-stone-600 transition hover:border-brand-300 hover:text-brand-600 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400 dark:hover:border-brand-600 dark:hover:text-brand-400"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function ChatPanel({ messages, onSend, loading, disabled }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSubmit = (e) => {
    e.preventDefault()
    const q = input.trim()
    if (!q || loading || disabled) return
    setInput('')
    onSend(q)
  }

  const pairs = []
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === 'user') {
      const assistant = messages[i + 1]?.role === 'assistant' ? messages[i + 1] : null
      pairs.push({ user: messages[i], assistant })
      if (assistant) i++
    }
  }

  return (
    <div className="flex flex-1 flex-col">
      <div className="scrollbar-thin flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-chat px-4 sm:px-6">
          {messages.length === 0 ? (
            <EmptyState onSend={onSend} disabled={disabled} loading={loading} />
          ) : (
            <div className="py-4">
              {pairs.map(({ user, assistant }) => (
                <div key={user.id} className="border-b border-stone-100 last:border-0 dark:border-zinc-900">
                  <UserQuery content={user.content} />
                  {assistant && (
                    <AssistantBlock
                      content={assistant.content}
                      sources={assistant.sources}
                      loading={assistant.loading}
                    />
                  )}
                </div>
              ))}
            </div>
          )}
          <div ref={bottomRef} className="h-4" />
        </div>
      </div>

      <div className="sticky bottom-0 border-t border-stone-200 bg-stone-50/95 px-4 py-4 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/95 sm:px-6">
        <form onSubmit={handleSubmit} className="mx-auto max-w-chat">
          <div className="relative rounded-2xl border border-stone-300 bg-white shadow-sm transition focus-within:border-brand-400 focus-within:ring-2 focus-within:ring-brand-400/20 dark:border-zinc-700 dark:bg-zinc-900 dark:focus-within:border-brand-500">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e)
                }
              }}
              rows={1}
              disabled={disabled || loading}
              placeholder={disabled ? 'Select an indexed repo to start…' : 'Ask a follow-up…'}
              className="max-h-36 min-h-[52px] w-full resize-none bg-transparent px-4 py-3.5 pr-14 text-sm text-stone-900 outline-none placeholder:text-stone-400 disabled:opacity-50 dark:text-zinc-100 dark:placeholder:text-zinc-500"
            />
            <button
              type="submit"
              disabled={disabled || loading || !input.trim()}
              className="absolute bottom-2.5 right-2.5 flex h-9 w-9 items-center justify-center rounded-xl bg-brand-500 text-white transition hover:bg-brand-600 disabled:opacity-30"
              aria-label="Send message"
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-stone-400 dark:text-zinc-600">
            Answers are grounded in indexed code · Shift+Enter for new line
          </p>
        </form>
      </div>
    </div>
  )
}
