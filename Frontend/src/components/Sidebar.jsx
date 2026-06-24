import { NavLink } from 'react-router-dom'
import { Home, MessageSquare, PlusCircle, X } from 'lucide-react'
import Logo from './Logo'
import ThemeToggle from './ThemeToggle'

const navItems = [
  { to: '/repos', label: 'Repositories', icon: Home },
  { to: '/ingest', label: 'Ingest Repo', icon: PlusCircle },
  { to: '/chat', label: 'Chat', icon: MessageSquare },
]

function linkClass({ isActive }) {
  return [
    'flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors',
    isActive
      ? 'bg-brand-50 text-brand-700 dark:bg-brand-900/30 dark:text-brand-300'
      : 'text-stone-600 hover:bg-stone-100 hover:text-stone-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100',
  ].join(' ')
}

export default function Sidebar({ open, onClose }) {
  return (
    <aside
      className={[
        'fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-stone-200 bg-white transition-transform duration-200 dark:border-zinc-800 dark:bg-zinc-950',
        open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
      ].join(' ')}
    >
      <div className="flex items-center justify-between border-b border-stone-200 px-5 py-4 dark:border-zinc-800">
        <div className="flex items-center gap-2.5">
          <Logo size={36} />
          <div>
            <p className="text-sm font-bold text-stone-900 dark:text-zinc-50">CodeQuery</p>
            <p className="text-xs text-stone-500 dark:text-zinc-500">RAG Assistant</p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg p-1.5 text-stone-500 hover:bg-stone-100 dark:text-zinc-400 dark:hover:bg-zinc-800 lg:hidden"
          aria-label="Close sidebar"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <nav className="flex-1 space-y-1 p-4">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} className={linkClass} onClick={onClose}>
            <Icon className="h-4 w-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="space-y-3 border-t border-stone-200 p-4 dark:border-zinc-800">
        <ThemeToggle />
        <p className="rounded-xl bg-stone-50 px-3 py-2 text-xs leading-relaxed text-stone-500 dark:bg-zinc-900 dark:text-zinc-500">
          Index a GitHub repo, then ask questions with hybrid BM25 + vector retrieval.
        </p>
      </div>
    </aside>
  )
}
