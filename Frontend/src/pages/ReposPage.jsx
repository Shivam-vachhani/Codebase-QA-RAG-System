import RepoCard, { AddRepoCard } from '../components/RepoCard'
import { useRepos } from '../hooks/useRepos.jsx'

export default function ReposPage() {
  const { repos, demoRepos, removeRepo } = useRepos()

  return (
    <div className="flex-1 px-4 py-6 sm:px-8 sm:py-8">
      <div className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-stone-900 dark:text-zinc-50">Your Indexed Repositories</h1>
          <p className="mt-1 text-sm text-stone-500 dark:text-zinc-400">
            Demo cards are previews — index a real repo to start chatting.
          </p>
        </div>
      </div>

      {repos.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-4 text-xs font-semibold uppercase tracking-wide text-stone-400 dark:text-zinc-600">
            Your Repositories
          </h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {repos.map((repo) => (
              <RepoCard key={repo.id} repo={repo} onRemove={removeRepo} />
            ))}
          </div>
        </section>
      )}

      <section>
        <h2 className="mb-4 text-xs font-semibold uppercase tracking-wide text-stone-400 dark:text-zinc-600">
          {repos.length > 0 ? 'Demo & Add' : 'Get Started'}
        </h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <AddRepoCard />
          {demoRepos.map((repo) => (
            <RepoCard key={repo.id} repo={repo} />
          ))}
        </div>
      </section>
    </div>
  )
}
