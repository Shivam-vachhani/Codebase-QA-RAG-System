import IngestWizard from '../components/IngestWizard'
import { useRepos } from '../hooks/useRepos.jsx'

export default function IngestPage() {
  const { addRepo } = useRepos()

  return (
    <div className="flex-1 px-4 py-6 sm:px-8 sm:py-8">
      <IngestWizard onComplete={(data, url) => addRepo(data, url)} />
    </div>
  )
}
