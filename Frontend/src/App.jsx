import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import ReposPage from './pages/ReposPage'
import IngestPage from './pages/IngestPage'

const ChatPage = lazy(() => import('./pages/ChatPage'))

function PageLoader() {
  return (
    <div className="flex flex-1 items-center justify-center p-12">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/repos" replace />} />
        <Route path="repos" element={<ReposPage />} />
        <Route path="ingest" element={<IngestPage />} />
        <Route
          path="chat/:repoId?"
          element={
            <Suspense fallback={<PageLoader />}>
              <ChatPage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/repos" replace />} />
      </Route>
    </Routes>
  )
}
