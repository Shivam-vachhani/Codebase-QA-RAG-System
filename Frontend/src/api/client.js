const API_BASE = import.meta.env.VITE_API_URL ?? '/api'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })

  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    const message = data.detail ?? data.error ?? `Request failed (${res.status})`
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message))
  }

  return data
}

export async function ingestRepo(repoUrl) {
  return request('/ingest', {
    method: 'POST',
    body: JSON.stringify({ repo_url: repoUrl }),
  })
}

export async function queryRepo({ question, repoId, model }) {
  return request('/query', {
    method: 'POST',
    body: JSON.stringify({ question, repo_id: repoId, model }),
  })
}

export async function healthCheck() {
  return request('/')
}
