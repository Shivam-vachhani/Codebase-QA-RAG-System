# Frontend — Codebase Q&A Assistant

React + Tailwind UI for the LangChain RAG backend.

## Setup

```bash
cd Frontend
npm install
npm run dev
```

Runs at **http://localhost:5173** and proxies API calls to `http://localhost:8000`.

## Backend

Start the FastAPI server first:

```bash
cd Backend
uvicorn app.main:server --reload --port 8000
```

## Production build

```bash
npm run build
npm run preview
```

Set `VITE_API_URL` to your deployed API base URL when building for production.

## Pages

| Route | Description |
|-------|-------------|
| `/repos` | Repository cards (demo + indexed) |
| `/ingest` | Multi-step ingest wizard → `POST /ingest` |
| `/chat/:repoId` | Chat interface → `POST /query` |

Indexed repos are stored in `localStorage`.
