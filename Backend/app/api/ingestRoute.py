import pathlib,shutil,asyncio
from fastapi import APIRouter,HTTPException
from fastapi.responses import JSONResponse
from app.models.ingest import IngestRequest
from app.services.git_service import clone_repo
from app.services.loader_service import get_code_files
from app.services.chunking_service import chunk_files
from app.services.vector_service import ingest_documents_to_chroma
from app.services.rag_service import invalidate_cache

router = APIRouter()

def _run_ingest(repo_url: str) -> dict:
    """Blocking ingest logic — runs in a thread pool via asyncio.to_thread."""

    repo_id, path = clone_repo(repo_url)
 
    try:
        if not repo_id:
            return {"status": "Failed", "error": "Clone failed — invalid URL or unreachable repo"}, ""
 
        print("Repo cloned....")
        files = get_code_files(path)
        print("Files loaded....")
        chunks = chunk_files(files)
        print("Files chunked....")
        result = ingest_documents_to_chroma(chunks, repo_id)
        print("Repo files stored....")
        return result,repo_id
 
    finally:
        if path and pathlib.Path(path).exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"[Ingest] Cleaned up clone at {path}")

@router.post("/ingest")
async def data_ingestion(req:IngestRequest):

    response,repo_id = await asyncio.to_thread(_run_ingest,str(req.repo_url))

    if(response['status'] == "Success"):
        invalidate_cache(repo_id)
        return JSONResponse(
            status_code=200,
            content=response
            )   
    else:
        raise HTTPException(
            status_code=500,
            detail="Some error occure in storing file in vector store"
            )