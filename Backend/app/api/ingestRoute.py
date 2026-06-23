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

        print("\n" + "="*50 + " INSPECTING PARENT CHUNKS " + "="*50)
        # Filter out only the parent chunks from the complete list
        parent_docs = [doc for doc in chunks if doc.metadata.get("chunk_type") == "parent"]
        for idx, doc in enumerate(parent_docs[100:130]):  # Limits to first 10 so it doesn't flood your console
            print(f"\n[Parent Chunk #{idx+1}]")
            print(f"File: {doc.metadata.get('file_path')}")
            print(f"Node Type: {doc.metadata.get('node_type')}")
            print(f"Lines: {doc.metadata.get('start_line')} to {doc.metadata.get('end_line')}")
            print(f"ID: {doc.metadata.get('chunk_id')}")
            print("-" * 40)
            # Show the first 3 lines of the actual code chunk
            code_lines = doc.page_content.splitlines()
            preview = "\n".join(code_lines[:3])
            print(preview)
            if len(code_lines) > 3:
                print("... (truncated) ...")
        
        print(f"\nTotal Parents Extracted: {len(parent_docs)}")
        print("="*126 + "\n")

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