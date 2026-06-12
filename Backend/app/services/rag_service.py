from app.services.vector_service import load_chroma,get_all_docs
from app.services.hybrid_retriver_service import HybridRetriever
from app.services.llm import build_rag_chain
from langchain_core.documents import Document
import time 

_rag_cache:dict[str,"RAGservice"] ={}

def get_rag_service(repo_id:str)-> "RAGservice" :

    if repo_id not in _rag_cache: 
        print(f"[RAGService] Building service for repo: {repo_id}")
        _rag_cache[repo_id] = RAGservice(repo_id)
        print(f"[RAGService] Cached. Total cached repos: {len(_rag_cache)}")
    else:
        print(f"[RAGService] Cache hit for repo: {repo_id}")
    
    return _rag_cache[repo_id]

def invalidate_cache(repo_id:str):
    """Call this after re-ingesting a repo so stale data is evicted."""
    if repo_id in _rag_cache:
        del _rag_cache[repo_id]
        print(f"[RAGService] Cache cleared for repo: {repo_id}")

class RAGservice():

    def __init__(self,repo_id:str):
        vectorestore = load_chroma(repo_id)
        chunks= get_all_docs(vectorestore)
        self.retriever = HybridRetriever(chunks,vectorestore)
        self.chain = build_rag_chain()

    def run(self,question:str)->dict:
        t0 = time.time()
        docs = self.retriever.retrieve(question,final_k=5)
        t1=time.time()
        print(f"[TIMER] Retrieval (BM25 + Vector + Rerank): {t1-t0:.2f}s")

        context = self._fomrat_context(docs)
        print("Chain is started running.....")
        t2 = time.time()
        awnser = self.chain.invoke({
            "context":context,
            "question":question
        })
        t3 = time.time()
        print(f"[TIMER] LLM generation: {t3-t2:.2f}s")
        print(f"[TIMER] Total: {t3-t0:.2f}s")
        return {
            "answer":awnser,
            "sources":[
                {
                    "file":d.metadata['file_path'],
                    "line":d.metadata['start_line'],
                    "language":d.metadata['language']
                }
                for d in docs
            ]
        }

    def _fomrat_context(self,docs:list[Document])->str:
        parts = []
        for doc in docs:
            header = (
            f"File: {doc.metadata['file_path']} "
            f"| Line: {doc.metadata['start_line']} "
            f"| Language: {doc.metadata['language']}"
        )
            parts.append(f"{header}\n```{doc.metadata['language']}\n{doc.page_content}\n```")
        return '\n\n---\n\n'.join(parts)