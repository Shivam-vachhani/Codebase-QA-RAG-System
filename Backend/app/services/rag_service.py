from app.services.vector_service import load_chroma,get_all_docs, get_file_summaries_by_path
from app.services.hybrid_retriver_service import HybridRetriever ,detect_symbol_serach
from app.services.llm import build_rag_chain
from app.services.query_analysis_service import analyze_query
from langchain_core.documents import Document
from cachetools import LRUCache
import time 

_rag_cache:LRUCache = LRUCache(maxsize=50) 

def get_rag_service(repo_id:str,model:str)-> "RAGservice" :
    cache_key = (repo_id,model)

    if cache_key not in _rag_cache: 
        print(f"[RAGService] Building service for repo: {repo_id}, model: {model}")
        _rag_cache[cache_key] = RAGservice(repo_id,model)
        print(f"[RAGService] Cached. Total cached repos: {len(_rag_cache)}")
    else:
         print(f"[RAGService] Cache hit for repo: {repo_id}, model: {model}")

    return _rag_cache[cache_key]

def invalidate_cache(repo_id:str):
    """Call this after re-ingesting a repo so stale data is evicted."""
    keys_to_delete = [k for k in _rag_cache.keys() if k[0]==repo_id]

    for k in keys_to_delete:
        del _rag_cache[k]
    if keys_to_delete:
        print(f"[RAGService] Cache cleared for repo: {repo_id} ({len(keys_to_delete)} entries)")

class RAGservice():

    def __init__(self,repo_id:str,model:str):
        self.repo_id = repo_id
        self.model = model

        child_vectorestore = load_chroma(repo_id,collection="child_chunks")
        parent_vectorestore = load_chroma(repo_id,collection="parent_chunks")
        summary_vectorestore = load_chroma(repo_id,collection="summaries")

        child_chunks= get_all_docs(child_vectorestore)
        self.retriever = HybridRetriever(child_chunks,child_vectorestore,parent_vectorestore,summary_vectorestore,max_workers=4)
        self.chains = {
            "CONCEPTUAL":build_rag_chain(model,"CONCEPTUAL"),
            "CODE_SPECIFIC":build_rag_chain(model,"CODE_SPECIFIC"),
        }


    def run(self,question:str,include_context:bool = False)->dict:
        t0 = time.time()

        symbol = detect_symbol_serach(question)
        if symbol:
           print(f"[RAGService] Symbol query detected, skipping query analysis")
           docs = self.retriever._symbol_search(symbol, k =5)
           classification="SYMBOL_LOOKUP"
        else:
            analysis = analyze_query(self.repo_id,question)
            t_analysis = time.time()
            print(f"[TIMER] Query analysis: {t_analysis - t0:.2f}s")

            docs = self.retriever.retrieve(
                original_query=question,
                expanded_queries=analysis.expanded_queries,
                classification=analysis.classification,
                confidence=analysis.confidence,
                final_k=5,
                min_code_slots=1
            )
            classification = analysis.classification
            
        if not docs:
            return {
             "answer": "Not found in the indexed codebase.",
             "sources": [],
             "summaries": {}
            } 
           
        t1=time.time()
        print(f"[TIMER] Retrieval (BM25 + Vector + Rerank) total : {t1-t0:.2f}s (classification={classification})")
    
        file_paths = list({d.metadata.get("file_path") for d in docs if {d.metadata.get("file_path")}})
        file_summaries = get_file_summaries_by_path(self.repo_id,file_paths)
        t2 = time.time()
        print(f"[TIMER] File summary enrichment: {t2 - t1:.2f}s")
        
        context = self._fomrat_context(docs,file_summaries)
        chain_key = "CONCEPTUAL" if classification == "CONCEPTUAL" else "CODE_SPECIFIC"
        chain = self.chains[chain_key]

        print("Chain is started running.....")
        t3 = time.time()
        awnser = chain.invoke({
            "context":context,
            "question":question
        })

        t4 = time.time()
        print(f"[TIMER] LLM generation: {t4-t3:.2f}s")
        print(f"[TIMER] Total: {t4-t0:.2f}s")

        response_dict = {
            "answer":awnser,
            "sources":[
                {
                    "file":d.metadata['file_path'],
                    "chunk_type":d.metadata["chunk_type"],
                    "language":d.metadata.get("language","plain text"),
                }
                for d in docs
            ],
            "summaries": { path:summary for path, summary in file_summaries.items() } ,
            "classification": classification,
        }

        if include_context:
            response_dict["context"] = [d.page_content for d in docs]
            
        return response_dict


    def _fomrat_context(self,docs:list[Document],file_summaries:dict[str,str])->str:
        """'High-level context' block from file summaries,
             prepended before the code chunks."""
        parts = []

        included_summaries = {
            path:summary for path,summary in file_summaries.items() 
            if path in {d.metadata.get("file_path") for d in docs} 
        }

        if included_summaries:
            summary_block = "\n\n".join(
                f"[File summary - {path}]\n{summary}"
                for path,summary in included_summaries.items()
            )
            parts.append(f"High-level context:\n{summary_block}")
        
        for doc in docs:
            if doc.metadata.get("chunk_type") in ["file_summary", "folder_summary", "repo_summary"]:
                header = (
                    f"File: {doc.metadata['file_path']} "
                    f"| Chunk type: {doc.metadata.get('chunk_type', 'summary')}"
                )
            else:
                header = (
                f"File: {doc.metadata['file_path']} "
                f"| Line: {doc.metadata['start_line']} "
                f"| Language: {doc.metadata['language']} "
                f"| Chunk type: {doc.metadata.get('chunk_type', 'code')}"
                )

            parts.append(f"{header}\n```{(doc.metadata.get('language') or '')}\n{doc.page_content}\n```")

        
        return '\n\n---\n\n'.join(parts)