from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document 
from concurrent.futures import ThreadPoolExecutor
import re
import hashlib

_reranker : CrossEncoder | None = None

def get_reranker():

    global _reranker

    if _reranker is None:
        try:
            print(f"[Reranker] Loding model into memory...")
            _reranker =CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",trust_remote_code=True)
            print(f"[Reranker] Ready.")
        except Exception as e:
            print(f"[Reranker] Failed to load model: {e}. Reranking will be skipped.")
            _reranker = None
    else:
        print("[Reranker] Using cached model.")
    
    return _reranker

SYMBOL_PATTERN = re.compile(r'where.*?[`"\']?(\w+)[`"\']?\s*(is\s*)?(called|used|imported|referenced)', re.IGNORECASE)

def detect_symbol_serach(question:str)->str | None:
    """Detects if the question is a symbol lookup (e.g., "where is foo called?") and extracts the symbol name."""
    match = SYMBOL_PATTERN.search(question)
    if match:
        return match.group(1)
    return None

class HybridRetriever():
    def __init__(self,child_chunks:list[Document],child_vectorestore,parent_vectorestore,summary_vectorestore,max_workers:int = 4):
        """ Called ONCE when sever starts.
        Builds the BM25 index and loads the code-optimized reranker model.
        """
        self.child_chunks = child_chunks
        self.child_vectorstore = child_vectorestore
        self.parent_vectorstore= parent_vectorestore
        self.summary_vectorstore = summary_vectorestore
        self.max_workers = max_workers

        tokenized = [self._tokenized_code(doc.page_content) for doc in child_chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.re_ranker = get_reranker()


    def _tokenized_code(self,text:str)->list[str]:
        """Advanced multi-language code tokenizer for BM25.
        Splits snake_case, camelCase, PascalCase, and kebab-case 
        into searchable sub-tokens while preserving original names"""

        base_tokens = re.findall(r'[a-zA-z0-9_]+',text.lower())

        final_tokens =[]

        for token in base_tokens:
            final_tokens.append(token)
            if '_' in token:
                sub_word = [w for w in token.split('_') if w]
                final_tokens.extend(sub_word)

        camel_tokens = re.findall(r'[a-zA-Z]+',text)
        
        for token in camel_tokens:
            splits= re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\b)',token)
            if len(splits) > 1:
                final_tokens.extend([w.lower() for w in splits])
        
        return final_tokens
    

    def _search_single_query(self,query:str)->tuple[list[Document],list[Document]]:
        """Helper for parallel search — returns BM25 and vector results for a single query."""
        tokenized_query = self._tokenized_code(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_index = sorted(range(len(bm25_scores)),key= lambda i : bm25_scores[i], reverse=True)[:30]
        bm25_docs = [self.child_chunks[i] for i in top_index]

        vector_docs = self.child_vectorstore.similarity_search(query,k=30)
        summary_docs = self.summary_vectorstore.similarity_search(query,k=6)
        return bm25_docs,vector_docs,summary_docs


    def retrieve(
            self,
            original_query:str, 
            expanded_queries:list[str], 
            classification:str = "CODE_SPECIFIC",
            confidence:float = 1.0, 
            final_k:int=5,
            min_code_slots:int=1
    )->list[Document]:
        """
        Called on EVERY user query.
        Runs BM25 + Vector + Summary for ALL query
        variants IN PARALLEL via ThreadPoolExecutor, RRF-merges everything
        (weighted by classification), reranks against the ORIGINAL query,
        returns parent chunks which guarantees at least this many non-summary (real code) docs
        survive into the final result.
        """

        symbol = detect_symbol_serach(original_query)
        if symbol:
            print(f"[Retriever] Symbol lookup detected: {symbol}")
            return self._symbol_search(symbol, final_k)
        
        all_queries = list(dict.fromkeys([original_query]+expanded_queries))

        bm25_lists = []
        vector_lists = []
        summary_lists = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers,len(all_queries))) as executor:
            future_to_query = {
                executor.submit(self._search_single_query,q): q
                for q in all_queries 
            }
            for future in future_to_query:
                try:
                    bm25_docs,vector_docs,summary_docs = future.result()
                    bm25_lists.append(bm25_docs)
                    vector_lists.append(vector_docs)
                    summary_lists.append(summary_docs)

                except Exception as e:
                    q = future_to_query[future]
                    print(f"[Retriever] Search failed for query {q!r}: {e}")
                    
        merged_children = self._rrf_merge(bm25_lists,vector_lists,summary_lists,classification,confidence,top_n=20)

        if self.re_ranker is None:
            print("[Retriever] Reranker unavailable — returning RRF-merged results.")
            floored = self._enforce_code_floor(merged_children,final_k,min_code_slots)
            return self._fetch_parents(floored)
        
        pairs = [[original_query,doc.page_content] for doc in merged_children]
        scores = self.re_ranker.predict(pairs)
        ranked = [doc for doc,_ in sorted(zip(merged_children,scores),key=lambda x:x[1],reverse=True)]

        top_children = self._enforce_code_floor(ranked,final_k,min_code_slots)
        parent_docs = self._fetch_parents(top_children)
        # for doc in parent_docs:
        #     print(doc.page_content)
        return parent_docs
            

    def _fetch_parents(self,child_docs:list[Document])->list[Document]:
        """
        Given child chunks, look up their full parent chunks from Chroma.
        Deduplicates — multiple children from the same function return one parent.
        """
        seen_parent_ids = set()
        seen_summary_paths = set()
        file_counts={}
        parents=[]

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            file_path = child.metadata.get("file_path", "")
            chunk_type = child.metadata.get("chunk_type", "")
            
            if chunk_type == "file_summary" or chunk_type == "folder_summary" or chunk_type == "repo_summary":
                if file_path in seen_summary_paths:
                    continue
                seen_summary_paths.add(file_path)
                parents.append(child)
                continue
        
            if not parent_id or parent_id in seen_parent_ids:
                continue
            
            if file_counts.get(file_path,0)>=2:
                continue

            seen_parent_ids.add(parent_id)
            file_counts[file_path] = file_counts.get(file_path, 0) + 1

            result = self.parent_vectorstore.get(
                where={"chunk_id":parent_id},
                include=["documents","metadatas"]
            )

            if result["documents"]:
                parents.append(Document(
                    page_content=result["documents"][0],
                    metadata = result["metadatas"][0]
                ))
            else:
                print(f"[Retriever] Parent {parent_id} not found — using child")
                parents.append(child)

        return parents 
    

    def _rrf_merge(self,
                   bm25_lists:list[list[Document]],
                   vector_lists:list[list[Document]],
                   summary_lists:list[list[Document]],
                   classification:str,
                   confidence:float =1.0,
                   k:int = 60,
                   top_n:int = 10)-> list[Document]:
        """Reciprocal Rank Fusion — merges three ranked nested lists.
        Classification sets a target weight profile; confidence controls how far
        we commit to it. Low confidence blends back toward neutral (1.0/1.0/1.0)
        instead of fully trusting a classification the model itself wasn't sure about."""
        
        NEUTRAL = {"bm25":1.0, "vector":1.0, "summary":1.0}
        if classification == "CODE_SPECIFIC":
            target = {"bm25": 1.5, "vector": 1.1, "summary": 0.4}
        else:
            target = {"bm25": 0.9, "vector": 0.7, "summary": 1.5} 

        c= max(0.0,min(1.0,confidence))
        bm25_weight    = NEUTRAL["bm25"]    + c * (target["bm25"]    - NEUTRAL["bm25"])
        vector_weight  = NEUTRAL["vector"]  + c * (target["vector"]  - NEUTRAL["vector"])
        summary_weight = NEUTRAL["summary"] + c * (target["summary"] - NEUTRAL["summary"])
        
        SUMMARY_TYPES = {"file_summary", "folder_summary", "repo_summary"}

        scores : dict[str,float] = {}
        doc_map : dict[str,Document] = {}
       
        def _procces(doc_lists,sorce_weight):
            for doc_list in doc_lists:
                for rank,doc in enumerate(doc_list):
                    if hasattr(doc,"id") and doc.id:
                        key = str(doc.id)
                    elif "chunk_id" in doc.metadata:
                        key = str(doc.metadata["chunk_id"])
                    else:
                        key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

                    base_score = (1.0 / (rank + k)) * sorce_weight

                    file_path = str(doc.metadata.get("file_path", "")).lower()
                    language = str(doc.metadata.get("language", "")).lower()
                    content_len = len(doc.page_content)

                    is_markdown = file_path.endswith(".md") or language == "markdown" or "readme" in file_path
                    if is_markdown and doc.metadata.get("chunk_type") not in SUMMARY_TYPES:
                        base_score *= 0.40
                    if content_len > 1500:
                        base_score *= 0.60

                    scores[key] = scores.get(key,0.0) + base_score

                    if key not in doc_map or len(doc.metadata) > len(doc_map[key].metadata):
                        doc_map[key] = doc

        _procces(bm25_lists,bm25_weight)
        _procces(vector_lists,vector_weight)
        _procces(summary_lists,summary_weight)

        sorted_keys = sorted(scores,key = lambda x:scores[x],reverse=True)
        return [doc_map[key] for key in sorted_keys[:top_n]]

    def _symbol_search(self,symbol:str,k:int)->list[Document]:
        """Exact string match across all child chunks, then fetch parents."""
        matches = [doc for doc in self.child_chunks if symbol in doc.page_content and doc.metadata.get("node_type") != "fallback"]  

        scored = sorted(matches,key= lambda d:d.page_content.count(symbol),reverse=True)
        top= scored[:k]
        return self._fetch_parents(top)

    def _enforce_code_floor(self, orderd_docs:list[Document],final_k:int,min_code_slots:int)->list[Document]:
        """ordered_docs is already sorted best -> worst (post-rerank or post-RRF).
        If the top final_k has fewer than min_code_slots real code docs, swap in
        the best-scoring code docs from beyond final_k, evicting the weakest
        summary docs currently occupying a slot."""
        SUMMARY_TYPES = {"file_summary", "folder_summary", "repo_summary"}
        
        top = orderd_docs[:final_k]
        code_in_top = [d for d in top if d.metadata.get("chunk_type") not in SUMMARY_TYPES]
        if len(code_in_top) >= min_code_slots:
            return top
        
        remaining_code = [d for d in orderd_docs[final_k:] if d.metadata.get("chunk_type") not in SUMMARY_TYPES]
        if not remaining_code:
            return top
        
        result = list(top)
        needed = min_code_slots - len(code_in_top)

        for _ in range(needed):
            if not remaining_code:
                break
            evict_indx=None
            for i in range(len(result)- 1, -1 ,-1):
                if result[i].metadata.get("chunk_type") in SUMMARY_TYPES:
                    evict_indx = i
                    break
            if evict_indx is None:
                break
            result.pop(evict_indx)
            result.append(remaining_code.pop(0))
        
        return result