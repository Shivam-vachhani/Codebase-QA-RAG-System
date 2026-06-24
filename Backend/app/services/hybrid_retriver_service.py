from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document 
import re
import hashlib

_reranker : CrossEncoder | None = None

def get_reranker():

    global _reranker

    if _reranker is None:
        try:
            print(f"[Reranker] Loding model into memory...")
            _reranker =CrossEncoder("BAAI/bge-reranker-v2-m3",trust_remote_code=True)
            print(f"[Reranker] Ready.")
        except Exception as e:
            print(f"[Reranker] Failed to load model: {e}. Reranking will be skipped.")
            _reranker = None
    else:
        print("[Reranker] Using cached model.")
    
    return _reranker

class HybridRetriever():
    def __init__(self,child_chunks:list[Document],child_vectorestore,parent_vectorestore):
        """ Called ONCE when sever starts.
        Builds the BM25 index and loads the code-optimized reranker model.
        """
        self.child_chunks = child_chunks
        self.child_vectorstore = child_vectorestore
        self.parent_vectorstore= parent_vectorestore

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
    
    def retrieve(self,query:str,final_k:int=5)->list[Document]:
        """
        Called on EVERY user query.
        Runs BM25 + Vector in parallel, merges via RRF, reranks, returns best child_chunks.
        """
        
        ref_match = re.search(
            r'where.*?[`"\']?(\w+)[`"\']?\s*(is\s*)?(called|used|imported|referenced)',
        query, re.IGNORECASE
         )
    
        if ref_match:
            symbol = ref_match.group(1)
            print(f"[Retriever] Symbol lookup detected: {symbol}")
            return self._symbol_search(symbol, final_k)
        
        tokenized_query = self._tokenized_code(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top10_index = sorted(range(len(bm25_scores)),
                             key= lambda i:bm25_scores[i],
                             reverse=True)[:30]
        bm25_docs = [self.child_chunks[i] for i  in top10_index ]

        vector_docs = self.child_vectorstore.similarity_search(query,k=30)

        merged_children = self._rrf_merge(bm25_docs,vector_docs,top_n=20)

        if self.re_ranker is None:
            print("[Retriever] Reranker unavailable — returning RRF-merged results.")
            return merged_children[:final_k]

        pairs = [[query,doc.page_content] for doc in merged_children]
        scores = self.re_ranker.predict(pairs)
        ranked = sorted(zip(merged_children,scores),key=lambda x:x[1],reverse=True)

        top_children = [doc for doc, _ in ranked[:final_k] ]

        parent_docs = self._fetch_parents(top_children)
        for doc in parent_docs:
            print(doc.page_content)
        return parent_docs
            

    def _fetch_parents(self,child_docs:list[Document])->list[Document]:
        """
        Given child chunks, look up their full parent chunks from Chroma.
        Deduplicates — multiple children from the same function return one parent.
        """
        seen_parent_ids = set()
        file_counts={}
        parents=[]

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            file_path = child.metadata.get("file_path", "")

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
                   bm25_docs:list[Document],
                   vector_docs:list[Document],
                   k:int = 60,
                   top_n:int = 10)-> list[Document]:
        """ Reciprocal Rank Fusion — merges two ranked lists. """

        scores : dict[str,float] = {}
        doc_map : dict[str,Document] = {}

        for doc_list in [bm25_docs,vector_docs]:
            for rank,doc in enumerate(doc_list):
                if hasattr(doc,"id") and doc.id:
                    key = str(doc.id)
                elif "chunk_id" in doc.metadata:
                    key = str(doc.metadata["chunk_id"])
                else:
                    key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                
                base_score = 1.0 / (rank + k)

                file_path = str(doc.metadata.get("file_path", "")).lower()
                language = str(doc.metadata.get("language", "")).lower()
                content_len = len(doc.page_content)

                if file_path.endswith(".md") or language == "markdown" or "readme" in file_path:
                    base_score *= 0.40
                if content_len > 1500:
                    base_score *= 0.60
                scores[key] = scores.get(key,0.0) + base_score

                if key not in doc_map or len(doc.metadata) > len(doc_map[key].metadata):
                    doc_map[key] = doc

        sorted_keys = sorted(scores,key = lambda x:scores[x],reverse=True)
        return [doc_map[key] for key in sorted_keys[:top_n]]

    def _symbol_search(self,symbol:str,k:int)->list[Document]:
        """Exact string match across all child chunks, then fetch parents."""
        matches = [doc for doc in self.child_chunks if symbol in doc.page_content and doc.metadata.get("node_type") != "fallback"]  

        scored = sorted(matches,key= lambda d:d.page_content.count(symbol),reverse=True)
        top= scored[:k]
        return self._fetch_parents(top)    
