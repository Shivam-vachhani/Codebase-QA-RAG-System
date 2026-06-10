from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document 
import re
import hashlib

class HybridRetriever:
    def __init__(self,chunks:list[Document],vectorestore):
        """ Called ONCE when your FastAPI app starts.
        Builds the BM25 index and loads the code-optimized reranker model.
        """
        self.chunks = chunks
        self.vectorstore = vectorestore

        tokenized = [self._tokenized_code(doc.page_content) for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)


        self.re_ranker =CrossEncoder("BAAI/bge-reranker-v2-m3",trust_remote_code=True)

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
        Runs BM25 + Vector in parallel, merges via RRF, reranks, returns best chunks.
        """

        tokenized_query = self._tokenized_code(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top10_index = sorted(range(len(bm25_scores)),
                             key= lambda i:bm25_scores[i],
                             reverse=True)[:30]
        bm25_docs = [self.chunks[i] for i  in top10_index ]

        vector_docs = self.vectorstore.similarity_search(query,k=30)

        merged_docs = self._rrf_merge(bm25_docs,vector_docs,top_n=25)

        pairs = [[query,doc.page_content] for doc in merged_docs]
        scores = self.re_ranker.predict(pairs)

        ranked = sorted(zip(merged_docs,scores),key=lambda x:x[1],reverse=True)

        return [doc for doc,_ in ranked[:final_k]]


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

                if file_path.endswith(".md") or language == "markdown" or "readme" in file_path:
                    base_score *= 0.40

                scores[key] = scores.get(key,0.0) + base_score

                if key not in doc_map or len(doc.metadata) > len(doc_map[key].metadata):
                    doc_map[key] = doc

        sorted_keys = sorted(scores,key = lambda x:scores[x],reverse=True)
        return [doc_map[key] for key in sorted_keys[:top_n]]       
