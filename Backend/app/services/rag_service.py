from app.services.vector_service import load_chroma,get_all_docs
from app.services.hybrid_retriver_service import HybridRetriever
from app.services.llm import build_rag_chain
from langchain_core.documents import Document


class RAGservice():
    def __init__(self,repo_id:str):
        vectorestore = load_chroma(repo_id)
        chunks= get_all_docs(vectorestore)
        self.retriever = HybridRetriever(chunks,vectorestore)
        self.chain = build_rag_chain()

    def run(self,question:str)->dict:
        docs = self.retriever.retrieve(question,final_k=5)
        context = self._fomrat_context(docs)
        print("Chain is started running.....")
        awnser = self.chain.invoke({
            "context":context,
            "question":question
        })

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
    
rag = RAGservice(repo_id="305704ec")
result = rag.run(question="how is the token implimentation done in this project ?")
print(result['answer'])