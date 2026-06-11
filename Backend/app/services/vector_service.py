from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from Backend.app.services.loader_service import get_code_files
from app.services.git_service import clone_repo
from Backend.app.services.chunking_service import chunk_files
from app.services.hybrid_retriver_service import HybridRetriever
from Backend.app.utils import config

def get_embedding_model()->OllamaEmbeddings:
    """Initializes and returns the local embedding model using Ollama."""
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=config.OLLAMA_HOST
    )


def ingest_documents_to_chroma(documents: list[Document],repo_id:str)->bool:
    """ Saves text chunks into Chroma DB.
    Injects repo_id into metadata to isolate codebase searches later."""
    try:
        embeddings = get_embedding_model()
        for doc in documents:
            doc.metadata['repo_id'] = repo_id     

        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=config.CHROMA_PATH + f"/{repo_id}",
            collection_name="codebase_assistant",
        )

        return {"Success":True,"repoId":repo_id}
    except Exception as e:
        print(f"Error ingesting documents to Chroma: {e}")
        return False

def load_chroma(repo_id:str):
    return Chroma(
        persist_directory=f"{config.CHROMA_PATH}/{repo_id}",
        embedding_function=get_embedding_model(),
        collection_name="codebase_assistant"
    )

def get_all_docs(vectorstore) ->list[Document]:
    chunks = []
    raw_data = vectorstore.get(include=["documents","metadatas"])
    for text,metadata in (zip(raw_data["documents"],raw_data["metadatas"])):
        chunks.append(Document(page_content=text,metadata=metadata))
    return chunks


# if __name__ == "__main__":         
#      repo_id, path = clone_repo("https://github.com/Shivam-vachhani/Amigo-Social-Media-Website.git")
#      files = get_code_files(path)
#      chunks = chunk_files(files)
#      success = ingest_documents_to_chroma(chunks,repo_id)
#      vector_store = load_chroma(repo_id)
#      if vector_store:
#         print("Vectorstore exist")
#         chunks = get_all_docs(vector_store)
#         print(f"Successfully loaded {len(chunks)} chunks from Chroma.")
#         retreiver = HybridRetriever(chunks, vector_store)
#         query = "where is chatting functions in this project ?"
#         relavent_docs = retreiver.retrieve(query=query,final_k=5)
#         print("\n--- Top Relevant Documents ---")
#         for doc in relavent_docs:
#             print(f"Source: {doc.metadata}")
#             print(doc.page_content)
#             print("-" * 40)
#      else:
#          print("got some problem in saving docs to chroma")
     

