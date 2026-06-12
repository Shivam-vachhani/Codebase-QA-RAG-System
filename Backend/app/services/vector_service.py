from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings 
from app.utils import config
from dotenv import load_dotenv

load_dotenv()

def get_embedding_model()->OllamaEmbeddings:
    """Initializes and returns the local embedding model using Ollama."""
    # return OllamaEmbeddings(
    #     model="nomic-embed-text",
    #     base_url=config.OLLAMA_HOST
    # )
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def ingest_documents_to_chroma(documents: list[Document],repo_id:str)->bool:
    """ Saves text chunks into Chroma DB.
    Injects repo_id into metadata to isolate codebase searches later."""
    try:
        embeddings = get_embedding_model()
        for doc in documents:
            doc.metadata['repo_id'] = repo_id     
        
        print(f"[Chroma] Starting embedding of {len(documents)} chunks...")

        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=config.CHROMA_PATH + f"/{repo_id}",
            collection_name="codebase_assistant",
        )

        return {"status":"Success","repoId":repo_id}
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

     

