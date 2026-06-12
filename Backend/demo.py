# main.py — complete codebase Q&A in one file

import os, subprocess, shutil, pathlib,json,re,hashlib
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


##-----configurations-----##
OLLAMA_HOST = os.getenv("OLLAMA_HOST","http://localhost:11434")
CHROMA_PATH = os.getenv("CHROMA_PATH","./data/chroma_db")
CLONE_TMP = pathlib.Path("/temp/repos")

SUPPORTED ={
    # Backend / Standard languages
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".rb": Language.RUBY,
    ".ipynb":Language.PYTHON,
    
    # Standard Frontend languages
    ".js": Language.JS,
    ".ts": Language.TS,
    ".jsx": Language.JS,
    ".tsx": Language.TS,
    
    # Markup & Styles
    ".html": Language.HTML,  
    ".md": Language.MARKDOWN,

    }

TEXT_FALLBACKS = {".json", ".yaml", ".yml", ".txt", ".css", ".toml", ".ini", ".conf", ".env"}


EXACT_TEXT_FILES = {"dockerfile", "jenkinsfile", "procfile", "makefile"}


BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".wav",                                   
    ".exe", ".dll", ".so", ".dylib", ".jar", ".pyc",            
    ".ttf", ".woff", ".woff2",                                  
    ".pdf", ".zip", ".tar", ".gz", ".rar",                     
    ".lock", ".db", ".sqlite" }                                  

IGNORE_DIRS= {"node_modules",".git","__pycache__","dist","build",".venv",".next", "out"}


# ── models ────────────────────────────────────────────────
class IngestRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    question: str
    repo_id:  str

# __ "Hybrid Retriver Class________________________

_reranker : CrossEncoder | None = None
def get_reranker():
    global _reranker

    if _reranker is None:
        print(f"[Reranker] Loding model into memory...")
        _reranker =CrossEncoder("BAAI/bge-reranker-v2-m3",trust_remote_code=True)
        print(f"[Reranker] Ready.")
    else:
        print("[Reranker] Using cached model.")
    
    return _reranker
class HybridRetriever():
    def __init__(self,chunks:list[Document],vectorestore):
        """ Called ONCE when your FastAPI app starts.
        Builds the BM25 index and loads the code-optimized reranker model.
        """
        self.chunks = chunks
        self.vectorstore = vectorestore

        tokenized = [self._tokenized_code(doc.page_content) for doc in chunks]
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
            splits= re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)',token)
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

        merged_docs = self._rrf_merge(bm25_docs,vector_docs,top_n=10)

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

# __ "Rag pipeline class"__________________________

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


# ──  "services" ───────────────────────────
def clone_repo(repo_url:str)->tuple[str,str]:
    
    repo_id = str(uuid4())[:8]
    clone_path = CLONE_TMP / repo_id
    CLONE_TMP.mkdir(parents=True,exist_ok=True)

    try:
        subprocess.run(
            ["git","clone","--depth","1",repo_url,str(clone_path)],
            check=True,text=True,capture_output=True
        )

        return repo_id,str(clone_path)
    
    except subprocess.CalledProcessError as e:
        print(f"Git Clone failed:{e.stderr}")
        return "",""

##----Helper function to extract and format .ipynb content for LLM ingestion----##
def extract_notebook_content(path_object)->str:
    """Parses .ipynb JSON structure and formats it cleanly for an LLM."""
    try:
        notebook_data = json.loads(path_object.read_text(encoding='utf-8',errors='ignore'))
        extracted_lines = []
    
        for cell in notebook_data.get('cells',[]):
            cell_type = cell.get("cell_type")
            source = cell.get('source',[])

            cell_text = "".join(source) if isinstance(source,list) else str(source)

            if cell_type == "code":
                extracted_lines.append(f"\n#----Code cell----\n{cell_text}\n")
            elif cell_type == "markdown" and cell_text.strip():
                extracted_lines.append(f"\n#----Documentation----\n'''\n{cell_text}\n'''\n")
        return "".join(extracted_lines)
    except Exception as e:
        print(f"Failed to cleanly extract notebook {path_object.name}:{e}")
        return ""



##----Main function to traverse cloned repo and extract code files----##
def get_code_files(clone_path:str)->list[dict]:
    files = []
    for p in pathlib.Path(clone_path).rglob("*"):

        if any(bad in p.parts for bad in IGNORE_DIRS):
            continue

        if p.is_file():
            ext = p.suffix.lower()
            file_name =p.name.lower()

            if ext in BINARY_EXTENSIONS:
                continue

            lang_enum =None
            should_read=False

            if ext in SUPPORTED:
                lang_enum = SUPPORTED[ext]
                should_read=True

            elif ext in TEXT_FALLBACKS or file_name in EXACT_TEXT_FILES:
                lang_enum =None 
                should_read=True
            
            if should_read:
                try:
                    if ext == ".ipynb":
                        content_to_save = extract_notebook_content(p)
                    else:
                        content_to_save = p.read_text(encoding="utf-8",errors="ignore")
                    
                    if content_to_save.strip():
                        files.append({
                            "path":str(p.relative_to(clone_path)),
                            "language":lang_enum.value if lang_enum else "plain_text",
                            "content": content_to_save
                      })
                        
                except Exception:
                    pass 
    return files

#----Function to split code files into manageable chunks for LLM processing----##
def chunk_files(files: list[dict])->list[Document]:
    all_chunks = []
    for f in files:
        raw_content = f['content']
        if f['language'] != 'plain_text':
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language(f['language']),
                chunk_size=1000,
                chunk_overlap=200
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n","\n"," ",""]
            )

        chunks = splitter.split_text(f['content'])
        total_chunks = len(chunks)
        current_search_index = 0


        for index,chunk in enumerate(chunks):
            start_char_pos = raw_content.find(chunk,current_search_index)

            if start_char_pos != -1:
                start_line = raw_content[:start_char_pos].count("\n")+1
                current_search_index = start_char_pos + 1

            else:
                start_line  = 1 

            doc = Document(
                page_content=chunk,
                metadata={
                    "file_path":f['path'],
                    "language":str(f['language']),
                    "chunk_index" : index,
                    "total_chunks": total_chunks - 1 ,
                    "chunk_id": f"{f['path']}_chunk_{index}",
                    "start_line": start_line
                }
              )
            all_chunks.append(doc)
    
    return all_chunks


def get_embedding_model()->OllamaEmbeddings:
    """Initializes and returns the local embedding model using Ollama."""
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_HOST
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
            persist_directory=CHROMA_PATH + f"/{repo_id}",
            collection_name="codebase_assistant",
        )

        return True
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

def llm_model():
    return ChatOllama(model="codellama:7b",base_url=OLLAMA_HOST)

def build_rag_chain():
    prompt = ChatPromptTemplate([
        ('system', """You are a senior software engineer and technical mentor helping a developer deeply understand a codebase.

        Your goal is not just to answer — but to make the developer genuinely understand what is happening, why it was built that way, and how the pieces connect.
        
        RULES:
        - Answer using ONLY the context provided below
        - Never hallucinate function names, file paths, or logic not present in the context
        - Always cite file path + line number when referencing specific code
        - Use markdown code blocks with the correct language tag for all code snippets
        - If the answer is not in the context, say exactly: "Not found in the indexed codebase."
        
        OUTPUT FORMAT:
        
        **Direct Answer**
        One or two sentences — what is happening and where.
        
        **Code**
        Show the relevant snippet with file path and line number cited above it.
        
        **What's Actually Happening**
        Explain the code like you are walking a junior developer through it step by step.
        - What does this code do?
        - Why was it written this way?
        - What problem is it solving?
        - How do the pieces connect to each other?
        Write 6–10 lines minimum. Be thorough, not brief.
        
        **How It Fits Into the Bigger Picture**
        One short paragraph explaining how this piece connects to the rest of the codebase — what calls it, what it returns to, what depends on it.
        
        **You Might Also Want To Explore**
        Suggest 2–3 natural follow-up questions the developer could ask, phrased as actual questions. For example:
        - "Where is this token being validated on protected routes?"
        - "How does the refresh token get stored in the cookie?"
        - "What happens if the JWT secret is missing from the environment?"
        
        Context:
        {context}"""),
        ("human", "{question}")
    ])

    return prompt | llm_model() | StrOutputParser()
  
# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(title="Codebase Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten this in real production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ingest")
async def data_ingestion(req:IngestRequest):

    repo_id,path = clone_repo(str(req.repo_url))

    try:
        if not repo_id:
            raise HTTPException(status_code=400,detail="Clone repo failed please provaid valid url")
        else:
            print("Repo clonned....")
        files = get_code_files(path)
        print("Files loded....")
        chunks = chunk_files(files)
        print("Files chunked....") 
        response = ingest_documents_to_chroma(chunks,repo_id)
        print("Repo files stored....")
        print(response)

    finally:
        if path and pathlib.Path(path).exists():
            shutil.rmtree(path,ignore_errors=True)
            print(f"[Ingest] Cleaned up clone at {path}")

    if(response['status'] == "Success"):
        return JSONResponse(
            status_code=200,
            content=response
            )   
    else:
        raise HTTPException(
            status_code=500,
            detail="Some error occure in storing file in vector store"
            )


@app.post("/query")
def ask_query(req:QueryRequest):
    
    pipeline = get_rag_service(req.repo_id) 
    response = pipeline.run(req.question)

    if response:
        return JSONResponse(
            status_code=200,
            content={
            "query":req.question,
            "response":response,
            }
         )
    else:
        raise HTTPException(status_code=400,detail="Can't genrate response")