# main.py — complete codebase Q&A in one file

import os, subprocess, shutil, pathlib,json,re,hashlib,asyncio
from uuid import uuid4
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from cachetools import LRUCache
from tree_sitter_language_pack import get_language, get_parser  
import threading

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

TREESITTER_LANG_MAP = {
    "python":     "python",
    "js":         "javascript",
    "ts":         "typescript",
    "jsx":        "javascript",
    "tsx":        "typescript",
    "java":       "java",
    "go":         "go",
    "rs":         "rust",
    "rb":         "ruby",
    "php":        "php",
}

MEANINGFUL_NODE_TYPES = {
    "typescript": {
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",    
        "enum_declaration",          
        "lexical_declaration",     
    },
    "javascript": {
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "lexical_declaration",
    },
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "java":       {"class_declaration", "method_declaration", "constructor_declaration"},
    "go":         {"function_declaration", "method_declaration", "type_declaration"},
    "rust":       {"function_item", "impl_item", "struct_item", "enum_item"},
    "ruby":       {"method", "class", "module"},
    "php":        {"function_definition", "class_declaration", "method_declaration"},
}

# ── models ────────────────────────────────────────────────
class IngestRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    question:str
    repo_id:str
    model:Literal["gpt-4o","qwen-2.5"]

# __ "Hybrid Retriver Class________________________

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

        tokenized_query = self._tokenized_code(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top10_index = sorted(range(len(bm25_scores)),
                             key= lambda i:bm25_scores[i],
                             reverse=True)[:30]
        bm25_docs = [self.child_chunks[i] for i  in top10_index ]

        vector_docs = self.child_vectorstore.similarity_search(query,k=30)

        merged_children = self._rrf_merge(bm25_docs,vector_docs,top_n=10)

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
        parents=[]

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")

            if not parent_id or parent_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_id)

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

                if file_path.endswith(".md") or language == "markdown" or "readme" in file_path:
                    base_score *= 0.40

                scores[key] = scores.get(key,0.0) + base_score

                if key not in doc_map or len(doc.metadata) > len(doc_map[key].metadata):
                    doc_map[key] = doc

        sorted_keys = sorted(scores,key = lambda x:scores[x],reverse=True)
        return [doc_map[key] for key in sorted_keys[:top_n]]       

# __ "Rag pipeline class"__________________________

_rag_cache:LRUCache = LRUCache(maxsize=50) 

def get_rag_service(repo_id:str,model:str)-> "RAGservice" :
    cache_key = (repo_id,model)

    if repo_id not in _rag_cache: 
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
        child_vectorestore = load_chroma(repo_id,collection="child_chunks")
        parent_vectorestore = load_chroma(repo_id,collection="parent_chunks")
        
        child_chunks= get_all_docs(child_vectorestore)
        self.retriever = HybridRetriever(child_chunks,child_vectorestore,parent_vectorestore)
        self.chain = build_rag_chain(model)

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

##----Function to split code files into manageable chunks for LLM processing----##
_thread_local = threading.local()

def _character_fallback(file: dict, chunk_size: int, lable: str) -> list[Document]:
    """Plain character splitter used when tree-sitter can't parse the file."""
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = spliter.split_text(file["content"])
    docs = []

    for i, chunk in enumerate(chunks):
        start_line = file["content"][:file["content"].find(chunk)].count("\n") + 1
        docs.append(Document(
            page_content=chunk,
            metadata={
                "file_path": file["path"],
                "language": file["language"],
                "start_line": start_line,
                "chunk_type": lable,
                "node_type": "fallback",
                "chunk_id": f"{file['path']}_{lable}_{i}",
            })
        )

    return docs


def get_ts_parser(ts_lang_name: str):
    """
    Returns a parser for the current thread.
    Creates it fresh if this thread doesn't have one yet.
    Thread-local means no object ever crosses thread boundaries.
    """
    if not hasattr(_thread_local, "parsers"):
        _thread_local.parsers = {}

    if ts_lang_name not in _thread_local.parsers:
        try:
            language = get_language(ts_lang_name)
            parser = get_parser(ts_lang_name)
            _thread_local.parsers[ts_lang_name] = (language, parser)
        except Exception as e:
            print(f"[TreeSitter] Failed to load parser for {ts_lang_name}: {e}")
            return None, None
    return _thread_local.parsers[ts_lang_name]


def extract_parent_chunks(file: dict) -> list[Document]:
    """
    Uses Tree-Sitter to split code at real syntactic boundaries.
    Returns one Document per function/class — never cuts mid-function.
    Falls back to character splitter for unsupported languages.
    """
    language_str = file['language']
    content = file['content']
    path = file["path"]

    if isinstance(content, bytes):
        content = content.decode("utf-8")

    if language_str not in TREESITTER_LANG_MAP:
        return _character_fallback(file, chunk_size=2000, lable="parent")

    ts_lang_name = TREESITTER_LANG_MAP[language_str]

    try:
        language, parser = get_ts_parser(ts_lang_name)

        if parser is None:
            return _character_fallback(file, chunk_size=2000, lable="parent")

    except Exception as e:
        print(f"[TreeSitter] Parser unavailable for {ts_lang_name}: {e} — falling back")
        return _character_fallback(file, chunk_size=2000, lable="parent")

    tree = None
    cursor = None
    try:
        # parser.parse() accepts str in this library's Rust bindings
        tree = parser.parse(content)
        target_node_types = MEANINGFUL_NODE_TYPES.get(ts_lang_name, set())
        source_line = content.splitlines()
        parent_chunks = []
        visited_ranges = []

        # Tree.walk() and Tree.root_node() are both methods (not properties)
        cursor = tree.walk()

        reached_end = False

        while not reached_end:
            # TreeCursor.node() — method call, returns a Node object
            current_node = cursor.node()

            # Node.kind() — method call, returns the node type string (NOT .type)
            node_kind = current_node.kind()

            if node_kind in target_node_types:
                # Node.start_position() / end_position() — method calls, return Point
                # Point has .row and .column attributes
                start_line = current_node.start_position().row
                end_line = current_node.end_position().row + 1

                already_covered = any(
                    s <= start_line and end_line <= e for s, e in visited_ranges
                )

                if not already_covered:
                    visited_ranges.append((start_line, end_line))
                    chunk_text = "\n".join(source_line[start_line:end_line])

                    if chunk_text.strip():
                        parent_chunks.append(Document(
                            page_content=chunk_text,
                            metadata={
                                "file_path": path,
                                "language": language_str,
                                "start_line": start_line + 1,
                                "end_line": end_line,
                                "chunk_type": "parent",
                                "node_type": node_kind,
                                "chunk_id": f"{path}_parent_{start_line}"
                            }
                        ))
                    if cursor.goto_next_sibling():
                        continue

            if cursor.goto_first_child():
                continue
            if cursor.goto_next_sibling():
                continue

            while True:
                if not cursor.goto_parent():
                    reached_end = True
                    break
                if cursor.goto_next_sibling():
                    break

        covered_lines = set()
        for s, e in visited_ranges:
            covered_lines.update(range(s, e))

        uncovered = [(i, line) for i, line in enumerate(source_line) if i not in covered_lines]

        if uncovered:
            module_text = "\n".join(line for _, line in uncovered)
            if module_text.strip():
                parent_chunks.append(Document(
                    page_content=module_text,
                    metadata={
                        "file_path": path,
                        "language": language_str,
                        "start_line": 1,
                        "end_line": len(source_line),
                        "chunk_type": "parent",
                        "node_type": "module_level",
                        "chunk_id": f"{path}_parent_module"
                    }
                ))

        if not parent_chunks:
            print(f"[TreeSitter] No meaningful nodes found in {path} — falling back")
            return _character_fallback(file, chunk_size=2000, lable="parent")

        return parent_chunks
    finally:
        if cursor is not None:
            del cursor
        if tree is not None:
            del tree


def create_child_chunks(parent_docs: list[Document]) -> list[Document]:
    """
    Splits each parent chunk into small child chunks for embedding search.
    Children store parent_id so we can look up the full parent at query time.
    """
    child_chunks = []

    for parent in parent_docs:
        language = parent.metadata.get("language", "plain_text")
        parent_id = parent.metadata["chunk_id"]

        if language in TREESITTER_LANG_MAP:
            try:
                lang_enum = Language(language)
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang_enum,
                    chunk_size=300,
                    chunk_overlap=30,
                )
            except Exception:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=30,
                )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=30,
            )

        pieces = splitter.split_text(parent.page_content)

        for i, piece in enumerate(pieces):
            child_chunks.append(Document(
                page_content=piece,
                metadata={
                    "file_path": parent.metadata["file_path"],
                    "language": parent.metadata["language"],
                    "start_line": parent.metadata["start_line"],
                    "chunk_type": "child",
                    "chunk_id": f"{parent_id}_child_{i}",
                    "parent_id": parent_id
                }
            ))

    return child_chunks


def chunk_files(files: list[dict]) -> list[Document]:
    all_docs = []
    for f in files:
        parents = extract_parent_chunks(f)
        children = create_child_chunks(parents)
        all_docs.extend(parents)
        all_docs.extend(children)
    return all_docs

##---- Embedding and Chroma DB functions ----##

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

        parent_docs = [d for d in documents if d.metadata.get("chunk_type") == "parent"]
        child_docs = [d for d in documents if d.metadata.get("chunk_type") == "child"]

        for doc in documents:
            doc.metadata['repo_id'] = repo_id     
        
        print(f"[Chroma] Storing {len(parent_docs)} parents, {len(child_docs)} children...")

        Chroma.from_documents(
            documents=child_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PATH + f"/{repo_id}",
            collection_name="child_chunks",
        )

        Chroma.from_documents(
            documents=parent_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PATH + f"/{repo_id}",
            collection_name="parent_chunks",
        )

        return {"status":"Success","repoId":repo_id,
                "parents":len(parent_docs),"childrens":len(child_docs)}
    
    except Exception as e:
        print(f"Error ingesting documents to Chroma: {e}")
        return {"status": "Failed", "error": str(e)}
    
    
def load_chroma(repo_id:str,collection:str = "child_chunks"):
    return Chroma(
        persist_directory=f"{config.CHROMA_PATH}/{repo_id}",
        embedding_function=get_embedding_model(),
        collection_name=collection
    )


def get_all_docs(vectorstore) ->list[Document]:
    chunks = []
    raw_data = vectorstore.get(include=["documents","metadatas"])
    for text,metadata in (zip(raw_data["documents"],raw_data["metadatas"])):
        chunks.append(Document(page_content=text,metadata=metadata))
    return chunks

##---- LLM and RAG chain functions ----##

def llm_model(model:str):
    if model == "gpt-4o":
            return ChatOpenAI(model="gpt-4o-mini")
    elif model == "qwen-2.5":
        return ChatOllama(model="codellama:7b",base_url=OLLAMA_HOST)


def build_rag_chain(model:str):
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

    return prompt | llm_model(model) | StrOutputParser()
  
# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(title="Codebase Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten this in real production
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        print("Files chunked....")
        result = ingest_documents_to_chroma(chunks, repo_id)
        print("Repo files stored....")
        return result,repo_id
 
    finally:
        if path and pathlib.Path(path).exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"[Ingest] Cleaned up clone at {path}")


@app.post("/ingest")
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


@app.post("/query")
def ask_query(req:QueryRequest):
    
    pipeline = get_rag_service(req.repo_id,req.model) 
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