import os,pathlib
from  langchain_text_splitters import Language

##-----configurations-----##
OLLAMA_HOST = os.getenv("OLLAMA_HOST","http://localhost:11434")
CHROMA_PATH = os.getenv("CHROMA_PATH","./data/chroma_db")
CLONE_TMP = pathlib.Path("/tmp/repos")

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

__all__ = [
    "OLLAMA_HOST",
    "CHROMA_PATH",
    "CLONE_TMP",
    "SUPPORTED",
    "TEXT_FALLBACKS",
    "EXACT_TEXT_FILES",
    "BINARY_EXTENSIONS",
    "IGNORE_DIRS",
]