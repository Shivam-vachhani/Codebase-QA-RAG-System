import os,pathlib
from  langchain_text_splitters import Language

##-----configurations-----##
OLLAMA_HOST = os.getenv("OLLAMA_HOST","http://localhost:11434")
CHROMA_PATH = os.getenv("CHROMA_PATH","./data/chroma_db")
CLONE_TMP = pathlib.Path("./tmp/repos")

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

IGNORE_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "Gemfile.lock",
    "poetry.lock",
    "Cargo.lock",
}

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
    },
    "javascript": {
        "function_declaration",
        "arrow_function",
        "class_declaration",
        "method_definition",
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


__all__ = [
    "OLLAMA_HOST",
    "CHROMA_PATH",
    "CLONE_TMP",
    "SUPPORTED",
    "TEXT_FALLBACKS",
    "EXACT_TEXT_FILES",
    "BINARY_EXTENSIONS",
    "IGNORE_DIRS",
    "TREESITTER_LANG_MA",
    "MEANINGFUL_NODE_TYPES"
]

