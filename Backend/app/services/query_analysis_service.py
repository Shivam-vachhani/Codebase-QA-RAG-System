import os
from cachetools import LRUCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.utils.rate_limiter import RateLimiter

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
model = "llama-3.1-8b-instant"

analysis_rate_limiter = RateLimiter(max_requested=30, period=60.0)

_analysis_cache:LRUCache = LRUCache(maxsize=200)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query analysis engine for a code search system.

    Given a developer's question and some high-level context about the codebase,
    do two things:
    
    1. Classify the question as CONCEPTUAL (asking how/why something works) or
       CODE_SPECIFIC (asking where something is defined, called, or implemented).
    2. Generate 2-3 alternative search queries that would help retrieve relevant
       code. Use real function/class/file names from the context when relevant,
       and include technical synonyms (e.g. "login" -> "authenticate", "auth").
    
    Keep expanded queries short (5-10 words), phrased as search queries not
    sentences."""),
        ("human", "Codebase context:\n{context}\n\nQuestion: {question}")
])


