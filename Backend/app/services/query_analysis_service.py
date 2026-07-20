import os,time
from cachetools import LRUCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.utils.rate_limiter import RateLimiter
from app.services.vector_service import load_chroma
from app.models.query_analysis import QeryAnalysis

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
ANALYSIS_LLM_MODEL= "llama-3.1-8b-instant"

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
     
    CRITICAL NOTE:Generate AT MOST 3 alternative search queries — never more than 3, even if more come to mind.
     
    Keep expanded queries short (5-10 words), phrased as search queries not
    sentences."""),
        ("human", "Codebase context:\n{context}\n\nQuestion: {question}")
])


def _get_summary_context(repo_id:str, question:str, k:int = 5) -> str:
    """cheap grounding lookup against summries that build at ingest."""

    try:
      summary_store = load_chroma(repo_id,collection="summaries")
      doc_results=summary_store.similarity_search(question,k=k)
      print(f"[QueryAnalysis] Found {len(doc_results)} summary docs for context lookup.")
      return "/n/n".join(
           f"[{d.metadata.get("chunk_type")}] {d.metadata.get('file_path')}:{d.page_content}"
           for d in doc_results
      )
    except Exception as e:
       print(f"[QueryAnalysis] Summary context lookup failed: {e}")
       return ""
    

def _get_analysis_llm():
   return ChatGroq(
      model=ANALYSIS_LLM_MODEL,
      temperature=0,
      groq_api_key = GROQ_API_KEY,
   ).with_structured_output(QeryAnalysis)


def _invoke_with_backoff(chain,payload,max_retries:int=3):
   for attempt in range(max_retries):
      analysis_rate_limiter.acquire()
      try:
         return chain.invoke(payload)
      except Exception as e:
         msg= str(e)
         is_429 = "429" in msg or "rate_limit" in msg.lower()
         if not is_429 or attempt == max_retries-1:
            raise
         wait = 2 ** attempt
         print(f"[QueryAnalysis] 429 — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
         time.sleep(wait)
   raise RuntimeError("Query analysis call failed after max retries")

def _fallback(question:str) -> QeryAnalysis:
   """CODE_SPECIFIC is the safer default under weighted routing — it excludes
    nothing downstream, it just under-weights the summary signal."""
   return QeryAnalysis(
      classification="CODE_SPECIFIC",
      confidence=0.0,
      expanded_queries=[question]
   )

def analyze_query(repo_id:str, question:str) ->QeryAnalysis:
   cache_key = (repo_id,question)
   if cache_key in _analysis_cache:
      return _analysis_cache[cache_key]
   
   try:
      context = _get_summary_context(repo_id,question,k=5)
      if not context:
         result = _fallback(question)
      else:
         chain = ANALYSIS_PROMPT | _get_analysis_llm()
         result = _invoke_with_backoff(chain,{"context":context,"question":question})

      print(f"[QueryAnalysis] {question!r} -> {result.classification} "
            f"(conf={result.confidence:.2f}) expansions={result.expanded_queries}")
      
   except Exception as e:
       print(f"[QueryAnalysis] Failed, falling back: {e}")
       result = _fallback(question)

   _analysis_cache[cache_key] = result
   return result


