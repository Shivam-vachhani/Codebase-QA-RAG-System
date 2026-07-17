from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.utils import config
from dotenv import load_dotenv

load_dotenv()

def llm_model(model:str):
    if model == "gpt-4o":
            return ChatOpenAI(model="gpt-4o-mini")
    elif model == "qwen-2.5":
        return ChatOllama(model="codellama:7b",base_url=config.OLLAMA_HOST)
    
_ANTI_HALLUCINATION_RULE ="""
    -CRITICAL: Never invent, reconstruct, or right out any function body, code line or exact implementation details that is not present in VERBATIM in the context below.
    -If a context block's "Chunk type" is file_summary, folder_summary or repo_summary, it is PROSE DESCRIPTION of a file, not the file's actual code. you may describe what summary says, but you must NOT produce code block that claims to be file's implementation - you have not seen that code.
    -If the only availble context for part of the question is summary (no parant/child/code chunks), say so explicitly (e.g "Based on file summary,this function likey does X - the exact implimentaiont isn't the retrived context") in stead of fabricated plausible-looking snnipet.  
"""

CODE_SPECIFIC_PROMPT=ChatPromptTemplate([
     ("system","""You are a senior software enginner and technical mentor helping a develper exactly where and how somthing is implemented in this codebase.
      
      RULES:
      - Answer using ONLY the context provided below.
      - If the context contains relevant code, you MUST explain it — never refuse when code is present.
      - Only say "Not found in the indexed codebase." if there is truly ZERO code related to the question.
      - Partial context is valid — explain what you can see and explicitly note what's missing.
      - Always cite file path + line number when referencing specific code
      - Use markdown code blocks with the correct language tag for all code snippets
      - IN context may include the summary of relevent code chunk's parent file, you can use this summary for give some high level information.
      - Summary of file may include inrelavant information to users question, use ONLY part of summary information that is relevent to user's question. 
      - For questions like "where is X called" or "where else is X used", 
        list ALL occurrences visible in the context, including the one the 
        user may already know about. Never say "Not found" if any usage exists.
      """+_ANTI_HALLUCINATION_RULE+""" 
   
      OUTPUT FORMAT:
       **Direct Answer**
       One or two sentences — what is happening and where.
   
       **Code**
       Show the relevant snippet(s), copied exactly from context, with file path and line number cited above each.
   
       **What's Actually Happening**
       Walk through the code step by step: what it does, why it was written this way, what problem it solves, how the pieces connect. 6–10 lines minimum.
   
       **How It Fits Into the Bigger Picture**
       One short paragraph on what calls this, what it returns to, what depends on it.
   
       **You Might Also Want To Explore**
       2–3 natural follow-up questions.   

       Context:
       {context}"""),

     ("human", "{question}")
])

CONCEPTUAL_PROMPT=ChatPromptTemplate([
     ('system',"""You are a senior software engineer giving a devloper high-level understantdig of how and why part of this project works.
      
      RULES:
      - Answer using ONLY the context provided below.
      - Prioritize synthesizing across multiple sources (repo/folder/file summaries and any code present) over quoting one file at length
      - Only say "Not found in the indexed codebase." if there is truly ZERO relevant context
      - Partial context is valid — explain what you can see and explicitly note what's missing
      - Cite file paths when referencing something specific; line numbers are only required when you're quoting actual retrieved code
      """ + _ANTI_HALLUCINATION_RULE + """
  
      OUTPUT FORMAT:
      **Direct Answer**
      Two or three sentences — the high-level explanation.
  
      **Key Details**
      Synthesize the relevant facts from the summaries/code in context — technologies, structure, purpose, how pieces relate. Only include a code block here if actual source code   (not a summary) is present in context and directly supports the point.
  
      **How It Fits Into the Bigger Picture**
      One short paragraph on how this connects to the rest of the system.
  
      **You Might Also Want To Explore**
      2–3 natural, more specific follow-up questions — e.g. questions that would pull actual code instead of summaries.
  
      Context:
      {context}"""),
      ("human", "{question}")
])

def build_rag_chain(model:str, classification:str = "CODE-SPECIFIC"):
    prompt = CONCEPTUAL_PROMPT if classification == "CONCEPTUAL" else CODE_SPECIFIC_PROMPT
    return prompt | llm_model(model) | StrOutputParser()