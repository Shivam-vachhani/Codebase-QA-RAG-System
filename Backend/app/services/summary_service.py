import time,pathlib,os
from langchain_groq import ChatGroq
from app.utils.rate_limiter import RateLimiter

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUMMARY_LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

groq_rate_limiter = RateLimiter(max_requested=25,period=60.0)

def _invoke_with_backoff(llm,prompt,max_retries:int = 5):
     for attempt in range(max_retries):
          groq_rate_limiter.acquire()
          try:
               return llm.invoke(prompt)
          except Exception as e:
               msg = str(e)
               is_429 = "429" in msg or "rate_limit" in msg.lower()

               if not is_429 or attempt == max_retries-1:
                    raise

               retry_after = getattr(getattr(e,"response",None),"headers",{}).get("retry-after")
               wait = float(retry_after) if retry_after else (2**attempt)
               print(f"[Groq] 429 — retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
               time.sleep(wait)
          raise RuntimeError("Groq call failed after max retries")




def _get_summary_llm():
    return ChatGroq(model=SUMMARY_LLM_MODEL,
                    temperature=0,
                    groq_api_key=GROQ_API_KEY)

def summarize_file(file:dict) -> str:
     """Generate a summary for a single file. Called during ingest."""
     llm = _get_summary_llm()

     content_priview = file["content"][:6000]

     prompt = f"""You are summarizing a source code file for a codebase search index. This summary will be embedded and retrieved by semantic search, so every sentence must carry specific, retrievable information.
     
     File: {file['path']}
     Language: {file['language']}
     
     ```{file['language']}
     {content_priview}
     ```
     
     Write a concise technical summary (4-6 sentences) covering:
     - This file's primary responsibility (one sentence, specific to what it actually does)
     - The key named units it defines — functions, types, classes, constants, exports, or any other top-level construct the language uses, referenced by their real names
     - What other specific files, modules, or packages it imports or depends on — use real paths/names, not "external libraries" in the abstract
     - Any genuinely notable detail visible in the code (a specific algorithm, a specific validation rule, a specific data structure, a specific config value) — only if there's a concrete detail to point to
     
     Hard rules:
     - Do NOT use the words "modular," "maintainable," "extensible," "scalable," "robust," or "separation of concerns" unless you can name the specific mechanism that makes it true.
     - Do NOT end with a generic closing sentence about code quality or design philosophy. End on the last concrete fact.
     - Do NOT assume or name any particular architecture style (e.g. "microservices," "MVC," "component-based") unless the code itself demonstrates it directly. Describe what the file does in plain terms if no such pattern is clearly visible.
     - Do NOT use terminology specific to one language/framework (e.g. "hook," "component," "route," "endpoint," "blueprint") unless that exact concept is what the code shows. Use neutral terms like "function," "unit," "data structure," or "entry point" when the file's own language/paradigm doesn't map to those framework-specific terms.
     - If the file is small/trivial (e.g. a config file, a static data file, a constants file), say so in one sentence and stop — do not pad a simple file with extra sentences to hit a length target.
     - Every sentence must reference something concrete from the code shown above. If you can't point to a specific name or behavior, cut the sentence.
     
     Be specific and technical. No fluff."""
          
     response = _invoke_with_backoff(llm,prompt)
     return response.content

def summerize_folder(folder_path:str,file_summeries:list[dict]) -> str :
     llm = _get_summary_llm()   

     files_overview = "\n".join(
          f"{s["file_path"]}:{s["summary"]}"
          for s in file_summeries[:20]
     ) 
      
     prompt = f"""You are summarizing a folder/module in a codebase for a search index. This will be embedded for semantic retrieval, so precision matters more than polish.

     Folder: {folder_path}

     File summaries in this folder:
     {files_overview}

     Write a concise technical summary (4-6 sentences) covering:
     - What this folder is responsible for, stated plainly and specifically (not "manages X" — say what X actually consists of, based on the files shown)
     - How the files in it relate to or depend on each other — name the actual files
     - What consumes this folder's output (which other folder or part of the system) — only if the evidence supports it; if unclear, say "likely used by" rather than asserting it
     - One concrete detail if one is visible (a specific data flow, a specific naming convention, a specific shared dependency across files) — skip this point entirely if nothing concrete stands out

     Hard rules:
     - Do NOT use "modular," "maintainable," "extensible," "scalable," or "separation of concerns" unless naming the specific mechanism.
     - Do NOT name or assume any particular architecture style ("microservices," "MVC," "layered," "component-based," etc.) unless the file summaries directly demonstrate it. Default to describing the folder's actual contents and relationships in plain terms.
     - Do NOT use language- or framework-specific terms (e.g. "hooks," "components," "endpoints," "models," "controllers") unless the file summaries themselves use that paradigm. Stick to neutral terms otherwise.
     - Do NOT end with a generic sentence praising the design. End on the last specific fact.
     - If the folder only contains config/boilerplate files, say that directly instead of inflating it with architecture language.

     Be specific and technical."""
    
     response = _invoke_with_backoff(llm,prompt)
     return response.content

def summerize_repo(repo_summary_input:dict) -> str:
     llm = _get_summary_llm()

     folder_overview = "\n".join(
          f"{path}:{summary}"
          for path,summary in list(repo_summary_input["folder_summaries"].items())[:15] 
     )

     file_tree_sample = "\n".join(repo_summary_input["file_tree"][:50])

     prompt = f"""You are generating a top-level overview of this codebase for a developer assistant. This will be embedded for semantic search, so accuracy about the actual structure matters more than sounding impressive.

     File tree (sample):
     {file_tree_sample}

     Module summaries:
     {folder_overview}

     Write a technical overview (8-10 sentences) covering:
     - What this project does and its purpose, stated plainly
     - The actual technologies/languages/frameworks in use, named directly, based only on what's shown in the file tree and module summaries
     - The actual high-level structure — count and name the real top-level parts (e.g. "two top-level folders: X and Y" or "a single package with three modules"); do not describe it as more distributed, layered, or service-oriented than the evidence supports
     - Key folders/modules and what each is responsible for, referencing real names from the file tree
     - How data or control actually flows between the named parts, step by step, based only on what the module summaries state
     - Who the likely users of this system are, based only on what the project's actual functionality suggests

     Hard rules:
     - Only describe the structure using an architecture label ("microservices," "event-driven," "layered," "monolith," etc.) if the file tree and module summaries clearly demonstrate it.Otherwise describe the structure in plain, neutral terms (e.g. "one backend and one frontend" or "a single script with helper modules").
     - Do NOT use terminology tied to one specific language or framework paradigm unless that paradigm is what the evidence shows. Use neutral terms ("program," "module," "service," "entry  point," "data file") when uncertain.
     - Do NOT close with a generic statement about the project being well-designed, maintainable, or scalable unless a specific fact supports it.
     - Reference actual folder and file names throughout, not generic descriptions like "the backend module" when a real name is available.

     Be specific."""
     response = _invoke_with_backoff(llm,prompt)
     return response.content

def build_summary_index(files:list[dict],repo_id:str) -> dict:
    """
    Builds the full summary hierarchy during ingest.
    Returns structured summary data ready for Chroma storage.
    """
    print(f"[Summaries] Generating {len(files)} file summaries...")

    file_summaries = []

    summarizable = [ f for f in files 
                     if f["language"] != "plain_text"
                     and len(f["content"].strip()) > 100
                    ]


    for i,file in enumerate(summarizable):
         try:
              summary_text = summarize_file(file)
              file_summaries.append({
                   "file_path":file["path"],
                   "language":file["language"],
                   "summary":summary_text,
              })
              
              if i % 10 == 0:
                   print(f"[Summaries] {i}/{len(summarizable)} files done...")
         except Exception as e:
            print(f"[Summaries] Failed to summarize {file['path']}: {e}")
            continue


    print("[Summaries] Generating folder summaries...")  

    folder_map: dict[str , list[dict]] = {}
    for fs in file_summaries:
         folder = str(pathlib.Path(fs["file_path"]).parent)
         folder_map.setdefault(folder,[]).append(fs) 
         
    folder_summaries={}
    for folder_path,folder_files in folder_map.items():
         if len(folder_files) == 0:
              continue
         try:
              folder_summaries[folder_path] = summerize_folder(folder_path,folder_files)
         except Exception as e:
              print(f"[Summaries] Failed to summarize folder {folder_path}: {e}")
    
    print("[Summaries] Generating repo-level summary...") 
    file_tree = sorted(f["path"] for f in files) 
    repo_summary_text = summerize_repo({
         "folder_summaries":folder_summaries,
         "file_tree":file_tree
    }) 
    return{
         "repo_summary":repo_summary_text,
         "folder_summary":folder_summaries,
         "file_summary":file_summaries,
    }
    

    

         