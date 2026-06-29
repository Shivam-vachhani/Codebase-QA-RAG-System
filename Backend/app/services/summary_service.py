import time,pathlib
from  langchain_openai import ChatOpenAI
SUMMARY_LLM_MODEL = "gpt-4o-mini"

def _get_summary_llm():
    return ChatOpenAI(model=SUMMARY_LLM_MODEL,temperature=0)

def summarize_file(file:dict) -> str:
     """Generate a summary for a single file. Called during ingest."""
     llm = _get_summary_llm()

     content_priview = file["content"][:6000]

     prompt = f""" you are summerizing a source code file for a codebase search index.

    File:{file['path']}
    Language:{file['language']}

    ```{file['language']}
    {content_priview}
    ```
    Write a concise technical summary(4-8 sentance) covering:
    -What this file's primary responsibility is 
    -key classes, funcions or exports its defines
    -What other parts of the system it depends on it
    -Any important patterns and design decisions visible here

    Be Specific and technical. No fluff."""
     
     response = llm.invoke(prompt)
     return response.content

def summerize_folder(folder_path:str,file_summeries:list[dict]) -> str :
     llm = _get_summary_llm()   

     files_overview = "\n".join(
          f"{s["file_path"]}:{s["summary"]}"
          for s in file_summeries[:20]
     ) 
      
     prompt = f""" you are summarizing a folder/module in codebase for a search index.
     Folder: {folder_path}
     Files in this folder:
     {files_overview}
 
     Write a concise technical summarie(4-6 sentence) covering:
     -what this folder/module is responsible for
     -How its file related to each other
     -What the rest of the codebase uses this module for 
     -Any key architecute patterns
 
     Be specific and technical """
    
     response = llm.invoke(prompt)
     return response.content

def summerize_repo(repo_summary_input:dict) -> str:
     llm = _get_summary_llm()

     folder_overview = "\n".join(
          f"{path}:{summary}"
          for path,summary in list(repo_summary_input["folder_summaries"].items())[:15] 
     )

     file_tree_sample = "\n".join(repo_summary_input["file_tree"][:50])

     prompt=f"""you are genarating a top-level over view of the codebase for devloper assistant.

     File tree(sample):
     {file_tree_sample}

     Module Summaries:
     {folder_overview}

     Write through technical overview (8-12 sentances) covering:
     -What this project does and its primary purpose
     -The main technology and framewoks used
     -The high level architecture (frontend/backend/services/etc.)
     -Key modules and what each is responsible for 
     -how data flows through the system at high level 
     -who the likely audiance/user of this system are

     Be spicific. Referance actual folde and file names."""

     response = llm.invoke(prompt)
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
    

    

         