import json,pathlib
from app.utils import config

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

        if any(bad in p.parts for bad in config.IGNORE_DIRS):
            continue

        if p.is_file():
    
            ext = p.suffix.lower()
            file_name =p.name.lower()

            if ext in config.BINARY_EXTENSIONS or file_name in config.IGNORE_FILES:
                continue

            lang_enum =None
            should_read=False

            if ext in config.SUPPORTED:
                lang_enum = config.SUPPORTED[ext]
                should_read=True

            elif ext in config.TEXT_FALLBACKS or file_name in config.EXACT_TEXT_FILES:
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

