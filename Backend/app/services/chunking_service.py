from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from app.services.git_service import clone_repo
from app.services.loader_service import get_code_files
from langchain_core .documents import Document

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



