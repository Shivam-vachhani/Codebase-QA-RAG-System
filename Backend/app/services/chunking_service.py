from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from tree_sitter_language_pack import get_language, get_parser
from app.utils import config
import threading

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

    if language_str not in config.TREESITTER_LANG_MAP:
        return _character_fallback(file, chunk_size=2000, lable="parent")

    ts_lang_name = config.TREESITTER_LANG_MAP[language_str]

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
        target_node_types = config.MEANINGFUL_NODE_TYPES.get(ts_lang_name, set())
        source_line = content.splitlines()
        parent_chunks = []
        visited_ranges = []

        cursor = tree.walk()

        reached_end = False

        while not reached_end:
            current_node = cursor.node()

            node_kind = current_node.kind()

            if node_kind in target_node_types:

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

        if language in config.TREESITTER_LANG_MAP:
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


# ---- Function to split code files into manageable chunks for LLM processing ---- #
def chunk_files(files: list[dict]) -> list[Document]:
    all_docs = []
    for f in files:
        parents = extract_parent_chunks(f)
        children = create_child_chunks(parents)
        all_docs.extend(parents)
        all_docs.extend(children)
    return all_docs