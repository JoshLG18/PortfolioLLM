from .chunking import Chunk
from .config import LOCAL_DOCS_DIR


def load_local_doc_chunks(docs_dir: str = LOCAL_DOCS_DIR) -> list[Chunk]:
    raise NotImplementedError("TODO: migrate from app.py load_chunks - walk docs_dir, extract, chunk")
