from .chunking import Chunk


def build_sparse_index(chunks: list[Chunk]):
    raise NotImplementedError("TODO: migrate from app.py build_sparse_index - 3x TF-IDF (word/char/title) + BM25")


def build_dense_index(chunks: list[Chunk]):
    raise NotImplementedError("TODO: migrate from app.py build_dense_index - sentence-transformers + FAISS IndexFlatIP")


def persist_index(index_id: str, chunks: list[Chunk], sparse, dense) -> None:
    raise NotImplementedError(
        "TODO: write artifacts to /data/index/by-id/{index_id}/, then atomically swap "
        "/data/index/current.json via write-tmp + os.rename"
    )
