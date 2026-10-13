from dataclasses import dataclass


@dataclass
class LoadedIndex:
    index_id: str
    chunks: list
    tfidf_word: object
    tfidf_char: object
    tfidf_title: object
    sparse_matrix: object
    bm25: object
    embeddings: object
    faiss_index: object


def load_current_index(index_data_dir: str) -> LoadedIndex:
    raise NotImplementedError(
        "TODO: read /data/index/current.json pointer, load joblib/pkl/npy/faiss artifacts from by-id/{index_id}/"
    )


def poll_for_new_index(current_index_id: str | None, index_data_dir: str) -> bool:
    raise NotImplementedError(
        "TODO: stat current.json mtime (~30s poll), compare index_id, return True if it changed"
    )
