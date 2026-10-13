def minmax(scores):
    raise NotImplementedError("TODO: migrate from app.py _minmax")


def expand_query(query: str, bm25, synonym_map: dict) -> str:
    raise NotImplementedError(
        "TODO: migrate from app.py expand_query - synonym map + BM25 pseudo-relevance-feedback titles"
    )


def retrieve(query: str, loaded_index, top_k: int, rewritten_query: str | None = None):
    raise NotImplementedError(
        "TODO: migrate from app.py retrieve - dense+sparse+BM25 score fusion "
        "(W_DENSE=0.50/W_SPARSE=0.30/W_BM25=0.20), cross-encoder rerank, cosine dedup"
    )
