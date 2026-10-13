CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None


def get_cross_encoder():
    """Lazy-loaded once at process startup. Query-dependent — never persisted
    to the index volume; reranking always happens at serve time.
    """
    raise NotImplementedError("TODO: migrate from app.py cross-encoder load")
