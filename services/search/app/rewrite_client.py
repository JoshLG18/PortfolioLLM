import os

GENERATION_SERVICE_URL = os.environ.get("GENERATION_SERVICE_URL", "http://generation:8000")


def rewrite_query(query: str, pseudo_relevance_titles: list[str]) -> str:
    raise NotImplementedError(
        "TODO: POST to f'{GENERATION_SERVICE_URL}/rewrite-query'; fall back to the "
        "original query (and static expand_query) if generation-service is unreachable"
    )
