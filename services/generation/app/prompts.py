def build_answer_prompt(question: str, sources: list[dict]) -> str:
    raise NotImplementedError("TODO: migrate from app.py llm_answer prompt/system assembly")


def build_rewrite_prompt(query: str, pseudo_relevance_titles: list[str]) -> str:
    raise NotImplementedError("TODO: new prompt for LLM-based query rewriting (see plan section 6)")
