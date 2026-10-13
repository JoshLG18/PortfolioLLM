from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    source_type: str  # "local_doc" | "github_readme" | "portfolio_header"
    source_id: str
    title: str
    text: str
    section_path: str | None
    token_count: int
    href: str | None = None


def chunk_text_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Fixed word-count splitter — kept as a fallback for unstructured content.
    TODO: migrate from app.py chunk_text_words.
    """
    raise NotImplementedError


def chunk_structured(text: str, source_type: str) -> list[str]:
    """Heading/section-boundary chunking for HTML/Markdown (READMEs, Blog.html,
    Website.html) and sentence-aware paragraph-merge chunking for PDFs.
    TODO: new structure-aware chunker, see plan section "Search pipeline improvements".
    """
    raise NotImplementedError


def file_to_chunks(path: Path, source_type: str) -> list[Chunk]:
    raise NotImplementedError("TODO: migrate from app.py file_to_chunks, dispatch structured vs word chunker")
