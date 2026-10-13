from pathlib import Path


def extract_text_pdf(path: Path) -> str:
    raise NotImplementedError("TODO: migrate from app.py _extract_text_pdf")


def extract_text_html(path: Path) -> str:
    raise NotImplementedError("TODO: migrate from app.py _extract_text_html")


def extract_text_md(path: Path) -> str:
    raise NotImplementedError("TODO: migrate from app.py _extract_text_md")


EXTRACTORS = {
    ".pdf": extract_text_pdf,
    ".html": extract_text_html,
    ".htm": extract_text_html,
    ".md": extract_text_md,
    ".markdown": extract_text_md,
}
