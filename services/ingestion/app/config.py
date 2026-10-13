import os

CHUNK_WORDS = 800
MIN_CHARS = 60
CHUNK_OVERLAP_WORDS = 100

EMBED_MODEL_NAME = os.environ.get(
    "EMBED_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
)

GITHUB_REPOS = [r.strip() for r in os.environ.get("GITHUB_REPOS", "").split(",") if r.strip()]
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_CACHE_TTL_SECONDS = int(os.environ.get("GITHUB_CACHE_TTL_SECONDS", "21600"))

PORTFOLIO_WEBSITE_REPO = os.environ.get("PORTFOLIO_WEBSITE_REPO", "JoshLG18/PortfolioWebsite")
PORTFOLIO_WEBSITE_BRANCH = os.environ.get("PORTFOLIO_WEBSITE_BRANCH", "main")
PORTFOLIO_ATTACHMENTS_PATH = os.environ.get("PORTFOLIO_ATTACHMENTS_PATH", "attachments")

INDEX_DATA_DIR = os.environ.get("INDEX_DATA_DIR", "/data/index")
GITHUB_CACHE_DIR = os.environ.get("GITHUB_CACHE_DIR", "/data/cache")

# Bump this whenever chunking logic or EMBED_MODEL_NAME changes, to force a
# rebuild even when source content hashes are otherwise unchanged.
CHUNKING_VERSION = "v1"

# TODO: migrate from app.py SYNONYM_MAP
SYNONYM_MAP: dict[str, list[str]] = {}
