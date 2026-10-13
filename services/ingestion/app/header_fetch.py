from dataclasses import dataclass


@dataclass
class HeaderFetchResult:
    html: str
    css: str
    bio_text: str
    stats: dict
    source_url: str
    fetched_at: str
    stale: bool


def fetch_portfolio_header() -> HeaderFetchResult:
    raise NotImplementedError(
        "TODO: fetch raw index.html from PORTFOLIO_WEBSITE_REPO, parse the <header> "
        "nav block, bio text, and #projects stats with BeautifulSoup; TTL cache + ETag"
    )
