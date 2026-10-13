from dataclasses import dataclass


@dataclass
class ReadmeFetchResult:
    repo: str
    sha: str | None
    text: str | None
    status: str  # "ok" | "cached" | "error"


def fetch_readme(repo: str) -> ReadmeFetchResult:
    raise NotImplementedError(
        "TODO: GET raw.githubusercontent.com or the GitHub API readme endpoint, "
        "with TTL cache + ETag conditional requests; serve last-known-good on failure"
    )


def fetch_all_readmes(repos: list[str]) -> list[ReadmeFetchResult]:
    raise NotImplementedError("TODO: call fetch_readme for each configured repo")
