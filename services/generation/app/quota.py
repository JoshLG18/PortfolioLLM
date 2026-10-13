import os

DAILY_QUOTA = int(os.environ.get("DAILY_QUOTA", "20"))
QUOTA_FILE = os.environ.get("QUOTA_FILE", "/data/quota/quota.json")


def load_quota() -> dict:
    raise NotImplementedError("TODO: migrate from app.py _load_quota, now reading/writing QUOTA_FILE on a volume")


def remaining_quota() -> int:
    raise NotImplementedError("TODO: migrate from app.py remaining_quota")


def increment_quota() -> None:
    """Single counter — shared by both /generate and /rewrite-query calls."""
    raise NotImplementedError("TODO: migrate from app.py increment_quota")
