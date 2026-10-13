from fastapi import FastAPI

app = FastAPI(title="ingestion-service")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/chunks")
def get_chunks(source: str | None = None, limit: int = 50, offset: int = 0):
    raise NotImplementedError("TODO")


@app.get("/sources")
def get_sources():
    raise NotImplementedError("TODO: local docs + github repos + their fetch status")


@app.post("/ingest/rebuild")
def rebuild(force: bool = False):
    raise NotImplementedError("TODO: compute_corpus_hash, skip if unchanged, else build+persist+notify search")


@app.get("/ingest/status")
def ingest_status():
    raise NotImplementedError("TODO")


@app.get("/header")
def get_header():
    raise NotImplementedError("TODO")


@app.post("/header/refresh")
def refresh_header():
    raise NotImplementedError("TODO: bypass TTL cache")
