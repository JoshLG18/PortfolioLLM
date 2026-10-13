from fastapi import FastAPI

app = FastAPI(title="search-service")

_state = {"index_id": None, "loaded": False}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "index_id": _state["index_id"], "loaded": _state["loaded"]}


@app.post("/search")
def search(payload: dict):
    raise NotImplementedError("TODO: call retrieval.retrieve with query/top_k/rewrite_query from payload")


@app.post("/search/reload")
def reload_index():
    raise NotImplementedError("TODO: reload artifacts via index_loader.load_current_index, update _state")
