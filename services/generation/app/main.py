from fastapi import FastAPI

app = FastAPI(title="generation-service")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/generate")
def generate(payload: dict):
    raise NotImplementedError("TODO: check quota, build prompt, call Gemini, return answer + usage + quota")


@app.post("/rewrite-query")
def rewrite_query_endpoint(payload: dict):
    raise NotImplementedError("TODO: build rewrite prompt, call Gemini, decrement shared quota, return rewritten_query")


@app.get("/quota")
def quota():
    raise NotImplementedError("TODO: return remaining/limit/resets_at")
