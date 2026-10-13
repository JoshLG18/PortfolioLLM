import os

GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

_gemini = None


def get_gemini_client():
    raise NotImplementedError(
        "TODO: migrate from app.py - genai.configure(api_key=GEMINI_API_KEY) + "
        "genai.GenerativeModel(GEMINI_MODEL_NAME), cached as a module-level singleton"
    )
