# app.py  —  Portfolio Q&A with Gemini 2.0 Flash-Lite + daily 1500-request guard
import os, re, glob, json, datetime
from typing import List, Dict
import streamlit as st

# ---------- Readers ----------
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ---------- Retrieval ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Gemini ----------
import google.generativeai as genai


# ==========================================================
# Config
# ==========================================================
DAILY_QUOTA = 1500
QUOTA_FILE = ".gemini_quota.json"   # stored next to the app
MODEL_NAME = "gemini-2.0-flash-lite"

st.set_page_config(page_title="Ask My Portfolio", page_icon="💬", layout="wide")
st.title("💬 Ask My Portfolio")
st.write("Ask about my CV or projects. Answers come from your local files (PDF/HTML/MD) via retrieval, "
         "then Gemini 2.0 Flash-Lite writes a concise answer with citations.")

with st.sidebar:
    st.header("⚙️ Settings")
    TOP_K = st.slider("Top-k chunks", 1, 8, 4)
    MIN_CHARS = st.slider("Min chunk length (chars)", 60, 600, 120, step=20)
    SHOW_CHUNKS = st.checkbox("Show retrieved chunks", False)


# ==========================================================
# Quota tracking (simple local file)
# ==========================================================
def _load_quota():
    today = datetime.date.today().isoformat()
    try:
        if os.path.exists(QUOTA_FILE):
            with open(QUOTA_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
    except Exception:
        pass
    return {"date": today, "count": 0}

def remaining_quota() -> int:
    return max(0, DAILY_QUOTA - _load_quota()["count"])

def check_and_increment_quota() -> bool:
    """Return True if we can make a request; also increments the counter."""
    data = _load_quota()
    if data["count"] >= DAILY_QUOTA:
        return False
    data["count"] += 1
    try:
        with open(QUOTA_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    return True

with st.sidebar:
    st.markdown(f"**🔒 Daily quota left:** {remaining_quota()} / {DAILY_QUOTA}")


# ==========================================================
# Utilities
# ==========================================================
def sent_tokenize(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def windowed_chunks(sentences: List[str], min_chars=120, max_chars=900, overlap_sents=1) -> List[str]:
    chunks, buf, cur = [], [], 0
    for s in sentences:
        if cur + len(s) <= max_chars or cur < min_chars:
            buf.append(s); cur += len(s)
        else:
            if buf: chunks.append(" ".join(buf))
            buf = buf[-overlap_sents:] + [s]
            cur = sum(len(x) for x in buf)
    if buf: chunks.append(" ".join(buf))
    return [c for c in chunks if len(c) >= min_chars]

def read_pdf_chunks(path: str, min_chars: int):
    out = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for p_i, p in enumerate(reader.pages, 1):
            txt = p.extract_text() or ""
            sents = sent_tokenize(txt)
            for j, ch in enumerate(windowed_chunks(sents, min_chars=min_chars, max_chars=1000), 1):
                out.append({"title": f"{os.path.basename(path)} — p{p_i}.{j}", "text": ch, "href": path})
    return out

def read_html_chunks(path: str, min_chars: int):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    out = []
    for i, para in enumerate([p for p in re.split(r"\n\s*\n", text) if p.strip()], 1):
        sents = sent_tokenize(para)
        for j, ch in enumerate(windowed_chunks(sents, min_chars=min_chars, max_chars=1000), 1):
            out.append({"title": f"{os.path.basename(path)} — ¶{i}.{j}", "text": ch, "href": path})
    return out

def read_md_chunks(path: str, min_chars: int):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    out = []
    for i, para in enumerate([p for p in re.split(r"\n\s*\n", text) if p.strip()], 1):
        sents = sent_tokenize(para)
        for j, ch in enumerate(windowed_chunks(sents, min_chars=min_chars, max_chars=1000), 1):
            out.append({"title": f"{os.path.basename(path)} — ¶{i}.{j}", "text": ch, "href": path})
    return out

READERS = {
    ".pdf": read_pdf_chunks,
    ".html": read_html_chunks,
    ".htm": read_html_chunks,
    ".md": read_md_chunks,
    ".markdown": read_md_chunks,
}

# ==========================================================
# Load & index docs
# ==========================================================
@st.cache_data(show_spinner=True)
def load_chunks(min_chars: int):
    files = [fp for fp in glob.glob("docs/*") if os.path.splitext(fp)[1].lower() in READERS]
    chunks, errors = [], []
    for fp in files:
        try:
            chunks.extend(READERS[os.path.splitext(fp)[1].lower()](fp, min_chars))
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")
    return files, chunks, errors

files, chunks, errors = load_chunks(MIN_CHARS)

if not chunks:
    st.info("Put .pdf/.html/.md files in `/docs` then reload.")
    st.stop()

@st.cache_resource(show_spinner=True)
def build_index(texts: List[str]):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=50000)
    X = vec.fit_transform(texts)
    return vec, X

vectorizer, X = build_index([c["text"] for c in chunks])

def retrieve(query: str, k: int):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idxs = sims.argsort()[::-1][:k]
    return [chunks[i] | {"score": float(sims[i])} for i in idxs]


# ==========================================================
# Gemini client
# ==========================================================
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY in .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=api_key)
_gemini = genai.GenerativeModel(MODEL_NAME)

def llm_answer_with_gemini(question: str, contexts: list[dict]) -> str:
    # compact, cited context
    blocks = []
    for i, c in enumerate(contexts[:4], 1):   # top-4
        txt = c["text"]
        if len(txt) > 1000: txt = txt[:1000] + " ..."
        blocks.append(f"[{i}] {c['title']}\n{txt}")
    context = "\n\n".join(blocks)

    system = (
        "Answer strictly from the provided sources. Be concise and factual. "
        "If asked about machine learning methods/algorithms/models, output a deduplicated bullet list of names. "
        "Always cite sources like [1], [2]. If not present in sources, say you can't find it."
    )
    prompt = f"{system}\n\nQuestion: {question}\n\nSources:\n{context}\n\nAnswer:"
    r = _gemini.generate_content(prompt)
    return (r.text or "").strip()


# ==========================================================
# UI — Ask
# ==========================================================
st.divider()
q = st.text_input("Ask something:")
if q:
    top = retrieve(q, TOP_K)
    if SHOW_CHUNKS:
        with st.expander("Retrieved chunks"):
            for i, c in enumerate(top, 1):
                st.markdown(f"**[{i}] {c['title']}** (score {c['score']:.3f})")
                st.write(c["text"])

    # Enforce free daily quota locally
    if not check_and_increment_quota():
        st.error(f"Daily limit reached ({DAILY_QUOTA} requests). Try again tomorrow — we block extra calls to avoid charges.")
    else:
        try:
            answer = llm_answer_with_gemini(q, top)
            st.markdown(answer)
        except Exception as e:
            st.error(f"Gemini call failed: {e}")

    with st.expander("Sources used"):
        for i, s in enumerate(top, 1):
            st.markdown(f"**[{i}] {s['title']}** — [{s['href']}]({s['href']})")
