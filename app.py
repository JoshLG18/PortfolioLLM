# ---- Portfolio RAG — Streamlit app ----
# ----------------------------------------
import os, re, glob, json, datetime, traceback
from typing import List, Dict
import numpy as np
import streamlit as st

from bs4 import BeautifulSoup
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

from rank_bm25 import BM25Okapi
from nltk.stem.snowball import SnowballStemmer

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

import google.generativeai as genai

# ----------------------------------------

st.set_page_config(
    page_title="Portfolio RAG",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# inject CSS before any spinner / widget so the theme is present on boot
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    :root {
        --bg-base:    #0d1117;
        --bg-panel:   #161d2b;
        --bg-card:    #1c2535;
        --bg-input:   #1c2535;
        --border:     #253047;
        --accent:     #00e5ff;
        --accent-dim: #0097a7;
        --text:       #d0dce8;
        --text-muted: #5a7190;
        --green:      #00e676;
        --red:        #ff5252;
        --font-ui:    'IBM Plex Sans', sans-serif;
        --font-mono:  'IBM Plex Mono', monospace;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main .block-container {
        background-color: var(--bg-base) !important;
        color: var(--text) !important;
        font-family: var(--font-ui) !important;
    }

    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; font-family: var(--font-ui) !important; }

    .main .block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px !important; }

    h1 {
        font-family: var(--font-mono) !important; font-size: 1.4rem !important; font-weight: 600 !important;
        color: var(--accent) !important; letter-spacing: 0.08em !important; text-transform: uppercase !important;
        border-bottom: 1px solid var(--border) !important; padding-bottom: 0.6rem !important; margin-bottom: 1.2rem !important;
    }
    h2, h3 {
        font-family: var(--font-mono) !important; color: var(--accent) !important;
        font-weight: 600 !important; letter-spacing: 0.06em !important;
        text-transform: uppercase !important; font-size: 0.8rem !important;
    }

    label, .stTextInput label, .stSlider label, .stCheckbox label,
    p, li, span, [data-testid="stMarkdownContainer"] p {
        font-family: var(--font-ui) !important; color: var(--text) !important; font-size: 0.88rem !important;
    }

    .stTextInput > div > div > input {
        background-color: var(--bg-input) !important; color: var(--text) !important;
        border: 1px solid var(--accent) !important; border-radius: 4px !important;
        font-family: var(--font-ui) !important; font-size: 0.92rem !important; padding: 0.5rem 0.8rem !important;
    }
    .stTextInput > div > div > input:focus { box-shadow: 0 0 0 2px var(--accent) !important; outline: none !important; }
    .stTextInput > div > div > input::placeholder { color: var(--text-muted) !important; }

    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background-color: var(--accent) !important; border-color: var(--accent) !important;
    }

    .stButton > button {
        background: transparent !important; color: var(--accent) !important;
        border: 1px solid var(--accent) !important; border-radius: 3px !important;
        font-family: var(--font-mono) !important; font-size: 0.78rem !important;
        letter-spacing: 0.05em !important; text-transform: uppercase !important;
        padding: 0.35rem 1rem !important; transition: background 0.15s, color 0.15s;
    }
    .stButton > button:hover { background: var(--accent) !important; color: var(--bg-base) !important; }

    [data-testid="stExpander"] {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        border-radius: 4px !important; margin-top: 0.6rem !important;
    }
    [data-testid="stExpander"] summary {
        background: var(--bg-card) !important; color: var(--accent) !important;
        font-family: var(--font-mono) !important; font-size: 0.78rem !important;
        letter-spacing: 0.06em !important; text-transform: uppercase !important;
        padding: 0.55rem 0.9rem !important; border-radius: 4px !important;
    }
    [data-testid="stExpander"] summary:hover { background: #1f2d42 !important; }
    [data-testid="stExpander"] > div > div { background: var(--bg-card) !important; padding: 0.8rem 1rem !important; }

    hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

    [data-testid="stMetric"] {
        background: var(--bg-card) !important; border: 1px solid var(--border) !important;
        border-radius: 4px !important; padding: 0.7rem 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important; font-family: var(--font-mono) !important;
        font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent) !important; font-family: var(--font-mono) !important;
        font-size: 1.5rem !important; font-weight: 600 !important;
    }

    [data-testid="stAlert"], .stAlert {
        background: var(--bg-card) !important; border-left: 3px solid var(--accent) !important;
        border-radius: 4px !important; color: var(--text) !important; font-size: 0.85rem !important;
    }

    [data-testid="stSpinner"] > div { color: var(--accent) !important; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }

    .answer-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-left: 3px solid var(--accent); border-radius: 4px;
        padding: 1rem 1.2rem; margin: 0.8rem 0;
        font-family: var(--font-ui); font-size: 0.92rem; line-height: 1.65; color: var(--text);
    }
    .section-label {
        font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600;
        color: var(--accent); letter-spacing: 0.1em; text-transform: uppercase;
        margin-bottom: 0.5rem; margin-top: 1.2rem;
    }
    .chunk-card {
        background: var(--bg-panel); border: 1px solid var(--border);
        border-radius: 4px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
        font-size: 0.83rem; line-height: 1.5;
    }
    .chunk-title { font-family: var(--font-mono); color: var(--accent); font-size: 0.72rem; letter-spacing: 0.05em; margin-bottom: 0.35rem; }
    .chunk-score { font-family: var(--font-mono); font-size: 0.7rem; float: right; }

    .source-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; }
    .source-idx { font-family: var(--font-mono); color: var(--accent); min-width: 1.5rem; font-size: 0.72rem; }
    .source-link { color: var(--text-muted) !important; text-decoration: none !important; font-size: 0.78rem; }
    .source-link:hover { color: var(--accent) !important; }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--accent) !important; font-family: var(--font-mono) !important;
        font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
        border-bottom: 1px solid var(--border) !important; padding-bottom: 0.4rem !important; margin-bottom: 0.7rem !important;
    }

    .badge {
        display: inline-block; font-family: var(--font-mono); font-size: 0.68rem;
        padding: 0.15rem 0.5rem; border-radius: 2px; letter-spacing: 0.05em;
        text-transform: uppercase; margin-right: 0.4rem;
    }
    .badge-cyan  { background: rgba(0,229,255,0.12); color: var(--accent); border: 1px solid rgba(0,229,255,0.3); }
    .badge-green { background: rgba(0,230,118,0.10); color: var(--green);  border: 1px solid rgba(0,230,118,0.3); }
    .badge-muted { background: rgba(90,113,144,0.15); color: var(--text-muted); border: 1px solid var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------

# Config
CHUNK_WORDS       = 800  # large enough to capture most project docs as a single chunk
CHUNK_OVERLAP_W   = 100  # overlap for the rare cases a doc exceeds one chunk
MIN_CHARS_DEFAULT = 60

W_DENSE  = 0.50  # FAISS dense score weight
W_SPARSE = 0.30  # TF-IDF sparse score weight
W_BM25   = 0.20  # BM25 score weight

DENSE_CANDIDATES = 64   # top-k from FAISS before fusion
FINAL_POOL       = 32   # candidates fed to cross-encoder (kept small for speed)
FINAL_K          = 6    # more chunks = better coverage for broad questions
DEDUP_COS_THRESH = 0.90 # cosine threshold for near-duplicate filtering

EMBED_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL_NAME  = "gemini-2.5-flash-preview-05-20"  # only model with free tier quota on this account

DAILY_QUOTA = 20        # Gemini 2.5 Flash free tier: 20 requests per day (RPD)
RPM_LIMIT   = 5        # free tier: 5 requests per minute — app will warn if exceeded
QUOTA_FILE  = ".gemini_quota.json"

SYNONYM_MAP = {
    "cv":          ["resume", "résumé"],
    "llm":         ["language model", "genai", "foundation model"],
    "genai":       ["generative ai", "foundation model"],
    "streamlit":   ["python web app", "web app"],
    "nlp":         ["natural language processing"],
    "retrieval":   ["rag", "document search"],
    "rag":         ["retrieval augmented generation", "retrieval"],
    "internship":  ["placement"],
    "dissertation":["The Intersection of Machine Learning and Type 1 Diabetes"],
}

# ----------------------------------------
# Quota helpers — persisted to QUOTA_FILE so the count survives app restarts
# ----------------------------------------
def _load_quota():
    today = datetime.date.today().isoformat()
    try:
        if os.path.exists(QUOTA_FILE):
            with open(QUOTA_FILE) as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
    except Exception:
        pass
    return {"date": today, "count": 0}

def remaining_quota():
    return max(0, DAILY_QUOTA - _load_quota()["count"])

def increment_quota():
    # returns True if a call is allowed and increments the counter, False if limit reached
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

# ----------------------------------------

# Sidebar
with st.sidebar:
    st.markdown("### ⚙ Settings")
    TOP_K       = st.slider("Top-k chunks", 1, 10, FINAL_K)
    MIN_CHARS   = st.slider("Min chunk chars", 40, 500, MIN_CHARS_DEFAULT, step=10)
    SHOW_CHUNKS = st.checkbox("Show retrieved chunks", False)
    SHOW_DIAG   = st.checkbox("Show CE score metrics", False)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Quota")
    rem = remaining_quota()
    pct = int(rem / DAILY_QUOTA * 100)
    bar_color = "var(--green)" if pct > 30 else "var(--red)"
    st.markdown(
        f"""
        <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Daily Remaining</div>
        <div style="font-family:var(--font-mono);font-size:1.3rem;font-weight:600;color:{bar_color};">
            {rem:,} <span style="font-size:0.7rem;color:var(--text-muted)">/ {DAILY_QUOTA:,}</span>
        </div>
        <div style="background:var(--bg-base);border:1px solid var(--border);border-radius:3px;height:6px;margin-top:4px;overflow:hidden;">
            <div style="width:{pct}%;height:6px;background:linear-gradient(90deg,var(--accent),var(--green));border-radius:3px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📁 Documents")
    # doc list injected after load_chunks

# Title
st.markdown(
    """
    <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:0.2rem;">
        <span style="font-family:var(--font-mono);font-size:1.25rem;font-weight:600;
                     color:var(--accent);text-transform:uppercase;letter-spacing:0.1em;">
            Portfolio RAG
        </span>
        <span class="badge badge-cyan">v2.0</span>
        <span class="badge badge-muted">hybrid · dense · cross-encoder</span>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------
# Text extraction helpers
# ----------------------------------------
def _extract_text_pdf(path):
    out = []
    with open(path, "rb") as f:
        for page in PdfReader(f).pages:
            out.append(page.extract_text() or "")
    return "\n".join(out)

def _extract_text_html(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return BeautifulSoup(f.read(), "html.parser").get_text(separator="\n", strip=True)

def _extract_text_md(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()

EXTRACTORS = {
    ".pdf": _extract_text_pdf, ".html": _extract_text_html,
    ".htm": _extract_text_html, ".md": _extract_text_md, ".markdown": _extract_text_md,
}

# ----------------------------------------
# Chunking (CW2 §2.3) — word-based sliding window with overlap
# 100-word windows preserve semantic boundaries; 25-word overlap
# prevents information loss at chunk edges.
# ----------------------------------------
def chunk_text_words(text, chunk_size=CHUNK_WORDS, overlap=CHUNK_OVERLAP_W, min_chars=60):
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(text) >= min_chars else []
    chunks, step = [], chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks

def file_to_chunks(path, min_chars):
    ext  = os.path.splitext(path)[1].lower()
    text = EXTRACTORS[ext](path)
    base = os.path.basename(path)
    return [
        {"title": f"{base} — chunk{i+1}", "text": ch, "href": path}
        for i, ch in enumerate(chunk_text_words(text, min_chars=min_chars))
    ]

# ----------------------------------------
# Load docs
# ----------------------------------------
@st.cache_data(show_spinner="Chunking documents…")
def load_chunks(min_chars):
    fps = [fp for fp in glob.glob("docs/*") if os.path.splitext(fp)[1].lower() in EXTRACTORS]
    chunks, errors = [], []
    for fp in fps:
        try:
            chunks.extend(file_to_chunks(fp, min_chars))
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")
    return fps, chunks, errors

files, chunks, errors = load_chunks(MIN_CHARS)

# populate sidebar doc list
with st.sidebar:
    if files:
        for fp in files:
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:0.5rem;padding:0.28rem 0;
                                border-bottom:1px solid var(--border);">
                        <span style="color:var(--accent);font-family:var(--font-mono);font-size:0.7rem;">▸</span>
                        <span style="font-size:0.78rem;">{os.path.basename(fp)}</span>
                    </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<span style="color:var(--text-muted);font-size:0.78rem;">No docs loaded</span>', unsafe_allow_html=True)

if not chunks:
    st.markdown(
        '<div class="answer-card" style="border-left-color:var(--red);">Add <code>.pdf</code> / '
        '<code>.html</code> / <code>.md</code> files to <code>/docs</code> and reload.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if errors:
    with st.expander("⚠ Parse errors"):
        for e in errors:
            st.warning(e)

# ----------------------------------------
# Sparse index — word + char + title TF-IDF and BM25
# ----------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
stemmer  = SnowballStemmer("english")

def stem_analyzer(text):
    return [stemmer.stem(t) for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]

@st.cache_resource(show_spinner="Building sparse index…")
def build_sparse_index(texts, titles):
    vw = TfidfVectorizer(analyzer=stem_analyzer, ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.9,  norm="l2", lowercase=False)
    vc = TfidfVectorizer(analyzer="char",         ngram_range=(3,5), sublinear_tf=True, min_df=2, max_df=0.95, norm="l2")
    vt = TfidfVectorizer(analyzer=stem_analyzer, ngram_range=(1,2), sublinear_tf=True, min_df=1, max_df=0.95, norm="l2", lowercase=False)
    Xw = vw.fit_transform(texts)
    Xc = vc.fit_transform(texts)
    Xt = vt.fit_transform(titles) * 2.0  # title match gets a 2x boost
    Xs = hstack([Xw, Xc, Xt], format="csr")
    bm = BM25Okapi([stem_analyzer(t) for t in texts])
    return vw, vc, vt, Xs, bm

vec_word, vec_char, vec_title, X_sparse, bm25 = build_sparse_index(
    [c["text"] for c in chunks], [c["title"] for c in chunks]
)

# ----------------------------------------
# Dense FAISS index (CW2 §2.4)
# all-MiniLM-L6-v2 validated in CW2; IndexFlatIP = exact cosine on normalised vecs
# chunk_vecs retained for O(k) dense dedup — avoids the slow sparse multiply loop
# ----------------------------------------
@st.cache_resource(show_spinner="Building dense FAISS index…")
def build_dense_index(texts):
    emb  = SentenceTransformer(EMBED_MODEL_NAME)
    vecs = emb.encode(texts, batch_size=64, show_progress_bar=False,
                      normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    idx  = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return emb, idx, vecs

embedder, faiss_index, chunk_vecs = build_dense_index([c["text"] for c in chunks])

# ----------------------------------------
# Cross-encoder re-ranker (CW2 §2.5)
# ms-marco-MiniLM-L-6-v2: full attention over (query, chunk) pairs
# ----------------------------------------
@st.cache_resource(show_spinner="Loading cross-encoder…")
def load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_NAME)

cross_encoder = load_cross_encoder()

# ----------------------------------------
# Gemini client — configured at startup, no heavy model to load locally
# ----------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.markdown(
        '<div class="answer-card" style="border-left-color:var(--red);">Missing <code>GEMINI_API_KEY</code> in <code>.streamlit/secrets.toml</code></div>',
        unsafe_allow_html=True,
    )
    st.stop()

genai.configure(api_key=api_key)
_gemini = genai.GenerativeModel(GEMINI_MODEL_NAME)

# ----------------------------------------
# Retrieval
# ----------------------------------------
def _minmax(v):
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if (hi - lo) < 1e-9 else (v - lo) / (hi - lo)

def expand_query(q, prf_titles):
    # synonym expansion + pseudo-relevance feedback from top BM25 doc titles
    extras = []
    for tok in TOKEN_RE.findall(q.lower()):
        extras += SYNONYM_MAP.get(tok, [])
    expanded = q
    if extras:
        expanded += " " + " ".join(sorted(set(extras)))
    if prf_titles:
        expanded += " " + " ".join(prf_titles[:2])
    return expanded

def retrieve(query, top_k=FINAL_K):
    n = len(chunks)

    # BM25 first-pass to get PRF titles for query expansion
    bm_scores  = bm25.get_scores(stem_analyzer(query))
    prf_titles = [chunks[i]["title"] for i in bm_scores.argsort()[::-1][:10]]
    qx         = expand_query(query, prf_titles)

    # dense retrieval via FAISS (CW2 §2.4)
    q_emb = embedder.encode([qx], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    d_scores, d_idxs = faiss_index.search(q_emb, min(DENSE_CANDIDATES, n))
    s_dense = np.zeros(n, dtype=float)
    for s, i in zip(d_scores[0], d_idxs[0]):
        if 0 <= i < n:
            s_dense[i] = float(s)

    # sparse retrieval — TF-IDF + BM25
    qw = vec_word.transform([qx])
    qc = vec_char.transform([qx])
    qt = vec_title.transform([qx]) * 2.0
    s_sparse = cosine_similarity(hstack([qw, qc, qt], format="csr"), X_sparse).ravel()
    s_bm25   = bm25.get_scores(stem_analyzer(qx)).astype(float)

    # hybrid score fusion
    score     = W_DENSE*_minmax(s_dense) + W_SPARSE*_minmax(s_sparse) + W_BM25*_minmax(s_bm25)
    pool_idxs = list(score.argsort()[::-1][:FINAL_POOL])

    # cross-encoder re-ranking (CW2 §2.5) — batch scored in one call
    pairs     = [[query, chunks[i]["text"][:800]] for i in pool_idxs]
    ce_scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked    = sorted(zip(pool_idxs, ce_scores), key=lambda x: x[1], reverse=True)

    # near-duplicate filtering using dense cosine (O(k²) on small k, avoids sparse multiply)
    selected, seen_vecs = [], []
    for idx, ce_score in ranked:
        if len(selected) >= top_k:
            break
        v = chunk_vecs[idx]
        if not any(float(np.dot(v, sv)) >= DEDUP_COS_THRESH for sv in seen_vecs):
            selected.append({"chunk": chunks[idx], "ce_score": float(ce_score), "idx": idx})
            seen_vecs.append(v)

    return selected

# ----------------------------------------
# Generation (CW2 §2.6 prompt structure)
# ----------------------------------------
def llm_answer(question, retrieved):
    blocks = []
    for i, r in enumerate(retrieved[:TOP_K], 1):
        txt = r["chunk"]["text"][:1500]  # larger chunks need more chars to preserve full content
        blocks.append(f"[{i}] {r['chunk']['title']}\n{txt}")

    # strict grounding prompt — mirrors CW2 rag_prompt design
    prompt = (
        "You are a helpful and precise question-answering assistant about this portfolio.\n"
        "Answer the question using ONLY the provided source chunks below.\n"
        "Be concise and factual. Always cite relevant sources as [1], [2], etc.\n"
        "If the answer is not present in the sources, say: 'I could not find that in the available documents.'\n\n"
        f"Question: {question}\n\nSources:\n" + "\n\n".join(blocks) + "\n\nAnswer:"
    )

    r = _gemini.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=1024),
    )
    return (r.text or "").strip()

# ----------------------------------------
# UI
# ----------------------------------------

# pipeline status strip
st.markdown(
    f"""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.2rem;">
        <span class="badge badge-green">● READY</span>
        <span class="badge badge-muted">{len(files)} doc{"s" if len(files)!=1 else ""}</span>
        <span class="badge badge-muted">{len(chunks):,} chunks</span>
        <span class="badge badge-cyan">dense + sparse + BM25</span>
        <span class="badge badge-cyan">cross-encoder rerank</span>
        <span class="badge badge-muted">gemini-2.5-flash</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)
q = st.text_input(label="query_input", label_visibility="collapsed", placeholder="Ask something about the portfolio…")

if q:
    try:
        with st.spinner("Retrieving…"):
            top = retrieve(q, top_k=TOP_K)

        if not increment_quota():
            st.markdown(
                f'<div class="answer-card" style="border-left-color:var(--red);">'
                f'Daily quota of {DAILY_QUOTA} requests reached (Gemini 2.5 Flash free tier: 20 RPD). Resets tomorrow.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Generating answer…"):
                answer = llm_answer(q, top)

            st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{answer or "_No answer returned._"}</div>', unsafe_allow_html=True)

            if SHOW_DIAG:
                st.markdown('<div class="section-label">Cross-encoder scores</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(top), 8))
                for col, r in zip(cols, top):
                    col.metric(label=r["chunk"]["title"].split(" — ")[-1], value=f"{r['ce_score']:.2f}")

            if SHOW_CHUNKS:
                with st.expander("🔍 Retrieved chunks"):
                    for i, r in enumerate(top, 1):
                        ce    = r["ce_score"]
                        color = "var(--green)" if ce > 5 else "var(--accent)" if ce > 0 else "var(--text-muted)"
                        st.markdown(
                            f"""<div class="chunk-card">
                                    <div>
                                        <span class="chunk-title">[{i}] {r["chunk"]["title"]}</span>
                                        <span class="chunk-score" style="color:{color};">CE {ce:.3f}</span>
                                    </div>
                                    <div style="clear:both;margin-top:0.3rem;color:var(--text);font-size:0.82rem;line-height:1.5;">
                                        {r["chunk"]["text"][:400]}{"…" if len(r["chunk"]["text"])>400 else ""}
                                    </div>
                                </div>""",
                            unsafe_allow_html=True,
                        )

            with st.expander("📚 Sources"):
                for i, r in enumerate(top, 1):
                    c = r["chunk"]
                    st.markdown(
                        f"""<div class="source-row">
                                <span class="source-idx">[{i}]</span>
                                <span style="color:var(--text);font-size:0.82rem;">{c["title"]}</span>
                                <a class="source-link" href="{c["href"]}">{c["href"]}</a>
                            </div>""",
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.markdown(
            f'<div class="answer-card" style="border-left-color:var(--red);"><strong style="color:var(--red);">Error:</strong> {e}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Traceback"):
            st.code(traceback.format_exc(), language="python")