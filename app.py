# ---- Portfolio RAG — Streamlit app ----
# ----------------------------------------
import os, re, glob, json, datetime, traceback, base64
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
    page_title="Josh Le Grice — Portfolio RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── helper: load local image as base64 data URI ─────────────────────────────
def _img_to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext  = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return ""

PROFILE_IMG = _img_to_b64("attachments/profile.jpeg")
PROFILE_TAG = (
    f'<img class="hero-photo" src="{PROFILE_IMG}" alt="Josh Le Grice" />'
    if PROFILE_IMG else
    '<div class="hero-photo-placeholder">JLG</div>'
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Design tokens ─────────────────────────────────────────────────── */
    :root {
        --bg:         #080e18;
        --bg2:        #0c1420;
        --card:       #0f1b2d;
        --card2:      #111f32;
        --ink:        #e4ddd0;
        --ink2:       #b8b0a2;
        --muted:      #5a7080;
        --border:     rgba(232,226,214,.08);
        --border2:    rgba(232,226,214,.14);
        --gold:       #b8922a;
        --gold2:      #d4a843;
        --gold-tint:  rgba(184,146,42,.10);
        --gold-glow:  rgba(184,146,42,.20);
        --navy:       #0a1525;
        --red:        #c0564a;
        --green:      #3fcf8e;
        --serif:      'DM Serif Display', Georgia, serif;
        --sans:       'DM Sans', ui-sans-serif, system-ui, sans-serif;
        --mono:       'DM Mono', ui-monospace, Menlo, monospace;
        --radius:     4px;
        --shadow:     0 4px 28px rgba(0,0,0,.5);
        --shadow-sm:  0 1px 6px rgba(0,0,0,.3);
    }

    /* ── Base reset ────────────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }

    html, body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    .main .block-container,
    [data-testid="stMainBlockContainer"] {
        background: var(--bg) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
    }

    /* header & decoration stripe */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: var(--bg) !important;
        border-bottom: 1px solid var(--border) !important;
    }
    [data-testid="stDecoration"] {
        background: var(--gold) !important;
        background-image: none !important;
        height: 2px !important;
    }
    #MainMenu, footer { visibility: hidden !important; }

    /* hide sidebar collapse/expand arrow button */
    [data-testid="collapsedControl"],
    button[kind="header"],
    [data-testid="stSidebarCollapseButton"],
    button[aria-label="Collapse sidebar"],
    button[aria-label="Expand sidebar"] {
        display: none !important;
    }

    .main .block-container {
        padding: 36px 28px 80px 28px !important;
        max-width: 1060px !important;
    }

    /* ── Typography ────────────────────────────────────────────────────── */
    h1 {
        font-family: var(--serif) !important; font-size: 2.2rem !important;
        font-weight: 400 !important; color: var(--ink) !important;
        letter-spacing: -.01em !important; margin: 0 0 .4rem !important;
        line-height: 1.1 !important;
    }
    h2 { font-family: var(--serif) !important; font-size: 1.3rem !important; font-weight: 400 !important; color: var(--ink) !important; margin: 0 !important; }
    h3 { font-family: var(--serif) !important; font-size: 1rem !important; font-weight: 400 !important; color: var(--ink) !important; }

    p, li, span, label,
    [data-testid="stMarkdownContainer"] p {
        font-family: var(--sans) !important;
        color: var(--ink2) !important;
        font-size: .92rem !important;
        line-height: 1.65 !important;
    }

    /* ── Sidebar ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebarNav"] {
        background: var(--bg2) !important;
        border-right: 1px solid var(--border2) !important;
    }
    /* uniform text colour in sidebar */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: var(--ink2) !important;
        font-family: var(--sans) !important;
    }
    /* sidebar section headings */
    .sb-heading {
        font-family: var(--mono) !important;
        font-size: .67rem; font-weight: 500;
        letter-spacing: .13em; text-transform: uppercase;
        color: var(--gold) !important;
        border-bottom: 1px solid var(--border2);
        padding-bottom: 8px;
        margin: 26px 0 14px 0;
    }
    .sb-heading:first-child { margin-top: 6px; }

    /* sidebar slider & checkbox colours */
    [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
        background: var(--gold) !important; border-color: var(--gold) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] div[role="progressbar"] {
        background: var(--gold) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] *,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] * {
        color: var(--ink2) !important;
    }

    /* ── Text input ──────────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    [data-testid="stTextInput"] input {
        background: var(--card2) !important;
        color: var(--ink) !important;
        border: 1px solid var(--border2) !important;
        border-radius: var(--radius) !important;
        font-family: var(--sans) !important;
        font-size: .95rem !important;
        padding: .6rem 1rem !important;
        transition: border-color .15s, box-shadow .15s;
    }
    .stTextInput > div > div > input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 3px var(--gold-glow) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--muted) !important; }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button,
    .stDownloadButton > button {
        background: transparent !important; color: var(--gold) !important;
        border: 1px solid var(--gold) !important; border-radius: var(--radius) !important;
        font-family: var(--mono) !important; font-size: .72rem !important;
        letter-spacing: .07em !important; text-transform: uppercase !important;
        padding: .38rem 1.1rem !important; transition: background .15s, color .15s;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover { background: var(--gold) !important; color: var(--navy) !important; }



    /* ── Expander ────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--card) !important; border: 1px solid var(--border2) !important;
        border-radius: var(--radius) !important; margin-top: .5rem !important;
    }
    [data-testid="stExpander"] summary {
        background: var(--card) !important; color: var(--gold) !important;
        font-family: var(--mono) !important; font-size: .72rem !important;
        letter-spacing: .08em !important; text-transform: uppercase !important;
        padding: .55rem 1rem !important;
    }
    [data-testid="stExpander"] summary:hover { background: var(--card2) !important; }
    [data-testid="stExpander"] > div > div { background: var(--card) !important; padding: .8rem 1rem !important; }

    /* ── Metrics ─────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--card) !important; border: 1px solid var(--border2) !important;
        border-radius: var(--radius) !important; padding: .7rem 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important; font-family: var(--mono) !important;
        font-size: .67rem !important; text-transform: uppercase !important; letter-spacing: .09em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--gold) !important; font-family: var(--mono) !important;
        font-size: 1.4rem !important; font-weight: 500 !important;
    }

    /* ── Alerts & spinner ────────────────────────────────────────────────── */
    [data-testid="stAlert"], .stAlert {
        background: var(--card) !important; border-left: 3px solid var(--gold) !important;
        border-radius: var(--radius) !important; color: var(--ink) !important; font-size: .88rem !important;
    }
    [data-testid="stSpinner"] > div { color: var(--gold) !important; }

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold); }

    /* ── Markdown content (answer) ───────────────────────────────────────── */
    [data-testid="stMarkdownContainer"] ul { padding-left: 1.2rem; margin: .4rem 0; }
    [data-testid="stMarkdownContainer"] li { color: var(--ink2) !important; margin-bottom: .2rem; }
    [data-testid="stMarkdownContainer"] strong { color: var(--gold) !important; }
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: var(--ink) !important; font-family: var(--serif) !important;
        font-size: 1.05rem !important; font-weight: 400 !important;
        letter-spacing: 0 !important; text-transform: none !important;
        margin: 1rem 0 .3rem !important;
    }
    [data-testid="stMarkdownContainer"] code {
        background: var(--card2); color: var(--gold2);
        padding: .1rem .35rem; border-radius: 3px;
        font-family: var(--mono); font-size: .83rem;
    }
    hr { border-color: var(--border) !important; margin: .6rem 0 !important; }

    /* ── Hero ────────────────────────────────────────────────────────────── */
    .hero-photo {
        width: 118px; height: 118px; border-radius: var(--radius);
        object-fit: cover; border: 1px solid var(--border2);
        box-shadow: var(--shadow); display: block; flex-shrink: 0;
    }
    .hero-photo-placeholder {
        width: 118px; height: 118px; border-radius: var(--radius);
        background: var(--card2); border: 1px solid var(--border2);
        display: grid; place-items: center; flex-shrink: 0;
        font-family: var(--serif); font-size: 1.5rem; color: var(--gold);
    }
    .hero-eyebrow {
        font-size: .7rem; font-weight: 500; letter-spacing: .14em;
        text-transform: uppercase; color: var(--gold); margin: 0 0 6px;
    }
    .hero-name {
        font-family: var(--serif);
        font-size: clamp(1.7rem, 3.5vw, 2.5rem);
        font-weight: 400; line-height: 1.08; color: var(--ink); margin: 0 0 9px;
    }
    .hero-sub {
        color: var(--muted); font-size: .93rem; line-height: 1.7; margin: 0 0 16px;
    }
    .hero-btn {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 10px 22px; font-size: .87rem;
        font-family: var(--sans); font-weight: 500;
        letter-spacing: .03em; border-radius: var(--radius);
        text-decoration: none; transition: .15s;
    }
    .hero-btn.outline { background: transparent; color: var(--ink); border: 1px solid var(--border2); }
    .hero-btn.outline:hover { border-color: var(--gold); color: var(--gold); }
    .hero-btn.filled { background: var(--gold); color: var(--navy); border: 1px solid var(--gold); font-weight: 600; }
    .hero-btn.filled:hover { background: var(--gold2); border-color: var(--gold2); }
    .statbar { display: flex; gap: 24px; padding-top: 13px; border-top: 1px solid var(--border); }
    .stat .k { font-family: var(--serif); font-size: 1.3rem; line-height: 1; color: var(--ink); }
    .stat .t { font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-top: 2px; }

    /* ── Divider ─────────────────────────────────────────────────────────── */
    .rule { display: flex; align-items: center; gap: 12px; margin: 26px 0; }
    .rule::before, .rule::after { content: ''; flex: 1; height: 1px; background: var(--border); }
    .rule-diamond { width: 5px; height: 5px; background: var(--gold); transform: rotate(45deg); flex-shrink: 0; }

    /* ── Section label ───────────────────────────────────────────────────── */
    .section-label {
        font-family: var(--mono); font-size: .67rem; font-weight: 500;
        color: var(--gold); letter-spacing: .13em; text-transform: uppercase;
        margin-bottom: .5rem; margin-top: 1.2rem;
    }

    /* ── Page title + badges ─────────────────────────────────────────────── */
    .rag-title {
        font-family: var(--serif) !important;
        font-size: clamp(2rem, 3vw, 2.5rem) !important;
        font-weight: 400 !important;
        color: var(--ink) !important;
        line-height: 1.05 !important;
        letter-spacing: -.02em !important;
        display: block !important;
        margin: 0 0 12px 0 !important;
    }
    .rag-badge {
        display: inline-block; font-family: var(--mono); font-size: .65rem;
        padding: .18rem .58rem; border-radius: var(--radius);
        letter-spacing: .07em; text-transform: uppercase; vertical-align: middle;
    }
    .b-gold  { background: var(--gold-tint); color: var(--gold2); border: 1px solid rgba(184,146,42,.28); }
    .b-muted { background: rgba(90,112,128,.12); color: var(--muted); border: 1px solid var(--border2); }
    .b-green { background: rgba(63,207,142,.10); color: var(--green); border: 1px solid rgba(63,207,142,.35); }

    .status-strip {
        display: flex; gap: .4rem; flex-wrap: wrap;
        align-items: center; margin-bottom: 1.4rem;
    }

    /* ── Answer card ─────────────────────────────────────────────────────── */
    .answer-card {
        background: var(--card); border: 1px solid var(--border2);
        border-left: 3px solid var(--gold); border-radius: var(--radius);
        padding: 1.1rem 1.3rem; margin: .6rem 0;
        font-family: var(--sans); font-size: .95rem; line-height: 1.7; color: var(--ink2);
    }

    /* ── Chunk card ──────────────────────────────────────────────────────── */
    .chunk-card {
        background: var(--bg2); border: 1px solid var(--border);
        border-radius: var(--radius); padding: .7rem 1rem;
        margin-bottom: .45rem; font-size: .86rem; line-height: 1.5;
    }
    .chunk-title { font-family: var(--mono); color: var(--gold); font-size: .69rem; letter-spacing: .05em; margin-bottom: .3rem; }
    .chunk-score { font-family: var(--mono); font-size: .67rem; float: right; }

    /* ── Source row ──────────────────────────────────────────────────────── */
    .source-row {
        display: flex; align-items: center; gap: .6rem;
        padding: .38rem 0; border-bottom: 1px solid var(--border); font-size: .87rem;
    }
    .source-idx { font-family: var(--mono); color: var(--gold); min-width: 1.6rem; font-size: .69rem; }

    /* ── Sidebar doc row ─────────────────────────────────────────────────── */
    .doc-row {
        display: flex; align-items: center; gap: .5rem;
        padding: .28rem 0; border-bottom: 1px solid var(--border);
    }
    .doc-bullet { color: var(--gold); font-family: var(--mono); font-size: .69rem; }
    .doc-name   { font-size: .82rem !important; color: var(--ink2) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHUNK_WORDS       = 800
CHUNK_OVERLAP_W   = 100
MIN_CHARS_DEFAULT = 60

W_DENSE  = 0.50
W_SPARSE = 0.30
W_BM25   = 0.20

DENSE_CANDIDATES = 64
FINAL_POOL       = 32
FINAL_K          = 6
DEDUP_COS_THRESH = 0.90

EMBED_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL_NAME  = "gemini-2.5-flash"

DAILY_QUOTA = 20
RPM_LIMIT   = 5
QUOTA_FILE  = ".gemini_quota.json"

SYNONYM_MAP = {
    "cv":           ["resume", "résumé"],
    "llm":          ["language model", "genai", "foundation model"],
    "genai":        ["generative ai", "foundation model"],
    "streamlit":    ["python web app", "web app"],
    "nlp":          ["natural language processing"],
    "retrieval":    ["rag", "document search"],
    "rag":          ["retrieval augmented generation", "retrieval"],
    "internship":   ["placement"],
    "dissertation": ["The Intersection of Machine Learning and Type 1 Diabetes"],
}

# ── QUOTA HELPERS ─────────────────────────────────────────────────────────────
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

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-heading">⚙ Settings</div>', unsafe_allow_html=True)
    TOP_K       = st.slider("Top-k chunks", 1, 10, FINAL_K)
    SHOW_CHUNKS = st.checkbox("Show retrieved chunks", False)
    SHOW_DIAG   = st.checkbox("Show CE score metrics", False)

    st.markdown('<div class="sb-heading">◈ Daily Quota</div>', unsafe_allow_html=True)
    rem     = remaining_quota()
    pct     = int(rem / DAILY_QUOTA * 100)
    bar_col = "var(--gold)" if pct > 30 else "var(--red)"
    st.markdown(
        f"""
        <div style="font-family:var(--mono);font-size:.66rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.11em;margin-bottom:4px;">
            Remaining today
        </div>
        <div style="font-family:var(--mono);font-size:1.45rem;font-weight:500;
                    color:{bar_col};margin-bottom:6px;">
            {rem} <span style="font-size:.73rem;color:var(--muted);">/ {DAILY_QUOTA}</span>
        </div>
        <div style="background:var(--bg);border:1px solid var(--border2);
                    border-radius:3px;height:4px;overflow:hidden;">
            <div style="width:{pct}%;height:4px;
                        background:linear-gradient(90deg,var(--gold),var(--gold2));
                        border-radius:3px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-heading">◈ Documents</div>', unsafe_allow_html=True)
    doc_placeholder = st.empty()


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex;align-items:flex-start;gap:28px;margin-bottom:18px;">
        {PROFILE_TAG}
        <div style="flex:1;min-width:0;">
            <div class="hero-eyebrow">MSc Data Science · University of Exeter</div>
            <div class="hero-name">Josh Le Grice</div>
            <div class="hero-sub">Quantitative developer and data scientist focused on machine
            learning, deep learning, and financial modelling.</div>
            <div style="margin-bottom:14px;">
                <a class="hero-btn outline"
                   href="https://joshlg18.github.io/PortfolioWebsite/"
                   target="_blank" rel="noopener">View Projects ↗</a>
            </div>
            <div class="statbar">
                <div class="stat"><div class="k">9</div><div class="t">Projects</div></div>
                <div class="stat"><div class="k">3</div><div class="t">Featured</div></div>
                <div class="stat"><div class="k">1</div><div class="t">Internship</div></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="rule"><div class="rule-diamond"></div></div>', unsafe_allow_html=True)

# ── RAG TITLE ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin:4px 0 12px 0;">
        <span class="rag-title" style="font-family:'DM Serif Display',Georgia,serif !important; clamp(1.8rem, 3vw, 2.8rem) !important;font-weight:400 !important;color:#e4ddd0 !important;letter-spacing:-.02em !important;display:block !important;">Portfolio RAG</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── TEXT EXTRACTION ───────────────────────────────────────────────────────────
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
    ".pdf":      _extract_text_pdf,
    ".html":     _extract_text_html,
    ".htm":      _extract_text_html,
    ".md":       _extract_text_md,
    ".markdown": _extract_text_md,
}

# ── CHUNKING ──────────────────────────────────────────────────────────────────
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

# ── LOAD DOCS ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chunking documents…")
def load_chunks(min_chars):
    fps = [fp for fp in glob.glob("docs/*")
           if os.path.splitext(fp)[1].lower() in EXTRACTORS]
    chunks, errors = [], []
    for fp in fps:
        try:
            chunks.extend(file_to_chunks(fp, min_chars))
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")
    return fps, chunks, errors

# placeholder shown while indexes build — replaced with green READY after all @cache calls complete
status_placeholder = st.empty()
status_placeholder.markdown(
    '<div class="status-strip"><span class="rag-badge b-muted">⟳ Loading…</span></div>',
    unsafe_allow_html=True,
)

files, chunks, errors = load_chunks(MIN_CHARS_DEFAULT)

# fill sidebar doc list
with doc_placeholder.container():
    if files:
        for fp in files:
            st.markdown(
                f'<div class="doc-row">'
                f'<span class="doc-bullet">▸</span>'
                f'<span class="doc-name">{os.path.basename(fp)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span style="font-size:.82rem;color:var(--muted);">No docs loaded</span>',
            unsafe_allow_html=True,
        )

# status strip — updated to green READY once all indexes are loaded
# (will be set after cross-encoder loads below)

if not chunks:
    st.markdown(
        '<div class="answer-card" style="border-left-color:var(--red);">'
        'Add <code>.pdf</code> / <code>.html</code> / <code>.md</code> '
        'files to <code>/docs</code> and reload.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if errors:
    with st.expander("⚠ Parse errors"):
        for e in errors:
            st.warning(e)

# ── SPARSE INDEX ──────────────────────────────────────────────────────────────
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
    Xt = vt.fit_transform(titles) * 2.0
    Xs = hstack([Xw, Xc, Xt], format="csr")
    bm = BM25Okapi([stem_analyzer(t) for t in texts])
    return vw, vc, vt, Xs, bm

vec_word, vec_char, vec_title, X_sparse, bm25 = build_sparse_index(
    [c["text"] for c in chunks], [c["title"] for c in chunks]
)

# ── DENSE FAISS INDEX ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building dense FAISS index…")
def build_dense_index(texts):
    emb  = SentenceTransformer(EMBED_MODEL_NAME)
    vecs = emb.encode(texts, batch_size=64, show_progress_bar=False,
                      normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    idx  = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return emb, idx, vecs

embedder, faiss_index, chunk_vecs = build_dense_index([c["text"] for c in chunks])

# ── CROSS-ENCODER ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading cross-encoder…")
def load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_NAME)

cross_encoder = load_cross_encoder()

# All indexes loaded — update status strip to green READY
status_placeholder.markdown(
    '<div class="status-strip"><span class="rag-badge b-green" '    'style="font-size:.78rem;padding:.25rem .8rem;letter-spacing:.1em;">● &nbsp;READY</span></div>',
    unsafe_allow_html=True,
)

# ── GEMINI CLIENT ─────────────────────────────────────────────────────────────
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.markdown(
        '<div class="answer-card" style="border-left-color:var(--red);">'
        'Missing <code>GEMINI_API_KEY</code> in <code>.streamlit/secrets.toml</code></div>',
        unsafe_allow_html=True,
    )
    st.stop()

genai.configure(api_key=api_key)
_gemini = genai.GenerativeModel(GEMINI_MODEL_NAME)

# ── RETRIEVAL ─────────────────────────────────────────────────────────────────
def _minmax(v):
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if (hi - lo) < 1e-9 else (v - lo) / (hi - lo)

def expand_query(q, prf_titles):
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
    n          = len(chunks)
    bm_scores  = bm25.get_scores(stem_analyzer(query))
    prf_titles = [chunks[i]["title"] for i in bm_scores.argsort()[::-1][:10]]
    qx         = expand_query(query, prf_titles)

    q_emb = embedder.encode([qx], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    d_scores, d_idxs = faiss_index.search(q_emb, min(DENSE_CANDIDATES, n))
    s_dense = np.zeros(n, dtype=float)
    for s, i in zip(d_scores[0], d_idxs[0]):
        if 0 <= i < n:
            s_dense[i] = float(s)

    qw       = vec_word.transform([qx])
    qc       = vec_char.transform([qx])
    qt       = vec_title.transform([qx]) * 2.0
    s_sparse = cosine_similarity(hstack([qw, qc, qt], format="csr"), X_sparse).ravel()
    s_bm25   = bm25.get_scores(stem_analyzer(qx)).astype(float)

    score     = W_DENSE*_minmax(s_dense) + W_SPARSE*_minmax(s_sparse) + W_BM25*_minmax(s_bm25)
    pool_idxs = list(score.argsort()[::-1][:FINAL_POOL])

    pairs     = [[query, chunks[i]["text"][:800]] for i in pool_idxs]
    ce_scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked    = sorted(zip(pool_idxs, ce_scores), key=lambda x: x[1], reverse=True)

    selected, seen_vecs = [], []
    for idx, ce_score in ranked:
        if len(selected) >= top_k:
            break
        v = chunk_vecs[idx]
        if not any(float(np.dot(v, sv)) >= DEDUP_COS_THRESH for sv in seen_vecs):
            selected.append({"chunk": chunks[idx], "ce_score": float(ce_score), "idx": idx})
            seen_vecs.append(v)

    return selected

# ── GENERATION ────────────────────────────────────────────────────────────────
def llm_answer(question, retrieved):
    blocks = []
    for i, r in enumerate(retrieved[:TOP_K], 1):
        txt = r["chunk"]["text"][:2000]
        blocks.append(f"[Source {i}] {r['chunk']['title']}\n{txt}")

    system = (
        "You are an expert portfolio assistant for Josh Le Grice, a Computer Science student and developer.\n"
        "Your job is to give rich, detailed, enthusiastic answers about Josh's work, skills, and experience.\n\n"
        "Rules:\n"
        "- Answer fully and in depth — never cut off mid-sentence or give a one-liner\n"
        "- Synthesise information across all provided sources\n"
        "- Use markdown formatting: **bold** for key terms, bullet points for lists, headings for long answers\n"
        "- Cite sources as [1], [2] etc. inline\n"
        "- Write in third person about Josh (e.g. 'Josh built...', 'His project...')\n"
        "- If asked about a specific project, cover: what it does, technologies used, motivation, outcomes\n"
        "- Only say information is unavailable if it is genuinely absent from ALL sources\n"
        "- Never make up information that isn't in the sources — if you don't know, say you don't know"
        "- Be enthusiastic and positive in tone, highlighting Josh's strengths and achievements but professional"
        "- Be concise and avoid unnecessary filler words or repetition, but don't sacrifice depth and detail"
    )

    prompt = (
        f"{system}\n"
        f"Question: {question}\n\n"
        f"Sources:\n" + "\n\n".join(blocks) +
        "\n\nProvide a thorough, well-structured answer:"
    )

    result = _gemini.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.3, top_p=0.95),
    )
    return (result.text or "").strip()

# ── QUERY UI ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)
q = st.text_input(
    label="query_input",
    label_visibility="collapsed",
    placeholder="Ask something about the portfolio…",
)

if q:
    try:
        with st.spinner("Retrieving…"):
            top = retrieve(q, top_k=TOP_K)

        if not increment_quota():
            st.markdown(
                f'<div class="answer-card" style="border-left-color:var(--red);">'
                f'Daily quota of {DAILY_QUOTA} requests reached '
                f'(Gemini 2.5 Flash free tier: 20 RPD). Resets tomorrow.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Generating answer…"):
                answer = llm_answer(q, top)

            st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(
                """<style>
                div[data-testid="stVerticalBlock"]:has(>div[data-testid="stMarkdownContainer"]>.answer-wrap) {
                    background: var(--card);
                    border: 1px solid var(--border2);
                    border-left: 3px solid var(--gold);
                    border-radius: var(--radius);
                    padding: 1.1rem 1.3rem;
                    margin: .4rem 0 1rem;
                }
                </style>""",
                unsafe_allow_html=True,
            )
            st.markdown('<span class="answer-wrap"></span>', unsafe_allow_html=True)
            st.markdown(answer or "_No answer returned._")

            if SHOW_DIAG:
                st.markdown('<div class="section-label">Cross-Encoder Scores</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(top), 8))
                for col, r in zip(cols, top):
                    col.metric(
                        label=r["chunk"]["title"].split(" — ")[-1],
                        value=f"{r['ce_score']:.2f}",
                    )

            if SHOW_CHUNKS:
                with st.expander("🔍 Retrieved Chunks"):
                    for i, r in enumerate(top, 1):
                        ce    = r["ce_score"]
                        color = "var(--green)" if ce > 5 else "var(--gold)" if ce > 0 else "var(--muted)"
                        st.markdown(
                            f"""<div class="chunk-card">
                                <div>
                                    <span class="chunk-title">[{i}] {r["chunk"]["title"]}</span>
                                    <span class="chunk-score" style="color:{color};">CE {ce:.3f}</span>
                                </div>
                                <div style="clear:both;margin-top:.3rem;color:var(--ink2);
                                            font-size:.85rem;line-height:1.5;">
                                    {r["chunk"]["text"][:400]}{"…" if len(r["chunk"]["text"]) > 400 else ""}
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            with st.expander("📚 Sources"):
                seen_files = []
                for i, res in enumerate(top, 1):
                    c         = res["chunk"]
                    fname     = os.path.basename(c["href"])
                    doc_title = c["title"].split(" — chunk")[0]
                    if fname in seen_files:
                        continue
                    seen_files.append(fname)
                    safe_title = doc_title.replace("<", "&lt;").replace(">", "&gt;")
                    safe_fname = fname.replace("<", "&lt;").replace(">", "&gt;")
                    st.markdown(
                        f'<div class="source-row">'
                        f'<span class="source-idx">[{len(seen_files)}]</span>'
                        f'<span style="color:var(--ink);font-size:.86rem;font-weight:500;">{safe_title}</span>'
                        f'<span style="color:var(--muted);font-family:var(--mono);'
                        f'font-size:.69rem;margin-left:auto;">{safe_fname}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.markdown(
            f'<div class="answer-card" style="border-left-color:var(--red);">'
            f'<strong style="color:var(--red);">Error:</strong> {e}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Traceback"):
            st.code(traceback.format_exc(), language="python")