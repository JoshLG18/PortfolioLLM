# app.py — Portfolio Q&A (sparse hybrid + Gemini re-rank; no sentence-transformers)

import os, re, glob, json, datetime, traceback
from typing import List, Dict
import streamlit as st

# ---------- Readers ----------
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ---------- Retrieval (sparse) ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# ---------- BM25 ----------
from rank_bm25 import BM25Okapi

# ---------- Stemming ----------
from nltk.stem.snowball import SnowballStemmer

# ---------- Gemini ----------
import google.generativeai as genai


# ==========================================================
# Config
# ==========================================================
DAILY_QUOTA = 1500
QUOTA_FILE = ".gemini_quota.json"
MODEL_NAME = "gemini-2.0-flash-lite"

# Retrieval constants tuned for accuracy
MIN_CHARS_DEFAULT = 1000
MAX_CHARS = 4000
OVERLAP_SENTS = 1

# Hybrid weights and sizes
W_SPARSE = 0.65
W_BM25   = 0.35
CANDIDATE_POOL = 128         # how many to retrieve before re-ranking
FINAL_K = 128                  # how many chunks to send to Gemini for answering
DEDUP_THRESHOLD = 0.90       # char-tfidf cosine for near-duplicate filtering

# Query expansion (extend for your domain)
SYNONYM_MAP = {
    "cv": ["resume", "résumé"],
    "llm": ["language model", "genai", "foundation model"],
    "genai": ["generative ai", "foundation model"],
    "streamlit": ["python web app", "web app"],
    "nlp": ["natural language processing"],
    "retrieval": ["rag", "document search"],
    "rag": ["retrieval augmented generation", "retrieval"],
    "internship": ["placement"],
    "dissertation": ["The Intersection of Machine Learning and Type 1 Diabetes: A Critical Analysis of Machine Learning Driven Innovations"]
}


st.set_page_config(page_title="Ask My Portfolio", page_icon="💬", layout="wide")
st.title("💬 Ask My Portfolio")

with st.sidebar:
    st.header("⚙️ Settings")
    TOP_K = st.slider("Top-k chunks (sent to Gemini)", 3, 128, FINAL_K)
    MIN_CHARS = st.slider("Min chunk length (chars)", 60, 2000, MIN_CHARS_DEFAULT, step=20)
    SHOW_CHUNKS = st.checkbox("Show final chunks", False)


# ==========================================================
# Quota tracking
# ==========================================================
def _load_quota(): # Load in the requests quota
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

def remaining_quota() -> int: # work out the remaining quota for the day
    return max(0, DAILY_QUOTA - _load_quota()["count"])

def check_and_increment_quota(n_calls: int = 1) -> bool: # increment the quota when requests are made
    """Reserve n_calls; return True if allowed and persist."""
    data = _load_quota()
    if data["count"] + n_calls > DAILY_QUOTA:
        return False
    data["count"] += n_calls
    try:
        with open(QUOTA_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    return True

with st.sidebar: # show the qouta on the sidebar
    st.markdown(f"**🔒 Daily quota left:** {remaining_quota()} / {DAILY_QUOTA}")


# ==========================================================
# Utilities
# ==========================================================
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+") 
stemmer = SnowballStemmer("english")

def sent_tokenize(text: str) -> List[str]: # sentence splitter based on punctuation
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def windowed_chunks(sentences: List[str], min_chars=120, max_chars=900, overlap_sents=1) -> List[str]: # create chunks using the tokenised sentences
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

def read_pdf_chunks(path: str, min_chars: int): # reads a pdf
    out = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for p_i, p in enumerate(reader.pages, 1):
            txt = p.extract_text() or ""
            sents = sent_tokenize(txt)
            for j, ch in enumerate(windowed_chunks(sents, min_chars=min_chars, max_chars=MAX_CHARS, overlap_sents=OVERLAP_SENTS), 1):
                out.append({"title": f"{os.path.basename(path)} — p{p_i}.{j}", "text": ch, "href": path})
    return out

def read_html_chunks(path: str, min_chars: int): # reads html
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    out = []
    for i, para in enumerate([p for p in re.split(r"\n\s*\n", text) if p.strip()], 1):
        sents = sent_tokenize(para)
        for j, ch in enumerate(windowed_chunks(sents, min_chars=min_chars, max_chars=MAX_CHARS, overlap_sents=OVERLAP_SENTS), 1):
            out.append({"title": f"{os.path.basename(path)} — ¶{i}.{j}", "text": ch, "href": path})
    return out

def read_md_chunks(path: str, min_chars: int): # reads markdown files
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    out = []
    for i, para in enumerate([p for p in re.split(r"\n\s*\n", text) if p.strip()], 1):
        sents = sent_tokenize(para)
        for j, ch in enumerate(windowed_chunks(sents, min_chars=min_chars, max_chars=MAX_CHARS, overlap_sents=OVERLAP_SENTS), 1):
            out.append({"title": f"{os.path.basename(path)} — ¶{i}.{j}", "text": ch, "href": path})
    return out

READERS = { # defines a dictionary for which function to use depending on the file being read
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
def load_chunks(min_chars: int): # read supported files from docs and return the files, chunks and any errors
    files = [fp for fp in glob.glob("docs/*") if os.path.splitext(fp)[1].lower() in READERS]
    chunks, errors = [], []
    for fp in files:
        try:
            chunks.extend(READERS[os.path.splitext(fp)[1].lower()](fp, min_chars))
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")
    return files, chunks, errors

files, chunks, errors = load_chunks(MIN_CHARS)

if not chunks: # if there are no chunks return an error message
    st.info("Put .pdf/.html/.md files in `/docs` then reload.")
    st.stop()

if errors: # if there are errors with the parsing print an error
    with st.expander("Parse errors"):
        for e in errors:
            st.warning(e)


# ==========================================================
# Robust sparse index (word + char + title) and BM25
# ==========================================================
def stem_analyzer(text: str):
    tokens = TOKEN_RE.findall(text.lower()) # splits up words and makes them lower case
    return [stemmer.stem(t) #returns the root form of some of the words
             for t in tokens if len(t) > 1] # removes any words that are only one letter

@st.cache_resource(show_spinner=True)
def build_sparse_and_bm25(texts: List[str], titles: List[str]):
    # Word TF-IDF
    vec_word = TfidfVectorizer( # builds a word level vectoriser
        analyzer=stem_analyzer, # uses the custom tokeniser
        ngram_range=(1, 2), # extracts both uni and bigrams
        sublinear_tf=True, # uses log scaling so frequently used words don't skew
        min_df=2, max_df=0.9, # ignores words that are used in less than 2 docs and more than 90% of docs
        norm="l2", # normalises each vector
        lowercase=False,
        max_features=None
    )
    # Char TF-IDF
    vec_char = TfidfVectorizer( # builds a character level vectoriser 
        analyzer="char", # splits sentences into sequences of characters
        ngram_range=(3, 5), # characters range from 3 to 5 in length
        sublinear_tf=True, # uses log scaling to stop frequency skew
        min_df=2, max_df=0.95, # ignores words that are used in less than 2 docs and more than 90% of docs
        norm="l2" # normalises each vector
    )
    # Title TF-IDF (boosted)
    vec_title = TfidfVectorizer( # creates a title vectoriser
        analyzer=stem_analyzer,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1, max_df=0.95,
        norm="l2", lowercase=False
    )

    X_word  = vec_word.fit_transform(texts) # creates a sparse matrix of words
    X_char  = vec_char.fit_transform(texts) # creates a sparse matrix of characters
    X_title = vec_title.fit_transform(titles) * 2.0 # creates a sparse matrix of titles and increases the magnitude
    X_sparse = hstack([X_word, X_char, X_title], format="csr") # horizontally stacks features into one large sparse matrix

    # BM25 corpus (stemmed)
    tokenized = [stem_analyzer(t) for t in texts] # tokenises each chunk
    bm25 = BM25Okapi(tokenized) # creates a score of how similar a chunk is to the query
    return (vec_word, vec_char, vec_title, X_sparse, X_char, bm25) # return each vector and the scores

vec_word, vec_char, vec_title, X_sparse, X_char, bm25 = build_sparse_and_bm25(
    [c["text"] for c in chunks],
    [c["title"] for c in chunks]
)

def normalize(v):
    import numpy as np
    v = np.asarray(v, dtype=float) # convert np array to floats
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-9:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin) # min max normalisation


# ==========================================================
# Retrieval: hybrid sparse + Gemini re-rank + dedup
# ==========================================================
def expand_query(q: str, prf_titles: List[str]) -> str:
    extras = []
    for raw in TOKEN_RE.findall(q.lower()): # splits the query into alpha numeric tokens
        extras += SYNONYM_MAP.get(raw, []) # looks up all the synonyms in the query
    prf = " ".join(prf_titles[:2]) if prf_titles else "" # takes top 5 most relevant document titles
    expanded = q
    if extras:
        expanded += " " + " ".join(sorted(set(extras)))
    if prf:
        expanded += " " + prf # expands the query to include all synonyms
    return expanded

def retrieve_candidates(query: str, pool: int) -> List[int]:
    # First-pass BM25 to harvest titles for PRF
    bm_first = bm25.get_scores(stem_analyzer(query)) # scores all the chunks with bm25
    bm_top = bm_first.argsort()[::-1][:10] # takes top 10 scoring chunks
    prf_titles = [chunks[i]["title"] for i in bm_top] # extracts their titles

    qx = expand_query(query, prf_titles) # expands the query and adds top 5 titles to the query

    # Sparse cosine
    qw = vec_word.transform([qx])
    qc = vec_char.transform([qx])
    qt = vec_title.transform([qx]) * 2.0
    q_sparse = hstack([qw, qc, qt], format="csr")
    s_sparse = cosine_similarity(q_sparse, X_sparse).ravel()

    # BM25
    s_bm25 = bm25.get_scores(stem_analyzer(qx)).astype(float) # runs bm25 but on the expanded query

    # Hybrid
    score = W_SPARSE*normalize(s_sparse) + W_BM25*normalize(s_bm25) # balances semantic and lexical matches
    cand = score.argsort()[::-1][:pool] # sorts chunks by hybrid score
    return list(cand)

def gemini_rerank(question: str, cand_idxs: List[int], top_k: int) -> List[int]:
    # Prepare compact chunks for ranking
    lines, mapping = [], []
    for rank, i in enumerate(cand_idxs, 1):
        txt = chunks[i]["text"]
        if len(txt) > 800: txt = txt[:800] + " ..."
        lines.append(f"[{rank}] {chunks[i]['title']}\n{txt}")
        mapping.append(i)  # map local rank -> global index

    context = "\n\n".join(lines)
    instruction = (
        "You are a re-ranker. Given a question and N document chunks, "
        "return the N indices ordered from most relevant to least. "
        "Only output a comma-separated list of integers (no text). "
        "Relevance means the chunk contains the most specific facts needed to answer. "
        "If two chunks are near-duplicates, keep the clearer one earlier."
    )
    prompt = f"{instruction}\n\nQuestion: {question}\n\nChunks:\n{context}\n\nIndices in best-to-worst order:"
    resp = _gemini.generate_content(prompt)
    raw = (getattr(resp, "text", None) or "").strip()

    # Parse "1, 5, 2, 3" → [1,5,2,3]
    order_local = []
    for x in re.findall(r"\d+", raw):
        v = int(x)
        if 1 <= v <= len(mapping):
            order_local.append(v)
    if not order_local:
        order_local = list(range(1, min(top_k, len(mapping)) + 1))

    # Map to global indices, trim to top_k
    out, seen = [], set()
    for v in order_local:
        gi = mapping[v-1]
        if gi not in seen:
            out.append(gi); seen.add(gi)
        if len(out) >= top_k:
            break
    return out

def dedup_by_char_tfidf(indices: List[int], limit: int) -> List[int]:
    selected, seen = [], []
    for i in indices:
        if len(selected) >= limit:
            break
        vi = X_char[i]
        is_dup = False
        for j in seen:
            vj = X_char[j]
            num = vi.multiply(vj).sum()
            den = (vi.multiply(vi).sum() ** 0.5) * (vj.multiply(vj).sum() ** 0.5)
            cos = float(num / (den + 1e-9))
            if cos >= DEDUP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            selected.append(i)
            seen.append(i)
    return selected


# ==========================================================
# Gemini client
# ==========================================================
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY in .streamlit/secrets.toml")
    st.stop()
genai.configure(api_key=api_key)
_gemini = genai.GenerativeModel(MODEL_NAME)

def llm_answer_with_gemini(question: str, contexts: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts[:5], 1):
        txt = c["text"]
        if len(txt) > 1000: txt = txt[:1000] + " ..."
        blocks.append(f"[{i}] {c['title']}\n{txt}")
    context = "\n\n".join(blocks)
    system = (
        "Read and understand the sources that you have been given. "
        "Answer strictly from the provided sources. Be concise and factual. "
        "Always cite sources like [1], [2]. If not present in sources, say you can't find it. "
        "Reply in deduplicated bullet points."
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
    try:
        # 1) Retrieve a strong candidate pool (cheap)
        cand = retrieve_candidates(q, pool=CANDIDATE_POOL)

        # 2) Ensure quota for 2 calls: re-rank + final answer
        if not check_and_increment_quota(n_calls=2):
            st.error(f"Daily limit would be exceeded by this query (needs 2 calls). "
                     f"Remaining: {remaining_quota()} / {DAILY_QUOTA}")
        else:
            # 3) Re-rank with Gemini and dedup
            ordered = gemini_rerank(q, cand, top_k=max(TOP_K*2, TOP_K))
            final_indices = dedup_by_char_tfidf(ordered, limit=TOP_K)
            top = [chunks[i] for i in final_indices]

            # 4) Show answer
            answer = llm_answer_with_gemini(q, top)
            st.markdown(answer or "_No answer returned._")

            if SHOW_CHUNKS:
                with st.expander("Final chunks sent to Gemini"):
                    for i, c in enumerate(top, 1):
                        st.markdown(f"**[{i}] {c['title']}**")
                        st.write(c["text"])

            with st.expander("Sources used"):
                for i, s in enumerate(top, 1):
                    st.markdown(f"**[{i}] {s['title']}** — [{s['href']}]({s['href']})")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
        st.text(traceback.format_exc())
