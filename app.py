import os
import sys
import tempfile
import time
import streamlit as st

# Add backend directory to sys.path
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from pdf_processor import extract_pdf_pages, chunk_pages
from visual_processor import SimpleVisualCaptioner, extract_pdf_images, create_visual_chunks
from embeddings import SimpleEmbedder
from retriever import build_faiss_index, retrieve
from llm import generate_answer
from citations import get_unique_sources, append_citations
from guardrails import check_retrieval_relevance, REFUSAL_MESSAGE
from cache import (
    make_cache_key,
    get_cached_answer,
    store_answer,
    clear_cache,
    get_cache_size
)

# ----------------------------------------------------------------------
# Streamlit Resource Caching: Load heavy models once across sessions
# ----------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SimpleEmbedder()

@st.cache_resource
def load_captioner():
    return SimpleVisualCaptioner()

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind | AI Document Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Comprehensive Reflective Slate-Silver & Purple Theme (Zero White Artifacts)
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ================= Global Reflective Canvas ================= */
.stApp {
    background-color: #0E1218 !important;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(168, 85, 247, 0.22), transparent 50%),
        radial-gradient(circle at 85% 25%, rgba(192, 132, 252, 0.1), transparent 40%),
        radial-gradient(circle at 15% 75%, rgba(147, 51, 234, 0.08), transparent 45%),
        linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 100% 100%, 40px 40px, 40px 40px !important;
    color: #F8FAFC !important;
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* High-Contrast Typography */
p, span, label, div {
    color: #E2E8F0;
}
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700;
}

/* ================= Floating Navbar ================= */
.brainwind-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(26, 32, 44, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 9999px;
    padding: 12px 28px;
    margin: 0 auto 35px auto;
    max-width: 950px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2);
}

.brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 34px;
    height: 34px;
    background: linear-gradient(135deg, #C084FC, #9333EA);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-weight: 900;
    font-size: 18px;
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.5);
}

.brand-name {
    font-size: 1.3rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.03em;
}

.brand-name span {
    color: #C084FC;
}

.nav-pill-group {
    display: flex;
    align-items: center;
    gap: 18px;
}

.nav-link-text {
    font-size: 0.9rem;
    color: #CBD5E1;
    font-weight: 600;
}

.nav-badge-local {
    background: rgba(168, 85, 247, 0.18);
    color: #E9D5FF;
    border: 1px solid rgba(192, 132, 252, 0.4);
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 0 18px rgba(168, 85, 247, 0.25);
}

.nav-badge-local::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #C084FC;
    box-shadow: 0 0 10px #C084FC;
}

/* ================= Hero Section ================= */
.hero-container {
    text-align: center;
    padding: 10px 20px 30px 20px;
    max-width: 900px;
    margin: 0 auto;
}

.hero-tagline {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(192, 132, 252, 0.35);
    padding: 8px 20px;
    border-radius: 9999px;
    color: #E9D5FF;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(168, 85, 247, 0.2);
}

.hero-title {
    font-size: 3.3rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.04em;
    color: #FFFFFF;
    margin-bottom: 18px;
}

.hero-gradient {
    background: linear-gradient(135deg, #E9D5FF 0%, #C084FC 45%, #A855F7 80%, #7E22CE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
    filter: drop-shadow(0 0 30px rgba(168, 85, 247, 0.45));
}

.hero-subtitle {
    font-size: 1.12rem;
    line-height: 1.7;
    color: #CBD5E1;
    max-width: 740px;
    margin: 0 auto;
    font-weight: 400;
}

/* ================= Feature Showcase Grid ================= */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin: 30px 0 40px 0;
}

.feature-box {
    background: linear-gradient(180deg, #1E2638 0%, #171E2C 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 18px;
    padding: 22px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
}

.feature-box:hover {
    border-color: rgba(192, 132, 252, 0.5);
    transform: translateY(-3px);
    box-shadow: 0 14px 35px rgba(168, 85, 247, 0.2);
    background: linear-gradient(180deg, #242D42 0%, #1A2234 100%);
}

.feature-icon-badge {
    width: 38px;
    height: 38px;
    background: rgba(168, 85, 247, 0.18);
    border: 1px solid rgba(192, 132, 252, 0.35);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin-bottom: 14px;
}

.feature-box h4 {
    font-size: 1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 6px 0;
}

.feature-box p {
    font-size: 0.85rem;
    color: #94A3B8;
    line-height: 1.5;
    margin: 0;
}

/* ================= Question Centerpiece Card ================= */
.qa-centerpiece-container {
    background: linear-gradient(180deg, #1A2130 0%, #141A26 100%);
    border: 1px solid rgba(192, 132, 252, 0.3);
    border-top: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 24px;
    padding: 28px 32px;
    margin: 15px 0 35px 0;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55), 0 0 35px rgba(168, 85, 247, 0.15);
}

.qa-centerpiece-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}

.qa-title-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
}

.qa-badge {
    background: rgba(168, 85, 247, 0.2);
    color: #E9D5FF;
    border: 1px solid rgba(192, 132, 252, 0.4);
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ================= Form Inputs & Buttons (Zero White Artifacts) ================= */
.stTextInput > div > div > input {
    background-color: #161D2A !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    padding: 16px 20px !important;
    box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.25s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: #C084FC !important;
    background-color: #1A2234 !important;
    box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.35), 0 0 25px rgba(168, 85, 247, 0.3) !important;
}

/* All Buttons (Regular & Form Submit) */
div[data-testid="stFormSubmitButton"] > button,
.stButton > button,
button[kind="primaryFormSubmit"],
button[kind="secondaryFormSubmit"],
button[kind="primary"],
button[kind="secondary"] {
    background: linear-gradient(135deg, #C084FC 0%, #A855F7 50%, #7E22CE 100%) !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 9999px !important;
    padding: 12px 28px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 6px 25px rgba(168, 85, 247, 0.45) !important;
    letter-spacing: -0.01em;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
}

div[data-testid="stFormSubmitButton"] > button:hover,
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 35px rgba(168, 85, 247, 0.65) !important;
    background: linear-gradient(135deg, #D8B4FE 0%, #C084FC 50%, #9333EA 100%) !important;
    color: #FFFFFF !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #121722 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(180deg, #1F2738 0%, #171E2D 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 10px 18px !important;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #A855F7 0%, #7E22CE 100%) !important;
    border-color: #C084FC !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.35) !important;
}

/* File Uploader Dropzone & Uploaded File Cards */
div[data-testid="stFileUploader"] {
    background: transparent !important;
}

div[data-testid="stFileUploader"] section {
    background: #1C2433 !important;
    border: 1px dashed rgba(192, 132, 252, 0.35) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: #C084FC !important;
    background: #222C3D !important;
}

div[data-testid="stFileUploader"] section span,
div[data-testid="stFileUploader"] section small,
div[data-testid="stFileUploader"] section div,
div[data-testid="stFileUploader"] section p {
    color: #CBD5E1 !important;
}

div[data-testid="stFileUploader"] section button {
    background: #2A3549 !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
}

/* Uploaded File Chip / Card in File Uploader (Fixing white background) */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] [data-testid="stFileUploadFileData"],
div[data-testid="stFileUploader"] [role="listitem"],
div[data-testid="stFileUploader"] ul,
div[data-testid="stFileUploader"] li,
div[data-testid="stFileUploader"] section + div > div,
div[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] + div > div,
div[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] ~ * > div {
    background: #1C2433 !important;
    background-color: #1C2433 !important;
    border: 1px solid rgba(192, 132, 252, 0.3) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] *,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
div[data-testid="stFileUploader"] [role="listitem"] * {
    color: #FFFFFF !important;
}

/* File Icon Box inside Chip */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] svg {
    fill: #C084FC !important;
    color: #C084FC !important;
}

/* Delete (x) Button & Add (+) Button */
div[data-testid="stFileUploader"] button[aria-label="Delete"],
div[data-testid="stFileUploader"] button[title="Delete file"],
div[data-testid="stFileUploader"] button[data-testid="stBaseButton-headerNoPadding"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button,
div[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] + div button {
    background: #252F42 !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #CBD5E1 !important;
}

div[data-testid="stFileUploader"] button[aria-label="Delete"]:hover,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button:hover {
    background: rgba(239, 68, 68, 0.25) !important;
    border-color: #EF4444 !important;
    color: #FCA5A5 !important;
}

/* Streamlit Status Widget, Alert & Spinner (Fixing white background) */
div[data-testid="stStatusWidget"],
div[data-testid="stExpander"],
details[data-testid="stExpander"] {
    background: #171E2D !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    color: #FFFFFF !important;
    margin: 15px 0 !important;
}

div[data-testid="stStatusWidget"] summary,
div[data-testid="stExpander"] summary {
    background: #1C2433 !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
}

div[data-testid="stStatusWidget"] *,
div[data-testid="stExpander"] * {
    color: #E2E8F0 !important;
}

/* Streamlit Inline Code Tags & Pre */
code, pre, .stCodeBlock {
    background: #1E2738 !important;
    color: #E9D5FF !important;
    border: 1px solid rgba(192, 132, 252, 0.3) !important;
    border-radius: 8px !important;
    padding: 2px 6px !important;
}

/* Streamlit Alerts & Messages */
div[data-testid="stAlert"] {
    background: #171E2D !important;
    border: 1px solid rgba(192, 132, 252, 0.35) !important;
    border-radius: 14px !important;
}

div[data-testid="stAlert"] * {
    color: #FFFFFF !important;
}

/* ================= Answer & Result Container ================= */
.answer-card {
    background: linear-gradient(180deg, #1B2333 0%, #141A28 100%);
    border: 1px solid rgba(192, 132, 252, 0.25);
    border-top: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 24px;
    padding: 28px 32px;
    margin: 25px 0;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.answer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.user-query-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 8px 18px;
    border-radius: 9999px;
    font-size: 0.95rem;
    color: #FFFFFF;
    font-weight: 500;
}

.badge-cache-hit {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(168, 85, 247, 0.2);
    color: #E9D5FF;
    border: 1px solid rgba(192, 132, 252, 0.45);
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    box-shadow: 0 0 18px rgba(168, 85, 247, 0.25);
}

.badge-cache-miss {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59, 130, 246, 0.2);
    color: #93C5FD;
    border: 1px solid rgba(59, 130, 246, 0.4);
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
}

.badge-grounded-allow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(168, 85, 247, 0.18);
    color: #E9D5FF;
    border: 1px solid rgba(192, 132, 252, 0.35);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}

.badge-grounded-refuse {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(239, 68, 68, 0.18);
    color: #FECACA;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}

.answer-text-content {
    font-size: 1.15rem;
    line-height: 1.8;
    color: #FFFFFF;
    font-weight: 400;
    margin: 18px 0;
}

.source-container-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.source-pill-card {
    background: #171E2C;
    border: 1px solid rgba(192, 132, 252, 0.35);
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: #FFFFFF;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
}

.source-num-badge {
    background: rgba(168, 85, 247, 0.3);
    color: #E9D5FF;
    font-weight: 800;
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 6px;
}

/* ================= Evidence Cards ================= */
.evidence-item-card {
    background: #151B27;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-left: 4px solid #C084FC;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
}

.evidence-meta-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #CBD5E1;
}

.evidence-type-badge {
    font-weight: 700;
    color: #C084FC;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.evidence-text-preview {
    font-size: 0.9rem;
    line-height: 1.6;
    color: #F1F5F9;
    font-family: 'Inter', monospace;
    white-space: pre-wrap;
}

.doc-pill-item {
    background: #18202E;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Floating Navbar
# ----------------------------------------------------------------------
st.markdown("""
<div class="brainwind-navbar">
    <div class="brand-container">
        <div class="brand-icon">🔮</div>
        <div class="brand-name">doc<span>mind</span></div>
    </div>
    <div class="nav-pill-group">
        <span class="nav-link-text">Multimodal RAG</span>
        <span class="nav-link-text">Ollama (Gemma 2B)</span>
        <span class="nav-link-text">FAISS FlatIP</span>
        <span class="nav-badge-local">100% Offline AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Hero Section
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-tagline">
        🔮 Next-Gen Local Document Intelligence
    </div>
    <h1 class="hero-title">
        Ask your documents with <br><span class="hero-gradient">privacy-first AI</span>
    </h1>
    <p class="hero-subtitle">
        Ingest PDFs, extract charts with OCR + BLIP vision, perform instant semantic search, and get grounded answers with verifiable page-level citations.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Feature Showcase Grid
# ----------------------------------------------------------------------
st.markdown("""
<div class="feature-grid">
    <div class="feature-box">
        <div class="feature-icon-badge">📄</div>
        <h4>Page-Aware Chunks</h4>
        <p>256-word sliding window preserving exact PDF page coordinates for zero-hallucination citations.</p>
    </div>
    <div class="feature-box">
        <div class="feature-icon-badge">🖼️</div>
        <h4>Vision + OCR Engine</h4>
        <p>PyMuPDF extracts figures, Tesseract reads text, and BLIP generates natural image descriptions.</p>
    </div>
    <div class="feature-box">
        <div class="feature-icon-badge">⚡</div>
        <h4>MiniLM + FAISS</h4>
        <p>384-dimensional dense semantic vectors with Cosine Inner-Product similarity lookup.</p>
    </div>
    <div class="feature-box">
        <div class="feature-icon-badge">🛡️</div>
        <h4>Grounding Guardrails</h4>
        <p>Deterministic 0.40 confidence gate verifies evidence relevance before invoking the LLM.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar: Ingestion, Cache, & System Health
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload business reports, financial statements, or technical documentation."
    )
    
    if uploaded_files:
        st.markdown("**Selected Documents:**")
        for f in uploaded_files:
            st.markdown(f"""
            <div class="doc-pill-item">
                <span style="font-size: 0.88rem; font-weight: 600; color: #FFFFFF;">📄 {f.name}</span>
                <span style="font-size: 0.75rem; color: #C084FC; font-weight: 700;">READY</span>
            </div>
            """, unsafe_allow_html=True)
            
    col_proc, col_clear = st.columns(2)
    process_clicked = col_proc.button("⚡ Process", use_container_width=True)
    clear_docs_clicked = col_clear.button("Reset", use_container_width=True)
    
    if clear_docs_clicked:
        st.session_state["processed"] = False
        st.session_state["chunks"] = []
        st.session_state["index"] = None
        st.session_state["doc_names"] = []
        st.success("Document index reset.")
        st.rerun()
        
    st.divider()
    
    st.markdown("### ⚡ Fast Query Cache")
    st.markdown(f"**Cached Answers:** `{get_cache_size()}`")
    if st.button("🗑️ Clear Cache", use_container_width=True):
        clear_cache()
        st.success("Query cache cleared.")
        st.rerun()
        
    st.divider()
    
    st.markdown("### ⚙️ System Status")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #E2E8F0; line-height: 2;">
        <div>🟣 <b>Local Ollama:</b> <code style="color:#E9D5FF; background:#1C2433;">gemma2:2b</code></div>
        <div>🟣 <b>Embeddings:</b> <code style="color:#E9D5FF; background:#1C2433;">all-MiniLM-L6-v2</code></div>
        <div>🟣 <b>Vector Store:</b> <code style="color:#E9D5FF; background:#1C2433;">FAISS FlatIP</code></div>
        <div>🟣 <b>Vision Model:</b> <code style="color:#E9D5FF; background:#1C2433;">BLIP Base</code></div>
        <div>🟣 <b>OCR Engine:</b> <code style="color:#E9D5FF; background:#1C2433;">Tesseract v5</code></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🔒 Privacy Guarantee")
    st.caption(
        "All ingestion, OCR, embeddings, and LLM inferences execute 100% locally. "
        "No document contents or queries ever touch external cloud servers."
    )

# ----------------------------------------------------------------------
# Document Processing Pipeline Execution
# ----------------------------------------------------------------------
if "processed" not in st.session_state:
    st.session_state["processed"] = False
    st.session_state["chunks"] = []
    st.session_state["index"] = None
    st.session_state["doc_names"] = []

if process_clicked:
    if not uploaded_files:
        st.error("Please select at least one PDF file to process.")
    else:
        with st.status("Ingesting documents through multimodal pipeline...", expanded=True) as status:
            try:
                st.write("🔄 Initializing local MiniLM and BLIP vision models...")
                embedder = load_embedder()
                captioner = load_captioner()
                
                all_chunks = []
                doc_names = []
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        doc_name = uploaded_file.name
                        doc_names.append(doc_name)
                        temp_pdf_path = os.path.join(temp_dir, doc_name)
                        
                        with open(temp_pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        st.write(f"📖 Extracting text pages from `{doc_name}`...")
                        pages = extract_pdf_pages(temp_pdf_path)
                        text_chunks = chunk_pages(pages, doc_name, chunk_size=256, overlap=50)
                        st.write(f"   ✓ Generated {len(text_chunks)} text chunks.")
                        
                        st.write(f"🖼️ Extracting charts & running OCR/BLIP on `{doc_name}`...")
                        extracted_images = extract_pdf_images(temp_pdf_path)
                        visual_chunks = create_visual_chunks(extracted_images, doc_name, captioner)
                        st.write(f"   ✓ Generated {len(visual_chunks)} visual chunks.")
                        
                        all_chunks.extend(text_chunks)
                        all_chunks.extend(visual_chunks)
                        
                st.write(f"🔢 Indexing {len(all_chunks)} total chunks into FAISS...")
                faiss_index = build_faiss_index(all_chunks, embedder)
                
                st.session_state["processed"] = True
                st.session_state["chunks"] = all_chunks
                st.session_state["index"] = faiss_index
                st.session_state["doc_names"] = doc_names
                
                status.update(label="✅ All documents indexed and ready for Q&A!", state="complete", expanded=False)
                st.success(f"Ready! Indexed {len(all_chunks)} chunks across {len(doc_names)} document(s).")
                
            except Exception as e:
                status.update(label="❌ Ingestion failed", state="error")
                st.error(f"Error during processing: {str(e)}")

# ----------------------------------------------------------------------
# Main Q&A Workspace (Centerpiece)
# ----------------------------------------------------------------------
if not st.session_state["processed"]:
    st.info("👈 Upload your PDF documents in the sidebar and click **'⚡ Process'** to activate intelligence.")
else:
    # Question Form Card
    st.markdown("""
    <div class="qa-centerpiece-container">
        <div class="qa-centerpiece-header">
            <div class="qa-title-wrap">
                <span style="font-size: 1.3rem; font-weight: 700; color: #FFFFFF;">💬 Ask DocMind</span>
                <span class="qa-badge">Grounded QA</span>
            </div>
            <span style="font-size: 0.85rem; color: #CBD5E1;">Indexed Context: <b style="color:#C084FC;">Active</b></span>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form(key="docmind_qa_form", clear_on_submit=False):
        user_question = st.text_input(
            "Question Input:",
            placeholder="Ask anything about financial metrics, revenue, charts, or strategic initiatives...",
            label_visibility="collapsed"
        )
        col_btn, _ = st.columns([2, 5])
        with col_btn:
            submit_btn = st.form_submit_button("Ask Question ➔", use_container_width=True)
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process Query Submission
    if submit_btn:
        if not user_question or not user_question.strip():
            st.warning("Please type a question before submitting.")
        else:
            q_clean = user_question.strip()
            doc_names = st.session_state["doc_names"]
            chunks = st.session_state["chunks"]
            index = st.session_state["index"]
            embedder = load_embedder()
            
            cache_key = make_cache_key(doc_names, q_clean)
            cached_response = get_cached_answer(cache_key)
            
            # ----------------- CACHE HIT -----------------
            if cached_response is not None:
                st.markdown(f"""
<div class="answer-card">
<div class="answer-header">
<div class="user-query-badge">👤 <b>Query:</b> {q_clean}</div>
<div class="badge-cache-hit">⚡ Cache: HIT (0.00s)</div>
</div>
<div style="font-size: 0.85rem; color: #C084FC; font-weight: 600; margin-bottom: 12px;">
✓ Answer retrieved instantly from local in-memory cache. Downstream models were bypassed.
</div>
<div class="answer-text-content">{cached_response.replace(chr(10), '<br>')}</div>
</div>
""", unsafe_allow_html=True)
                
            # ----------------- CACHE MISS -----------------
            else:
                start_time = time.time()
                
                # Step 1: Semantic Vector Retrieval
                retrieved_chunks = retrieve(q_clean, chunks, index, embedder, top_k=5)
                
                # Step 2: Relevance Guardrail Check
                relevance = check_retrieval_relevance(retrieved_chunks, threshold=0.40)
                best_score = relevance["best_score"]
                is_allowed = relevance["allowed"]
                
                guard_badge_html = (
                    f'<div class="badge-grounded-allow">🛡️ Grounding: ALLOW ({best_score:.4f} ≥ 0.40)</div>'
                    if is_allowed else
                    f'<div class="badge-grounded-refuse">🛡️ Grounding: REFUSE ({best_score:.4f} &lt; 0.40)</div>'
                )
                
                if not is_allowed:
                    final_response = REFUSAL_MESSAGE
                    store_answer(cache_key, final_response)
                    elapsed = time.time() - start_time
                    
                    st.markdown(f"""
<div class="answer-card">
<div class="answer-header">
<div class="user-query-badge">👤 <b>Query:</b> {q_clean}</div>
<div style="display: flex; gap: 8px; align-items: center;">
<div class="badge-cache-miss">🔵 Cache: MISS</div>
{guard_badge_html}
</div>
</div>
<div class="answer-text-content" style="color: #CBD5E1;">{final_response}</div>
<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 14px;">
⏱️ Guardrail refused generation in {elapsed:.2f}s due to insufficient evidence.
</div>
</div>
""", unsafe_allow_html=True)
                    
                else:
                    with st.spinner("Generating grounded answer with local Gemma 2B..."):
                        llm_context = retrieved_chunks[:3]
                        raw_answer = generate_answer(q_clean, llm_context)
                        
                        if raw_answer.startswith("Error:"):
                            final_response = raw_answer
                            sources_list = []
                        else:
                            unique_sources = get_unique_sources(llm_context)
                            final_response = append_citations(raw_answer, unique_sources)
                            sources_list = unique_sources
                            store_answer(cache_key, final_response)
                            
                    elapsed = time.time() - start_time
                    
                    # Clean and format answer text (convert newlines to <br>)
                    formatted_answer = final_response.replace("\n", "<br>")
                    
                    # Build source cards cleanly without line indentation
                    sources_html = ""
                    if sources_list:
                        pill_elements = "".join([
                            f'<div class="source-pill-card"><span class="source-num-badge">[{s["citation_id"]}]</span><span>{s["document"]} · Page {s["page"]}</span></div>'
                            for s in sources_list
                        ])
                        sources_html = f'<div class="source-container-row"><span style="font-size: 0.85rem; font-weight: 700; color: #CBD5E1; margin-right: 8px;">SOURCES:</span>{pill_elements}</div>'
                        
                    st.markdown(f"""
<div class="answer-card">
<div class="answer-header">
<div class="user-query-badge">👤 <b>Query:</b> {q_clean}</div>
<div style="display: flex; gap: 8px; align-items: center;">
<div class="badge-cache-miss">🔵 Cache: MISS</div>
{guard_badge_html}
</div>
</div>
<div class="answer-text-content">{formatted_answer}</div>
{sources_html}
<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 14px;">
⏱️ Answer generated in {elapsed:.2f} seconds via local pipeline.
</div>
</div>
""", unsafe_allow_html=True)
                    
                # Step 4: Retrieved Evidence Accordion
                with st.expander("🔎 View Retrieved Evidence & Multimodal Chunks", expanded=False):
                    if not retrieved_chunks:
                        st.write("No evidence retrieved.")
                    else:
                        for idx, chunk in enumerate(retrieved_chunks, start=1):
                            chunk_type = chunk.get("type", "text")
                            badge_icon = "🖼️ Visual Figure (OCR + BLIP)" if chunk_type == "image" else "📄 PDF Text Chunk"
                            doc = chunk.get("document", "Unknown")
                            page = chunk.get("page", "?")
                            score = chunk.get("score", 0.0)
                            chunk_text = chunk.get("text", "")
                            
                            st.markdown(f"""
<div class="evidence-item-card">
<div class="evidence-meta-row">
<span class="evidence-type-badge">#{idx} {badge_icon}</span>
<span><b>Doc:</b> {doc} | <b>Page:</b> {page} | <b>Similarity:</b> {score:.4f}</span>
</div>
<div class="evidence-text-preview">{chunk_text}</div>
</div>
""", unsafe_allow_html=True)
