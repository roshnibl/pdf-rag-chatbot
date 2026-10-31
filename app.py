"""Streamlit RAG chatbot UI.

Shows the full deliverable: a chat interface over the ingested PDF corpus with
per-answer source citations, retrieval visualization (top chunks + scores), and
end-to-end latency measurement.

Launch:  streamlit run app.py
"""
from __future__ import annotations

import threading
import time

import streamlit as st

import config

# NOTE: heavy pipeline imports (torch via the reranker, fastembed/ONNX) are done
# lazily inside the functions below so the page renders instantly instead of
# paying ~10s of import cost on every load. They load on the first query.

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Visual theme (custom CSS) ---------------------------------------------
st.markdown(
    """
    <style>
      /* Hide Streamlit's toolbar (Deploy/menu) + footer; keep header so the
         sidebar expand arrow stays available. */
      [data-testid="stToolbar"], #MainMenu, footer {display: none;}
      [data-testid="stHeader"] {background: transparent;}
      /* Keep the collapsed-sidebar expand arrow visible and clickable. */
      [data-testid="stSidebarCollapsedControl"] {
        display: flex !important; visibility: visible !important; opacity: 1 !important;
      }

      /* Ambient background */
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(1100px 560px at 15% -10%, rgba(139,124,255,0.14), transparent 60%),
          radial-gradient(900px 480px at 100% 0%, rgba(95,208,230,0.08), transparent 55%),
          #0d0f16;
      }

      /* Hero header */
      .hero {padding: 6px 0 2px 0;}
      .hero-title {
        font-size: 2.35rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;
        background: linear-gradient(90deg, #b7acff, #8b7cff 42%, #5fd0e6);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .hero-sub {color: #9aa0b4; font-size: 1.0rem; margin-top: 6px; max-width: 720px;}
      .hero-rule {height: 1px; margin: 16px 0 4px;
        background: linear-gradient(90deg, rgba(139,124,255,0.5), rgba(139,124,255,0) 70%);}

      /* Chat bubbles */
      [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 14px 16px; margin-bottom: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.22);
      }

      /* Latency metric chips */
      [data-testid="stMetric"] {
        background: rgba(139,124,255,0.08);
        border: 1px solid rgba(139,124,255,0.18);
        border-radius: 12px; padding: 8px 12px;
      }
      [data-testid="stMetricValue"] {font-size: 1.05rem;}
      [data-testid="stMetricLabel"] {opacity: 0.75;}

      /* Source expanders */
      [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        overflow: hidden; background: rgba(255,255,255,0.02);
      }
      /* Hide the Material-symbols chevron — on slow machines it flashes as raw
         "arrow_down" text before the icon font loads, overlapping the label.
         The expander still toggles when its header is clicked. */
      [data-testid="stExpanderToggleIcon"] {display: none !important;}
      summary[data-testid="stExpanderHeader"] {cursor: pointer;}

      /* Chat input */
      [data-testid="stChatInput"] {border-radius: 14px;}
      [data-testid="stChatInput"] textarea {font-size: 1.0rem;}

      /* Sidebar */
      [data-testid="stSidebar"] {
        background: #11131d; border-right: 1px solid rgba(255,255,255,0.06);
      }

      /* Alerts / info boxes */
      [data-testid="stAlert"] {border-radius: 12px;}

      /* Buttons */
      .stButton > button {
        border-radius: 10px; border: 1px solid rgba(139,124,255,0.3);
        background: rgba(139,124,255,0.12); color: #e8eaf2; font-weight: 600;
      }
      .stButton > button:hover {
        background: rgba(139,124,255,0.22); border-color: #8b7cff;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _collection_stats():
    """Read collection size (cached as data so it refreshes after a document is
    added/removed — clearing data cache does not reload the models).

    Imports chromadb directly (lightweight) rather than the embedder module, so
    the page can show the chunk count without triggering the fastembed/torch
    imports that make startup slow.
    """
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=config.CHROMA_DIR,
            # Settings MUST match embedder.get_chroma_collection(), or ChromaDB
            # raises "instance already exists with different settings".
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        col = client.get_or_create_collection(
            config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        return col.count()
    except Exception:
        return 0


def _start_background_warmup():
    """Load the embedding + reranker models in a daemon thread so the page is
    usable immediately. The first query uses them if ready, or waits briefly
    inside its own spinner. Runs once per session.

    The thread only touches module-level singletons (no st.* calls), so it is
    safe to run outside Streamlit's script context.
    """
    if st.session_state.get("_warm_started"):
        return
    st.session_state["_warm_started"] = True

    def _load():
        try:
            from pipeline.embedder import get_embedding_model
            from pipeline.retriever import _get_reranker

            get_embedding_model()
            _get_reranker()
        except Exception:
            pass

    threading.Thread(target=_load, daemon=True).start()


@st.cache_data(show_spinner=False)
def _corpus_documents():
    """List the ingested PDF names (for the welcome screen)."""
    import os

    try:
        files = sorted(
            f for f in os.listdir(config.PDF_DIR) if f.lower().endswith(".pdf")
        )
        return [os.path.splitext(f)[0].replace("_", " ") for f in files]
    except Exception:
        return []


EXAMPLE_QUESTIONS = [
    "What were the main intelligence failures before the 9/11 attacks?",
    "How did inflation change between 2021 and 2024?",
    "What is the difference between a list and a tuple in Python?",
    "What is a confidence interval and how is it computed?",
]


def _render_sources(sources):
    """Render the retrieval visualization + citation list for one answer."""
    if not sources:
        st.info("No source passages were retrieved.")
        return

    st.markdown("**Sources & retrieved passages**")
    for i, s in enumerate(sources, start=1):
        score = s.rerank_score if s.rerank_score is not None else s.vector_score
        label = "rerank" if s.rerank_score is not None else "cosine"
        header = f"[{i}] {s.filename} — p.{s.page_number}  ({label}: {score})"
        with st.expander(header):
            st.caption(f"Vector (cosine) similarity: {s.vector_score}")
            if s.rerank_score is not None:
                st.caption(f"Cross-encoder rerank score: {s.rerank_score}")
            st.write(s.text)


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("📚 PDF RAG Chatbot")
    st.caption("Open-source embeddings · Chroma · Gemini")

    chunk_count = _collection_stats()
    if chunk_count == 0:
        st.error(
            "Vector DB is empty. Run `python ingest.py` after adding PDFs to "
            "the `pdfs/` folder."
        )
    else:
        st.success(f"Indexed chunks: {chunk_count:,}")

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Passages sent to LLM (top-K)", 1, 10, config.TOP_K_FINAL)
    show_retrieval = st.checkbox("Show retrieval visualization", value=True)

    st.divider()
    st.markdown(
        f"**Embedding:** `{config.EMBEDDING_MODEL}`\n\n"
        f"**Reranker:** `{config.RERANKER_MODEL if config.USE_RERANKER else 'off'}`\n\n"
        f"**LLM:** `{config.GEMINI_MODEL}`"
    )
    if not config.GEMINI_API_KEY:
        st.warning("GEMINI_API_KEY not set — add it to `.env`.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# --- Main area header ------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">📚 PDF RAG Chatbot</div>
      <div class="hero-sub">Ask anything across the document corpus — every answer
      is grounded in the sources, with page-level citations.</div>
    </div>
    <div class="hero-rule"></div>
    """,
    unsafe_allow_html=True,
)

# Load models in the background so the page is instantly usable.
if chunk_count > 0:
    _start_background_warmup()

# --- Chat state ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent docs + examples panel. Uses a plain button toggle (not st.expander)
# to avoid Streamlit's Material-icon chevron flashing as raw "arrow_down" text
# before the icon font loads.
example_clicked = None
if "show_help" not in st.session_state:
    st.session_state.show_help = True  # open on first load
if st.button("💡 Documents & example questions  (show / hide)",
             use_container_width=True, key="toggle_help"):
    st.session_state.show_help = not st.session_state.show_help

if st.session_state.show_help:
    docs = _corpus_documents()
    if docs:
        st.markdown("**I can answer questions about these documents:**")
        st.markdown("\n".join(f"- {d}" for d in docs))
        st.write("")
    st.markdown("**Try one of these — or type your own below:**")
    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        if st.button(ex, use_container_width=True, key=f"example_q_{i}"):
            example_clicked = ex

# Document management (add / remove PDFs) — main area, always reachable.
if "show_manage" not in st.session_state:
    st.session_state.show_manage = False
if st.button("📁 Manage documents  (show / hide)",
             use_container_width=True, key="toggle_manage"):
    st.session_state.show_manage = not st.session_state.show_manage

if st.session_state.show_manage:
    import os as _os

    up = st.file_uploader("Add a PDF to the knowledge base", type="pdf", key="pdf_uploader")
    if up is not None and st.button(f"Ingest '{up.name}'", use_container_width=True):
        _os.makedirs(config.PDF_DIR, exist_ok=True)
        dest = _os.path.join(config.PDF_DIR, up.name)
        with st.spinner(f"Ingesting {up.name} — this can take a few minutes..."):
            with open(dest, "wb") as _f:
                _f.write(up.getbuffer())
            from pipeline.manage import ingest_pdf_file

            pages, nchunks = ingest_pdf_file(dest)
        st.cache_data.clear()
        st.success(f"Added {up.name}: {pages} pages, {nchunks} chunks.")
        st.rerun()

    current = sorted(
        f for f in (_os.listdir(config.PDF_DIR) if _os.path.isdir(config.PDF_DIR) else [])
        if f.lower().endswith(".pdf")
    )
    if current:
        st.caption(f"Current documents ({len(current)}) — click 🗑 to remove:")
        for fn in current:
            c1, c2 = st.columns([0.85, 0.15])
            c1.write(f"📄 {fn}")
            if c2.button("🗑", key=f"rm_{fn}", help=f"Remove {fn}"):
                with st.spinner(f"Removing {fn}..."):
                    from pipeline.manage import remove_pdf

                    remove_pdf(fn)
                st.cache_data.clear()
                st.rerun()

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            cols = st.columns(3)
            cols[0].metric("Retrieval", f"{msg['t_retrieve']:.2f}s")
            cols[1].metric("Generation", f"{msg['t_generate']:.2f}s")
            cols[2].metric("Total", f"{msg['t_total']:.2f}s")
            if show_retrieval:
                _render_sources(msg["sources"])


# --- Handle new input ------------------------------------------------------
prompt = st.chat_input("Ask a question about the PDF corpus...") or example_clicked

if prompt:
    if chunk_count == 0:
        st.error("Cannot answer — the vector DB is empty. Run `python ingest.py` first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            # Heavy imports (torch reranker, fastembed) happen here on first use.
            from pipeline.generator import generate_answer
            from pipeline.retriever import retrieve

            t0 = time.time()
            sources = retrieve(prompt, top_k_final=top_k)
            t1 = time.time()
            answer = generate_answer(prompt, chunks=sources)
            t2 = time.time()

        st.markdown(answer.text)

        t_retrieve = t1 - t0
        t_generate = t2 - t1
        t_total = t2 - t0
        cols = st.columns(3)
        cols[0].metric("Retrieval", f"{t_retrieve:.2f}s")
        cols[1].metric("Generation", f"{t_generate:.2f}s")
        cols[2].metric("Total", f"{t_total:.2f}s")

        if show_retrieval:
            _render_sources(answer.sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "sources": answer.sources,
            "t_retrieve": t_retrieve,
            "t_generate": t_generate,
            "t_total": t_total,
        }
    )
