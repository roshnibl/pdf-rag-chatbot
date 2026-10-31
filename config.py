"""Central configuration for the RAG chatbot.

All tunable knobs live here so the pipeline stays deterministic and reproducible.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
# Vector DB location, resolved in priority order:
#   1. CHROMA_DIR env var (explicit override, e.g. deployment bundles)
#   2. a DB bundled inside the project at data/chroma_db (transport to another
#      machine: unzip and it "just works")
#   3. a fast local path OFF OneDrive (this machine, for ingestion speed)
_BUNDLED_DB = os.path.join(BASE_DIR, "data", "chroma_db")
CHROMA_DIR = (
    os.getenv("CHROMA_DIR")
    or (_BUNDLED_DB if os.path.isdir(_BUNDLED_DB) else "")
    or os.path.join(
        os.getenv("LOCALAPPDATA") or os.path.join(BASE_DIR, "data"),
        "rag_chatbot_db",
        "chroma_db",
    )
)
COLLECTION_NAME = "pdf_corpus"

# --- Chunking (deterministic) ---
# bge-small embeds at most 512 tokens; keep chunks under that so no text is
# silently truncated out of its own embedding.
CHUNK_SIZE_TOKENS = 480
CHUNK_OVERLAP_TOKENS = 48        # 10% overlap
TOKENIZER_ENCODING = "cl100k_base"

# --- Embedding model (free / open-source) ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"          # "cuda" if a GPU is available
# Threads for the ONNX embedding backend (fastembed). Use all cores by default.
EMBEDDING_THREADS = os.cpu_count() or 4
# bge models recommend this prefix on the *query* only, not on documents.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- OCR ---
# Pages with fewer than this many extracted characters are treated as scanned
# and sent through OCR as a fallback.
OCR_CHAR_THRESHOLD = 100
OCR_DPI = 200
# On Windows, set this to the tesseract.exe path if it is not on PATH, e.g.
#   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

# --- Retrieval ---
TOP_K_RETRIEVE = 30              # candidates pulled from the vector DB
TOP_K_FINAL = 8                  # passages sent to the LLM after reranking
USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- LLM (Gemini) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# flash-lite is less congested and has more generous free-tier limits than the
# full flash models (which were returning 503/429). Plenty capable for RAG.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_TEMPERATURE = 0.2
GEMINI_MAX_OUTPUT_TOKENS = 1024
