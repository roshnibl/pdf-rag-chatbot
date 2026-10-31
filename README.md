# 📚 PDF RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions from a large
private corpus of PDFs (≥10 PDFs, each ≥200 pages) using a **free / open-source
embedding model** and a **free / open-source vector DB**, with end-to-end
latency in the **2–5 second** range and **source citations** (PDF filename +
page number) on every answer.

Built for **AI/ML Challenge 1: RAG Chatbot**.

---

## Architecture

```
OFFLINE INGESTION (run once)
  PDFs ──► PyMuPDF (native text)          ──► clean + detect language
       └─► Tesseract OCR (scanned pages)  ──► chunk (500 tok, 20% overlap)
                                          ──► bge-small-en-v1.5 embeddings
                                          ──► Chroma (HNSW, cosine)

ONLINE QUERY (2–5s)
  query ──► embed ──► Chroma top-20 ──► cross-encoder rerank ──► top-5
        ──► Gemini 2.0 Flash (grounded + cited) ──► answer + sources
```

| Component | Choice | Free / OSS |
|---|---|---|
| PDF text | PyMuPDF | ✅ OSS |
| OCR | Tesseract (`pytesseract`) | ✅ OSS |
| Chunking | LangChain token splitter | ✅ OSS |
| Embeddings | `BAAI/bge-small-en-v1.5` | ✅ OSS |
| Vector DB | Chroma (HNSW index) | ✅ OSS |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ✅ OSS |
| LLM | Google Gemini 2.0 Flash | ✅ Free API key |
| UI | Streamlit | ✅ OSS |

> The embedding model and vector DB are fully open-source, as required. The LLM
> uses a **free** Gemini API key (no credit card) — the challenge permits a
> hosted LLM ("open-source or hosted").

---

## Setup

### 1. Install system dependencies

**Tesseract OCR** (needed only if your PDFs have scanned pages):

- **Windows:** Install from https://github.com/UB-Mannheim/tesseract/wiki, then
  set `TESSERACT_CMD` in `.env` to e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt-get install tesseract-ocr`

### 2. Install Python dependencies

```bash
cd rag-chatbot
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure your free Gemini key

```bash
cp .env.example .env
```

Get a free key at https://aistudio.google.com/app/apikey (no credit card), then
put it in `.env`:

```
GEMINI_API_KEY=your_key_here
```

---

## Usage

### Step 1 — Get the PDF corpus

Either drop your own PDFs into `pdfs/`, or download a free 10-PDF corpus
(OpenStax textbooks, CC-BY):

```bash
python download_pdfs.py
```

### Step 2 — Ingest (build the vector DB)

```bash
python ingest.py
```

This extracts text (+ OCR fallback), chunks, embeds, and persists everything to
`data/chroma_db/`. Run it once; re-run only when the corpus changes.

### Step 3 — Launch the chatbot

```bash
python -m streamlit run app.py
```
> Use `python -m streamlit` (not the `streamlit` command) — on locked-down
> corporate machines the generated `streamlit.exe` can be blocked by AppLocker,
> while invoking it through Python always works.

Open http://localhost:8501 and start asking questions. Each answer shows:
- the generated response with inline `[filename p.N]` citations,
- retrieval/generation/total **latency metrics**,
- an expandable **retrieval visualization** (top chunks + similarity & rerank scores).

### Step 4 (optional) — Evaluate

Create `eval_set.json` (format in `evaluate.py`), then:

```bash
python evaluate.py
```

Reports latency p50/p95, Recall@1/3/5, MRR, and citation accuracy.

---

## Deployment (public demo)

### Hugging Face Spaces (free, recommended)

1. Create a new **Streamlit** Space.
2. Push this folder to the Space repo.
3. In **Settings → Secrets**, add `GEMINI_API_KEY`.
4. Commit the prebuilt `data/chroma_db/` so the Space starts already-ingested
   (recommended — ingestion of large PDFs is slow on free CPU tiers).

Add a `packages.txt` with `tesseract-ocr` if your corpus needs OCR.

### Streamlit Community Cloud (free)

Point it at `app.py`, add `GEMINI_API_KEY` in **Secrets**, deploy.

---

## Tuning latency (2–5s target)

All knobs are in `config.py`:

| Symptom | Knob | Suggestion |
|---|---|---|
| Retrieval too slow | `TOP_K_RETRIEVE` | lower to 10 |
| Rerank too slow (CPU) | `USE_RERANKER` | set `False` |
| Generation too slow | `GEMINI_MAX_OUTPUT_TOKENS` | lower to 512 |
| Better recall needed | `TOP_K_FINAL` | raise to 7–8 |
| Faster embedding | `EMBEDDING_DEVICE` | `"cuda"` if GPU |

---

## Project layout

```
rag-chatbot/
├── app.py              # Streamlit chat UI (deliverable)
├── ingest.py           # one-shot ingestion pipeline
├── evaluate.py         # latency / recall / MRR / citation metrics
├── download_pdfs.py    # fetch a free 10-PDF corpus
├── config.py           # all tunable settings
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── pdf_loader.py   # PyMuPDF + Tesseract OCR, page-level provenance
│   ├── chunker.py      # cleaning, header/footer removal, token chunking
│   ├── embedder.py     # bge-small embeddings + Chroma persistence
│   ├── retriever.py    # HNSW retrieval + cross-encoder rerank
│   └── generator.py    # Gemini, grounded + cited answers
├── pdfs/               # your PDF corpus
└── data/chroma_db/     # persisted vector DB
```

---

## Receiving this project on another machine

If someone sent you `rag-chatbot-bundle.zip`, the vector DB is **already built and
included** — you do NOT need to run ingestion. Steps:

1. **Unzip** anywhere, then open a terminal in the `rag-chatbot/` folder.
2. **Install Python 3.12 or 3.13** (3.14 also works). Avoid zipped virtual
   environments — create a fresh one here:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
   > For an exact match to the sender's versions, use `pip install -r requirements.lock` instead.
   > If `.venv` won't execute inside a OneDrive/corporate folder, create it outside that folder.
3. **Add your own free Gemini key**:
   ```bash
   cp .env.example .env     # then edit .env and set GEMINI_API_KEY
   ```
   Get one free at https://aistudio.google.com/app/apikey
4. **Run it** — the bundled DB at `data/chroma_db` is auto-detected:
   ```bash
   python -m streamlit run app.py
   ```

**First run needs internet once** to download the embedding + reranker models
(~210 MB, cached afterward). After that it works offline except for the Gemini
answer call. Your CPU only embeds one short query at a time — no heavy ingestion.

> The `.venv` and the sender's `.env` are intentionally NOT in the bundle: virtual
> environments aren't portable across machines, and API keys must not be shared.

---

## Sending this project to someone else

```bash
python package.py      # produces dist/rag-chatbot-bundle.zip
```
This bundles the code, `requirements.txt`/`.lock`, the **prebuilt vector DB**, and
the PDFs — and excludes `.venv`, `.env`, and caches. Run `python ingest.py` first
if you want the latest corpus baked in. Send the resulting zip.

---

## Managing documents from the UI

Open the sidebar (the `»` arrow, top-left) → **📁 Manage documents**:
- **Add a PDF:** upload a file → click *Ingest* → it's extracted, chunked, embedded,
  and immediately queryable — no restart, no full re-ingest.
- **Remove:** click 🗑 next to a document → its chunks are deleted from the vector DB
  and the file is removed from `pdfs/`.

## Publishing to GitHub (challenge submission)

The repository is **code-only by design** — `.gitignore` excludes the virtual env,
your API key, the vector DB, the PDFs, and build artifacts.

```bash
git init
git add .
git commit -m "PDF RAG chatbot"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Anyone cloning the repo builds it with:
```bash
pip install -r requirements.txt
cp .env.example .env          # add their own free Gemini key
python download_pdfs.py       # fetch the corpus
python ingest.py              # build the vector DB (~40 min CPU, one-time)
python -m streamlit run app.py
```

> The prebuilt vector DB (`chroma.sqlite3` ≈ 125 MB) is intentionally **not** committed:
> it exceeds GitHub's 100 MB file limit and large binaries don't belong in git. To let
> a reviewer skip the ingest, share the DB via the zip (see *Sending this project*) or a
> cloud link.

---

## How requirements are met

- **Open/Free:** embeddings (bge-small) and vector DB (Chroma) are OSS; Gemini key is free.
- **Scalability:** streaming per-page extraction + batched embedding handle 200+ page PDFs.
- **Latency:** HNSW ANN + lightweight reranker + Gemini Flash target 2–5s.
- **Explainability:** every answer carries `[filename p.N]` citations + a source panel.
- **Reproducibility:** deterministic chunking (fixed token size/overlap) and a reset-and-rebuild ingest.
