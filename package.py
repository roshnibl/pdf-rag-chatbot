"""Build a portable transport bundle (zip) of this project for another machine.

Includes: source code, requirements, the PREBUILT vector DB, and the PDFs.
Excludes: .venv (not portable), .env (your API key), caches.

The recipient recreates the environment with `pip install -r requirements.txt`
(pip fetches the correct wheels for THEIR OS/Python), adds their own Gemini key,
and runs the app against the bundled DB — no CPU-heavy ingestion needed.

Run:  python package.py
Produces:  dist/rag-chatbot-bundle.zip
"""
from __future__ import annotations

import os
import shutil
import zipfile

import config

BASE = config.BASE_DIR
DIST_DIR = os.path.join(BASE, "dist")
ZIP_PATH = os.path.join(DIST_DIR, "rag-chatbot-bundle.zip")

# Top-level files to include (code + config templates + docs).
TOP_LEVEL_FILES = [
    "app.py",
    "ingest.py",
    "evaluate.py",
    "download_pdfs.py",
    "package.py",
    "config.py",
    "requirements.txt",
    "README.md",
    ".env.example",
    ".gitignore",
]
# Optional lock file, included if present.
OPTIONAL_FILES = ["requirements.lock"]


def _add_file(zf: zipfile.ZipFile, abs_path: str, arcname: str):
    zf.write(abs_path, arcname)


def _add_tree(zf: zipfile.ZipFile, src_dir: str, arc_prefix: str,
              skip_pycache: bool = True):
    for root, dirs, files in os.walk(src_dir):
        if skip_pycache:
            dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".pyc"):
                continue
            abs_path = os.path.join(root, f)
            rel = os.path.relpath(abs_path, src_dir)
            _add_file(zf, abs_path, os.path.join(arc_prefix, rel))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build a transport bundle zip.")
    parser.add_argument(
        "--no-pdfs", action="store_true",
        help="Exclude the source PDFs (~74MB). The prebuilt DB already holds "
             "their text; download_pdfs.py can re-fetch them if ever needed.",
    )
    args = parser.parse_args()

    os.makedirs(DIST_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    if not os.path.isdir(config.CHROMA_DIR):
        raise SystemExit(
            f"No vector DB found at {config.CHROMA_DIR}. Run `python ingest.py` "
            "first so there is a DB to bundle."
        )

    added = []
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Source + config + docs
        for name in TOP_LEVEL_FILES + OPTIONAL_FILES:
            src = os.path.join(BASE, name)
            if os.path.isfile(src):
                _add_file(zf, src, os.path.join("rag-chatbot", name))
                added.append(name)

        # 2. pipeline package
        _add_tree(zf, os.path.join(BASE, "pipeline"), "rag-chatbot/pipeline")
        added.append("pipeline/")

        # 2b. Streamlit theme
        streamlit_cfg = os.path.join(BASE, ".streamlit", "config.toml")
        if os.path.isfile(streamlit_cfg):
            _add_file(zf, streamlit_cfg, "rag-chatbot/.streamlit/config.toml")
            added.append(".streamlit/config.toml")

        # 3. PDFs (optional)
        pdf_dir = config.PDF_DIR
        if not args.no_pdfs and os.path.isdir(pdf_dir):
            _add_tree(zf, pdf_dir, "rag-chatbot/pdfs", skip_pycache=False)
            added.append("pdfs/")
        elif args.no_pdfs:
            added.append("pdfs/ (SKIPPED via --no-pdfs; run download_pdfs.py to restore)")

        # 4. Prebuilt vector DB -> bundled at data/chroma_db so config finds it
        _add_tree(zf, config.CHROMA_DIR, "rag-chatbot/data/chroma_db",
                  skip_pycache=False)
        added.append(f"data/chroma_db (from {config.CHROMA_DIR})")

    size_mb = os.path.getsize(ZIP_PATH) / 1e6
    print("Bundle created:")
    print(f"  {ZIP_PATH}  ({size_mb:.1f} MB)")
    print("\nIncluded:")
    for a in added:
        print(f"  - {a}")
    print("\nExcluded (intentionally): .venv/, .env, __pycache__/, model caches")
    print("\nSend her the zip. Setup steps are in README.md "
          "('Receiving this project on another machine').")


if __name__ == "__main__":
    main()
