"""Download a free, public-domain PDF corpus (10 large PDFs, each >= 200 pages).

Sources: US government reports (govinfo.gov, public domain) plus one open-license
textbook. All URLs verified downloadable. Run:  python download_pdfs.py

Use --limit N to fetch only the first N (e.g. `--limit 2` for a quick test).
If a URL ever changes, download manually into pdfs/; the pipeline only needs
pdfs/ to contain PDFs of >= 200 pages each.
"""
from __future__ import annotations

import os
import urllib.request

import config

# (filename, url). All verified 200 OK, application/pdf, >= 200 pages.
# Deliberately moderate-sized (no 1000+ page docs) to keep CPU ingest time down
# while still meeting the "10 PDFs, each >= 200 pages" requirement.
# Ordered smallest-first so a quick `--limit 2` test downloads fast.
CORPUS = [
    ("911_Commission_Report.pdf",
     "https://www.govinfo.gov/content/pkg/GPO-911REPORT/pdf/GPO-911REPORT.pdf"),          # ~585 pg
    ("ThinkStats2.pdf",
     "https://greenteapress.com/thinkstats2/thinkstats2.pdf"),                            # ~226 pg
    ("ThinkPython2.pdf",
     "https://greenteapress.com/thinkpython2/thinkpython2.pdf"),                          # ~290 pg
    ("Economic_Report_President_2024.pdf",
     "https://www.govinfo.gov/content/pkg/ERP-2024/pdf/ERP-2024.pdf"),                    # ~450 pg
    ("Economic_Report_President_2023.pdf",
     "https://www.govinfo.gov/content/pkg/ERP-2023/pdf/ERP-2023.pdf"),
    ("Economic_Report_President_2022.pdf",
     "https://www.govinfo.gov/content/pkg/ERP-2022/pdf/ERP-2022.pdf"),
    ("Economic_Report_President_2021.pdf",
     "https://www.govinfo.gov/content/pkg/ERP-2021/pdf/ERP-2021.pdf"),
    ("Economic_Report_President_2020.pdf",
     "https://www.govinfo.gov/content/pkg/ERP-2020/pdf/ERP-2020.pdf"),
    ("Economic_Report_President_2019.pdf",
     "https://www.govinfo.gov/content/pkg/ERP-2019/pdf/ERP-2019.pdf"),
    ("Economic_Report_President_2018.pdf",
     "https://www.govinfo.gov/content/pkg/ERP-2018/pdf/ERP-2018.pdf"),
]


def _download(filename: str, url: str, dest_dir: str):
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  skip (exists): {filename}")
        return
    print(f"  downloading: {filename} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    size_mb = os.path.getsize(dest) / 1e6
    print(f"    done ({size_mb:.1f} MB)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download the free PDF corpus.")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Download only the first N PDFs (0 = all). Useful for a quick test.",
    )
    args = parser.parse_args()

    corpus = CORPUS[: args.limit] if args.limit > 0 else CORPUS

    os.makedirs(config.PDF_DIR, exist_ok=True)
    print(f"Downloading {len(corpus)} PDF(s) into {config.PDF_DIR}")
    for filename, url in corpus:
        try:
            _download(filename, url, config.PDF_DIR)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"    FAILED: {filename} -> {e}")
            print("    Download it manually from openstax.org into pdfs/.")
    print("\nDone. Next:  python ingest.py")


if __name__ == "__main__":
    main()
