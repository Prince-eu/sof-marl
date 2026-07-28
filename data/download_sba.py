"""Download the SBA 7(a) FOIA loan dataset and record a reproducible snapshot.

Reads the CSV URL from config/env.yaml (sba.csv_url), writes the file to sba.raw_path,
and prints the row count and sha256 so the exact snapshot can be recorded in the
technical report (the SBA portal refreshes quarterly, so the snapshot must be pinned).

Usage:
    python data/download_sba.py [--config config/env.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
import yaml


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/env.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["sba"]
    url = cfg["csv_url"]
    if url.startswith("TODO"):
        print(
            "SBA csv_url is not set.\n"
            f"  1. Open the dataset page: {cfg['dataset_page']}\n"
            "  2. Choose the 7(a) resource (e.g. '7(a) (FY2020-Present)').\n"
            "  3. Copy its CSV download URL into config/env.yaml under sba.csv_url.\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        return 2

    out = Path(cfg["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with out.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    with out.open("r", errors="replace") as f:
        n_rows = sum(1 for _ in f) - 1  # exclude header
    digest = sha256_of(out)
    print(f"Saved {out}  rows={n_rows}  sha256={digest}")
    print("Record file name, date, row count, and sha256 in docs/technical_report.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
