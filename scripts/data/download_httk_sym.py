#!/usr/bin/env python3
"""Download and extract the httk-symgen symmetry reference datasets.

Data from the httk Spacegroup Data Explorer (https://symdata.anyterial.se/),
distributed under CC-BY 4.0. The downloaded files are gitignored: reference
data, regenerated on demand, not source.

Fetch and extract with:

    python3 scripts/data/download_httk_sym.py
"""
import gzip
import shutil
import urllib.request
from pathlib import Path

BASE = "https://symdata.anyterial.se/data"
FILES = [
    "symmetry_basics.json.gz",
    "transformations_hm_entry.json.gz",
    "transformations_std.json.gz",
]
OUT = Path(__file__).resolve().parents[2] / "httk-sym"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name in FILES:
        gz_path = OUT / name
        json_path = gz_path.with_suffix("")  # strip .gz
        url = f"{BASE}/{name}"
        if json_path.exists():
            print(f"{json_path.name}: already extracted, skipping")
            continue
        print(f"{url} -> {gz_path.name}")
        urllib.request.urlretrieve(url, gz_path)
        print(f"  downloaded {gz_path.stat().st_size / 1e6:.1f} MB, extracting...")
        with gzip.open(gz_path, "rb") as fin, open(json_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        gz_path.unlink()
        print(f"  wrote {json_path.name} ({json_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
