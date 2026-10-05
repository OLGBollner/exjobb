#!/usr/bin/env python3
"""Download the httk-symgen symmetry reference datasets into httk-sym/.

Data from the httk Spacegroup Data Explorer (https://symdata.anyterial.se/),
distributed under CC-BY 4.0. The files are gitignored: they are reference
data, regenerated on demand, not source.
"""
from pathlib import Path
import urllib.request

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
        dest = OUT / name
        url = f"{BASE}/{name}"
        print(f"{url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
