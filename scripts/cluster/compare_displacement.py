#!/usr/bin/env python3
"""compare_displacement.py -- how large is a perturbed POSCAR, really?

Thin CLI wrapper; the logic lives in beyblade.vasp (see its docstrings).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from beyblade.vasp import compare_displacement_cli


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report max/rms Cartesian displacement of one or more "
        "perturbed POSCARs relative to the relaxed reference"
    )
    ap.add_argument("reference", type=Path, help="relaxed reference POSCAR (e.g. relax/ZFS_hyp/POSCAR)")
    ap.add_argument("perturbed", type=Path, nargs="+", help="perturbed POSCAR(s) to compare (Q runs, old vs new)")
    args = ap.parse_args()

    if not args.reference.is_file():
        sys.exit(f"Error: reference not found: {args.reference}")
    missing = [p for p in args.perturbed if not p.is_file()]
    if missing:
        sys.exit(f"Error: not found: {', '.join(map(str, missing))}")

    compare_displacement_cli(args.reference, args.perturbed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
