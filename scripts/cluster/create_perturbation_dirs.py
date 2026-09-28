#!/usr/bin/env python3
"""create_perturbation_dirs.py -- build ready-to-go perturbed ZFS run folders.

Creates, inside a defect folder:

    <defect>/
        first_order/
            pert_<scale>/
                all_bands/
                    input/            INCAR, POSCAR, KPOINTS, POTCAR, CHGCAR
                    run_perturbation_first_order.sh
                defect_band_approx/
                    input/            same files
                    run_perturbation_first_order.sh
        second_order/
            pert_<scale>/
                all_bands/ ...
                defect_band_approx/ ...

The defect name is read from the --output folder name.

Input files are copied verbatim, no rewriting:
  - all_bands input  <- <output>/ZFS_hyp    (fully occupied bands)
  - defect_band_approx input <- <output>/ZFS_occup (defect-band occupation)

Both ZFS folders must exist; otherwise the script aborts before creating
anything.  CHGCAR is copied when present.  Existing pert_<scale> folders are
skipped, never overwritten.

Each generated SLURM script has DEFECT, PHONON_PATH and PERT prefilled with
absolute paths, so it is ready for `sbatch` as-is.

Usage:
    python create_perturbation_dirs.py --phonon NV_512/data/phonon_data.npz \
        [--pert 0.025 0.05] [--output /path/to/NV_512]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

FIRST_ORDER_SCRIPT = "run_perturbation_first_order.sh"
SECOND_ORDER_SCRIPT = "run_perturbation_second_order.sh"
COPIED_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "CHGCAR")
REQUIRED_FILES = COPIED_FILES[:4]  # CHGCAR optional

# basis -> source ZFS folder (relative to the defect/output folder)
BASIS_SOURCE = {
    "all_bands": "ZFS_hyp",
    "defect_band_approx": "ZFS_occup",
}

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def replace_tag(incar: str, tag: str, value: str) -> str:
    """Replace the value of every occurrence of a tag."""
    return re.sub(rf"^(\s*{tag}\s*=\s*)(.+?)\s*(?=$|!|#)",
                  rf"\g<1>{value}", incar, flags=re.M | re.I)


# --------------------------------------------------------------------------- #
# directory preparation
# --------------------------------------------------------------------------- #
def prepare_basis(src: Path, dst: Path, order: str, phonon: Path,
                  defect: str, pert: float) -> None:
    """Fill dst/input/ and write the prefilled SLURM script."""
    inp = dst / "input"
    inp.mkdir(parents=True, exist_ok=True)

    for name in COPIED_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, inp / name)

    script_name = FIRST_ORDER_SCRIPT if order == "first" else SECOND_ORDER_SCRIPT
    tmpl = Path(__file__).resolve().parent / script_name
    if not tmpl.is_file():
        fail(f"missing template script {tmpl}")
        return
    text = tmpl.read_text()
    text = replace_tag(text, "DEFECT", defect)
    text = replace_tag(text, "PHONON_PATH", str(phonon.resolve()))
    text = replace_tag(text, "PERT", str(pert))
    (dst / script_name).write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phonon", type=Path, required=True,
                    help="phonon npz used as-is for both orders")
    ap.add_argument("--pert", type=float, nargs="+", default=[0.025],
                    help="one or more perturbation scales (default: 0.025)")
    ap.add_argument("--output", type=Path, default=None,
                    help="defect folder to write into; the basename is used as "
                         "the defect name (default: ./<defect> requires a name)")
    args = ap.parse_args()

    if args.output is None:
        sys.exit("Error: --output is required (the defect name is read from it)")
    out = args.output.resolve()
    defect = out.name
    phonon = args.phonon.resolve()

    if not phonon.is_file():
        sys.exit(f"Error: phonon file {phonon} does not exist")

    # both source ZFS folders must exist before anything is created
    sources: dict[str, Path] = {}
    for basis, folder in BASIS_SOURCE.items():
        src = out / folder
        if not src.is_dir():
            sys.exit(f"Error: missing {src} -- both ZFS_hyp and ZFS_occup must "
                     "exist before creating perturbation folders")
        for name in REQUIRED_FILES:
            if not (src / name).is_file():
                sys.exit(f"Error: missing {name} in {src}")
        sources[basis] = src

    print(f"=== {defect}: phonon {phonon} -> {out} ===")
    for order in ("first_order", "second_order"):
        for basis, src in sources.items():
            for pert in args.pert:
                dst = out / order / f"pert_{pert}" / basis
                if dst.exists():
                    print(f"  note: {dst} already exists -- skipping")
                    continue
                print(f"  creating {dst}")
                prepare_basis(src, dst, order.split("_")[0], phonon,
                              defect, pert)

    if FAILURES:
        print(f"\n{len(FAILURES)} problem(s) found -- inspect before launching")
        return 1
    print("\nDone. All folders ready for sbatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
