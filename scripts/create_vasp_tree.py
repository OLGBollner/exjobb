#!/usr/bin/env python3
"""Generate the VASP directory tree for a point-defect calculation.

Input: a prepared directory (e.g. <path-to-data>/<defect>_<size>/) containing

    <input>/
        data/       POSCAR, phonons.yaml (copied as-is)
        template/   the relax template (all files used by the run folders)

Output: a tree like the one used on the cluster:

    <output>/
        data/               copy of <input>/data
        templates/          copy of <input>/template
        relaxation_data/    all files from template/ + pristine POSCAR
        ZFS_hyp/            same
        ZFS_occup/          same

Usage:
    python create_vasp_tree.py --input <path-to-data>/<defect>_<size>/
"""
import argparse
import shutil
import sys
from pathlib import Path

RUN_STAGES = ("relaxation_data", "ZFS_hyp", "ZFS_occup")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="prepared input directory with data/ and template/")
    args = ap.parse_args()

    inp = args.input.resolve()
    data_src = inp / "data"
    template_src = inp / "template"
    if not template_src.is_dir():
        template_src = inp / "templates"
    if not data_src.is_dir() or not template_src.is_dir():
        sys.exit(f"Error: {inp} must contain 'data/' and 'template/' (or 'templates/') subfolders.")
    poscar = data_src / "POSCAR"
    if not poscar.is_file():
        sys.exit(f"Error: {poscar} not found.")

    name = inp.name
    out = Path(name)
    if out.exists():
        sys.exit(f"Error: output directory {out} already exists, refusing to overwrite.")

    print(f"Creating {out}/ from {inp}")
    shutil.copytree(data_src, out / "data")
    shutil.copytree(template_src, out / "templates")

    for stage in RUN_STAGES:
        dst = out / stage
        dst.mkdir(parents=True)
        shutil.copy2(poscar, dst / "POSCAR")
        for f in template_src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        print(f"  {stage}/  <- template/ + POSCAR")


if __name__ == "__main__":
    main()
