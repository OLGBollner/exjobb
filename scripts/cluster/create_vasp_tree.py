#!/usr/bin/env python3
"""Generate the VASP directory tree for a point-defect calculation.

Input: a prepared directory (e.g. <path-to-data>/<defect>_<size>/) containing

    <input>/
        data/          POSCAR (or structure.vasp), phonons.yaml, ...
        template/      relax/   INCAR, KPOINTS, POTCAR, run_vasp, ...

Output: a tree like the one used on the cluster:

    <output>/
        data/               copy of <input>/data
        relaxation_data/    files from template/relax + pristine POSCAR
        ZFS_hyp/            files from template/relax + pristine POSCAR
        ZFS_occup/          files from template/relax + pristine POSCAR

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
    ap.add_argument("--run-dirs", nargs="+", default=list(RUN_STAGES),
                    help="run directories to create, each seeded from template/relax")
    args = ap.parse_args()

    inp = args.input.resolve()
    data_src = inp / "data"
    relax_src = inp / "template" / "relax"
    if not data_src.is_dir() or not relax_src.is_dir():
        sys.exit(f"Error: {inp} must contain 'data/' and 'template/relax/'.")

    poscar = data_src / "POSCAR"
    struct_file = poscar if poscar.is_file() else data_src / "structure.vasp"
    if not struct_file.is_file():
        sys.exit(f"Error: no POSCAR or structure.vasp in {data_src}.")

    out = Path(inp.name)
    if out.exists():
        sys.exit(f"Error: output directory {out} already exists, refusing to overwrite.")

    print(f"Creating {out}/ from {inp}")
    shutil.copytree(data_src, out / "data")

    for stage in args.run_dirs:
        dst = out / stage
        shutil.copytree(relax_src, dst)
        shutil.copy2(struct_file, dst / "POSCAR")
        print(f"  {stage}/  <- template/relax + POSCAR")

    print("Done.")


if __name__ == "__main__":
    main()
