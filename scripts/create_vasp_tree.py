#!/usr/bin/env python3
"""Generate the VASP directory tree for a point-defect calculation.

Input: a prepared directory (e.g. <path-to-data>/<defect>_<size>/) containing

    <input>/
        data/       POSCAR, phonons.yaml (copied as-is)
        template/   one subfolder per calculation stage (e.g. relax/),
                    each with INCAR, KPOINTS, POTCAR, run_vasp, ...

Output: a tree like the one used on the cluster:

    <output>/
        data/               copy of <input>/data
        templates/          copy of <input>/template
        relaxation_data/    ready-to-submit run dir for the relaxed structure
        ZFS_hyp/            ZFS calculation (method 1), same files as relax
        ZFS_occup/          ZFS calculation (method 2), same files as relax

Each run dir gets the template files plus a copy of the pristine POSCAR.
INCAR/KPOINTS for the ZFS folders are meant to be modified manually afterwards;
perturbed mode runs are created later by the run_vasp script itself.

Usage:
    python create_vasp_tree.py --input <path-to-data>/<defect>_<size>/
"""
import argparse
import shutil
import sys
from pathlib import Path

RUN_STAGES = ("relaxation_data", "ZFS_hyp", "ZFS_occup")


def copy_tree(src: Path, dst: Path):
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="prepared input directory with data/ and template/")
    ap.add_argument("--run-dirs", nargs="+", default=list(RUN_STAGES),
                    help="run directories to create, each seeded from templates/relax")
    ap.add_argument("--seed-from", default="relax",
                    help="template stage to seed the run dirs from (default: relax)")
    args = ap.parse_args()

    inp = args.input.resolve()
    data_src = inp / "data"
    template_src = inp / "template"
    if not data_src.is_dir() or not template_src.is_dir():
        sys.exit(f"Error: {inp} must contain 'data/' and 'template/' subfolders.")

    name = inp.name
    out = Path(name)
    if out.exists():
        sys.exit(f"Error: output directory {out} already exists, refusing to overwrite.")

    print(f"Creating {out}/ from {inp}")
    copy_tree(data_src, out / "data")
    copy_tree(template_src, out / "templates")

    seed_dir = template_src / args.seed_from
    if not seed_dir.is_dir():
        sys.exit(f"Error: template stage '{args.seed_from}' not found in {template_src}.")
    poscar = data_src / "POSCAR"
    if not poscar.is_file():
        sys.exit(f"Error: {poscar} not found.")

    for stage in args.run_dirs:
        dst = out / stage
        copy_tree(seed_dir, dst)
        shutil.copy2(poscar, dst / "POSCAR")
        print(f"  {stage}/  <- {args.seed_from} template + POSCAR")

    print("Done. Edit INCAR/KPOINTS in the ZFS folders manually before submitting.")


if __name__ == "__main__":
    main()
