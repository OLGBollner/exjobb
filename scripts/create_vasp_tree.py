#!/usr/bin/env python3
"""Generate the VASP directory tree for a point-defect calculation.

Input: a prepared directory (e.g. <path-to-data>/<defect>_<size>/) containing

    <input>/
        data/       POSCAR, phonons.yaml (copied as-is)
        template/   one subfolder per calculation stage (relax/, ZFS_hyp/,
                    ZFS_occup/), each with staged files such as
                    INCAR.relax, KPOINTS.relax, INCAR.ZFS, KPOINTS.ZFS,
                    plus shared POTCAR and run_vasp

Output: a tree like the one used on the cluster:

    <output>/
        data/               copy of <input>/data
        templates/          copy of <input>/template
        relaxation_data/    seeded from templates/relax
        ZFS_hyp/            seeded from templates/ZFS_hyp
        ZFS_occup/          seeded from templates/ZFS_occup

Each run dir gets its stage's INCAR.<stage> -> INCAR and
KPOINTS.<stage> -> KPOINTS (renamed), together with POTCAR, run_vasp and a
copy of the pristine POSCAR. Files in the template that already have the
plain name are copied as-is.

Usage:
    python create_vasp_tree.py --input <path-to-data>/<defect>_<size>/
"""
import argparse
import shutil
import sys
from pathlib import Path

RUN_STAGES = ("relaxation_data", "ZFS_hyp", "ZFS_occup")
# template subfolder for each run stage
TEMPLATE_FOR_STAGE = {
    "relaxation_data": "relax",
    "ZFS_hyp": "ZFS_hyp",
    "ZFS_occup": "ZFS_occup",
}
SUFFIXED = ("INCAR", "KPOINTS")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="prepared input directory with data/ and template/")
    ap.add_argument("--stages", nargs="+", default=list(RUN_STAGES),
                    help="run directories to create (default: relax, ZFS_hyp, ZFS_occup)")
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

    for stage in args.stages:
        template_stage = TEMPLATE_FOR_STAGE.get(stage, stage)
        seed_dir = template_src / template_stage
        if not seed_dir.is_dir():
            sys.exit(f"Error: template stage '{template_stage}' not found in {template_src}.")

        dst = out / stage
        dst.mkdir(parents=True)
        shutil.copy2(poscar, dst / "POSCAR")
        for f in seed_dir.iterdir():
            if not f.is_file():
                continue
            target = f.name
            for kind in SUFFIXED:
                if f.name.startswith(kind + "."):
                    target = kind
                    break
            shutil.copy2(f, dst / target)
        print(f"  {stage}/  <- templates/{template_stage} + POSCAR")

    print("Done. Edit INCAR/KPOINTS in the ZFS folders manually before submitting.")


if __name__ == "__main__":
    main()
