#!/usr/bin/env python3
"""create_phonon_struct.py -- perturb a POSCAR along phonon normal modes.

Thin CLI wrapper; the logic lives in beyblade.vasp (apply_perturbation).
"""
from __future__ import annotations

import argparse
import sys

from pymatgen.io.vasp.inputs import Poscar

from beyblade.vasp import apply_perturbation, load_phonon_data, load_poscar


def parse_index(s: str) -> list[int]:
    if "," in s:
        return [int(x) for x in s.split(",")]
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("poscar_file")
    ap.add_argument("phonon_data")
    ap.add_argument("mode_indices", type=parse_index)
    ap.add_argument("amplitude", type=float,
                    help="Normal coordinate amplitude Q in Angstrom*sqrt(amu)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    structure = load_poscar(args.poscar_file)
    data = load_phonon_data(args.phonon_data)

    for idx in args.mode_indices:
        perturbed = apply_perturbation(structure, data.eigenvectors,
                                       data.atomic_masses, idx, args.amplitude)
        out = args.output if args.output else f"POSCAR_pert_{args.amplitude}_mode_{idx}"
        Poscar(perturbed).comment = f"Mode {idx}, Q={args.amplitude} Ang*sqrt(amu)"
        Poscar(perturbed).write_file(out)
        print(f"Saved: {out}")
    print("\nSuccess!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
