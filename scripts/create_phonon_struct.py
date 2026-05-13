#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from pathlib import Path

try:
    from pymatgen.core.structure import Structure
    from pymatgen.io.vasp.inputs import Poscar
except ImportError:
    print("Error: pymatgen is required.")
    sys.exit(1)


def parse_index(value):
    try:
        if '-' in value:
            start, end = map(int, value.split('-'))
            return list(range(start, end + 1))
        return [int(value)]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid index format: {value}")


def load_poscar(poscar_file: str) -> Structure:
    try:
        return Poscar.from_file(poscar_file).structure
    except FileNotFoundError:
        raise FileNotFoundError(f"POSCAR not found: {poscar_file}")


def load_phonon_data(npz_file: str) -> dict:
    """
    Loads phonon eigenvectors and masses from an npz file produced by PhononManager.
    Expects keys: 'eigs' (n_modes, n_atoms, 3), 'freqs' (n_modes,), and optionally 'masses' (n_atoms,).
    The eigenvectors are the raw phonopy eigenvectors of the dynamical matrix,
    normalized as sum_ja |e_ja|^2 = 1.
    """
    try:
        data = np.load(npz_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    if 'eigs' not in data:
        raise KeyError(f"'eigs' not found. Available: {list(data.keys())}")
    if data['eigs'].ndim != 3 or data['eigs'].shape[2] != 3:
        raise ValueError(f"Expected eigs shape (n_modes, n_atoms, 3), got {data['eigs'].shape}")

    return data


def apply_perturbation(structure: Structure,
                       eigs: np.ndarray,
                       masses: np.ndarray,
                       mode_index: int,
                       amplitude: float) -> Structure:
    """
    Applies a mass-weighted phonon perturbation to a structure.

    The phonopy eigenvector e_alpha(j) is the column vector of the dynamical matrix,
    normalized as sum_ja |e_ja|^2 = 1. The actual atomic displacement for a normal
    coordinate perturbation Q is:

        Delta r_alpha(j) [Angstrom] = Q [Angstrom*sqrt(amu)] * e_alpha(j) / sqrt(m_j [amu])

    Here `amplitude` is Q in Angstrom*sqrt(amu). For a diamond NV center with C at 12 amu,
    Q=0.1 gives per-atom displacements on the order of 0.1/sqrt(12) ~ 0.029 Angstrom.
    """
    n_modes, n_atoms, _ = eigs.shape
    idx = mode_index - 1

    if not (0 <= idx < n_modes):
        raise ValueError(f"Mode {mode_index} out of range [1, {n_modes}]")
    if len(structure) != n_atoms:
        raise ValueError(f"Atom count mismatch: structure={len(structure)}, eigs={n_atoms}")
    if len(masses) != n_atoms:
        raise ValueError(f"Mass array length {len(masses)} != n_atoms {n_atoms}")

    disp = amplitude * eigs[idx] / np.sqrt(masses[:, None])  # (n_atoms, 3) in Angstrom

    perturbed = structure.copy()
    for j in range(n_atoms):
        perturbed.sites[j].coords = structure.sites[j].coords + disp[j]

    print(f"Mode {mode_index}: max atomic displacement = {np.max(np.linalg.norm(disp, axis=1)):.6f} Å")
    return perturbed


def main():
    parser = argparse.ArgumentParser(
        description="Perturb a VASP POSCAR along a phonon mode using mass-weighted normal coordinates."
    )
    parser.add_argument('poscar_file')
    parser.add_argument('npz_file')
    parser.add_argument('mode_indices', type=parse_index)
    parser.add_argument('amplitude', type=float,
                        help='Normal coordinate amplitude Q in Angstrom*sqrt(amu)')
    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()

    structure = load_poscar(args.poscar_file)
    data = load_phonon_data(args.npz_file)
    eigs = data['eigs']

    if 'masses' not in data:
        raise KeyError("'masses' not found in npz file. Ensure PhononManager stores masses (see read_yaml fix).")
    masses = data['masses']

    for idx in args.mode_indices:
        perturbed = apply_perturbation(structure, eigs, masses, idx, args.amplitude)
        out = args.output if args.output else f"POSCAR_pert_{args.amplitude}_mode_{idx}"
        poscar = Poscar(perturbed)
        poscar.comment = f"Mode {idx}, Q={args.amplitude} Ang*sqrt(amu)"
        poscar.write_file(out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
