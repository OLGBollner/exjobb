#!/usr/bin/env python3
"""
Script to generate perturbed VASP structures based on a combination of two phonon eigenvectors.

Usage:
    python create_combined_phonon_struct.py <poscar_file> <phonon_file> <mode_i> <mode_j> <perturbation_angstrom>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from beyblade.models import PhononSpectrum
from beyblade.parsers import parse_phonon_data

try:
    from pymatgen.core.structure import Structure
    from pymatgen.io.vasp.inputs import Poscar
except ImportError:
    print("Error: pymatgen is required. Install with: pip install pymatgen")
    sys.exit(1)


def load_phonon_data(phonon_file: str) -> PhononSpectrum:
    """
    Loads phonon data (frequencies, eigenvectors, masses) via the general
    parser, accepting both npz files and phonopy yaml files.
    The eigenvectors are the raw phonopy eigenvectors of the dynamical matrix,
    normalized as sum_ja |e_ja|^2 = 1.
    """
    if not Path(phonon_file).exists():
        raise FileNotFoundError(f"Phonon data file not found: {phonon_file}")

    spectrum = parse_phonon_data(phonon_file)
    if spectrum.eigenvectors.ndim != 3 or spectrum.eigenvectors.shape[2] != 3:
        raise ValueError(
            f"Expected eigenvectors shape (n_modes, n_atoms, 3), got {spectrum.eigenvectors.shape}"
        )

    return spectrum


def load_poscar(poscar_file: str) -> Structure:
    """Load VASP POSCAR file using pymatgen."""
    try:
        poscar = Poscar.from_file(poscar_file)
        return poscar.structure
    except FileNotFoundError:
        raise FileNotFoundError(f"POSCAR file not found: {poscar_file}")
    except Exception as e:
        raise Exception(f"Error reading POSCAR file {poscar_file}: {e!s}")


def apply_combined_perturbation(structure: Structure,
                               eigenvectors: np.ndarray,
                               mode_i: int,
                               mode_j: int,
                               masses: np.ndarray,
                               amplitude: float,
                               original_indices: np.ndarray | None = None) -> Structure:
    """
    Apply a combined phonon perturbation (Mode I + Mode J) to a structure.

    Args:
        structure: pymatgen Structure object
        eigenvectors: phonon eigenvectors array (n_modes, n_atoms, 3), trimmed
        mode_i: First phonon mode index (1-based, in the original DFT numbering)
        mode_j: Second phonon mode index (1-based, in the original DFT numbering)
        amplitude: Perturbation amplitude in Angstroms
        original_indices: original-mode index of each row of `eigenvectors` (the
            "idx" array of the sym-classified file). When given, `mode_i`/`mode_j`
            are resolved to array rows via this mapping instead of assumed contiguous.
    """
    n_modes, n_atoms, _ = eigenvectors.shape

    # Resolve 1-based original-mode labels to rows of the (possibly trimmed)
    # eigenvector array. Without the mapping the labels are 1-based rows.
    if original_indices is not None:
        def _row(label: int) -> int:
            hits = np.flatnonzero(original_indices == label - 1)
            if hits.size != 1:
                raise ValueError(
                    f"Mode {label} not found (or not unique) in original indices "
                    f"[1..{int(original_indices.max()) + 1}]"
                )
            return int(hits[0])
    else:
        def _row(label: int) -> int:
            if not (1 <= label <= n_modes):
                raise ValueError(f"Mode indices must be between 1 and {n_modes}")
            return label - 1

    idx_i = _row(mode_i)
    idx_j = _row(mode_j)

    if len(structure) != n_atoms:
        raise ValueError(f"Structure atoms ({len(structure)}) != Eigenvector atoms ({n_atoms})")


    disp_i = amplitude * eigenvectors[idx_i] / np.sqrt(masses[:, None])  # (n_atoms, 3) in Angstrom
    disp_j = amplitude * eigenvectors[idx_j] / np.sqrt(masses[:, None])  # (n_atoms, 3) in Angstrom

    total_displacements = disp_i + disp_j

    perturbed = structure.copy()
    for atom_idx in range(n_atoms):
        perturbed.sites[atom_idx].coords = structure.sites[atom_idx].coords + total_displacements[atom_idx]

    print(f"Applied combined perturbation: Mode {mode_i} + Mode {mode_j}")
    print(f"Amplitude: {amplitude} Å")
    print(f"Max atom displacement for modes: {np.max(np.linalg.norm(disp_i, axis=1)):.6f} Å and {np.max(np.linalg.norm(disp_j, axis=1)):.6f} Å")

    return perturbed


def save_perturbed_poscar(structure: Structure,
                         output_file: str,
                         mode_i: int,
                         mode_j: int,
                         amplitude: float) -> None:
    """Save perturbed structure to POSCAR file."""
    poscar = Poscar(structure)
    poscar.comment = f"Combined perturbation - Modes {mode_i}+{mode_j}, Amplitude {amplitude} Å"
    poscar.write_file(output_file)
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VASP structure perturbed by two combined phonon modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python create_combined_phonon_struct.py POSCAR phonon_modes.npz 5 6 0.1
  python create_combined_phonon_struct.py POSCAR phonopy.yaml 5 6 0.1
        """
    )

    parser.add_argument('poscar_file', help='Path to VASP POSCAR file')
    parser.add_argument('phonon_file', help='Path to phonon data file (.npz or phonopy .yaml)')
    parser.add_argument('mode_i', type=int, help='First phonon mode index (1-based)')
    parser.add_argument('mode_j', type=int, help='Second phonon mode index (1-based)')
    parser.add_argument('amplitude', type=float, help='Perturbation amplitude in Angstroms')
    parser.add_argument('-o', '--output', default=None, help='Output filename')

    args = parser.parse_args()

    try:
        print("Loading files...")
        structure = load_poscar(args.poscar_file)
        phonon_data = load_phonon_data(args.phonon_file)
        eigenvectors = phonon_data.eigenvectors
        masses = phonon_data.atomic_masses

        print("Applying combined perturbation...")
        perturbed_structure = apply_combined_perturbation(
            structure, 
            eigenvectors, 
            args.mode_i, 
            args.mode_j, 
            masses,
            args.amplitude,
            original_indices=phonon_data.original_indices,
        )

        if args.output is None:
            output_file = f"POSCAR_combined_{args.mode_i}_{args.mode_j}_amp_{args.amplitude}"
        else:
            output_file = args.output

        save_perturbed_poscar(
            perturbed_structure, 
            output_file, 
            args.mode_i, 
            args.mode_j, 
            args.amplitude
        )

        print("\nSuccess!")

    except Exception as e:
        print(f"Error: {e!s}")   # stdout: the slurm wrapper greps $MSG
        sys.exit(1)


if __name__ == "__main__":
    main()
