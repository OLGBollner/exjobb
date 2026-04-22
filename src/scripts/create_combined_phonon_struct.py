#!/usr/bin/env python3
"""
Script to generate perturbed VASP structures based on a combination of two phonon eigenvectors.

Usage:
    python create_combined_phonon_struct.py <poscar_file> <npz_file> <mode_i> <mode_j> <perturbation_angstrom>
"""

import argparse
import sys
import numpy as np
from pathlib import Path

try:
    from pymatgen.core.structure import Structure
    from pymatgen.io.vasp.inputs import Poscar
except ImportError:
    print("Error: pymatgen is required. Install with: pip install pymatgen")
    sys.exit(1)


def load_phonon_eigenvectors(npz_file: str) -> np.ndarray:
    """Load phonon eigenvectors from npz file."""
    try:
        data = np.load(npz_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    if 'eigs' not in data:
        available_keys = list(data.keys())
        raise KeyError(f"'eigs' key not found in {npz_file}. Available keys: {available_keys}")

    eigs = data['eigs']
    if len(eigs.shape) != 3 or eigs.shape[2] != 3:
        raise ValueError(f"Expected 'eigs' shape (n_modes, n_atoms, 3), got {eigs.shape}")

    print(f"Loaded {eigs.shape[0]} modes for {eigs.shape[1]} atoms.")
    return eigs


def load_poscar(poscar_file: str) -> Structure:
    """Load VASP POSCAR file using pymatgen."""
    try:
        poscar = Poscar.from_file(poscar_file)
        return poscar.structure
    except FileNotFoundError:
        raise FileNotFoundError(f"POSCAR file not found: {poscar_file}")
    except Exception as e:
        raise Exception(f"Error reading POSCAR file {poscar_file}: {str(e)}")


def apply_combined_perturbation(structure: Structure,
                               eigenvectors: np.ndarray,
                               mode_i: int,
                               mode_j: int,
                               amplitude: float) -> Structure:
    """
    Apply a combined phonon perturbation (Mode I + Mode J) to a structure.

    Args:
        structure: pymatgen Structure object
        eigenvectors: phonon eigenvectors array (n_modes, n_atoms, 3)
        mode_i: First phonon mode index (1-based)
        mode_j: Second phonon mode index (1-based)
        amplitude: Perturbation amplitude in Angstroms
    """
    n_modes, n_atoms, _ = eigenvectors.shape

    # Convert to 0-based indexing
    idx_i = mode_i - 1
    idx_j = mode_j - 1

    if not (0 <= idx_i < n_modes) or not (0 <= idx_j < n_modes):
        raise ValueError(f"Mode indices must be between 1 and {n_modes}")

    if len(structure) != n_atoms:
        raise ValueError(f"Structure atoms ({len(structure)}) != Eigenvector atoms ({n_atoms})")

    # Combine the eigenvectors
    combined_vector = eigenvectors[idx_i] + eigenvectors[idx_j]
    
    # Calculate actual displacements
    total_displacements = amplitude * combined_vector

    # Create a copy and apply perturbation
    perturbed_structure = structure.copy()
    for atom_idx in range(n_atoms):
        old_coords = structure.sites[atom_idx].coords
        new_coords = old_coords + total_displacements[atom_idx]
        perturbed_structure.sites[atom_idx].coords = new_coords

    print(f"Applied combined perturbation: Mode {mode_i} + Mode {mode_j}")
    print(f"Amplitude: {amplitude} Å")
    print(f"Max atom displacement: {np.max(np.linalg.norm(total_displacements, axis=1)):.6f} Å")

    return perturbed_structure


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
        """
    )

    parser.add_argument('poscar_file', help='Path to VASP POSCAR file')
    parser.add_argument('npz_file', help="Path to npz file containing 'eigs' array")
    parser.add_argument('mode_i', type=int, help='First phonon mode index (1-based)')
    parser.add_argument('mode_j', type=int, help='Second phonon mode index (1-based)')
    parser.add_argument('amplitude', type=float, help='Perturbation amplitude in Angstroms')
    parser.add_argument('-o', '--output', default=None, help='Output filename')

    args = parser.parse_args()

    try:
        print("Loading files...")
        structure = load_poscar(args.poscar_file)
        eigenvectors = load_phonon_eigenvectors(args.npz_file)

        print("Applying combined perturbation...")
        perturbed_structure = apply_combined_perturbation(
            structure, 
            eigenvectors, 
            args.mode_i, 
            args.mode_j, 
            args.amplitude
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
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()