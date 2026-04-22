#!/usr/bin/env python3
"""
Script to generate perturbed VASP structures based on phonon eigenvectors.

This script takes a POSCAR file, phonon eigenvector data, a mode index, and a
perturbation amplitude to create a new perturbed structure.

Usage:
    python perturb_phonon.py <poscar_file> <npz_file> <mode_index> <perturbation_angstrom>

Arguments:
    poscar_file: Path to VASP POSCAR file
    npz_file: Path to npz file containing 'eigs' array with shape (n_modes, n_atoms, 3)
    mode_index: Phonon mode index (starting from 1)
    perturbation_angstrom: Perturbation amplitude in Angstroms

Example:
    python perturb_phonon.py POSCAR phonon_modes.npz 1 0.1
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
    """
    Load phonon eigenvectors from npz file.

    Args:
        npz_file: Path to npz file containing 'eigs' array

    Returns:
        numpy array with shape (n_modes, n_atoms, 3) containing eigenvector perturbations

    Raises:
        FileNotFoundError: If npz file doesn't exist
        KeyError: If 'eigs' key not found in npz file
        ValueError: If 'eigs' array doesn't have expected shape
    """
    try:
        data = np.load(npz_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    if 'eigs' not in data:
        available_keys = list(data.keys())
        raise KeyError(f"'eigs' key not found in {npz_file}. Available keys: {available_keys}")

    eigs = data['eigs']

    if len(eigs.shape) != 3:
        raise ValueError(f"Expected 'eigs' to have 3 dimensions (n_modes, n_atoms, 3), "
                        f"but got shape {eigs.shape}")

    if eigs.shape[2] != 3:
        raise ValueError(f"Expected 'eigs' to have 3 components per atom (x,y,z), "
                        f"but got shape {eigs.shape}")

    print(f"Loaded phonon eigenvectors with shape: {eigs.shape}")
    print(f"Number of modes: {eigs.shape[0]}")
    print(f"Number of atoms: {eigs.shape[1]}")

    return eigs


def load_poscar(poscar_file: str) -> Structure:
    """
    Load VASP POSCAR file using pymatgen.

    Args:
        poscar_file: Path to POSCAR file

    Returns:
        pymatgen Structure object

    Raises:
        FileNotFoundError: If POSCAR file doesn't exist
        Exception: If POSCAR file cannot be parsed
    """
    try:
        poscar = Poscar.from_file(poscar_file)
        structure = poscar.structure
        print(f"Loaded structure from {poscar_file}")
        print(f"Number of atoms: {len(structure)}")
        print(f"Lattice: {structure.lattice}")
        return structure
    except FileNotFoundError:
        raise FileNotFoundError(f"POSCAR file not found: {poscar_file}")
    except Exception as e:
        raise Exception(f"Error reading POSCAR file {poscar_file}: {str(e)}")


def apply_phonon_perturbation(structure: Structure,
                             eigenvectors: np.ndarray,
                             mode_index: int,
                             amplitude: float) -> Structure:
    """
    Apply phonon perturbation to a structure.

    Args:
        structure: pymatgen Structure object
        eigenvectors: phonon eigenvectors array with shape (n_modes, n_atoms, 3)
        mode_index: phonon mode index (1-based)
        amplitude: perturbation amplitude in Angstroms

    Returns:
        New pymatgen Structure object with applied perturbation

    Raises:
        ValueError: If mode_index is out of range or number of atoms doesn't match
    """
    n_modes, n_atoms, n_components = eigenvectors.shape

    # Convert to 0-based indexing
    mode_idx = mode_index - 1

    if mode_idx < 0 or mode_idx >= n_modes:
        raise ValueError(f"Mode index {mode_index} is out of range. "
                        f"Available modes: 1 to {n_modes}")

    if len(structure) != n_atoms:
        raise ValueError(f"Number of atoms in structure ({len(structure)}) doesn't match "
                        f"number of atoms in eigenvectors ({n_atoms})")

    # Get the eigenvector for the specified mode
    mode_eigenvector = eigenvectors[mode_idx]  # shape: (n_atoms, 3)

    # Create a copy of the structure
    perturbed_structure = structure.copy()

    # Apply perturbation to each atom
    for atom_idx in range(n_atoms):
        # Get current cartesian coordinates
        old_coords = structure.sites[atom_idx].coords  # cartesian coordinates in Angstroms

        # Get perturbation vector for this atom (already in Angstroms)
        perturbation = amplitude * mode_eigenvector[atom_idx]

        # Apply perturbation
        new_coords = old_coords + perturbation

        # Update the site coordinates
        #perturbed_structure.sites[atom_idx] = perturbed_structure.sites[atom_idx].to_unit_cell()
        perturbed_structure.sites[atom_idx].coords = new_coords

    print(f"Applied perturbation for mode {mode_index} with amplitude {amplitude} Å")
    print(f"Maximum displacement: {np.max(np.linalg.norm(amplitude * mode_eigenvector, axis=1)):.6f} Å")

    return perturbed_structure


def save_perturbed_poscar(structure: Structure,
                         output_file: str,
                         mode_index: int,
                         amplitude: float) -> None:
    """
    Save perturbed structure to POSCAR file.

    Args:
        structure: pymatgen Structure object
        output_file: output POSCAR filename
        mode_index: phonon mode index used for perturbation
        amplitude: perturbation amplitude used
    """
    poscar = Poscar(structure)

    # Add comment with perturbation info
    comment = f"Perturbed structure - Mode {mode_index}, Amplitude {amplitude} Å"
    poscar.comment = comment

    poscar.write_file(output_file)
    print(f"Saved perturbed structure to: {output_file}")

def parse_index(value):
    try:
      if '-' in value:
        start, end = map(int, value.split('-'))
        return list(range(start, end+1))
      else:
        return [int(value)]
    except ValueError:
      raise argparse.ArgumentTypeError(f"Invalid index format: {value}")

def main():
    """Main function to run the phonon perturbation script."""
    parser = argparse.ArgumentParser(
        description="Generate perturbed VASP structure based on phonon eigenvectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python perturb_phonon.py POSCAR phonon_modes.npz 1 0.1
  python perturb_phonon.py input.poscar modes.npz 5 0.05
        """
    )

    parser.add_argument('poscar_file',
                       help='Path to VASP POSCAR file')
    parser.add_argument('npz_file',
                       help="Path to npz file containing 'eigs' array")
    parser.add_argument('mode_indices', type=parse_index,
                       help='Phonon mode index (starting from 1)')
    parser.add_argument('perturbation_amplitude', type=float,
                       help='Perturbation amplitude in Angstroms')
    parser.add_argument('-o', '--output', default=None,
                       help='Output POSCAR filename (default: POSCAR_perturbed_mode_X)')

    args = parser.parse_args()

    try:
        # Load input files
        print("Loading input files...")
        structure = load_poscar(args.poscar_file)
        eigenvectors = load_phonon_eigenvectors(args.npz_file)

        for idx in args.mode_indices:
            # Apply perturbation
            print(f"\nApplying perturbation...")
            perturbed_structure = apply_phonon_perturbation(
                structure, eigenvectors, idx, args.perturbation_amplitude
            )

            # Generate output filename if not provided
            if args.output is None:
                output_file = f"POSCAR_pert_{args.perturbation_amplitude}_mode_{idx}"
            else:
                output_file = args.output

            # Save result
            print(f"\nSaving result...")
            save_perturbed_poscar(perturbed_structure, output_file,
                                  idx, args.perturbation_amplitude)

            print(f"\nSuccess! Perturbed structure saved to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
