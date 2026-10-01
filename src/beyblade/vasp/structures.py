"""Perturb VASP structures along phonon normal coordinates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar

from beyblade.models import PhononSpectrum
from beyblade.parsers import parse_phonon_data


def load_poscar(poscar_file: str | Path) -> Structure:
    """Load a VASP POSCAR file into a pymatgen Structure."""
    return Poscar.from_file(poscar_file).structure


def load_phonon_data(phonon_file: str | Path) -> PhononSpectrum:
    """Load phonon data (npz or phonopy yaml) via the general parser.

    The eigenvectors are the raw phonopy eigenvectors of the dynamical
    matrix, normalized as sum_ja |e_ja|^2 = 1.
    """
    if not Path(phonon_file).exists():
        raise FileNotFoundError(f"Phonon data file not found: {phonon_file}")

    spectrum = parse_phonon_data(phonon_file)
    if spectrum.eigenvectors.ndim != 3 or spectrum.eigenvectors.shape[2] != 3:
        raise ValueError(
            f"Expected eigenvectors shape (n_modes, n_atoms, 3), "
            f"got {spectrum.eigenvectors.shape}"
        )
    return spectrum


def _displacement(eigenvectors: np.ndarray, masses: np.ndarray,
                  index0: int, amplitude: float) -> np.ndarray:
    """Mass-weighted displacement for one mode, in Angstrom.

    amplitude is the normal coordinate Q in Angstrom*sqrt(amu):
    Delta r_alpha(j) = Q * e_alpha(j) / sqrt(m_j).
    """
    return amplitude * eigenvectors[index0] / np.sqrt(masses[:, None])


def _check_mode(eigenvectors: np.ndarray, structure: Structure,
                masses: np.ndarray, index0: int, n_modes: int) -> None:
    if not (0 <= index0 < n_modes):
        raise ValueError(f"Mode index {index0 + 1} out of range [1, {n_modes}]")
    if len(structure) != eigenvectors.shape[1]:
        raise ValueError(
            f"Structure atoms ({len(structure)}) != "
            f"Eigenvector atoms ({eigenvectors.shape[1]})"
        )
    if len(masses) != len(structure):
        raise ValueError(
            f"Mass array length {len(masses)} != n_atoms {len(structure)}"
        )


def apply_perturbation(structure: Structure, eigs: np.ndarray,
                       masses: np.ndarray, mode_index: int,
                       amplitude: float) -> Structure:
    """Apply a mass-weighted single-mode perturbation to a structure.

    mode_index is 1-based; amplitude is Q in Angstrom*sqrt(amu). For a
    carbon (12 amu), Q=0.1 gives per-atom displacements ~0.1/sqrt(12) ~
    0.029 Angstrom.
    """
    n_modes, _, _ = eigs.shape
    idx = mode_index - 1
    _check_mode(eigs, structure, masses, idx, n_modes)

    disp = _displacement(eigs, masses, idx, amplitude)  # (n_atoms, 3)

    perturbed = structure.copy()
    for j in range(len(structure)):
        perturbed.sites[j].coords = structure.sites[j].coords + disp[j]

    print(f"Mode {mode_index}: max atomic displacement = "
          f"{np.max(np.linalg.norm(disp, axis=1)):.6f} Å")
    return perturbed


def apply_combined_perturbation(structure: Structure,
                                eigenvectors: np.ndarray,
                                mode_i: int, mode_j: int,
                                masses: np.ndarray,
                                amplitude: float) -> Structure:
    """Apply a two-mode combined perturbation (Mode I + Mode J, 1-based)."""
    n_modes, _, _ = eigenvectors.shape
    idx_i, idx_j = mode_i - 1, mode_j - 1
    _check_mode(eigenvectors, structure, masses, idx_i, n_modes)
    _check_mode(eigenvectors, structure, masses, idx_j, n_modes)

    disp = (_displacement(eigenvectors, masses, idx_i, amplitude)
            + _displacement(eigenvectors, masses, idx_j, amplitude))

    perturbed = structure.copy()
    for j in range(len(structure)):
        perturbed.sites[j].coords = structure.sites[j].coords + disp[j]

    print(f"Applied combined perturbation: Mode {mode_i} + Mode {mode_j}, "
          f"amplitude {amplitude} Å; max displacements "
          f"{np.max(np.linalg.norm(_displacement(eigenvectors, masses, idx_i, amplitude), axis=1)):.6f} Å "
          f"and {np.max(np.linalg.norm(_displacement(eigenvectors, masses, idx_j, amplitude), axis=1)):.6f} Å")
    return perturbed


def compare_displacement(reference: str | Path,
                         perturbed: str | Path) -> dict:
    """Displacement of a perturbed structure relative to a reference.

    Returns per-atom displacement vectors (n_atoms, 3) plus summary
    statistics. Atoms are matched by index: both files must come from the
    same phonon/defect tree, so the atom order is identical.
    """
    ref = load_poscar(reference)
    pert = load_poscar(perturbed)
    if len(ref) != len(pert):
        raise ValueError(f"Atom count mismatch: {len(ref)} vs {len(pert)}")
    if ref.composition.reduced_formula != pert.composition.reduced_formula:
        raise ValueError(
            f"Composition mismatch: {ref.composition.reduced_formula} vs "
            f"{pert.composition.reduced_formula}")
    disp = np.array([pert.sites[j].coords - ref.sites[j].coords
                     for j in range(len(ref))])
    norms = np.linalg.norm(disp, axis=1)
    return {
        "disp": disp,
        "max": float(norms.max()),
        "rms": float(np.sqrt((norms ** 2).mean())),
        "max_index": int(norms.argmax()),
        "n_atoms": len(ref),
    }


def compare_displacement_cli(reference: str | Path,
                             perturbed: list[str | Path]) -> None:
    """Print a table of displacement statistics for one or more structures."""
    rows = []
    for path in perturbed:
        try:
            stats = compare_displacement(reference, path)
        except (FileNotFoundError, ValueError) as e:
            print(f"{path}: {e}")
            continue
        rows.append((path, stats))
        print(f"{path}\n    max |disp| = {stats['max']:.6f} Å "
              f"(atom {stats['max_index'] + 1})"
              f"\n    rms        = {stats['rms']:.6f} Å"
              f"\n    n_atoms    = {stats['n_atoms']}\n")
    if len(rows) > 1:
        print("Relative sizes (max |disp|, first entry = 1.000):")
        base = rows[0][1]["max"]
        for path, stats in rows:
            print(f"    {stats['max'] / base:8.3f}  {path}")


def write_perturbed_poscar(structure: Structure, output_file: str,
                           comment: str) -> None:
    poscar = Poscar(structure)
    poscar.comment = comment
    poscar.write_file(output_file)
