"""Generalized phonon-mode symmetry classification.

Pipeline:
1. Detect the point group of the defect structure via pymatgen's
   SpacegroupAnalyzer (spglib backend).
2. For each phonon mode, compute its character under each group
   operation (atom permutation + displacement rotation).
3. Project the character vector onto each irrep of the group's
   character table; degenerate pairs are matched with summed
   characters (2D irreps).

The analytic selection of coupling coefficients from the ZFS matrix
remains a per-group mapping table; only irrep labels cross
this seam.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ---------------------------------------------------------------------------
# Character tables (Hermann-Mauguin spglib point-group symbols as keys).
# χ vectors are ordered to match _operation_keys().
# ---------------------------------------------------------------------------

CHARACTER_TABLES: dict[str, dict[str, list[float]]] = {
    # C3v: operations (E, C3, C3^2, σv1, σv2, σv3)
    "3m": {
        "A1": [1, 1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1, -1],
        "E": [2, -1, -1, 0, 0, 0],
    },
}


@dataclass(frozen=True)
class PointGroup:
    """Detected point group of a structure."""
    symbol: str                # Hermann-Mauguin point-group symbol (spglib)
    order: int                 # number of symmetry operations
    operations: tuple          # rotation matrices (3x3)
    space_group: str           # international space group symbol


def detect_point_group(structure: Structure, symprec: float = 1e-3) -> PointGroup:
    """Detect the point group of a (defect) supercell structure."""
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    symm_ops = sga.get_symmetry_operations()
    return PointGroup(
        symbol=sga.get_point_group_symbol(),
        order=len(symm_ops),
        operations=tuple(op.rotation_matrix for op in symm_ops),
        space_group=sga.get_space_group_symbol(),
    )


def detect_point_group_from_spectrum(spectrum, symprec: float = 1e-3) -> PointGroup:
    """Build a pymatgen Structure from a PhononSpectrum and detect its group."""
    structure = Structure(
        lattice=spectrum.lattice,
        species=spectrum.atom_symbols,
        coords=spectrum.atom_frac_coords,
        coords_are_cartesian=False,
    )
    return detect_point_group(structure, symprec=symprec)


def classify_modes(spectrum, tol_mev: float = 0.05, symprec: float = 1e-3) -> list[str]:
    """Classify each phonon mode into irreps of the detected point group.

    Returns a list of irrep labels, one per mode. Degenerate partners of
    multi-dimensional irreps share the same label (optionally with a
    partner suffix when distinguishable, e.g. "Ex"/"Ey" for C3v).
    """
    pg = detect_point_group_from_spectrum(spectrum, symprec=symprec)
    table = CHARACTER_TABLES.get(pg.symbol)
    if table is None:
        raise NotImplementedError(
            f"No character table for point group '{pg.symbol}'. "
            "Add one to CHARACTER_TABLES."
        )

    chars = _mode_characters(spectrum, pg)
    labels = _match_irreps(spectrum, chars, table, tol_mev)
    spectrum.symmetries = labels
    return labels


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _mode_characters(spectrum, pg: PointGroup) -> np.ndarray:
    """χ_m(R) = <ψ_m | R ψ_m> for each mode m and operation R.

    R acts on the eigenvector by permuting atoms and rotating their
    displacement vectors: ψ'_i = R_cart · ψ_{σ(i)}.
    """
    eigs = np.asarray(spectrum.eigenvectors)
    n_modes = eigs.shape[0]
    if eigs.ndim == 3:
        vecs = eigs                     # (n_modes, n_atoms, 3)
    else:
        n_atoms = eigs.shape[-1] // 3   # flat (n_modes, 3N)
        vecs = eigs.reshape(n_modes, n_atoms, 3)
    n_atoms = vecs.shape[1]

    frac = np.mod(np.asarray(spectrum.atom_frac_coords, dtype=float), 1.0)
    inv_lat = np.linalg.inv(np.asarray(spectrum.lattice, dtype=float))
    cart_atoms = frac @ np.asarray(spectrum.lattice, dtype=float)
    symbols = np.asarray(spectrum.atom_symbols)

    chars = np.zeros((n_modes, pg.order))
    for k, R in enumerate(pg.operations):
        R = np.asarray(R, dtype=float)
        mapping = _atom_mapping(R, cart_atoms, frac, inv_lat, symbols)
        rotated = vecs @ R.T                 # rotate displacements
        permuted = rotated[:, mapping, :]    # move to where R sends each atom
        # <ψ | Rψ> summed over atoms
        chars[:, k] = np.sum(vecs * permuted, axis=(1, 2))
    return chars


def _atom_mapping(R, cart_atoms, frac, inv_lat, symbols):
    """σ: for each atom i, index of the atom R sends i onto."""
    n_atoms = cart_atoms.shape[0]
    mapping = np.zeros(n_atoms, dtype=int)
    rotated_cart = cart_atoms @ R.T
    rot_frac = np.mod(rotated_cart @ inv_lat, 1.0)
    orig_frac = frac
    for i in range(n_atoms):
        diffs = np.mod(orig_frac - rot_frac[i] + 0.5, 1.0) - 0.5
        dists = np.linalg.norm(diffs @ np.asarray(inv_lat).T, axis=1)
        valid = np.where(symbols == symbols[i])[0]
        mapping[i] = valid[np.argmin(dists[valid])] if len(valid) else i
    return mapping


def _match_irreps(spectrum, chars, table, tol_mev):
    """Assign irrep labels by projection; pair degenerate modes for E."""
    n = spectrum.n_modes
    freqs = spectrum.frequencies_mev
    labels: list[Optional[str]] = [None] * n
    assigned: set[int] = set()

    ops_sorted = np.argsort(-np.abs(chars[:, 0]))  # noop, keep op order
    # Pair candidates by frequency degeneracy first.
    pairs: list[tuple[int, int]] = []
    singles: list[int] = []
    for i in range(n):
        if i in assigned:
            continue
        partner = None
        for j in range(n):
            if j != i and j not in assigned and abs(freqs[j] - freqs[i]) < tol_mev:
                partner = j
                break
        if partner is not None:
            pairs.append((i, partner))
            assigned.update([i, partner])
        else:
            singles.append(i)
            assigned.add(i)

    # Match pairs against 2D irreps (summed characters).
    for i, j in pairs:
        summed = chars[i] + chars[j]
        best, best_score = None, -1.0
        for name, chi in table.items():
            chi = np.asarray(chi)
            score = abs(np.dot(summed, chi)) / (np.linalg.norm(summed) * np.linalg.norm(chi) + 1e-12)
            if score > best_score:
                best, best_score = name, score
        labels[i] = labels[j] = best
        if best == "E":
            labels[i], labels[j] = _split_e_partners(spectrum, chars, i, j)

    # Match singles against 1D irreps only.
    for i in singles:
        best, best_score = None, -1.0
        for name, chi in table.items():
            chi = np.asarray(chi)
            if np.allclose(chi, chi[0]) and len(set(np.asarray(chi).round(6))) == 1 and chi[0] != 2:
                pass  # 1D irrep
            # Heuristic: 2D irreps have χ(E)=2, skip for singles.
            if chi[0] != 1:
                continue
            score = abs(np.dot(chars[i], chi)) / (np.linalg.norm(chars[i]) * np.linalg.norm(chi) + 1e-12)
            if score > best_score:
                best, best_score = name, score
        labels[i] = best
    return labels


def _split_e_partners(spectrum, chars, i, j):
    """Distinguish the two partners of a 2D irrep (Ex/Ey for C3v-like groups).

    Use the sign of the character under the first reflection-like
    operation (χ = 0 for both partners of C3v E under σv, so fall back
    to the sign of the inner product between partners).
    """
    if i == j:
        return "Ex", "Ey"
    inner = float(np.dot(chars[i], chars[j]))
    return ("Ex", "Ey") if inner >= 0 else ("Ey", "Ex")
