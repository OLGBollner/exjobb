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
from typing import Any, Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ---------------------------------------------------------------------------
# Character tables (Hermann-Mauguin spglib point-group symbols as keys).
# χ vectors are ordered to match _operation_keys().
# ---------------------------------------------------------------------------

CHARACTER_TABLES: dict[str, dict[str, Any]] = {
    # C3v: operations (E, C3, C3^2, σv1, σv2, σv3)
    # 2D irreps carry sublabels: the character pattern of each component,
    # used for per-mode matching and for pairing (Ex/Ey etc.).
    "3m": {
        "A1": [1, 1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1, -1],
        "E": {
            "sublabels": {
                "Ex": [1, -1, -1, 1, 1, 1],
                "Ey": [1, -1, -1, -1, -1, -1],
            }
        },
    },
    # C2v: operations (E, C2, σv, σv')
    "mm2": {
        "A1": [1, 1, 1, 1],
        "A2": [1, 1, -1, -1],
        "B1": [1, -1, 1, -1],
        "B2": [1, -1, -1, 1],
    },
    # D3h: operations (E, 2C3, 3C2', σh, 2S3, 3σv)
    "-6m2": {
        "A1'": [1, 1, 1, 1, 1, 1],
        "A2'": [1, 1, -1, 1, 1, -1],
        "E'": [1, -1, 0, 1, -1, 0],
        "A1''": [1, 1, 1, -1, -1, -1],
        "A2''": [1, 1, -1, -1, -1, 1],
        "E''": [1, -1, 0, -1, 1, 0],
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


def classify_modes(spectrum, tol_mev: float = 0.01, symprec: float = 1e-3) -> tuple[list[str], list[list[int]]]:
    """Classify each phonon mode into irreps of the detected point group.

    Returns (labels, deg_groups): a list of irrep labels, one per mode, and
    the degeneracy groups — lists of mode indices sharing a frequency and
    irrep. Degenerate partners of multi-dimensional irreps end up in the
    same group; accidental degeneracies between different irreps do not.
    """
    pg = detect_point_group_from_spectrum(spectrum, symprec=symprec)
    table = CHARACTER_TABLES.get(pg.symbol)
    if table is None:
        raise NotImplementedError(
            f"No character table for point group '{pg.symbol}'. "
            "Add one to CHARACTER_TABLES."
        )

    ops = defect_frame_operations(spectrum)
    chars = _mode_characters(spectrum, ops)
    labels, deg_groups = _match_irreps(spectrum, chars, table, tol_mev)
    spectrum.symmetries = labels
    return labels, deg_groups


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _defect_frame(spectrum) -> tuple[np.ndarray, np.ndarray]:
    """Returns (axis, reflection_normal) of the defect in Cartesian coords.

    Same convention as the legacy analyze_c3v_symmetry: NV (C+N) has its
    principal axis along [1, 1, 1] with a sigma_v mirror normal [1, -1, 0];
    everything else (e.g. ClV) is taken as [0, 0, 1] / [1, 0, 0].
    """
    symbols = list(spectrum.atom_symbols)
    if "C" in symbols and "N" in symbols:
        return np.array([1.0, 1.0, 1.0]), np.array([1.0, -1.0, 0.0])
    return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])


def _defect_center(spectrum) -> np.ndarray:
    """Fractional coords of the defect center, wrapped into [0, 1).

    N site for NV, Cl-pair midpoint for ClV, else the cell origin. Matches
    PhononSpectrum.translate_defect_to_origin. Callers are responsible for
    periodic wrapping of the result.
    """
    frac = np.mod(np.asarray(spectrum.atom_frac_coords, dtype=float), 1.0)
    symbols_list = list(spectrum.atom_symbols)
    if symbols_list.count("N") == 1:
        return frac[symbols_list.index("N")]
    if "Cl" in symbols_list:
        return frac[np.asarray(symbols_list) == "Cl"].mean(axis=0)
    return np.zeros(3)


def defect_frame_operations(spectrum) -> list[np.ndarray]:
    """Builds the point-group operations in the defect-aligned frame.

    The defect is centered at the origin and the principal axis is taken as
    the group's z axis, so the operations are the standard rotations/reflections
    about that frame. Operations act about the *defect center*, not the cell
    origin: applying them to an off-center structure corrupts the atom mapping
    and hence the characters.
    """
    from beyblade.utils import MathUtils

    axis, normal = _defect_frame(spectrum)
    axis = axis / np.linalg.norm(axis)
    ops = [np.eye(3)]
    ops.append(MathUtils.rotation_around_symmetry_axis(axis, 3))
    ops.append(MathUtils.rotation_around_symmetry_axis(axis, 3).T)  # C3^2
    for sign in (+1.0, -1.0, -1.0):  # three sigma_v mirrors of C3v
        n = normal / np.linalg.norm(normal)
        M = np.eye(3) - 2.0 * np.outer(n, n)
        if sign < 0:  # rotate plane about axis by +-120 deg for the other mirrors
            C = ops[1]
            M = C @ M
        ops.append(M)
    return ops


def _mode_characters(spectrum, operations: list[np.ndarray]) -> np.ndarray:
    """chi_m(R) = <psi_m | R psi_m> for each mode m and operation R.

    R acts on the eigenvector by permuting atoms and rotating their
    displacement vectors. The structure is first centered on the defect
    (PBC-aware, matching the legacy code) and the operations are built in
    the defect-aligned frame, so an off-center defect does not corrupt the
    characters.
    """
    eigs = np.asarray(spectrum.eigenvectors)
    n_modes = eigs.shape[0]
    if eigs.ndim == 3:
        vecs = eigs                     # (n_modes, n_atoms, 3)
    else:
        n_atoms = eigs.shape[-1] // 3   # flat (n_modes, 3N)
        vecs = eigs.reshape(n_modes, n_atoms, 3)

    lattice = np.asarray(spectrum.lattice, dtype=float)
    inv_lat = np.linalg.inv(lattice)
    frac = np.mod(np.asarray(spectrum.atom_frac_coords, dtype=float), 1.0)
    center = _defect_center(spectrum)
    frac = np.mod(frac - center, 1.0)   # PBC-aware centering, defect at origin
    cart_atoms = frac @ lattice
    symbols = np.asarray(spectrum.atom_symbols)

    chars = np.zeros((n_modes, len(operations)))
    for k, R in enumerate(operations):
        R = np.asarray(R, dtype=float)
        mapping = _atom_mapping(R, cart_atoms, frac, inv_lat, symbols, lattice)
        # Legacy convention: chi = trace(eig[mapping] @ (R @ eig.T))
        # expands to sum_i v_{sigma(i)} . R v_i: rotate the original atom's
        # displacement, compare with the mapped atom's.
        rotated = np.einsum("ij,naj->nai", R, vecs)
        chars[:, k] = np.sum(vecs[:, mapping, :] * rotated, axis=(1, 2))
    return chars


def _atom_mapping(R, cart_atoms, frac, inv_lat, symbols, lattice):
    """sigma: for each atom i, index of the atom R sends i onto."""
    n_atoms = cart_atoms.shape[0]
    mapping = np.zeros(n_atoms, dtype=int)
    rotated_cart = cart_atoms @ R.T
    rot_frac = np.mod(rotated_cart @ inv_lat, 1.0)
    for i in range(n_atoms):
        diffs = np.mod(frac - rot_frac[i] + 0.5, 1.0) - 0.5
        dists = np.linalg.norm(diffs @ np.asarray(lattice, dtype=float), axis=1)
        valid = np.where(symbols == symbols[i])[0]
        mapping[i] = valid[np.argmin(dists[valid])] if len(valid) else i
    return mapping


def _parent_label(label: str) -> str:
    """Parent irrep of a (possibly sub-)label: "Ex" -> "E", "E" -> "E"."""
    return label.rstrip("xy") if label not in ("A1", "A2") else label


def _match_irreps(spectrum, chars, table, tol_mev):
    """Assign irrep labels per mode, mirroring the legacy semantics.

    Each mode is matched independently against 1D irreps and the
    single-component character patterns of degenerate (multidimensional)
    irreps. Frequency degeneracy is used only afterwards, to build
    deg_groups: components of the same multidimensional irrep sit at the
    same frequency. Accidental degeneracies between different irreps are
    therefore harmless.
    """
    n = spectrum.n_modes
    freqs = spectrum.frequencies_mev

    # Per-irrep component patterns: 1D irreps appear as-is; for an irrep of
    # dimension d>1 the stored sublabels are the characters of one component.
    patterns: list[tuple[str, np.ndarray]] = []
    parent_of: dict[str, str] = {}
    for name, entry in table.items():
        if isinstance(entry, dict):
            for sub, chi in entry["sublabels"].items():
                patterns.append((sub, np.asarray(chi, dtype=float)))
                parent_of[sub] = name
        else:
            patterns.append((name, np.asarray(entry, dtype=float)))
            parent_of[name] = name

    labels: list[Optional[str]] = []
    parents: list[Optional[str]] = []  # parent irrep of each sublabel
    for i in range(n):
        v = chars[i]
        nv = np.linalg.norm(v) + 1e-12
        best, best_score, best_parent = None, -1.0, None
        for name, chi in patterns:
            score = abs(np.dot(v, chi)) / (nv * (np.linalg.norm(chi) + 1e-12))
            if score > best_score:
                best, best_score, best_parent = name, score, parent_of[name]
        labels.append(best)
        parents.append(best_parent)

    # Degeneracy groups: complete sets of one irrep's sublabels at the same
    # frequency. Modes of the same parent irrep whose sublabels cover each
    # component exactly once form a group; accidental coincidences between
    # different parents never merge. Unpaired members (e.g. a lone Ex whose
    # Ey partner sits beyond tol) are reported as their own group.
    deg_groups: list[list[int]] = []
    by_parent: dict[str, list[int]] = {}
    for i, par in enumerate(parents):
        by_parent.setdefault(par or "?", []).append(i)

    for par, idxs in by_parent.items():
        remaining = list(idxs)
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            # partners: modes sharing the seed's frequency (within tol)
            partners = [
                j for j in remaining
                if abs(freqs[j] - freqs[seed]) < tol_mev
            ]
            # keep at most one partner per distinct sublabel beyond the seed
            taken: set[str] = set()
            for j in partners:
                if labels[j] == labels[seed]:
                    continue  # same sublabel: accidental, not a partner
                if labels[j] in taken:
                    continue
                taken.add(labels[j])
                group.append(j)
            for j in group[1:]:
                remaining.remove(j)
            deg_groups.append(group)
    return labels, deg_groups
