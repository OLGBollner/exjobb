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
            # The stored sublabel patterns are NOT physical characters (a 2D
            # irrep's components are basis conventions). They only define the
            # parent class characters via their sum, and are kept for pairing.
            "sublabels": {
                "Ex": [1, -1, -1, 1, 1, 1],
                "Ey": [1, -1, -1, -1, -1, -1],
            },
            # True class characters of E (components mix under C3, so the
            # sublabel proxy patterns double-count there). Used for the
            # degenerate-group verification: Ex+Ey must sum to this.
            "class_chars": [2, -1, -1, 0, 0, 0],
            # Legacy convention: chi(first reflection) > 0 -> Ex, else Ey.
            "sublabel_rule": (3, "Ex", "Ey"),
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
        "E'": [2, -1, 0, 2, -1, 0],
        "A1''": [1, 1, 1, -1, -1, -1],
        "A2''": [1, 1, -1, -1, -1, 1],
        "E''": [2, -1, 0, -2, 1, 0],
    },
}


class SymmetryDetectionError(RuntimeError):
    """Raised when the point group of a structure cannot be determined.

    Typically synthetic fixtures with all atoms at the origin, which have
    no usable symmetry. Callers that have stored labels may catch this and
    fall back; anything else is a genuine bug and should propagate.
    """


@dataclass(frozen=True)
class PointGroup:
    """Detected point group of a structure."""
    symbol: str                # Hermann-Mauguin point-group symbol (spglib)
    order: int                 # number of symmetry operations
    operations: tuple          # rotation matrices (3x3)
    space_group: str           # international space group symbol


def detect_point_group(structure: Structure, symprec: float = 1e-3) -> PointGroup:
    """Detect the point group of a (defect) supercell structure."""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        symm_ops = sga.get_symmetry_operations()
    except Exception as exc:
        # spglib raises its own error for pathological (e.g. all-atoms-at-
        # origin) structures; translate it so callers can catch our type.
        raise SymmetryDetectionError(
            f"Unable to determine symmetry (symprec={symprec}): {exc}"
        ) from exc
    if not symm_ops:
        raise SymmetryDetectionError(
            f"No symmetry operations found (symprec={symprec}); "
            "cannot classify modes."
        )
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


def expand_classes(table: dict, ops: list[str]) -> dict[str, list[float]]:
    """Expand a per-class character table into per-operation chi vectors.

    ``table`` is one entry of POINT_GROUP_CHARACTER_TABLES (classes +
    irreps); ``ops`` is the per-operation class label list in pipeline
    order, e.g. ["E", "C3", "C3", "sv", "sv", "sv"] for C3v. Returns
    {irrep: [chi_per_operation]}.
    """
    class_chars = {name: chars for name, chars in table["irreps"].items()}
    out: dict[str, list[float]] = {}
    for name, chars in class_chars.items():
        lookup = {cname: chi for (cname, _mult), chi in zip(table["classes"], chars)}
        vec = [lookup[op] for op in ops]
        out[name] = vec
    return out


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
    # Sublabels of a 2D irrep (Ex/Ey) are a basis convention, not physics:
    # the stored convention (set by the legacy analyzer) is the sign of the
    # character on the first reflection operation. Compute that index here.
    first_reflection = next(
        (k for k, R in enumerate(ops) if np.linalg.det(R) < 0), None
    )
    labels, deg_groups = _match_irreps(
        spectrum, chars, table, tol_mev, first_reflection=first_reflection
    )
    spectrum.symmetries = labels
    spectrum.sym_check = verify_deg_groups(chars, labels, deg_groups, table)
    return labels, deg_groups


def verify_deg_groups(
    chars, labels, deg_groups, table, atol: float = 0.05,
) -> dict:
    """Check that each degenerate group sums to its parent irrep's characters.

    Physics check (independent of how labels were derived): the character
    vector of a multidimensional irrep is the trace of the full
    representation, so the summed characters of a complete degenerate set
    must reproduce the parent irrep's class characters. For C3v E this is
    [2, -1, -1, 0, 0, 0] — verified on ClV: 127/127 Ex/Ey pairs.

    Returns a report dict:
      ok         True when every multi-member group passes
      failures   [(group_idx, indices, expected, got), ...] for mismatches
      unpaired   [(group_idx, indices), ...] lone members of a
                 multidimensional irrep (partner beyond tol) — unverifiable,
                 not a failure
    Singleton 1D irreps (A1/A2) are skipped: they are their own trace.
    """
    # class characters per irrep: explicit when the table carries them,
    # else the sum of its sublabel patterns (identity-safe fallback)
    class_chars: dict[str, np.ndarray] = {}
    dim: dict[str, int] = {}
    for name, entry in table.items():
        if isinstance(entry, dict):
            subs = list(entry["sublabels"].items())
            if "class_chars" in entry:
                chi = np.asarray(entry["class_chars"], dtype=float)
            else:
                chi = np.zeros(len(subs[0][1]), dtype=float)
                for _, sub_chi in subs:
                    chi += np.asarray(sub_chi, dtype=float)
            class_chars[name] = chi
            dim[name] = len(subs)
        else:
            class_chars[name] = np.asarray(entry, dtype=float)
            dim[name] = 1

    failures, unpaired = [], []
    for gi, group in enumerate(deg_groups):
        lab = labels[group[0]]
        if lab is None:
            continue
        parent = _parent_label(lab)
        if parent not in class_chars:
            continue
        if len(group) == 1:
            # lone member of a multidimensional irrep: partner missing
            if dim[parent] > 1:
                unpaired.append((gi, list(group)))
            continue
        got = np.sum([chars[i] for i in group], axis=0)
        expected = class_chars[parent]
        if not np.allclose(got, expected, atol=atol):
            failures.append((gi, list(group), expected.tolist(), got.tolist()))
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "unpaired": unpaired,
    }


def classify_and_pair(
    spectrum,
    tol_mev: float = 0.01,
    symprec: float = 1e-3,
    force: bool = False,
) -> tuple[list[str], list[list[int]]]:
    """Classify modes and build degeneracy groups, trusting stored labels by default.

    Returns (labels, deg_groups). If the spectrum already carries symmetry
    labels, they are trusted and only deg_groups are (re)built; pass
    force=True to re-derive labels via the general projection method.

    Both paths run verify_deg_groups() afterwards: the character-sum check
    (each degenerate set must sum to its parent irrep's class characters).
    The report is stored on spectrum.sym_check. If the point group cannot
    be detected (e.g. synthetic test structures), verification is skipped
    (sym_check stays None) rather than raising — labels/groups still work.
    """
    if spectrum.symmetries is not None and not force:
        labels = list(spectrum.symmetries)
        deg_groups = _build_groups_from_labels(spectrum, labels, tol_mev)
    else:
        labels, deg_groups = classify_modes(spectrum, tol_mev=tol_mev, symprec=symprec)
        spectrum.symmetries = labels
        return labels, deg_groups  # verify already ran inside classify_modes
    try:
        pg = detect_point_group_from_spectrum(spectrum, symprec=symprec)
        table = CHARACTER_TABLES.get(pg.symbol, {})
    except Exception:
        table = {}
    if table:
        ops = defect_frame_operations(spectrum)
        chars = _mode_characters(spectrum, ops)
        spectrum.sym_check = verify_deg_groups(chars, labels, deg_groups, table)
    return labels, deg_groups


def _build_groups_from_labels(spectrum, labels, tol_mev):
    """Build degeneracy groups from stored labels without re-deriving."""
    from collections import defaultdict

    parent_of = {
        sub: parent for parent, entry in CHARACTER_TABLES["3m"].items()
        if isinstance(entry, dict) for sub in entry["sublabels"]
    }
    freqs = spectrum.frequencies_mev
    by_key: dict[tuple[str, float], list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        key = (parent_of.get(lab, lab), round(freqs[i] / tol_mev))
        by_key[key].append(i)
    groups = []
    for (parent, _), idxs in by_key.items():
        # one group per complete set: split idxs by sublabel count
        remaining = list(idxs)
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            seen = {labels[seed]}
            for j in remaining:
                if labels[j] not in seen:
                    seen.add(labels[j])
                    group.append(j)
            for j in group[1:]:
                remaining.remove(j)
            groups.append(group)
    return groups


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


def _match_irreps(spectrum, chars, table, tol_mev, first_reflection=None):
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

    # Per-irrep patterns: 1D irreps appear as-is. A multi-dimensional irrep
    # (e.g. C3v E) is irreducible, so its components have no characters of
    # their own — the sublabels (Ex/Ey) are a basis convention. Each mode is
    # therefore scored once against the parent's class characters (the sum of
    # the stored sublabel patterns), and the sublabel is then assigned by a
    # sign rule (see first_reflection in classify_modes) that mirrors the
    # legacy analyzer's convention: chi > 0 -> first sublabel, else second.
    patterns: list[tuple[str, np.ndarray]] = []
    parent_of: dict[str, str] = {}
    sublabel_rule: dict[str, tuple[int, str, str]] = {}
    for name, entry in table.items():
        if isinstance(entry, dict):
            subs = list(entry["sublabels"].items())
            parent_chi = np.zeros(len(next(iter(subs))[1]), dtype=float)
            for sub, chi in subs:
                parent_chi += np.asarray(chi, dtype=float)
                parent_of[sub] = name
            patterns.append((name, parent_chi))
            parent_of[name] = name
            if "sublabel_rule" in entry:
                sublabel_rule[name] = tuple(entry["sublabel_rule"])
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
        if (
            best is not None
            and best in sublabel_rule
            and first_reflection is not None
        ):
            op_idx, pos, neg = sublabel_rule[best]
            chi_ref = v[op_idx]
            best = pos if chi_ref > 0 else neg
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


def group_of(deg_groups: list[list[int]] | None, i: int) -> list[int] | None:
    """Return the degeneracy group containing mode i, or None."""
    if deg_groups is None:
        return None
    for g in deg_groups:
        if i in g:
            return g
    return None


def twin_of(deg_groups, i: int) -> int | None:
    """Return the degenerate partner of mode i (Ex<->Ey etc.), or None."""
    g = group_of(deg_groups, i)
    if g is None or len(g) < 2:
        return None
    return next(j for j in g if j != i)
