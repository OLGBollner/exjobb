"""Symmetry projection of spin operators and ZFS derivative tensors.

Two sides of the spin-phonon Hamiltonian meet here:

    H_s-ph = sum_{Gamma,mu} V_{Gamma,mu} F_{Gamma,mu}

Spin side
---------
``rank2_spin_operators(S)`` builds the quadratic spin operators from the
general-S ladder formulas (thesis eqs.; hold for any S, not just S=1):

    F_z  = 3 (S_z^2 - S(S+1)/3)
    F_x  = (S_+^2 + S_-^2)/2          (E, |m|=2 pair)
    F_y  = -i (S_+^2 - S_-^2)/2
    F_x' = sqrt(2) {S_x, S_z}         (E, |m|=1 pair)
    F_y' = sqrt(2) {S_y, S_z}

For S=1 these are exactly the thesis matrices (validation anchor, see
tests/test_projection.py).

Tensor side
-----------
The derivative tensor dD/dQ_l (3x3, symmetric, defect frame) is decomposed
into irreducible rank-2 pieces. For C3v the real tensor basis reproducing
the Appendix-A channel normalization is:

    A1: B_A1        = diag(-1/2, -1/2, 1) / 3
    E (|m|=1):      (xz+zx)/(2*sqrt2), (yz+zy)/(2*sqrt2)   -> V_0pm
    E (|m|=2):      (xx-yy)/2, (xy+yx)/2                   -> V_pm

These scalings are not arbitrary: they are fixed by the spin-operator
normalization (e.g. F_z = 3 S_z^2 - S(S+1) 1 means the A1 coefficient is
(dDzz - (dDxx+dDyy)/2)/3), so contractions with this basis give channel
values V that multiply the F matrices directly.

``project_tensor`` does the same decomposition generically: character
projection on the 6-dim space of symmetric 3x3 tensors, then contraction
with the physics-normalized basis above (valid whenever the defect frame
aligns with the group's canonical z axis, which holds for ZFS tensors by
construction).
"""
import numpy as np

from beyblade.spin import spin_raise, spin_lower, spin_z, spin_x, spin_y
from beyblade.symmetry import CHARACTER_TABLES

# ---------------------------------------------------------------------------
# Spin side
# ---------------------------------------------------------------------------


def rank2_spin_operators(S: float) -> dict:
    """Quadratic (rank-2) spin operators for arbitrary S, thesis definitions."""
    dim = int(round(2 * S + 1))
    Sp, Sm = spin_raise(S), spin_lower(S)
    Sz, Sx, Sy = spin_z(S), spin_x(S), spin_y(S)
    return {
        "F_z": 3.0 * (Sz @ Sz - S * (S + 1) / 3.0 * np.eye(dim)),
        "F_x": 0.5 * (Sp @ Sp + Sm @ Sm),
        "F_y": -0.5j * (Sp @ Sp - Sm @ Sm),
        "F_xp": np.sqrt(2.0) * (Sx @ Sz + Sz @ Sx),
        "F_yp": np.sqrt(2.0) * (Sy @ Sz + Sz @ Sy),
    }


# ---------------------------------------------------------------------------
# Tensor side
# ---------------------------------------------------------------------------

# Physics-normalized real rank-2 tensor basis (C3v, defect frame along z).
# Each entry: (irrep, component tensor). Frobenius contraction Tr(B^T dD)
# gives the channel coefficient that multiplies the matching F operator.
_TENSOR_BASIS = {
    "A1": [np.diag([-0.5, -0.5, 1.0]) / 3.0],
    "E1": [np.array([[0, 0, 1 / (2 * np.sqrt(2))],
                     [0, 0, 0],
                     [1 / (2 * np.sqrt(2)), 0, 0]]),
           np.array([[0, 0, 0],
                     [0, 0, 1 / (2 * np.sqrt(2))],
                     [0, 1 / (2 * np.sqrt(2)), 0]])],
    "E2": [np.diag([0.5, -0.5, 0.0]),
           np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]])],
}


def c3v_channel_components(dD: np.ndarray, label: str):
    """Appendix-A channel values for one mode with symmetry label `label`.

    label in {"A1", "Ex", "Ey"}. Returns the |contraction| in the same
    normalization as zfs_manager.calculate_first_order_derivatives, so the
    legacy path and the new path agree exactly.
    """
    dD = np.asarray(dD, dtype=float)
    if label == "A1":
        return abs(np.trace(_TENSOR_BASIS["A1"][0].T @ dD))
    if label in ("E", "Ex", "Ey"):
        E2 = [np.trace(B.T @ dD) for B in _TENSOR_BASIS["E2"]]
        E1 = [np.trace(B.T @ dD) for B in _TENSOR_BASIS["E1"]]
        return {
            # components: c1 = (dDxx-dDyy)/2, c2 = dDxy
            "V_p_m": np.hypot(E2[0], E2[1]),
            "V_0_pm": np.hypot(*E1),
        }
    raise ValueError(f"unknown C3v label: {label}")


def _transform(T, R):
    """g . T . g^T."""
    return R @ T @ R.T


_F_TENSORS = {
    "F_z": np.diag([-1.0, -1.0, 2.0]),
    "F_x": np.diag([1.0, -1.0, 0.0]),
    "F_y": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    "F_xp": np.sqrt(2.0) * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
    "F_yp": np.sqrt(2.0) * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
}
# F operator -> point-group irrep for C3v: F_z alone in A1; (F_x, F_y) are
# the |m|=2 E copy (in-plane quadrupole), (F_xp, F_yp) the |m|=1 copy.
# The plain "E" key is used when the character table names the irrep E.
_F_IRREPS = {
    "A1": ["F_z"],
    "E2": ["F_x", "F_y"],
    "E1": ["F_xp", "F_yp"],
    "E": ["F_x", "F_y", "F_xp", "F_yp"],
}


def _channel_values(dD: np.ndarray, names) -> np.ndarray:
    """Contractions Tr(f dD)/||f||^2 for the named F tensors."""
    dD = np.asarray(dD, dtype=float)
    vals = []
    for name in names:
        f = _F_TENSORS[name]
        vals.append(float(np.sum(f * dD) / np.sum(f * f)))
    return np.array(vals)


def characters_for_group(symbol: str) -> dict[str, list[float]]:
    """Flatten symmetry.CHARACTER_TABLES into {irrep: class_chars}.

    Uses the same operation order as symmetry._operation_keys. Entries that
    are dicts (2D irreps with sublabel machinery) expose 'class_chars'.
    """
    from beyblade.symmetry import _TABLE_META_KEYS

    raw = CHARACTER_TABLES[symbol]
    out = {}
    for irrep, val in raw.items():
        if irrep in _TABLE_META_KEYS:
            continue
        out[irrep] = val["class_chars"] if isinstance(val, dict) else val
    return out

# Symmetric 3x3 monomial basis used by the generic projector. Coordinates:
# (xx, yy, zz, xy, xz, yz); each monomial is the symmetrized matrix, so a
# tensor T = sum_k a_k M_k has a_xy = T_xy etc.
_MONOMIALS = {
    "xx": np.diag([1.0, 0, 0]),
    "yy": np.diag([0, 1.0, 0]),
    "zz": np.diag([0, 0, 1.0]),
    "xy": np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
    "xz": np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]]),
    "yz": np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]]),
}


def _to_monomial_coeffs(dD):
    """Coefficients a with dD = sum_k a_k B_k (monomials mutually orthogonal)."""
    b = np.array([np.sum(B * dD) for B in _MONOMIALS.values()])
    n2 = np.array([np.sum(B * B) for B in _MONOMIALS.values()])
    return b / n2


def _transform(T, R):
    """g . T . g^T."""
    return R @ T @ R.T


def _group_action(operations):
    """Monomial-coefficient transfer matrices: T = sum_j a_j B_j (a in monomial
    coords), then gTg^T = sum_k a'_k B_k with a' = M_g @ a, where
    M_g[j,k] = <B_j, g B_k g^T> / <B_k, B_k> (monomials are not orthonormal)."""
    norms2 = np.array([np.sum(B * B) for B in _MONOMIALS.values()])
    Ms = []
    for R in operations:
        W = np.array([[np.sum(Bj * _transform(Bk, R))
                       for Bk in _MONOMIALS.values()]
                      for Bj in _MONOMIALS.values()])
        Ms.append(W / norms2[None, :])
    return np.array(Ms)  # (n_g, 6, 6)


def _character_projector(Ms, chi):
    """P = (d/|G|) sum_g chi(g)^* M_g, as a 6x6 matrix (d = chi[E])."""
    d = chi[0]
    P = np.zeros((6, 6))
    for g_idx, M in enumerate(Ms):
        P += np.conj(chi[g_idx]) * M
    return d * P / len(Ms)


def _irrep_subspace(P):
    """Orthonormal basis (columns) of the range of projector P."""
    w, V = np.linalg.eigh(P)
    keep = w > 1e-9
    return V[:, keep], int(keep.sum())


def _split_e_copies(P):
    """Split a doubly-occurring E subspace by O(3) parent structure.

    Group theory alone cannot separate two isomorphic E copies (any
    G-equivariant map is block-scalar), but the O(3) parent does: the
    m=1 copy (|m|=1) carries exactly one z index (xz, yz monomials), the
    m=2 copy none (xx-yy, xy). We project those monomial seeds through the
    character projector and orthonormalize within each copy.

    Returns two real matrices whose columns are orthonormal copy bases:
    (m=1 copy, m=2 copy).
    """
    seeds_m1 = [np.array([0, 0, 0, 0, 1, 0], float),
                np.array([0, 0, 0, 0, 0, 1], float)]
    seeds_m2 = [np.array([1, -1, 0, 0, 0, 0], float),
                np.array([0, 0, 0, 1, 0, 0], float)]

    def orthonormalize(seeds):
        basis = []
        for s in seeds:
            v = P @ s
            for u in basis:
                v = v - (u @ v) * u
            n = np.linalg.norm(v)
            if n > 1e-10:
                basis.append(v / n)
        return np.array(basis).T if basis else np.zeros((6, 0))

    return orthonormalize(seeds_m1), orthonormalize(seeds_m2)


def project_tensor(dD: np.ndarray, operations, characters=None,
                   spin_operators=None) -> dict:
    """Character-project a symmetric 3x3 tensor into irrep channel values.

    operations: list of 3x3 matrices of the point group (defect frame).
    characters: {irrep: [chi(g)]} aligned with operations; defaults to the
    C3v table when len(operations) == 6.
    spin_operators: {name: F matrix}; when given, the irrep assignment of
    each F is verified by projecting its tensor f (defaults to the C3v
    assignment in _F_IRREPS).

    Returns {irrep: value or [values]}: channel coefficients in the
    physics normalization V_mu = Tr(f_mu dD)/||f_mu||^2 that multiplies
    the matching F operators. For C3v: A1 -> V_0_0, E -> [V_0pm, V_pm].
    """
    dD = np.asarray(dD, dtype=float)
    if characters is None:
        if len(operations) != 6:
            raise ValueError("characters required for non-C3v groups")
        characters = C3V_CLASS_CHARS
    Ms = _group_action(operations)

    # Assign the F tensors to irreps (verifies or derives the pairing).
    if spin_operators is None:
        f_assign = {irrep: names for irrep, names in _F_IRREPS.items()
                    if irrep in characters}
    else:
        f_assign = _assign_f_to_irreps(spin_operators, Ms, characters)

    out = {}
    for irrep, f_names in f_assign.items():
        if len(f_names) > 1:
            # Doubly-occurring irrep: one value per copy.
            P = _character_projector(Ms, characters[irrep])
            copy1, copy2 = _split_e_copies(P)
            a_f = {n: _to_monomial_coeffs(_F_TENSORS[n]) for n in f_names}
            copies = []
            for cb in (copy1, copy2):
                members = [n for n, af in a_f.items()
                           if np.linalg.norm(cb.T @ af) > 1e-8 * np.linalg.norm(af)]
                if not members:
                    raise ValueError(f"no F operators project onto copy of {irrep}")
                copies.append(np.linalg.norm(_channel_values(dD, members)))
            out[irrep] = copies
        else:
            out[irrep] = abs(_channel_values(dD, f_names)[0])

    # Cross-check the character projector agrees on subspace structure.
    for irrep, chi in characters.items():
        P = _character_projector(Ms, chi)
        sub, rank = _irrep_subspace(P)
        if irrep in out and rank == 0:
            raise ValueError(f"projector says {irrep} is empty but F "
                             f"assignment gives a channel")
    return out


def _assign_f_to_irreps(spin_operators, Ms, characters):
    """Project each F tensor through the character projectors and keep the
    irrep(s) where it has support. E copies are kept together (the
    component pairing within E follows the mode's internal basis)."""
    f_assign = {}
    for irrep, chi in characters.items():
        P = _character_projector(Ms, chi)
        names = []
        for name, F in spin_operators.items():
            f = _F_TENSORS.get(name)
            if f is None:
                raise ValueError(f"no tensor representation for {name}")
            a = _to_monomial_coeffs(f)
            if np.linalg.norm(P @ a) > 1e-8 * np.linalg.norm(a):
                names.append(name)
        f_assign[irrep] = names
    return f_assign


C3V_CLASS_CHARS = characters_for_group("3m")
